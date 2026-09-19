"""What a benchmark item is, what it should cost, and where the run stands.

Three things live here and nothing else:

* :class:`Item` -- one job: a spec, an identity, and the question it answers.
* :class:`Calibration` -- the cost model, which starts from the figures
  measured in CLAUDE.md and refits itself from every completed item.
* :class:`State` -- what has run, written after every item so that killing the
  process loses at most the item in flight.

Stdlib only. ``bench plan``, ``bench status`` and ``bench report`` all import
this and must stay fast.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
SPECS_DIR = BENCH_DIR / "specs"
RUNS_DIR = BENCH_DIR / "runs"
STATE_FILE = BENCH_DIR / "state.json"
PLAN_FILE = BENCH_DIR / "plan.json"
CALIBRATION_FILE = BENCH_DIR / "calibration.json"

# Measured on the target machine (CLAUDE.md, "The hardware this was built for"):
# 23-26 s per step at 1344x768 and 10-13 s at 768x768. Those are 1.032 and
# 0.590 megapixels, so the cost per megapixel-step is not constant -- 23.7 at
# the larger size against 19.5 at the smaller. The fit below is deliberately
# crude: it only has to be close enough to order the list and to give a first
# ETA, and it is replaced by measurement as soon as anything finishes.
SECONDS_PER_MPX_STEP = 23.0
# Loading a checkpoint: ~40 s for SDXL base, 30-90 s for a single-file model.
DEFAULT_LOAD_S = 60.0
# One VAE decode, GPU tiled, at roughly one megapixel.
DEFAULT_DECODE_S_PER_MPX = 25.0

STATUSES = ("pending", "running", "paused", "done", "failed", "skipped")


def _atomic_write(path: Path, payload: str) -> None:
    """Write via a temporary file in the same directory, then replace.

    The state file is rewritten after every item and read by ``bench status``
    while a run is going. A half-written state file would lose the whole run's
    bookkeeping, and this is a multi-week run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One job in the benchmark: a spec to render and the question it answers."""

    id: str
    phase: str
    wave: str
    title: str
    why: str
    spec: dict[str, Any]
    model: str
    # Items sharing a ``group`` are meant to be looked at side by side; the
    # report lays them out together and diffs their measurements.
    group: str
    # Free-form columns for the CSV, e.g. {"axis": "vae_decode", "value": "cpu"}.
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def spec_path(self) -> Path:
        return SPECS_DIR / f"{self.id}.json"

    @property
    def bundle_dir(self) -> Path:
        return RUNS_DIR / self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Item":
        return cls(**data)

    # -- cost ---------------------------------------------------------------

    def stages(self) -> list[tuple[str, int, float]]:
        """``(stage, steps, megapixels)`` for each stage this item will run.

        Mirrors what the compiler resolves rather than calling it: importing
        the compiler here would make ``bench plan`` load pydantic and the
        vocabulary for a number that only has to be approximately right.
        """
        render = self.spec.get("render", {})
        steps = int(render.get("steps") or self._intent_steps())
        width, height = self.size()
        mpx = width * height / 1e6
        plan: list[tuple[str, int, float]] = [("base", steps, mpx)]

        refiner = self.spec.get("refiner", {})
        hires = self.spec.get("hires", {})
        maximum = render.get("quality") == "max"
        if refiner.get("enabled", maximum and "enabled" not in refiner):
            handoff = float(refiner.get("handoff", 0.8))
            plan[0] = ("base", max(1, round(steps * handoff)), mpx)
            plan.append(("refiner", max(1, round(steps * (1.0 - handoff))), mpx))
        if hires.get("enabled", maximum and "enabled" not in hires):
            scale = float(hires.get("scale", 1.5 if maximum else 1.5))
            strength = float(hires.get("strength", 0.3))
            hires_steps = hires.get("steps") or max(1, int(steps * strength))
            plan.append(("hires", int(hires_steps), mpx * scale * scale))
        return plan

    def _intent_steps(self) -> int:
        # The intent defaults from vocabulary/intents.yaml, inlined for the
        # same reason stages() does not call the compiler.
        return {
            "photoreal": 32, "painterly": 34, "game_asset": 30,
            "render3d": 32, "graphic": 30,
        }.get(self.spec.get("intent", "photoreal"), 32)

    def size(self) -> tuple[int, int]:
        render = self.spec.get("render", {})
        if render.get("width") and render.get("height"):
            return int(render["width"]), int(render["height"])
        aspect = self.spec.get("composition", {}).get("aspect", "1:1")
        return ASPECT_SIZES.get(aspect, (1024, 1024))

    @property
    def variations(self) -> int:
        return int(self.spec.get("render", {}).get("variations", 1))

    def work_units(self) -> float:
        """Megapixel-steps, the unit the progress bar counts in."""
        per = sum(steps * mpx for _stage, steps, mpx in self.stages())
        return per * self.variations


# The SDXL aspect buckets, as spec.py resolves them.
ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024), "5:4": (1152, 896), "4:5": (896, 1152),
    "3:2": (1216, 832), "2:3": (832, 1216), "16:9": (1344, 768),
    "9:16": (768, 1344), "21:9": (1536, 640), "9:21": (640, 1536),
}


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Seconds per step, learned from this machine as the run goes.

    Keyed by ``"{width}x{height}|{precision}"``, because the relationship
    between pixels and seconds is not linear here and the measured points are
    worth more than any curve fitted through them. Sizes never yet run fall
    back to the megapixel rate, itself refitted from everything measured.
    """

    per_size: dict[str, list[float]] = field(default_factory=dict)
    load_s: list[float] = field(default_factory=list)
    decode_s_per_mpx: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "Calibration":
        if not path.exists():
            return cls()
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a corrupt cache must not stop a run
            return cls()

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        _atomic_write(path, json.dumps(asdict(self), indent=1))

    @staticmethod
    def _key(width: int, height: int, precision: str) -> str:
        return f"{width}x{height}|{precision}"

    @staticmethod
    def _recent(values: list[float], keep: int = 12) -> float:
        """The median of the most recent readings.

        A median because a paused-and-resumed stage records one enormous first
        step; recent only, because the machine throttles as it warms up and an
        ETA should follow that rather than average it away.
        """
        return statistics.median(values[-keep:])

    def observe_step(self, width: int, height: int, precision: str, seconds: float) -> None:
        self.per_size.setdefault(self._key(width, height, precision), []).append(seconds)

    def observe_load(self, seconds: float) -> None:
        if seconds > 1.0:  # a warm, already-resident model is not a load
            self.load_s.append(seconds)

    def observe_decode(self, mpx: float, seconds: float) -> None:
        if mpx > 0:
            self.decode_s_per_mpx.append(seconds / mpx)

    def _points(self) -> list[tuple[float, float]]:
        """``(megapixels, seconds)`` for every size measured, one point each."""
        points = []
        for key, times in self.per_size.items():
            if not times:
                continue
            width, height = key.split("|")[0].split("x")
            points.append((int(width) * int(height) / 1e6, self._recent(times)))
        return points

    def fit(self) -> tuple[float, float]:
        """``(intercept, slope)`` seconds against megapixels, by least squares.

        An affine fit rather than a proportional one, because a sampler step
        has a fixed cost that does not scale with pixels, and extrapolating
        proportionally from a small measurement badly overestimates a large
        one. Measured here: the first 640x640 reading alone projected the whole
        benchmark at 92 days against an expected 64, purely from that.

        One measured size cannot separate the intercept from the slope, so
        until there are two the documented rate stands.
        """
        points = self._points()
        if len({round(mpx, 4) for mpx, _s in points}) < 2:
            return 0.0, SECONDS_PER_MPX_STEP
        mean_x = statistics.fmean(x for x, _y in points)
        mean_y = statistics.fmean(y for _x, y in points)
        spread = sum((x - mean_x) ** 2 for x, _y in points)
        if spread <= 0:
            return 0.0, SECONDS_PER_MPX_STEP
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
        if slope <= 0:
            # Noise, or a big render that happened to run while the machine was
            # idle. A negative cost per pixel is never the right answer.
            return 0.0, SECONDS_PER_MPX_STEP
        return mean_y - slope * mean_x, slope

    def mpx_rate(self) -> float:
        """Seconds per megapixel-step at one megapixel, for reporting."""
        intercept, slope = self.fit()
        return intercept + slope

    def step_seconds(self, width: int, height: int, precision: str) -> float:
        measured = self.per_size.get(self._key(width, height, precision))
        if measured:
            return self._recent(measured)
        intercept, slope = self.fit()
        seconds = max(0.1, intercept + slope * width * height / 1e6)
        # fp32 was never measured on this card. CLAUDE.md is explicit that
        # fp16 buys memory and not speed on a card with no tensor cores, so
        # the penalty guessed here is the offload change, not the arithmetic.
        return seconds * (2.5 if precision == "fp32" else 1.0)

    def load_seconds(self) -> float:
        return self._recent(self.load_s) if self.load_s else DEFAULT_LOAD_S

    def decode_seconds(self, mpx: float) -> float:
        rate = (
            self._recent(self.decode_s_per_mpx)
            if self.decode_s_per_mpx
            else DEFAULT_DECODE_S_PER_MPX
        )
        return rate * mpx

    def estimate(self, item: Item, warm: bool = False) -> float:
        """Wall-clock seconds this item should take."""
        precision = item.spec.get("render", {}).get("precision") or "fp16"
        total = 0.0 if warm else self.load_seconds()
        # With a refiner the base is released, the refiner loaded, then the
        # base loaded again: three loads rather than one.
        stages = item.stages()
        if any(stage == "refiner" for stage, _s, _m in stages) and not warm:
            total += self.load_seconds() * 2
        for _stage, steps, mpx in stages:
            width = height = int((mpx * 1e6) ** 0.5)
            total += steps * self.step_seconds(width, height, precision) * item.variations
        # One decode per variation, plus the hi-res pass's extra decode/encode.
        last_mpx = stages[-1][2]
        total += self.decode_seconds(last_mpx) * item.variations
        if any(stage == "hires" for stage, _s, _m in stages):
            total += self.decode_seconds(stages[0][2]) * 2 * item.variations
        return total


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class Record:
    """What happened to one item."""

    id: str
    status: str = "pending"
    attempts: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: float = 0.0
    images: int = 0
    bundle_dir: Optional[str] = None
    error: Optional[str] = None
    # Copied out of the bundle so the report and the ETA need not reopen it.
    s_per_step: Optional[float] = None
    decode: Optional[str] = None
    ram_free_start_gb: Optional[float] = None
    ram_free_min_gb: Optional[float] = None
    vram_peak_gb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class State:
    """The run's bookkeeping, rewritten after every item.

    A state remembers the file it came from, so ``save()`` with no argument
    goes back where ``load()`` found it. That is not a convenience: the runner
    saves from several places, and a default pointing at the module-level path
    means a state loaded from anywhere else silently overwrites the real run's
    bookkeeping. A test doing exactly that is how this was found.
    """

    def __init__(
        self, records: Optional[dict[str, Record]] = None, path: Path = STATE_FILE
    ) -> None:
        self.records: dict[str, Record] = records or {}
        self.path = path

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "State":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({key: Record(**value) for key, value in data.items()}, path=path)

    def save(self, path: Optional[Path] = None) -> None:
        payload = {key: record.to_dict() for key, record in sorted(self.records.items())}
        _atomic_write(path or self.path, json.dumps(payload, indent=1))

    def get(self, item_id: str) -> Record:
        return self.records.setdefault(item_id, Record(id=item_id))

    def status(self, item_id: str) -> str:
        record = self.records.get(item_id)
        return record.status if record else "pending"

    def is_finished(self, item_id: str) -> bool:
        return self.status(item_id) in {"done", "skipped"}

    def outstanding(self, items: Iterable[Item]) -> list[Item]:
        """Items still to run, in list order, resumable ones first.

        A paused item goes to the front of its own position rather than the
        front of the queue: the list is ordered for a reason, and a resume
        that jumped the whole list would break the phase ordering that makes
        stopping early safe.
        """
        return [item for item in items if not self.is_finished(item.id)]


def load_plan(path: Path = PLAN_FILE) -> list[Item]:
    if not path.exists():
        raise FileNotFoundError(
            f"no plan at {path}. Run:  python -m bench plan"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Item.from_dict(entry) for entry in data["items"]]


def save_plan(items: list[Item], path: Path = PLAN_FILE) -> None:
    payload = {
        "version": 1,
        "items": [item.to_dict() for item in items],
    }
    _atomic_write(path, json.dumps(payload, indent=1, ensure_ascii=False))


__all__ = [
    "ASPECT_SIZES",
    "BENCH_DIR",
    "CALIBRATION_FILE",
    "Calibration",
    "Item",
    "PLAN_FILE",
    "ROOT",
    "RUNS_DIR",
    "Record",
    "SPECS_DIR",
    "STATE_FILE",
    "State",
    "load_plan",
    "save_plan",
]
