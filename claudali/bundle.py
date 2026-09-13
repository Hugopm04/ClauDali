"""Writing a render to disk as a self-describing bundle.

A bundle is the unit the refine loop operates on. It holds the full-size images,
a small preview of each, a contact sheet when there is more than one variation,
the exact spec and seeds, and the diagnostics -- everything needed to judge a
result and produce the next spec from it.

The previews are not a convenience. A 1024x1024 PNG is expensive for an agent to
read and mostly redundant: composition, colour and gross errors are all legible
at 512px. The full-size file stays on disk for when it is actually wanted.

A bundle is written *as the render goes*. Each variation lands on disk the
moment it is decoded and ``result.json`` is rewritten with a ``status`` every
time, so a crash, a Ctrl+C or a pause never costs a finished image. While a job
can still be resumed, the bundle also holds a ``checkpoint/`` folder; see
``engine/checkpoint.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

from .compiler import CompiledPrompt
from .compose import finish, load_font
from .config import OUTPUTS_DIR, SETTINGS
from .diagnostics import analyse
from .engine.checkpoint import (
    RenderStopped,
    ResumeState,
    delete_checkpoint,
    mark_checkpoint,
    read_state,
    save_checkpoint,
    summarize_state,
    write_json,
)
from .engine.render import RenderedImage, RenderResult
from .spec import SceneSpec, load_spec

CONTACT_SHEET_MAX_COLUMNS = 4
CONTACT_SHEET_CELL = 384
LABEL_HEIGHT = 26

# What ``result.json``'s ``status`` can say. "aborted" was stopped hard and can
# still be resumed from the start of the variation it was on; "cancelled" was
# stopped on request and cannot.
STATUSES = ("running", "paused", "done", "aborted", "cancelled", "error")


@dataclass
class VariationRecord:
    """One rendered image as it exists on disk."""

    index: int
    seed: int
    image: str
    preview: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    """The on-disk result of one job."""

    job_id: str
    directory: str
    spec: dict[str, Any]
    compiled: dict[str, Any]
    variations: list[VariationRecord]
    contact_sheet: Optional[str] = None
    control_image: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    task: str = "txt2img"
    device: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    # Bundles written before pausing existed carry no status and were complete.
    status: str = "done"
    checkpoint: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bundle":
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        values["variations"] = [VariationRecord(**item) for item in data.get("variations", [])]
        return cls(**values)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _preview_path(image_path: Path, transparent: bool) -> Path:
    return image_path.with_suffix(".preview.png" if transparent else ".preview.jpg")


def _write_preview(image: Image.Image, path: Path) -> Path:
    """Downscale for cheap reading, preserving alpha only when it exists."""
    transparent = image.mode == "RGBA"
    target = _preview_path(path, transparent)

    scale = SETTINGS.preview_max_side / max(image.size)
    if scale < 1.0:
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        preview = image.resize(size, Image.LANCZOS)
    else:
        preview = image.copy()

    if transparent:
        preview.save(target, format="PNG", optimize=True)
    else:
        preview.convert("RGB").save(target, format="JPEG", quality=85, optimize=True)
    return target


def _build_contact_sheet(
    images: list[tuple[int, int, Image.Image]], path: Path
) -> Optional[Path]:
    """A labelled grid of every variation, for judging a batch in one look.

    Each cell carries its index and seed, which is what makes "render 3 again at
    higher steps" expressible without opening files to find the seed.
    """
    if len(images) < 2:
        return None

    columns = min(CONTACT_SHEET_MAX_COLUMNS, len(images))
    rows = math.ceil(len(images) / columns)
    cell_w = CONTACT_SHEET_CELL
    cell_h = CONTACT_SHEET_CELL + LABEL_HEIGHT

    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = load_font(None, 15)

    for position, (index, seed, image) in enumerate(images):
        column, row = position % columns, position // columns
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((cell_w - 8, CONTACT_SHEET_CELL - 8), Image.LANCZOS)

        x = column * cell_w + (cell_w - thumb.width) // 2
        y = row * cell_h + (CONTACT_SHEET_CELL - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        draw.text(
            (column * cell_w + 6, row * cell_h + CONTACT_SHEET_CELL + 4),
            f"#{index + 1}  seed {seed}",
            fill=(200, 200, 205),
            font=font,
        )

    sheet.save(path, format="JPEG", quality=88, optimize=True)
    return path


def bundle_directory(spec: SceneSpec, job_id: str) -> Path:
    """A sortable, human-readable directory name for one job."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return OUTPUTS_DIR / f"{stamp}_{spec.slug()}_{job_id[:8]}"


class BundleWriter:
    """Writes one job's bundle incrementally, from the first variation to the last.

    Every method leaves ``result.json`` describing exactly what is on disk, so a
    reader -- the API, the UI, a resume after a crash -- can trust it at any
    moment.
    """

    def __init__(self, bundle: Bundle, spec: SceneSpec) -> None:
        self.bundle = bundle
        self.spec = spec
        self.directory = Path(bundle.directory)
        # Time spent in earlier sessions of a resumed job.
        self._earlier_duration = bundle.duration_s

    @classmethod
    def create(
        cls,
        spec: SceneSpec,
        compiled: CompiledPrompt,
        job_id: str,
        directory: Optional[Path] = None,
    ) -> "BundleWriter":
        directory = directory or bundle_directory(spec, job_id)
        directory.mkdir(parents=True, exist_ok=True)
        bundle = Bundle(
            job_id=job_id,
            directory=str(directory),
            spec=spec.model_dump(mode="json"),
            compiled=compiled.to_dict(),
            variations=[],
            created_at=_now(),
            status="running",
        )
        (directory / "spec.json").write_text(
            json.dumps(bundle.spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        writer = cls(bundle, spec)
        writer._save()
        return writer

    @classmethod
    def open(cls, directory: Path | str) -> "BundleWriter":
        """Reopen a bundle to carry on writing into it, typically for a resume."""
        directory = Path(directory)
        manifest = read_bundle(directory)
        if manifest is None:
            raise FileNotFoundError(f"no bundle in {directory}: result.json is missing")
        spec = load_spec(json.loads((directory / "spec.json").read_text(encoding="utf-8")))
        bundle = Bundle.from_dict(manifest)
        bundle.directory = str(directory)
        return cls(bundle, spec)

    def completed_indices(self) -> list[int]:
        return [record.index for record in self.bundle.variations]

    # -- as the render goes ----------------------------------------------

    def add_variation(self, rendered: RenderedImage) -> VariationRecord:
        """Finish, save and measure one image, and record it at once."""
        # Overlays and postprocessing happen here rather than in the renderer, so
        # the diagnostics describe the image the caller actually receives.
        final = finish(rendered.image, self.spec)

        image_path = self.directory / f"{rendered.index + 1:03d}_seed{rendered.seed}.png"
        final.save(image_path, format="PNG", optimize=True)
        preview_path = _write_preview(final, image_path)

        record = VariationRecord(
            index=rendered.index,
            seed=rendered.seed,
            image=str(image_path),
            preview=str(preview_path),
            diagnostics=analyse(final),
        )
        others = [item for item in self.bundle.variations if item.index != rendered.index]
        self.bundle.variations = sorted([*others, record], key=lambda item: item.index)
        self._write_contact_sheet()
        self._save()
        return record

    def record_checkpoint(self, state: ResumeState) -> None:
        """Save a variation-boundary resume point. Passed to ``render`` as ``on_checkpoint``."""
        save_checkpoint(self.directory, state, self.bundle.job_id)
        self.bundle.checkpoint = summarize_state(state.to_json())
        self._save()

    # -- endings -----------------------------------------------------------

    def complete(self, result: RenderResult) -> Bundle:
        self._absorb(result)
        self.bundle.status = "done"
        self.bundle.checkpoint = None
        self.bundle.error = None
        delete_checkpoint(self.directory)
        self._save()
        return self.bundle

    def pause(self, stopped: RenderStopped, reason: str = "pause") -> Bundle:
        """Save the exact resume point and mark the bundle paused."""
        state = stopped.state
        state.reason = reason
        state.paused_at = _now()
        save_checkpoint(self.directory, state, self.bundle.job_id)
        self._absorb(stopped.result)
        self.bundle.status = "paused"
        self.bundle.checkpoint = summarize_state(state.to_json())
        self._save()
        return self.bundle

    def abort(self, duration_s: Optional[float] = None) -> Bundle:
        """Stopped hard, with no chance to save step state.

        The last variation-boundary checkpoint stays, so a resume restarts the
        interrupted variation from step 0 with the same seed.
        """
        meta = mark_checkpoint(self.directory, reason="abort", paused_at=_now())
        if duration_s is not None:
            self.bundle.duration_s = round(self._earlier_duration + duration_s, 2)
        self.bundle.status = "aborted"
        self.bundle.checkpoint = summarize_state(meta) if meta else None
        self._save()
        return self.bundle

    def cancel(self, result: Optional[RenderResult] = None) -> Bundle:
        """Stopped on request, for good: finished images stay, the checkpoint goes."""
        self._absorb(result)
        self.bundle.status = "cancelled"
        self.bundle.checkpoint = None
        delete_checkpoint(self.directory)
        self._save()
        return self.bundle

    def fail(self, message: str) -> Bundle:
        """An error. Any checkpoint is kept: the cause may be fixable and the job resumable."""
        self.bundle.status = "error"
        self.bundle.error = message
        meta = read_state(self.directory)
        self.bundle.checkpoint = summarize_state(meta) if meta else None
        self._save()
        return self.bundle

    # -- internals ---------------------------------------------------------

    def _absorb(self, result: Optional[RenderResult]) -> None:
        if result is None:
            return
        if result.control_image is not None and self.bundle.control_image is None:
            control_path = self.directory / "control.png"
            result.control_image.save(control_path)
            self.bundle.control_image = str(control_path)
        for note in result.notes:
            if note not in self.bundle.notes:
                self.bundle.notes.append(note)
        self.bundle.duration_s = round(self._earlier_duration + result.duration_s, 2)
        self.bundle.task = result.task
        self.bundle.device = result.device

    def _write_contact_sheet(self) -> None:
        images: list[tuple[int, int, Image.Image]] = []
        for record in self.bundle.variations:
            with Image.open(record.image) as image:
                images.append((record.index, record.seed, image.copy()))
        sheet_path = _build_contact_sheet(images, self.directory / "contact-sheet.jpg")
        self.bundle.contact_sheet = str(sheet_path) if sheet_path else None

    def _save(self) -> None:
        write_json(self.directory / "result.json", self.bundle.to_dict())


def read_bundle(directory: Path) -> Optional[dict[str, Any]]:
    """Load a previously written bundle manifest."""
    manifest = Path(directory) / "result.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))


def _bundle_directories() -> list[Path]:
    if not OUTPUTS_DIR.is_dir():
        return []
    return sorted(
        (path for path in OUTPUTS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def list_bundles(limit: int = 50) -> list[dict[str, Any]]:
    """Recent bundles, newest first. Backs the gallery and the history endpoint."""
    bundles = []
    for directory in _bundle_directories()[:limit]:
        manifest = read_bundle(directory)
        if manifest:
            bundles.append(manifest)
    return bundles


def list_resumable(limit: int = 50) -> list[dict[str, Any]]:
    """Bundles on disk that still hold a checkpoint, newest first. No torch.

    That includes bundles whose process died mid-render: their ``status`` still
    says ``running`` and nothing is running them.
    """
    found = []
    for directory in _bundle_directories():
        try:
            meta = read_state(directory)
        except ValueError:
            continue
        if meta is None:
            continue
        manifest = read_bundle(directory) or {}
        spec = manifest.get("spec") or {}
        variations = manifest.get("variations", [])
        found.append(
            {
                "directory": str(directory),
                "job_id": manifest.get("job_id"),
                "name": spec.get("name") or (spec.get("subject") or {}).get("primary", "")[:60],
                "status": manifest.get("status", "running"),
                "images": len(variations),
                "preview": variations[0]["preview"] if variations else None,
                "checkpoint": summarize_state(meta),
            }
        )
        if len(found) >= limit:
            break
    return found


__all__ = [
    "STATUSES",
    "Bundle",
    "BundleWriter",
    "VariationRecord",
    "bundle_directory",
    "list_bundles",
    "list_resumable",
    "read_bundle",
]
