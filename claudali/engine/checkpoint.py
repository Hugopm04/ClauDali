"""Pausing a render at any sampler step and resuming it bit for bit.

A resumed render has to continue as if it had never stopped: same latents, same
sampler history, same random stream. diffusers has no way to start its loop
partway, so this gets there with two small hooks on the pipeline *instance*,
both removed again when the call returns:

1. ``prepare_latents`` is wrapped. The original still runs, so the generator is
   consumed exactly as in an uninterrupted call and img2img and inpainting still
   get their noise and image latents; then the saved latents replace the fresh
   ones.
2. From inside that wrapper, ``pipe._interrupt`` is replaced by a
   :class:`_SkipUntil`. Every SDXL pipeline opens each loop iteration with
   ``if self.interrupt: continue`` and resets ``_interrupt`` to False *before*
   it prepares latents, so a value set here survives the reset. It is truthy for
   the steps already done, which the loop then skips without touching the UNet.
   At the first step still to run it restores the scheduler's stepping state and
   the generator's state, then stays falsy.

The restore has to happen inside the loop rather than before the call, because
the pipeline itself calls ``scheduler.set_timesteps`` and ``set_begin_index``,
which reset exactly the state being restored.

``tests/test_claudali.py`` pauses and resumes a tiny random SDXL pipeline on the
CPU and compares the latents with ``torch.equal``. If a diffusers upgrade moves
either hook, that test fails. Fix the hooks; do not loosen the test.

A checkpoint is two files in ``<bundle>/checkpoint/``: ``state.json``, readable
without torch by the API and UI, and ``state.pt``, holding only tensors and
primitives so it loads with ``torch.load(weights_only=True)``. A job with a
refiner or hi-res stage runs every variation's base stage before the next stage
starts, so ``state.pt`` also carries the latents of variations waiting between
stages (``ResumeState.staged``); format 2 added them.

Stdlib only at import time: ``jobs``, ``bundle`` and ``api`` import this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

FORMAT_VERSION = 2
# Format 1 is a base-stage-only checkpoint, which reads the same way.
READABLE_FORMATS = (1, 2)
CHECKPOINT_DIR = "checkpoint"
STATE_JSON = "state.json"
STATE_PT = "state.pt"

# The sampler stages a variation can go through, in order, with their names for people.
STAGE_NAMES = {"base": "base", "refiner": "refiner", "hires": "hi-res"}

# Packages whose version can change the numbers a sampler step produces.
# accelerate only moves tensors between devices, so it is left out.
FINGERPRINT_PACKAGES = ("torch", "diffusers", "transformers", "compel")


# ---------------------------------------------------------------------------
# Control and state
# ---------------------------------------------------------------------------


@dataclass
class RenderController:
    """How a caller asks a running render to stop.

    Plain flags, set from any thread and read at step and variation boundaries.
    A pause is honoured when the step in progress ends, which on a 6 GB card can
    be ~25 s away.
    """

    pause_requested: bool = False
    abort_requested: bool = False


@dataclass
class StepState:
    """Exactly where the sampler stands inside one variation."""

    next_step: int
    latents: Any  # torch.Tensor on the CPU, in the dtype the loop produced
    scheduler: dict[str, Any]
    generator: Any  # torch.ByteTensor from Generator.get_state()


@dataclass
class ResumeState:
    """Everything needed to carry on with a job.

    ``step`` is None at a stage boundary: that stage of the variation then starts
    from its first step with the original seed, which is still exact. ``stage``
    is ``base``, ``refiner`` or ``hires``.
    """

    seeds: list[int]
    variation: int
    completed: list[int]
    compiled: dict[str, Any]
    fingerprint: dict[str, Any]
    stage: str = "base"
    reason: str = "running"  # running | pause | abort | shutdown
    paused_at: Optional[str] = None
    step: Optional[StepState] = None
    # Variations that finished a stage and wait for the next: {variation: (stage
    # finished, latents on the CPU)}. Empty for a job with only the base stage.
    staged: dict[int, tuple[str, Any]] = field(default_factory=dict)
    # How many sampler steps the stage in progress runs, for display.
    stage_steps: Optional[int] = None

    @property
    def next_step(self) -> int:
        return self.step.next_step if self.step is not None else 0

    @property
    def has_tensors(self) -> bool:
        return self.step is not None or bool(self.staged)

    def to_json(self, job_id: Optional[str] = None) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "job_id": job_id,
            "stage": self.stage,
            "variation": self.variation,
            "next_step": self.next_step,
            "stage_steps": self.stage_steps,
            "has_step_state": self.step is not None,
            "staged": [
                {"variation": index, "stage": stage}
                for index, (stage, _latents) in sorted(self.staged.items())
            ],
            "seeds": list(self.seeds),
            "completed": sorted(self.completed),
            "reason": self.reason,
            "paused_at": self.paused_at,
            "fingerprint": self.fingerprint,
            "compiled": self.compiled,
        }


def summarize_state(meta: dict[str, Any]) -> dict[str, Any]:
    """The part of ``state.json`` worth showing a person: no prompt, no fingerprint."""
    compiled = meta.get("compiled") or {}
    return {
        "stage": meta.get("stage", "base"),
        "stages": compiled.get("stages", ["base"]),
        "variation": meta.get("variation", 0),
        "next_step": meta.get("next_step", 0),
        "steps": meta.get("stage_steps") or compiled.get("steps"),
        "variations": len(meta.get("seeds", [])),
        "completed": list(meta.get("completed", [])),
        "staged": list(meta.get("staged", [])),
        "mid_image": bool(meta.get("has_step_state")),
        "reason": meta.get("reason"),
        "paused_at": meta.get("paused_at"),
    }


class RenderStopped(Exception):
    """A render stopped because its caller asked. Carries where, and what finished."""

    def __init__(self, state: ResumeState, result: Any) -> None:
        super().__init__(
            f"render stopped at variation {state.variation + 1}, step {state.next_step}"
        )
        self.state = state
        self.result = result  # a RenderResult holding the images finished in this call


class RenderPaused(RenderStopped):
    """Stopped at a step or variation boundary, with enough saved to resume exactly."""


class RenderAborted(RenderStopped):
    """Stopped for good. Finished images are kept; the one in progress is not."""


class StepPaused(Exception):
    """Raised out of the sampler loop by :func:`claudali.engine.render.denoise`."""

    def __init__(self, step: StepState) -> None:
        super().__init__(f"paused before step {step.next_step}")
        self.step = step


class StepAborted(Exception):
    """Raised out of the sampler loop when an abort was requested."""


class ResumeMismatch(Exception):
    """Something that shapes the pixels changed between the pause and the resume."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__(
            "this render would not resume identically, because what shapes its pixels "
            "changed since it was paused: " + "; ".join(differences)
        )
        self.differences = list(differences)


# ---------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------

_SKIPPED_SCHEDULER_FIELDS = frozenset({"config", "_internal_dict"})
_UNSUPPORTED = object()


def capture_scheduler_state(scheduler: Any) -> dict[str, Any]:
    """Snapshot a scheduler's stepping state, whichever sampler it is.

    Generic on purpose: each sampler keeps different fields (DPM-Solver a model
    output history, Heun the previous derivative, UniPC a timestep list), and a
    per-class list would silently go stale. Values that are tensors, primitives
    or lists of those are kept; anything else is skipped. Every sampler in
    ``pipelines.SAMPLERS`` was checked to hold nothing else that changes while
    stepping.
    """
    state: dict[str, Any] = {}
    for name, value in vars(scheduler).items():
        if name in _SKIPPED_SCHEDULER_FIELDS:
            continue
        captured = _capture(value)
        if captured is not _UNSUPPORTED:
            state[name] = captured
    return state


def _capture(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_capture(item) for item in value]
        if any(item is _UNSUPPORTED for item in items):
            return _UNSUPPORTED
        return items if isinstance(value, list) else tuple(items)

    import torch

    if isinstance(value, torch.Tensor):
        # A dict marks a tensor, since no dict value is ever captured as itself.
        return {"tensor": value.detach().to("cpu", copy=True), "device": str(value.device)}
    return _UNSUPPORTED


def restore_scheduler_state(scheduler: Any, state: dict[str, Any]) -> None:
    """Write a snapshot back, each tensor onto the device it was taken from."""
    for name, value in state.items():
        setattr(scheduler, name, _restore(value))


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        import torch

        device = value["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        return value["tensor"].to(device, copy=True)
    if isinstance(value, list):
        return [_restore(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Resuming inside a pipeline call
# ---------------------------------------------------------------------------


class _SkipUntil:
    """Truthy for the first ``count`` checks of ``if self.interrupt:``, then restores state once."""

    def __init__(self, count: int, restore: Callable[[], None]) -> None:
        self.remaining = count
        self.restore: Optional[Callable[[], None]] = restore

    def __bool__(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return True
        if self.restore is not None:
            restore, self.restore = self.restore, None
            restore()
        return False


@contextlib.contextmanager
def resume_into(pipe: Any, step: StepState, generator: Any) -> Iterator[None]:
    """Make the next call of ``pipe`` continue from ``step``. See the module docstring."""
    original = pipe.prepare_latents

    def restore() -> None:
        restore_scheduler_state(pipe.scheduler, step.scheduler)
        generator.set_state(step.generator)

    def prepare_latents(*args: Any, **kwargs: Any) -> Any:
        prepared = original(*args, **kwargs)
        pipe._interrupt = _SkipUntil(step.next_step, restore)
        # Inpainting returns (latents, noise[, image_latents]); only the first is
        # the loop's state, the rest must stay as freshly computed.
        if isinstance(prepared, tuple):
            return (step.latents.to(prepared[0].device, copy=True), *prepared[1:])
        return step.latents.to(prepared.device, copy=True)

    pipe.prepare_latents = prepare_latents
    try:
        yield
    finally:
        # The pipeline is cached and shared by every later job.
        pipe.__dict__.pop("prepare_latents", None)
        pipe._interrupt = False


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _replace(source: Path, target: Path) -> bool:
    """Swap a finished file into place, retrying while another process holds the target.

    On Windows the swap fails for a moment while anything has the target open,
    and a bundle lives inside a sync folder whose client does exactly that.
    """
    for attempt in range(10):
        try:
            os.replace(source, target)
            return True
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    return False


def write_json(path: Path, data: Any) -> None:
    """Write JSON so that a reader polling the file never sees half of it."""
    path = Path(path)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    if not _replace(temporary, path):
        path.write_text(text, encoding="utf-8")
        temporary.unlink(missing_ok=True)


def checkpoint_dir(bundle_dir: Path | str) -> Path:
    return Path(bundle_dir) / CHECKPOINT_DIR


def save_checkpoint(bundle_dir: Path | str, state: ResumeState, job_id: Optional[str] = None) -> Path:
    """Write ``state.pt`` (when there are tensors to keep) and then ``state.json``.

    The metadata goes last and is the file a loader trusts: if the process dies
    between the two writes, the previous ``state.json`` still describes a
    consistent checkpoint and the new tensors are refused as not matching it.
    """
    directory = checkpoint_dir(bundle_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = directory / STATE_PT

    if state.has_tensors:
        import torch

        step = state.step
        payload = {
            "format": FORMAT_VERSION,
            "stage": state.stage,
            "variation": state.variation,
            "next_step": state.next_step,
            "latents": step.latents if step is not None else None,
            "scheduler": step.scheduler if step is not None else None,
            "generator": step.generator if step is not None else None,
            "staged": {
                index: {"stage": stage, "latents": latents}
                for index, (stage, latents) in state.staged.items()
            },
        }
        temporary = tensors.with_name(tensors.name + ".tmp")
        torch.save(payload, temporary)
        if not _replace(temporary, tensors):
            raise OSError(f"could not write {tensors}: the file stayed locked")

    write_json(directory / STATE_JSON, state.to_json(job_id))
    if not state.has_tensors:
        tensors.unlink(missing_ok=True)
    return directory


def read_state(bundle_dir: Path | str) -> Optional[dict[str, Any]]:
    """The checkpoint metadata, without torch, or None when there is no checkpoint."""
    path = checkpoint_dir(bundle_dir) / STATE_JSON
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def load_checkpoint(bundle_dir: Path | str) -> ResumeState:
    """Read a checkpoint back, tensors included."""
    meta = read_state(bundle_dir)
    if meta is None:
        raise FileNotFoundError(f"no checkpoint in {bundle_dir}: nothing to resume")
    if meta.get("format") not in READABLE_FORMATS:
        raise ValueError(
            f"checkpoint format {meta.get('format')} is not one this version reads "
            f"({', '.join(str(version) for version in READABLE_FORMATS)})"
        )

    step = None
    staged: dict[int, tuple[str, Any]] = {}
    if meta.get("has_step_state") or meta.get("staged"):
        import torch

        payload = torch.load(
            checkpoint_dir(bundle_dir) / STATE_PT, map_location="cpu", weights_only=True
        )
        staged = {
            int(index): (entry["stage"], entry["latents"])
            for index, entry in (payload.get("staged") or {}).items()
        }
        saved = (payload.get("stage"), payload.get("variation"), payload.get("next_step"), sorted(staged))
        expected = (
            meta.get("stage"),
            meta.get("variation"),
            meta.get("next_step"),
            sorted(entry["variation"] for entry in meta.get("staged", [])),
        )
        if saved != expected:
            raise ValueError(
                f"checkpoint/state.pt is at {saved} but state.json says {expected}; "
                "the checkpoint was only half written and cannot be resumed exactly"
            )
        if meta.get("has_step_state"):
            step = StepState(
                next_step=payload["next_step"],
                latents=payload["latents"],
                scheduler=payload["scheduler"],
                generator=payload["generator"],
            )

    return ResumeState(
        seeds=list(meta["seeds"]),
        variation=meta["variation"],
        completed=list(meta.get("completed", [])),
        compiled=meta["compiled"],
        fingerprint=meta.get("fingerprint", {}),
        stage=meta.get("stage", "base"),
        reason=meta.get("reason", "running"),
        paused_at=meta.get("paused_at"),
        step=step,
        staged=staged,
        stage_steps=meta.get("stage_steps"),
    )


def mark_checkpoint(bundle_dir: Path | str, **changes: Any) -> Optional[dict[str, Any]]:
    """Change metadata fields in ``state.json`` in place, e.g. the stop reason."""
    meta = read_state(bundle_dir)
    if meta is None:
        return None
    meta.update(changes)
    write_json(checkpoint_dir(bundle_dir) / STATE_JSON, meta)
    return meta


def delete_checkpoint(bundle_dir: Path | str) -> None:
    shutil.rmtree(checkpoint_dir(bundle_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that decide a step's numbers. No imports."""
    versions = {}
    for name in FINGERPRINT_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def image_digest(image: Any) -> Optional[str]:
    """A short hash of an image's pixels, or None when there is no image."""
    if image is None:
        return None
    digest = hashlib.sha256(f"{image.mode}{image.size}".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()[:16]


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def diff_fingerprints(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return [
        f"{key}: {saved.get(key)!r} -> {current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]


def check_fingerprint(saved: dict[str, Any], current: dict[str, Any], force: bool) -> list[str]:
    """Refuse a resume whose pixels could differ, unless forced. Returns notes."""
    differences = diff_fingerprints(saved, current)
    if not differences:
        return []
    if not force:
        raise ResumeMismatch(differences)
    return [
        "resumed with force although what shapes the pixels changed since the pause, so "
        "this image may not match an uninterrupted render: " + "; ".join(differences)
    ]


__all__ = [
    "STAGE_NAMES",
    "RenderAborted",
    "RenderController",
    "RenderPaused",
    "RenderStopped",
    "ResumeMismatch",
    "ResumeState",
    "StepAborted",
    "StepPaused",
    "StepState",
    "capture_scheduler_state",
    "check_fingerprint",
    "delete_checkpoint",
    "load_checkpoint",
    "read_state",
    "restore_scheduler_state",
    "resume_into",
    "save_checkpoint",
    "summarize_state",
]
