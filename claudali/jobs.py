"""The render queue.

Renders take minutes, not milliseconds, so ClauDali's API is asynchronous by
necessity: a caller submits a job, gets an id back immediately, and polls. A
synchronous endpoint would exceed the default timeout of essentially every HTTP
client before the first image finished.

Exactly one worker thread runs, because there is one GPU. Two concurrent SDXL
renders on a 6 GB card do not go twice as fast; they thrash the offload buffers
and both finish later than either would have alone.

A job can be paused at its next sampler step and resumed later, exactly. A
pause either lets the next queued job run or holds the whole queue until the
paused job resumes; a resumed job goes to the front. When the server stops, the
paused and not-yet-started jobs are written to ``runs/queue.json`` and come back
on the next start, held until someone resumes one or releases the queue.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .bundle import Bundle, BundleWriter, read_bundle
from .compiler import compile_spec
from .config import RUNS_DIR, SETTINGS
from .engine.checkpoint import RenderController, read_state, write_json
from .spec import SceneSpec, load_spec

logger = logging.getLogger(__name__)

QUEUE_STATE_FILE = RUNS_DIR / "queue.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _same_path(first: Path | str, second: Path | str) -> bool:
    return Path(first).resolve() == Path(second).resolve()


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


FINISHED = frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})
ACTIVE = frozenset({JobStatus.RUNNING, JobStatus.PAUSING})


class QueueFull(RuntimeError):
    """The queue already holds as many waiting jobs as it allows."""


class JobStateError(RuntimeError):
    """The job is not in a state that allows what was asked."""


@dataclass
class Job:
    """One render request and its lifecycle."""

    id: str
    spec: SceneSpec
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    stage: str = "base"
    step: int = 0
    total_steps: int = 0
    variation: int = 0
    total_variations: int = 1
    eta_s: Optional[float] = None

    bundle: Optional[Bundle] = None
    bundle_dir: Optional[str] = None
    error: Optional[str] = None
    # "resume_mismatch" when a resume was refused because something that shapes
    # the pixels changed. ``mismatch`` then lists what, and the checkpoint is
    # intact, so the job can be resumed again with force.
    error_type: Optional[str] = None
    mismatch: list[str] = field(default_factory=list)
    force_resume: bool = False
    controller: RenderController = field(default_factory=RenderController)

    _started_monotonic: Optional[float] = None
    # Work done and to do across every stage and variation, as the renderer counts it.
    _done: float = 0.0
    _total: float = 0.0

    @property
    def resumable(self) -> bool:
        return self.status is JobStatus.PAUSED or self.error_type == "resume_mismatch"

    @property
    def progress(self) -> float:
        """Overall completion across every stage of every variation, 0..1."""
        if self.status is JobStatus.DONE:
            return 1.0
        if self._total <= 0:
            return 0.0
        return min(1.0, self._done / self._total)

    def to_dict(self, include_bundle: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "step": self.step,
            "total_steps": self.total_steps,
            "variation": self.variation,
            "total_variations": self.total_variations,
            "eta_s": round(self.eta_s, 1) if self.eta_s else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "name": self.spec.name or self.spec.subject.primary[:60],
            "error": self.error,
            "resumable": self.resumable,
            "bundle_dir": self.bundle_dir,
        }
        if self.error_type:
            data["error_type"] = self.error_type
            data["mismatch"] = self.mismatch
        if self.bundle is not None:
            data["checkpoint"] = self.bundle.checkpoint
            if include_bundle:
                data["bundle"] = self.bundle.to_dict()
        return data


class JobQueue:
    """A single-worker render queue with progress, pausing, holding and cancellation."""

    def __init__(self, max_size: int = 64, retention: int = 200, autostart: bool = True) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        # A deque rather than queue.Queue, because a resumed job jumps the line.
        self._pending: deque[str] = deque()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._held = False
        self._hold_owner: Optional[str] = None
        self._current: Optional[str] = None
        self._max_size = max_size
        self._retention = retention
        self._autostart = autostart
        self._worker: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._shutting_down = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="claudali-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._wake:
            self._wake.notify_all()

    def shutdown(
        self,
        path: Path = QUEUE_STATE_FILE,
        timeout: float = 300.0,
        abandon: Callable[[], bool] = lambda: False,
    ) -> int:
        """Pause and hold, then save the queue. What Ctrl+C on the server does.

        Waits for the running job's checkpoint to land -- at most one sampler step
        plus the save, unless the model is still loading -- or until ``abandon()``
        says to stop waiting. An abandoned job keeps its last variation-boundary
        checkpoint, so it still resumes from the start of the variation it was on.
        Returns how many jobs were saved.
        """
        with self._wake:
            self._shutting_down = True
            self._held = True
            current = self._jobs.get(self._current) if self._current else None
            if current is not None and current.status in ACTIVE:
                current.status = JobStatus.PAUSING
                current.controller.pause_requested = True
                self._hold_owner = current.id

        if current is not None:
            deadline = time.monotonic() + timeout
            with self._wake:
                while (
                    self._current == current.id
                    and time.monotonic() < deadline
                    and not abandon()
                ):
                    self._wake.wait(0.25)

        self.stop()
        return self.save(path)

    # -- submission and control -------------------------------------------

    def submit(self, spec: SceneSpec) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            spec=spec,
            total_steps=spec.render.steps,
            total_variations=spec.render.variations,
        )
        with self._wake:
            if len(self._pending) >= self._max_size:
                raise QueueFull(f"the queue already holds {self._max_size} jobs waiting to start")
            self._register_locked(job)
            self._pending.append(job.id)
            self._wake.notify_all()
        self._maybe_start()
        return job

    def pause(self, job_id: str, hold_queue: bool = False) -> Job:
        """Pause a job: a running one at its next step, a queued one at once.

        With ``hold_queue`` no other job starts until this one resumes or the
        queue is released.
        """
        with self._wake:
            job = self._require_locked(job_id)
            if job.status in ACTIVE:
                job.status = JobStatus.PAUSING
                job.controller.pause_requested = True
            elif job.status is JobStatus.QUEUED:
                self._pending.remove(job.id)
                job.status = JobStatus.PAUSED
            elif job.status is not JobStatus.PAUSED:
                raise JobStateError(f"a {job.status.value} job cannot be paused")
            if hold_queue:
                self._held = True
                self._hold_owner = job.id
            self._wake.notify_all()
            return job

    def resume(self, job_id: str, force: bool = False) -> Job:
        """Put a paused job back at the front of the queue."""
        with self._wake:
            job = self._require_locked(job_id)
            self._resume_locked(job, force)
        self._maybe_start()
        return job

    def resume_bundle(self, directory: Path | str, force: bool = False) -> Job:
        """Queue a paused bundle found on disk, e.g. one the CLI left behind."""
        directory = Path(directory)
        meta = read_state(directory)
        if meta is None:
            raise FileNotFoundError(f"no checkpoint in {directory}: nothing to resume")
        manifest = read_bundle(directory)
        if manifest is None:
            raise FileNotFoundError(f"no result.json in {directory}")
        spec = load_spec(json.loads((directory / "spec.json").read_text(encoding="utf-8")))

        with self._wake:
            for existing in self._jobs.values():
                if not existing.bundle_dir or not _same_path(existing.bundle_dir, directory):
                    continue
                if existing.resumable:
                    self._resume_locked(existing, force)
                    job = existing
                    break
                if existing.status not in FINISHED:
                    raise JobStateError(
                        f"that bundle is already {existing.status.value} as job {existing.id}"
                    )
            else:
                job_id = manifest.get("job_id") or uuid.uuid4().hex
                if job_id in self._jobs:
                    job_id = uuid.uuid4().hex
                job = Job(
                    id=job_id,
                    spec=spec,
                    total_steps=(meta.get("compiled") or {}).get("steps", spec.render.steps),
                    total_variations=len(meta.get("seeds", [])) or spec.render.variations,
                    bundle=Bundle.from_dict(manifest),
                    bundle_dir=str(directory),
                    force_resume=force,
                )
                self._register_locked(job)
                self._pending.appendleft(job.id)
                self._wake.notify_all()
        self._maybe_start()
        return job

    def release(self) -> None:
        """Let queued jobs start again, without resuming whichever job held the queue."""
        with self._wake:
            self._held = False
            self._hold_owner = None
            self._wake.notify_all()

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. A queued or paused job ends at once, a running one at its
        next sampler step. A paused job's finished images are kept and its
        checkpoint deleted."""
        with self._wake:
            job = self._jobs.get(job_id)
            if job is None or job.status in FINISHED:
                return False
            if job.status in ACTIVE:
                job.controller.abort_requested = True
                return True
            if job.status is JobStatus.QUEUED:
                self._pending.remove(job.id)
            job.status = JobStatus.CANCELLED
            job.error = "cancelled by request"
            job.finished_at = _now()
            if self._hold_owner == job.id:
                self._hold_owner = None
            bundle_dir = job.bundle_dir
        if bundle_dir is not None and read_bundle(Path(bundle_dir)) is not None:
            job.bundle = BundleWriter.open(bundle_dir).cancel()
        return True

    # -- inspection -------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._wake:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._wake:
            return [self._jobs[job_id] for job_id in reversed(self._order[-limit:])]

    def current(self) -> Optional[Job]:
        with self._wake:
            return self._jobs.get(self._current) if self._current else None

    def stats(self) -> dict[str, Any]:
        with self._wake:
            jobs = list(self._jobs.values())
            held = self._held

        def count(*statuses: JobStatus) -> int:
            return sum(1 for job in jobs if job.status in statuses)

        return {
            "queued": count(JobStatus.QUEUED),
            "running": count(JobStatus.RUNNING, JobStatus.PAUSING),
            "pausing": count(JobStatus.PAUSING),
            "paused": count(JobStatus.PAUSED),
            "done": count(JobStatus.DONE),
            "error": count(JobStatus.ERROR),
            "cancelled": count(JobStatus.CANCELLED),
            "held": held,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: Path = QUEUE_STATE_FILE) -> int:
        """Write every unfinished job to ``path``; remove the file when there are none."""
        path = Path(path)
        with self._wake:
            stopped = [
                self._jobs[job_id]
                for job_id in self._order
                if self._jobs[job_id].status not in FINISHED | {JobStatus.QUEUED}
            ]
            waiting = [self._jobs[job_id] for job_id in self._pending]
            entries = [
                {
                    # A job still running here was abandoned mid-pause; it resumes
                    # from its last checkpoint all the same.
                    "status": "queued" if job.status is JobStatus.QUEUED else "paused",
                    "id": job.id,
                    # exclude_unset keeps inferred values inferred; see BundleWriter.create.
                    "spec": job.spec.model_dump(mode="json", exclude_unset=True),
                    "bundle_dir": job.bundle_dir,
                    "created_at": job.created_at,
                }
                for job in [*stopped, *waiting]
            ]
        if not entries:
            path.unlink(missing_ok=True)
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"saved_at": _now(), "held": True, "jobs": entries})
        return len(entries)

    def restore(self, path: Path = QUEUE_STATE_FILE) -> int:
        """Bring back the jobs a shutdown saved, with the queue held."""
        path = Path(path)
        if not path.is_file():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        restored = 0
        with self._wake:
            for entry in data.get("jobs", []):
                try:
                    spec = load_spec(entry["spec"])
                except Exception as exc:  # noqa: BLE001 - one bad entry must not lose the rest
                    logger.warning("could not restore job %s: %s", entry.get("id"), exc)
                    continue
                bundle_dir = entry.get("bundle_dir")
                manifest = read_bundle(Path(bundle_dir)) if bundle_dir else None
                job = Job(
                    id=entry["id"],
                    spec=spec,
                    created_at=entry.get("created_at") or _now(),
                    total_steps=(manifest or {}).get("compiled", {}).get("steps", spec.render.steps),
                    total_variations=spec.render.variations,
                    bundle=Bundle.from_dict(manifest) if manifest else None,
                    bundle_dir=bundle_dir,
                )
                if entry.get("status") == "queued":
                    self._pending.append(job.id)
                else:
                    job.status = JobStatus.PAUSED
                self._register_locked(job)
                restored += 1
            if restored:
                self._held = data.get("held", True)
        # Restored jobs now live in memory until the next shutdown writes them out
        # again; leaving the file would restore them twice.
        path.unlink(missing_ok=True)
        return restored

    # -- internals --------------------------------------------------------

    def _maybe_start(self) -> None:
        if self._autostart:
            self.start()

    def _require_locked(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _register_locked(self, job: Job) -> None:
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._prune_locked()

    def _resume_locked(self, job: Job, force: bool) -> None:
        if not job.resumable:
            raise JobStateError(f"a {job.status.value} job cannot be resumed")
        job.status = JobStatus.QUEUED
        job.force_resume = force
        job.error = None
        job.error_type = None
        job.mismatch = []
        job.finished_at = None
        job.eta_s = None
        job.controller = RenderController()
        if job.id in self._pending:
            self._pending.remove(job.id)
        self._pending.appendleft(job.id)
        if self._hold_owner == job.id:
            self._held = False
            self._hold_owner = None
        self._wake.notify_all()

    def _prune_locked(self) -> None:
        """Forget the oldest finished jobs once retention is exceeded.

        Only in-memory records of finished jobs are dropped; the bundles stay on
        disk and remain readable through the history endpoint.
        """
        while len(self._order) > self._retention:
            oldest = next(
                (job_id for job_id in self._order if self._jobs[job_id].status in FINISHED), None
            )
            if oldest is None:
                return
            self._order.remove(oldest)
            self._jobs.pop(oldest, None)

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            job = self._take_next(timeout=0.5)
            if job is None:
                continue
            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001 - one bad job must not kill the worker
                logger.exception("job %s failed", job.id)
                job.status = JobStatus.ERROR
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._settle(job)

    def _take_next(self, timeout: float) -> Optional[Job]:
        """The next job to run, marked running, or None when there is none or the queue is held."""
        with self._wake:
            if self._held or not self._pending:
                self._wake.wait(timeout)
            if self._stopping.is_set() or self._held or not self._pending:
                return None
            job = self._jobs.get(self._pending.popleft())
            if job is None or job.status is not JobStatus.QUEUED:
                return None
            job.status = JobStatus.RUNNING
            job.started_at = _now()
            self._current = job.id
            return job

    def _settle(self, job: Job) -> None:
        with self._wake:
            self._current = None
            if job.status in FINISHED:
                job.finished_at = _now()
                if self._hold_owner == job.id and not job.resumable:
                    # The queue stays held -- someone asked for that -- but by nobody's job.
                    self._hold_owner = None
            self._wake.notify_all()

    def _execute(self, job: Job) -> None:
        # Imported here so that submitting a job, listing history, or starting the
        # server never pays the multi-second cost of importing torch.
        from .engine.checkpoint import RenderAborted, RenderPaused, ResumeMismatch, load_checkpoint
        from .engine.render import render

        job._started_monotonic = time.monotonic()
        job.eta_s = None

        resume = None
        compiled = None
        if job.bundle_dir is not None and read_state(job.bundle_dir) is not None:
            writer = BundleWriter.open(job.bundle_dir)
            resume = load_checkpoint(job.bundle_dir)
            resume.completed = sorted(set(resume.completed) | set(writer.completed_indices()))
            job.total_steps = resume.compiled["steps"]
            job.total_variations = len(resume.seeds)
        else:
            compiled = compile_spec(job.spec)
            writer = BundleWriter.create(job.spec, compiled, job.id)
            job.bundle_dir = str(writer.directory)
            job.total_steps = compiled.steps
            job.total_variations = job.spec.render.variations
        job.bundle = writer.bundle

        def progress(update: Any) -> None:
            job.stage = update.stage
            job.step, job.total_steps = update.step, update.steps
            job.variation, job.total_variations = update.variation, update.variations
            job._done, job._total = update.done, update.total

            # A resumed job starts partway, and only the work run in this session
            # says how fast it is going.
            ran = update.done - update.resumed_from
            elapsed = time.monotonic() - (job._started_monotonic or time.monotonic())
            remaining = max(0.0, update.total - update.done)
            job.eta_s = elapsed / ran * remaining if ran > 0 else None

        try:
            result = render(
                job.spec,
                progress=progress,
                compiled=compiled,
                controller=job.controller,
                resume=resume,
                force=job.force_resume,
                on_variation=writer.add_variation,
                on_checkpoint=writer.record_checkpoint,
            )
        except RenderPaused as stopped:
            writer.pause(stopped, reason="shutdown" if self._shutting_down else "pause")
            job.status = JobStatus.PAUSED
            job.eta_s = None
            job.controller = RenderController()
            return
        except RenderAborted as stopped:
            writer.cancel(stopped.result)
            job.status = JobStatus.CANCELLED
            job.error = "cancelled by request"
            return
        except ResumeMismatch as exc:
            # The bundle is left paused and its checkpoint intact, so that a
            # forced resume can follow.
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.error_type = "resume_mismatch"
            job.mismatch = exc.differences
            return
        except Exception as exc:
            writer.fail(f"{type(exc).__name__}: {exc}")
            raise

        writer.complete(result)
        job.status = JobStatus.DONE
        job.step, job.variation = job.total_steps, job.total_variations
        job.eta_s = 0.0


QUEUE = JobQueue(max_size=SETTINGS.max_queue, retention=SETTINGS.job_retention)

__all__ = [
    "QUEUE",
    "QUEUE_STATE_FILE",
    "Job",
    "JobQueue",
    "JobStateError",
    "JobStatus",
    "QueueFull",
]
