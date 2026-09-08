"""The render queue.

Renders take minutes, not milliseconds, so ClauDali's API is asynchronous by
necessity: a caller submits a job, gets an id back immediately, and polls. A
synchronous endpoint would exceed the default timeout of essentially every HTTP
client before the first image finished.

Exactly one worker thread runs, because there is one GPU. Two concurrent SDXL
renders on a 6 GB card do not go twice as fast; they thrash the offload buffers
and both finish later than either would have alone.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .bundle import Bundle, write_bundle
from .compiler import compile_spec
from .config import SETTINGS
from .spec import SceneSpec

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class JobCancelled(Exception):
    """Raised from inside the sampler loop to abort a running render."""


@dataclass
class Job:
    """One render request and its lifecycle."""

    id: str
    spec: SceneSpec
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    step: int = 0
    total_steps: int = 0
    variation: int = 0
    total_variations: int = 1
    eta_s: Optional[float] = None

    bundle: Optional[Bundle] = None
    error: Optional[str] = None
    cancel_requested: bool = False

    _started_monotonic: Optional[float] = None

    @property
    def progress(self) -> float:
        """Overall completion across every variation, 0..1."""
        if self.status is JobStatus.DONE:
            return 1.0
        total = max(1, self.total_steps * self.total_variations)
        done = self.variation * self.total_steps + self.step
        return min(1.0, done / total)

    def to_dict(self, include_bundle: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "progress": round(self.progress, 4),
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
        }
        if include_bundle and self.bundle is not None:
            data["bundle"] = self.bundle.to_dict()
        return data


class JobQueue:
    """A single-worker render queue with progress reporting and cancellation."""

    def __init__(self, max_size: int = 64, retention: int = 200) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=max_size)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._retention = retention
        self._stopping = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="claudali-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stopping.set()

    # -- submission -------------------------------------------------------

    def submit(self, spec: SceneSpec) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            spec=spec,
            total_steps=spec.render.steps,
            total_variations=spec.render.variations,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._prune_locked()
        self._queue.put(job.id)
        self.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order[-limit:])]

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. A queued job dies immediately; a running one at
        its next sampler step, which is at most a few seconds away."""
        job = self.get(job_id)
        if job is None or job.status in {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}:
            return False
        job.cancel_requested = True
        if job.status is JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now().isoformat(timespec="seconds")
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        return {
            "queued": sum(1 for job in jobs if job.status is JobStatus.QUEUED),
            "running": sum(1 for job in jobs if job.status is JobStatus.RUNNING),
            "done": sum(1 for job in jobs if job.status is JobStatus.DONE),
            "error": sum(1 for job in jobs if job.status is JobStatus.ERROR),
            "worker_alive": bool(self._worker and self._worker.is_alive()),
        }

    def _prune_locked(self) -> None:
        """Forget the oldest finished jobs once retention is exceeded.

        Only in-memory records are dropped; the bundles stay on disk and remain
        readable through the history endpoint.
        """
        while len(self._order) > self._retention:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            job = self.get(job_id)
            if job is None or job.status is JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            try:
                self._execute(job)
            except JobCancelled:
                job.status = JobStatus.CANCELLED
                job.error = "cancelled by request"
            except Exception as exc:  # noqa: BLE001 - one bad job must not kill the worker
                logger.exception("job %s failed", job.id)
                job.status = JobStatus.ERROR
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                self._queue.task_done()

    def _execute(self, job: Job) -> None:
        # Imported here so that submitting a job, listing history, or starting the
        # server never pays the multi-second cost of importing torch.
        from .engine.render import render

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat(timespec="seconds")
        job._started_monotonic = time.monotonic()

        compiled = compile_spec(job.spec)
        job.total_steps = compiled.steps
        job.total_variations = job.spec.render.variations

        def progress(step: int, total: int, variation: int, variations: int) -> None:
            if job.cancel_requested:
                raise JobCancelled(job.id)
            job.step, job.total_steps = step, total
            job.variation, job.total_variations = variation, variations

            elapsed = time.monotonic() - (job._started_monotonic or time.monotonic())
            done = variation * total + step
            remaining = max(0, total * variations - done)
            if done > 0:
                job.eta_s = elapsed / done * remaining

        result = render(job.spec, progress=progress, compiled=compiled)
        job.bundle = write_bundle(job.spec, result, job.id)
        job.status = JobStatus.DONE
        job.step, job.variation = job.total_steps, job.total_variations
        job.eta_s = 0.0


QUEUE = JobQueue(max_size=SETTINGS.max_queue, retention=SETTINGS.job_retention)

__all__ = ["QUEUE", "Job", "JobCancelled", "JobQueue", "JobStatus"]
