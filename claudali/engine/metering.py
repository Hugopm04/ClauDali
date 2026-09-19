"""What a render cost: time per stage, time per step, and memory as it ran.

``duration_s`` on its own cannot answer the questions a benchmark asks. Two
renders of the same length differ because one loaded a second checkpoint, or
because one decoded on the CPU and paged for it, or because the first step
waited two minutes for offloaded weights to come back from the pagefile. This
module records those separately so the total can be taken apart afterwards.

Three things are measured, and each is measured because a real question needed
it:

* **Spans.** Named, nested-free intervals: loading a model, running one stage of
  one variation, one VAE decode, one upscale, one encode. Rolled up per stage,
  so ``s_per_step`` is a measured number rather than a division of the total.
* **Steps.** The wall time of every individual sampler step. The mean is what
  you quote; the *first* step is the one that catches paging, and it is reported
  on its own because averaging it in hides exactly the fault it reveals.
* **Memory.** Free RAM sampled on a background thread for the whole render, so
  the low-water mark is caught even though it falls between the decode plan's
  reading and the report at the end. Peak VRAM comes from torch's own counters.

Everything here degrades to ``None`` rather than failing. A measurement that
breaks a render is worse than no measurement, so every reading is guarded and a
meter that could not measure says so instead of guessing.

Stdlib only at import time: ``engine/render.py`` imports this module at the top,
and the import-cost rule in CLAUDE.md forbids that from reaching torch.
"""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .. import sysinfo

# Two seconds is fine against steps that take 10-26 s here, and 450 readings
# over a three-hour render is nothing to hold in memory. The point is to catch
# the trough during a CPU decode, which lasts tens of seconds.
SAMPLE_INTERVAL_S = 2.0


def _round(value: Optional[float], places: int = 2) -> Optional[float]:
    return None if value is None else round(value, places)


@dataclass
class Span:
    """One named interval, with whatever identifies it."""

    name: str
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seconds": round(self.seconds, 3), **self.detail}


class _RamSampler:
    """Free RAM on a background thread, for the low-water mark.

    A daemon thread, so an abandoned render can never keep the process alive.
    It holds no lock the render needs: the readings go into a plain list, and
    only ``stop`` reads it, after the thread has been joined.
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL_S) -> None:
        self.interval = interval
        self.readings: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="claudali-ram", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            free = sysinfo.available_ram_gb()
            if free is not None:
                self.readings.append(free)
            # wait() rather than sleep(): stop() returns promptly at the end of
            # a render instead of after a whole interval.
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None

    @property
    def low_water(self) -> Optional[float]:
        return min(self.readings) if self.readings else None


def _vram_peaks() -> dict[str, Optional[float]]:
    """Peak VRAM from torch's counters, in decimal GB. All None without CUDA."""
    empty: dict[str, Optional[float]] = {
        "vram_peak_allocated_gb": None,
        "vram_peak_reserved_gb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return empty
        return {
            "vram_peak_allocated_gb": _round(torch.cuda.max_memory_allocated() / 1e9),
            "vram_peak_reserved_gb": _round(torch.cuda.max_memory_reserved() / 1e9),
        }
    except Exception:  # noqa: BLE001 - a reading must never break a render
        return empty


def _reset_vram_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


class Meter:
    """Collects spans, step times and memory readings for one render call.

    A resumed render gets a fresh meter, so its numbers describe the session
    that ran, not the whole image. ``bundle.py`` keeps them per session for
    exactly that reason: adding a resume's load time to the original's would
    invent a render that never happened.
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL_S, enabled: bool = True) -> None:
        self.enabled = enabled
        self.spans: list[Span] = []
        # stage name -> the wall time of each of its sampler steps
        self.step_times: dict[str, list[float]] = {}
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.ram_start: Optional[float] = None
        self.ram_end: Optional[float] = None
        self.ram_total: Optional[float] = None
        self._sampler = _RamSampler(interval) if enabled else None
        self._interval = interval
        self._step_clock: Optional[float] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.started = time.time()
        reading = sysinfo.memory_gb()
        if reading is not None:
            self.ram_start, self.ram_total = reading
        _reset_vram_peaks()
        if self._sampler is not None:
            self._sampler.start()

    def stop(self) -> None:
        self.finished = time.time()
        self.ram_end = sysinfo.available_ram_gb()
        self._peaks = _vram_peaks()
        if self._sampler is not None:
            self._sampler.stop()

    # -- recording ----------------------------------------------------------

    @contextlib.contextmanager
    def span(self, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
        """Time a block and file it under ``name``.

        Yields the detail dict, so a caller that only learns what it did while
        doing it -- which device a decode chose, say -- can add that before the
        span closes. The span is recorded even when the block raises, because a
        pause is an exception and the work it did still happened.
        """
        clock = time.time()
        record = dict(detail)
        try:
            yield record
        finally:
            self.spans.append(Span(name=name, seconds=time.time() - clock, detail=record))

    def begin_steps(self) -> None:
        """Start the per-step clock. Called once before a stage's sampler runs."""
        self._step_clock = time.time()

    def step(self, stage: str) -> None:
        """One sampler step of ``stage`` finished."""
        if self._step_clock is None:
            self._step_clock = time.time()
            return
        now = time.time()
        self.step_times.setdefault(stage, []).append(now - self._step_clock)
        self._step_clock = now

    # -- reports ------------------------------------------------------------

    def _stage_rollup(self) -> dict[str, dict[str, Any]]:
        """Per-stage totals, with seconds-per-step measured rather than divided."""
        rollup: dict[str, dict[str, Any]] = {}
        for span in self.spans:
            if span.name != "stage":
                continue
            stage = str(span.detail.get("stage", "?"))
            entry = rollup.setdefault(stage, {"runs": 0, "total_s": 0.0, "steps": 0})
            entry["runs"] += 1
            entry["total_s"] += span.seconds
            entry["steps"] += int(span.detail.get("steps") or 0)
        for stage, entry in rollup.items():
            entry["total_s"] = round(entry["total_s"], 2)
            entry["mean_s"] = round(entry["total_s"] / entry["runs"], 2) if entry["runs"] else None
            times = self.step_times.get(stage) or []
            # Prefer the measured step times; fall back to dividing the stage
            # total, which includes the pipeline's own setup and so reads high.
            if times:
                entry["s_per_step"] = round(statistics.fmean(times), 3)
            elif entry["steps"]:
                entry["s_per_step"] = round(entry["total_s"] / entry["steps"], 3)
            else:
                entry["s_per_step"] = None
        return rollup

    def _step_rollup(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for stage, times in self.step_times.items():
            if not times:
                continue
            report[stage] = {
                "count": len(times),
                "mean_s": round(statistics.fmean(times), 3),
                "median_s": round(statistics.median(times), 3),
                "min_s": round(min(times), 3),
                "max_s": round(max(times), 3),
                # The first step of a stage is where paging shows up: after a
                # CPU decode short of RAM it has been 12x the others here.
                "first_s": round(times[0], 3),
            }
        return report

    def timing(self) -> dict[str, Any]:
        total = None if self.started is None else (self.finished or time.time()) - self.started
        load_s = sum(span.seconds for span in self.spans if span.name == "load")
        return {
            "total_s": _round(total),
            "load_s": round(load_s, 2),
            "stages": self._stage_rollup(),
            "steps": self._step_rollup(),
            "spans": [span.to_dict() for span in self.spans],
        }

    def memory(self) -> dict[str, Any]:
        peaks = getattr(self, "_peaks", None) or _vram_peaks()
        readings = self._sampler.readings if self._sampler is not None else []
        return {
            "ram_total_gb": _round(self.ram_total, 1),
            "ram_free_start_gb": _round(self.ram_start),
            "ram_free_end_gb": _round(self.ram_end),
            "ram_free_min_gb": _round(self._sampler.low_water if self._sampler else None),
            "ram_samples": len(readings),
            "sample_interval_s": self._interval,
            **peaks,
        }


__all__ = ["Meter", "Span", "SAMPLE_INTERVAL_S"]
