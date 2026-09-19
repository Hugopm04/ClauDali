"""Running the plan: spawning workers, following them, and keeping the books.

**Stopping and resuming.** Press Ctrl+C once. The worker pauses the render it
is inside at the end of the current sampler step, writes an exact checkpoint,
and exits; the runner records the item as paused and stops. Running the same
command again picks that item up from the step it stopped on and carries on
down the list. Nothing is recomputed and nothing is lost. Ctrl+C twice aborts
the item instead, which costs the variation in flight but keeps the ones
already saved; a later run restarts that variation from step 0 with its seed.

**Why a subprocess at all.** Two reasons, and they pull in opposite
directions. Measurement wants a fresh process per item, because a forced CPU
decode leaves the machine paging and the next render in the same process pays
for it -- CLAUDE.md records a 768x768 step going from 12 s to 90 s that way.
Throughput wants a long-lived process, because reloading a 7 GB checkpoint
costs about a minute and there are 1,600 items. So the settings phase, whose
whole purpose is measurement, gets a process per item, and the survey, which
is measuring pictures, runs consecutive same-model items together.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from .model import BENCH_DIR, ROOT, Calibration, Item, Record, State
from .progress import ProgressBar, RunProgress, human_duration

LOGS_DIR = BENCH_DIR / "logs"

# How many items one worker process may render before it is replaced. Only
# applies to grouped phases. A cap at all is belt and braces: every item writes
# its own bundle as it goes, so a crashed worker loses nothing but the item in
# flight -- but a process that has loaded and released checkpoints twenty times
# is not a process whose timings anyone should trust.
MAX_GROUP = 12

# Phases whose numbers are the point, so each item gets a clean process.
FRESH_PHASES = {"settings"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Plan:
    """The items to run this session, and the whole-run totals to show against."""

    all_items: list[Item]
    outstanding: list[Item]
    total_units: float
    done_units: float


def reconcile(items: list[Item], state: State, save: bool = True) -> int:
    """Believe the bundles on disk over the state file, and return how many changed.

    The state file is written between items; the bundles are written *during*
    them. So a runner killed outright -- a crash, a power cut, Task Manager --
    leaves items marked ``running`` whose work is actually finished and saved.
    Re-rendering those would silently repeat hours of work, and over a run this
    long that happens more than once. Every session therefore starts by asking
    each unfinished item's bundle what really happened to it.
    """
    changed = 0
    for item in items:
        record = state.records.get(item.id)
        if record is not None and record.status in {"done", "skipped", "failed"}:
            continue
        manifest = item.bundle_dir / "result.json"
        if not manifest.exists():
            continue
        try:
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = bundle.get("status")
        if status not in {"done", "paused", "aborted"}:
            continue
        record = state.get(item.id)
        record.status = "done" if status == "done" else "paused"
        record.finished_at = record.finished_at or bundle.get("created_at")
        record.bundle_dir = str(item.bundle_dir)
        record.images = len(bundle.get("variations") or [])
        record.duration_s = float(bundle.get("duration_s") or 0.0)
        timing = bundle.get("timing") or {}
        record.s_per_step = ((timing.get("stages") or {}).get("base") or {}).get("s_per_step")
        memory = bundle.get("memory") or {}
        record.ram_free_start_gb = memory.get("ram_free_start_gb")
        record.ram_free_min_gb = memory.get("ram_free_min_gb")
        record.vram_peak_gb = memory.get("vram_peak_reserved_gb")
        variations = bundle.get("variations") or []
        if variations:
            decode = variations[0].get("decode") or {}
            record.decode = f"{decode.get('device')}{'/tiled' if decode.get('tiled') else ''}"
        changed += 1
    if changed and save:
        state.save()
    return changed


def prepare(items: list[Item], state: State) -> Plan:
    """Work out what is left, and what fraction of the whole that is.

    The bar counts against the *whole* plan, not against what remains, so a
    run resumed at 40% opens at 40% rather than at zero.
    """
    reconcile(items, state)
    total = sum(item.work_units() for item in items)
    done = sum(item.work_units() for item in items if state.is_finished(item.id))
    return Plan(
        all_items=items,
        outstanding=state.outstanding(items),
        total_units=total,
        done_units=done,
    )


def group(items: list[Item], fresh_phases: set[str] = FRESH_PHASES) -> Iterator[list[Item]]:
    """Consecutive items a single worker process may render together."""
    batch: list[Item] = []
    for item in items:
        solo = item.phase in fresh_phases
        compatible = (
            batch
            and not solo
            and batch[0].phase not in fresh_phases
            and batch[-1].model == item.model
            and batch[-1].phase == item.phase
            and len(batch) < MAX_GROUP
        )
        if compatible:
            batch.append(item)
            continue
        if batch:
            yield batch
        batch = [item]
        if solo:
            yield batch
            batch = []
    if batch:
        yield batch


def observe(calibration: Calibration, item: Item, timing: dict[str, Any]) -> None:
    """Fold one finished item's measurements into the cost model."""
    if not timing:
        return
    sizes = {stage: mpx for stage, _steps, mpx in item.stages()}
    precision = item.spec.get("render", {}).get("precision") or "fp16"
    for stage, entry in (timing.get("steps") or {}).items():
        mpx = sizes.get(stage)
        mean = entry.get("mean_s")
        if not mpx or not mean:
            continue
        # The stage's own pixel count, not the item's: a hi-res step at 1.5x
        # scale costs 2.25 times a base step and must not be averaged with one.
        side = int((mpx * 1e6) ** 0.5)
        calibration.observe_step(side, side, precision, float(mean))
    for span in timing.get("spans") or []:
        if span.get("name") == "load":
            calibration.observe_load(float(span.get("seconds", 0.0)))
        elif span.get("name") == "decode":
            width, height = span.get("width"), span.get("height")
            if width and height:
                calibration.observe_decode(width * height / 1e6, float(span.get("seconds", 0.0)))


class Runner:
    """Walks the outstanding items, one worker process at a time."""

    def __init__(
        self,
        state: State,
        calibration: Calibration,
        bar: Optional[ProgressBar] = None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.state = state
        self.calibration = calibration
        self.bar = bar or ProgressBar()
        self.force = force
        self.dry_run = dry_run
        self.interrupted = False

    # -- one worker ---------------------------------------------------------

    def _spawn(self, batch: list[Item]) -> subprocess.Popen:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log = LOGS_DIR / f"{stamp}_{batch[0].id.replace('/', '_')}.log"
        command = [sys.executable, "-m", "bench.worker", *[item.id for item in batch]]
        if self.force:
            command.append("--force")
        # The project lives under a path with a non-ASCII character in it, and
        # Python on Windows defaults its pipes to the ANSI code page. Without
        # this the worker mangles every path it reports and the runner then
        # cannot print what it read back.
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            # The child's stderr goes to a file rather than the terminal: torch
            # and diffusers write progress and warnings there, and they would
            # tear the progress bar apart. The path is printed if an item fails.
            stderr=open(log, "w", encoding="utf-8"),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def _follow(self, process: subprocess.Popen, batch: list[Item], progress: RunProgress) -> None:
        """Read the worker's event stream until it closes, updating everything."""
        by_id = {item.id: item for item in batch}
        current: Optional[Item] = None

        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.bar.log(f"  worker said: {line[:200]}")
                continue
            current = self._handle(event, by_id, progress, current)
            self.bar.draw(progress)

    def _handle(
        self,
        event: dict[str, Any],
        by_id: dict[str, Item],
        progress: RunProgress,
        current: Optional[Item],
    ) -> Optional[Item]:
        kind = event.get("event")
        item = by_id.get(event.get("item", ""), current)

        if kind == "start" and item is not None:
            record = self.state.get(item.id)
            record.status = "running"
            record.attempts += 1
            record.started_at = _now()
            record.bundle_dir = str(item.bundle_dir)
            self.state.save()
            width, height = item.size()
            progress.item_id = item.id
            progress.item_title = item.title
            progress.item_units = item.work_units()
            progress.item_fraction = 0.0
            progress.model = item.model
            progress.size = f"{width}x{height}"
            progress.stage = ""
            return item

        if kind == "resuming" and item is not None:
            self.bar.log(
                f"  resuming {item.id} in the {event.get('stage')} stage, "
                f"{event.get('done', 0)} image(s) already done"
            )
            return item

        if kind == "step" and item is not None:
            total = float(event.get("total") or 0.0)
            progress.item_fraction = min(1.0, float(event.get("done", 0.0)) / total) if total else 0.0
            progress.stage = str(event.get("stage", ""))
            progress.step = int(event.get("step", 0))
            progress.steps = int(event.get("steps", 0))
            progress.variation = int(event.get("variation", 0))
            progress.variations = int(event.get("variations", 0))
            rate = progress.rate()
            progress.s_per_step = (
                progress.item_units / progress.steps / rate
                if rate and progress.steps
                else None
            )
            return item

        if kind == "done" and item is not None:
            self._finish(item, event, progress)
            return None

        if kind in {"paused", "aborted"} and item is not None:
            record = self.state.get(item.id)
            record.status = "paused"
            record.images = int(event.get("images", 0))
            record.bundle_dir = event.get("bundle")
            self.state.save()
            self.bar.log(
                f"  paused {item.id} with {record.images} image(s) saved; "
                "run the same command again to carry on"
            )
            return None

        if kind == "failed" and item is not None:
            record = self.state.get(item.id)
            record.status = "failed"
            record.finished_at = _now()
            record.error = str(event.get("error"))[:500]
            self.state.save()
            progress.failed_items += 1
            self.bar.log(f"  FAILED {item.id}: {record.error}")
            return None

        if kind == "asset":
            self.bar.log(f"  wrote {event.get('name')} -> {event.get('path')}")
        elif kind == "pausing":
            self.bar.log("  pausing after the current step...")
        elif kind == "interrupted":
            self.interrupted = True
        return current

    def _finish(self, item: Item, event: dict[str, Any], progress: RunProgress) -> None:
        timing = event.get("timing") or {}
        memory = event.get("memory") or {}
        observe(self.calibration, item, timing)
        self.calibration.save()

        stages = timing.get("stages") or {}
        base = stages.get("base") or {}
        record = self.state.get(item.id)
        record.status = "done"
        record.finished_at = _now()
        record.duration_s = float(event.get("duration_s") or 0.0)
        record.images = int(event.get("images", 0))
        record.bundle_dir = event.get("bundle")
        record.error = None
        record.s_per_step = base.get("s_per_step")
        decode = event.get("decode") or {}
        record.decode = (
            f"{decode.get('device')}{'/tiled' if decode.get('tiled') else ''}"
            if decode
            else None
        )
        record.ram_free_start_gb = memory.get("ram_free_start_gb")
        record.ram_free_min_gb = memory.get("ram_free_min_gb")
        record.vram_peak_gb = memory.get("vram_peak_reserved_gb")
        self.state.save()

        progress.done_units += item.work_units()
        progress.done_items += 1
        progress.item_fraction = 0.0
        progress.item_units = 0.0

        flags = event.get("flags") or []
        note = f"  done {item.id}  {human_duration(record.duration_s)}"
        if record.s_per_step:
            note += f"  {record.s_per_step:.1f} s/step"
        if flags:
            note += f"  flags: {', '.join(flags)}"
        self.bar.log(note)

    # -- the loop -----------------------------------------------------------

    def run(self, plan: Plan, limit: Optional[int] = None) -> RunProgress:
        outstanding = plan.outstanding[:limit] if limit else plan.outstanding
        progress = RunProgress(
            total_units=plan.total_units,
            done_units=plan.done_units,
            resumed_units=plan.done_units,
            total_items=len(plan.all_items),
            done_items=len(plan.all_items) - len(plan.outstanding),
        )
        self.bar.log(
            f"{len(outstanding)} item(s) to run of {len(plan.all_items)}; "
            f"{human_duration(sum(self.calibration.estimate(i) for i in outstanding))} estimated"
        )
        self.bar.draw(progress, force=True)

        for batch in group(outstanding):
            if self.interrupted:
                break
            if self.dry_run:
                for item in batch:
                    self.bar.log(
                        f"  would run {item.id} "
                        f"(~{human_duration(self.calibration.estimate(item))})"
                    )
                continue
            process = self._spawn(batch)
            try:
                self._follow(process, batch, progress)
            except KeyboardInterrupt:
                # The console delivered Ctrl+C to the worker too, so it is
                # already pausing. Keep reading until it closes its pipe:
                # killing it here would throw away the checkpoint it is in the
                # middle of writing, which is the one thing that must not happen.
                self.interrupted = True
                self.bar.log("  interrupted; waiting for the worker to save its checkpoint...")
                try:
                    self._follow(process, batch, progress)
                except KeyboardInterrupt:
                    self.bar.log("  second interrupt: leaving the worker to finish on its own")
            finally:
                if process.stderr is not None:
                    process.stderr.close()
                process.wait()
            self.state.save()

        self.bar.draw(progress, force=True)
        self.bar.clear()
        return progress


__all__ = ["FRESH_PHASES", "LOGS_DIR", "MAX_GROUP", "Plan", "Runner", "group", "observe", "prepare"]
