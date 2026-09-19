"""The whole-run progress bar and its ETA.

A per-job bar is no use over a run measured in weeks: what you want to know at
a glance is how much of the *benchmark* is done and when it will finish. So
the unit here is the megapixel-step, summed over every item in the plan, and
the bar counts the ones actually completed plus the fraction of the item in
flight.

The percentage is printed to four decimals on purpose. At this scale one
sampler step is around 0.0005% of the total, so two decimals would sit still
for ten minutes at a time and look like a hang. Four decimals move on every
step, which is the point: you should be able to tell in thirty seconds that it
is still working.

Stdlib only.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, TextIO

BAR_WIDTH = 34


def human_duration(seconds: Optional[float]) -> str:
    """``3d 04h 12m`` -- fixed vocabulary, so the line does not jump about."""
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN-safe
        return "unknown"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@dataclass
class RunProgress:
    """Where the whole benchmark stands.

    ``done_units`` is work finished before the current item; ``item_units`` and
    ``item_fraction`` describe the one in flight. Keeping them apart is what
    lets the bar move smoothly within an item without ever going backwards when
    one finishes.
    """

    total_units: float
    done_units: float = 0.0
    total_items: int = 0
    done_items: int = 0
    failed_items: int = 0

    item_id: str = ""
    item_title: str = ""
    item_units: float = 0.0
    item_fraction: float = 0.0
    stage: str = ""
    step: int = 0
    steps: int = 0
    variation: int = 0
    variations: int = 0
    model: str = ""
    size: str = ""
    s_per_step: Optional[float] = None

    started: float = field(default_factory=time.monotonic)
    # Work units finished before this session started, so the rate is measured
    # only over what this session actually ran. Resuming a run that is 60% done
    # must not claim credit for the 60% when estimating the remainder.
    resumed_units: float = 0.0

    # -- derived ------------------------------------------------------------

    @property
    def units_now(self) -> float:
        return self.done_units + self.item_units * self.item_fraction

    @property
    def fraction(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return min(1.0, self.units_now / self.total_units)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def rate(self) -> Optional[float]:
        """Work units per second, measured over this session only."""
        ran = self.units_now - self.resumed_units
        return ran / self.elapsed if ran > 0 and self.elapsed > 0 else None

    def remaining_seconds(self) -> Optional[float]:
        rate = self.rate()
        if not rate:
            return None
        return max(0.0, self.total_units - self.units_now) / rate

    def finish_time(self) -> Optional[datetime]:
        remaining = self.remaining_seconds()
        return None if remaining is None else datetime.now() + timedelta(seconds=remaining)


class ProgressBar:
    """Draws :class:`RunProgress` over three lines, in place."""

    def __init__(self, stream: Optional[TextIO] = None, enabled: bool = True) -> None:
        self.stream = stream or sys.stdout
        # Without a terminal -- piped to a file, or under CI -- redrawing in
        # place fills the log with escape codes, so fall back to a line per
        # item instead of a live bar.
        self.enabled = enabled and self.stream.isatty()
        self._lines = 0
        self._last = 0.0

    def _width(self) -> int:
        return max(60, shutil.get_terminal_size((100, 24)).columns - 1)

    def render(self, progress: RunProgress) -> list[str]:
        width = self._width()
        filled = int(progress.fraction * BAR_WIDTH)
        bar = "#" * filled + "." * (BAR_WIDTH - filled)

        head = (
            f"[{bar}] {progress.fraction * 100:9.4f}%   "
            f"item {progress.done_items + 1}/{progress.total_items}"
        )
        if progress.failed_items:
            head += f"   {progress.failed_items} failed"

        detail = f"  {progress.item_id}"
        if progress.stage:
            detail = (
                f"  {progress.stage:<8s} variation {progress.variation}/{progress.variations}"
                f"  step {progress.step}/{progress.steps}"
            )
            if progress.s_per_step:
                detail += f"   {progress.s_per_step:5.1f} s/step"
            detail += f"   {progress.model} {progress.size}"

        finish = progress.finish_time()
        when = finish.strftime("%a %d %b %H:%M") if finish else "unknown"
        tail = (
            f"  {human_duration(progress.elapsed)} elapsed"
            f"  ·  {human_duration(progress.remaining_seconds())} left"
            f"  ·  finish {when}"
        )
        return [line[:width] for line in (head, detail, tail)]

    def draw(self, progress: RunProgress, force: bool = False) -> None:
        # Redraw at most five times a second: a sampler step takes seconds, but
        # the elapsed clock should still look alive.
        now = time.monotonic()
        if not force and now - self._last < 0.2:
            return
        self._last = now
        lines = self.render(progress)
        if not self.enabled:
            return
        if self._lines:
            self.stream.write(f"\033[{self._lines}A")
        for line in lines:
            self.stream.write("\r\033[K" + line + "\n")
        self._lines = len(lines)
        self.stream.flush()

    def clear(self) -> None:
        """Leave the last drawn frame in place and stop tracking it."""
        self._lines = 0
        if self.enabled:
            self.stream.flush()

    def log(self, message: str) -> None:
        """Print a line above the bar, keeping the bar at the bottom."""
        if self.enabled and self._lines:
            self.stream.write(f"\033[{self._lines}A")
            self.stream.write("\r\033[J")
            self._lines = 0
        self.stream.write(message.rstrip() + "\n")
        self.stream.flush()


__all__ = ["BAR_WIDTH", "ProgressBar", "RunProgress", "human_duration"]
