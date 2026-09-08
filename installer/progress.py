"""Progress rendering for the installer and uninstaller.

Written against the standard library only, on purpose. A progress bar that first
has to ``pip install`` a progress bar library cannot show progress for its own
bootstrap, and would leave a package behind that the uninstaller then has to
know about. Everything here is carriage-return redraws, which work identically
in PowerShell, cmd, Windows Terminal and any POSIX shell.

Bars only ever report measured quantities: bytes actually written, files
actually deleted, steps actually completed. Where a stage's progress genuinely
cannot be measured -- pip resolving a dependency tree -- it shows an elapsed
timer and the live output line instead of an animation that implies knowledge
the installer does not have.
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Optional

BLOCK_FULL = "#"
BLOCK_EMPTY = "-"


def _make_stdout_safe() -> None:
    """Ensure writing to stdout can never itself raise.

    Redirecting the installer to a log file gives stdout the Windows ANSI
    codepage, which cannot encode a path containing an accented character. That
    turns a useful error message into a UnicodeEncodeError traceback and hides
    whatever actually went wrong -- exactly when the user most needs to see it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Already detached or not a real stream; nothing to do.
            pass


_make_stdout_safe()


def human_bytes(count: float) -> str:
    """Format a byte count in decimal units (1 GB = 1000 MB).

    Decimal rather than binary so that a 7.14 GB file reported by the model
    catalogue does not appear as 6.65 GB in the download bar. Two numbers for
    one file is the kind of detail that makes a user distrust the whole tool.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1000 or unit == "TB":
            return f"{int(count)} B" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1000
    return f"{count:.1f} TB"


def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def terminal_width(default: int = 100) -> int:
    try:
        return max(60, min(shutil.get_terminal_size().columns, 160))
    except Exception:  # noqa: BLE001 - not worth failing an install over
        return default


class Bar:
    """A single-line progress bar over a known total.

    ``unit='bytes'`` formats amounts and rate in human units; anything else
    counts plain items.
    """

    def __init__(self, label: str, total: float, unit: str = "bytes", width: int = 28) -> None:
        self.label = label
        self.total = max(total, 1e-9)
        self.unit = unit
        self.width = width
        self.current = 0.0
        self.started = time.monotonic()
        self._last_draw = 0.0

    def advance(self, amount: float) -> None:
        self.current += amount
        self.draw()

    def set(self, value: float) -> None:
        self.current = value
        self.draw()

    def _format_amounts(self) -> str:
        if self.unit == "bytes":
            return f"{human_bytes(self.current)}/{human_bytes(self.total)}"
        return f"{int(self.current)}/{int(self.total)}"

    def draw(self, force: bool = False) -> None:
        now = time.monotonic()
        # Redrawing on every chunk of a 7 GB download would spend more time on
        # the terminal than on the network.
        if not force and now - self._last_draw < 0.1:
            return
        self._last_draw = now

        fraction = min(1.0, self.current / self.total)
        elapsed = max(now - self.started, 1e-6)
        rate = self.current / elapsed
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        columns = terminal_width() - 1
        label = self.label if len(self.label) <= 22 else self.label[:21] + "~"

        # Progressively drop detail rather than truncating the label or the bar
        # into uselessness: a narrow terminal should lose the transfer rate, not
        # the name of what is being transferred.
        amounts = self._format_amounts()
        suffixes = []
        if self.unit == "bytes" and rate > 0:
            suffixes.append(f" {amounts}  {human_bytes(rate)}/s  ETA {human_time(remaining)}")
        if rate > 0:
            suffixes.append(f" {amounts}  ETA {human_time(remaining)}")
        suffixes.extend([f" {amounts}", ""])

        fixed = len(label) + 12  # two leading spaces, brackets, percentage
        for suffix in suffixes:
            bar_width = min(self.width, columns - fixed - len(suffix))
            if bar_width >= 10:
                break
        else:
            suffix, bar_width = "", max(4, columns - fixed)

        filled = int(bar_width * fraction)
        bar = BLOCK_FULL * filled + BLOCK_EMPTY * (bar_width - filled)

        line = f"  {label} [{bar}] {fraction*100:5.1f}%{suffix}"
        sys.stdout.write("\r" + line[:columns].ljust(columns))
        sys.stdout.flush()

    def finish(self, note: str = "") -> None:
        self.current = self.total
        self.draw(force=True)
        sys.stdout.write(f"  {note}\n" if note else "\n")
        sys.stdout.flush()


class Spinner:
    """For stages whose progress genuinely cannot be measured.

    Shows elapsed time and the most recent output line, so the user can see the
    stage is alive and what it is doing, without a bar pretending to know how
    far along it is.
    """

    FRAMES = "|/-\\"

    def __init__(self, label: str) -> None:
        self.label = label
        self.started = time.monotonic()
        self.frame = 0
        self.detail = ""
        self._last_draw = 0.0

    def update(self, detail: str = "") -> None:
        if detail:
            self.detail = detail.strip()
        now = time.monotonic()
        if now - self._last_draw < 0.12:
            return
        self._last_draw = now
        self.frame = (self.frame + 1) % len(self.FRAMES)

        elapsed = human_time(now - self.started)
        head = f"  {self.FRAMES[self.frame]} {self.label}  [{elapsed}]  "
        detail = self.detail[: max(0, terminal_width() - len(head) - 2)]
        sys.stdout.write("\r" + (head + detail).ljust(terminal_width() - 1))
        sys.stdout.flush()

    def finish(self, note: str = "done") -> None:
        elapsed = human_time(time.monotonic() - self.started)
        line = f"  + {self.label}  [{elapsed}]  {note}"
        sys.stdout.write("\r" + line.ljust(terminal_width() - 1) + "\n")
        sys.stdout.flush()


class Steps:
    """The overall installer progress: N discrete stages, and where we are."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.index = 0

    def start(self, name: str) -> None:
        self.index += 1
        total = len(self.names)
        header = f"[{self.index}/{total}] {name}"
        rule = "=" * min(terminal_width() - 1, max(len(header) + 2, 60))
        print(f"\n{rule}\n{header}\n{rule}")

    def overall(self) -> str:
        done = self.index
        total = len(self.names)
        width = 28
        filled = int(width * done / max(1, total))
        return f"[{BLOCK_FULL * filled}{BLOCK_EMPTY * (width - filled)}] {done}/{total} stages"


def rule(title: str = "") -> None:
    width = min(terminal_width() - 1, 78)
    if title:
        print(f"\n{title}\n{'-' * width}")
    else:
        print("-" * width)


def confirm(question: str, expected: Optional[str] = None) -> bool:
    """Ask before doing something irreversible.

    When ``expected`` is given the user must type that exact word, which is the
    right friction for a command that deletes tens of gigabytes.
    """
    if expected:
        answer = input(f"{question}\n  Type '{expected}' to confirm: ").strip()
        return answer == expected
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


__all__ = ["Bar", "Spinner", "Steps", "confirm", "human_bytes", "human_time", "rule"]
