"""What the machine's power settings are, and whether they spoil a measurement.

A laptop renders slower on battery, and slower again in a power-saving
profile. Nothing fails when that happens: the images are identical, the run
carries on, and only the *numbers* are wrong -- which for the settings phase is
the whole product. Two items rendered on different evenings stop being
comparable and nothing says so.

That is not hypothetical. ``features/full/init-source`` ran at 29.9 s/step
against 23.2 s/step for every other 1024x1024 item in the same phase, +29%,
and the Windows event log puts 85% of its window on battery. It was found
weeks later by correlating ``state.json`` against event 105 in the system log,
which is not a thing anyone should have to do twice.

So this module reads the power state up front, prints it, and makes the person
confirm before a run that may take weeks. Three facts, in the order they matter:

1. **Mains or battery**, from ``GetSystemPowerStatus``. Unambiguous, and the
   one that costs the most. A warning, never a refusal -- the caller may have
   a reason.
2. **The active power scheme**, by its friendly name. Deliberately *not*
   judged by that name: the schemes on this machine are a mix of localised
   Windows ones ("Equilibrado") and English ones Armoury Crate created, so any
   table of "bad" names would be wrong in the next language or the next OEM
   utility. What is checked instead is whether it matches the scheme the
   finished items ran under. A benchmark cares about comparability, not about
   which profile is objectively best, and that question needs no table.
3. **The Windows power-mode overlay**, whose GUIDs *are* fixed by Windows and
   so can be named. Reported for completeness; on this ASUS machine Armoury
   Crate switches the scheme rather than the overlay, so it reads as default.

Stdlib only, and every call is wrapped: an unreadable power state must never
be the reason a four-week run does not start. Anything unmeasurable reads
``None`` and says "unknown", in the manner of ``claudali.sysinfo``.
"""

from __future__ import annotations

import ctypes
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, TextIO

# Fixed by Windows, unlike the scheme GUIDs, so these are safe to name.
DEFAULT_OVERLAY = "00000000-0000-0000-0000-000000000000"
OVERLAYS = {
    DEFAULT_OVERLAY: "the scheme's own default",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}

UNKNOWN = "unknown"


@dataclass(frozen=True)
class PowerState:
    """The power settings at one moment. Any field may be ``None``."""

    on_ac: Optional[bool] = None
    battery_percent: Optional[int] = None
    scheme_guid: Optional[str] = None
    scheme_name: Optional[str] = None
    overlay_guid: Optional[str] = None
    overlay_name: Optional[str] = None

    @property
    def tag(self) -> str:
        """One short string, stored per item and compared between them.

        Readable in ``state.json`` and in the CSV, and stable enough to group
        by: ``"ac/High performance"``, ``"battery/Power saver"``. The battery
        percentage is deliberately left out, because it changes every minute
        and would make every item look like it ran under different conditions.
        """
        if self.on_ac is None and self.scheme_name is None:
            return UNKNOWN
        source = {True: "ac", False: "battery", None: "unknown"}[self.on_ac]
        return f"{source}/{self.scheme_name or UNKNOWN}"

    @property
    def scheme(self) -> str:
        """The scheme half of the tag, which is what comparability turns on."""
        return self.tag.split("/", 1)[-1]

    def describe(self) -> str:
        """Two or three lines a person can check against the machine."""
        if self.on_ac is None:
            power = "  power source     unknown (could not be read)"
        elif self.on_ac:
            charge = f", battery at {self.battery_percent}%" if self.battery_percent is not None else ""
            power = f"  power source     mains{charge}"
        else:
            charge = f" at {self.battery_percent}%" if self.battery_percent is not None else ""
            power = f"  power source     BATTERY{charge}"
        lines = [power, f"  power plan       {self.scheme_name or UNKNOWN}"]
        if self.overlay_name and self.overlay_guid not in (None, DEFAULT_OVERLAY):
            lines.append(f"  power mode       {self.overlay_name}")
        return "\n".join(lines)

    def complaints(self, baseline: Optional[str] = None) -> list[str]:
        """What about this state would make the next measurements untrustworthy.

        ``baseline`` is the scheme the already-finished items ran under, as
        ``history`` returns it. A different one is reported even when it is the
        faster of the two: a benchmark is spoiled by the *change*, not by the
        direction of it.
        """
        out: list[str] = []
        if self.on_ac is False:
            out.append(
                "This machine is on BATTERY. Renders measured now cannot be compared "
                "with ones measured on mains: the last item that ran this way came out "
                "29% slower than its peers."
            )
        elif self.on_ac is None:
            out.append("The power source could not be read, so nothing here checked it.")
        if baseline and self.scheme_name and self.scheme != baseline:
            out.append(
                f"The power plan is {self.scheme_name!r}, but the items already finished "
                f"ran under {baseline!r}. Their timings and these will not be comparable."
            )
        return out


def _windows_state() -> PowerState:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    powrprof = ctypes.windll.powrprof  # type: ignore[attr-defined]

    class _Status(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    class _Guid(ctypes.Structure):
        _fields_ = [
            ("d1", ctypes.c_ulong),
            ("d2", ctypes.c_ushort),
            ("d3", ctypes.c_ushort),
            ("d4", ctypes.c_ubyte * 8),
        ]

        def __str__(self) -> str:
            tail = bytes(self.d4)
            return f"{self.d1:08x}-{self.d2:04x}-{self.d3:04x}-{tail[:2].hex()}-{tail[2:].hex()}"

    on_ac: Optional[bool] = None
    percent: Optional[int] = None
    status = _Status()
    if kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        # 255 means the machine genuinely does not know, which is not the same
        # as being on battery and must not be reported as it.
        if status.ACLineStatus in (0, 1):
            on_ac = bool(status.ACLineStatus)
        if status.BatteryLifePercent != 255:
            percent = int(status.BatteryLifePercent)

    scheme_guid = scheme_name = None
    pointer = ctypes.POINTER(_Guid)()
    if powrprof.PowerGetActiveScheme(None, ctypes.byref(pointer)) == 0:
        try:
            scheme_guid = str(pointer.contents)
            size = ctypes.c_ulong(1024)
            buffer = ctypes.create_string_buffer(size.value)
            if powrprof.PowerReadFriendlyName(
                None, pointer, None, None, buffer, ctypes.byref(size)
            ) == 0:
                scheme_name = buffer.raw[: size.value].decode("utf-16-le").rstrip("\x00") or None
        finally:
            kernel32.LocalFree(pointer)

    overlay_guid = overlay_name = None
    overlay = _Guid()
    try:
        if powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(overlay)) == 0:
            overlay_guid = str(overlay)
            overlay_name = OVERLAYS.get(overlay_guid, overlay_guid)
    except AttributeError:
        pass  # Older Windows: no power-mode slider to read.

    return PowerState(on_ac, percent, scheme_guid, scheme_name, overlay_guid, overlay_name)


def _posix_state() -> PowerState:
    from pathlib import Path

    for supply in sorted(Path("/sys/class/power_supply").glob("A*/online")):
        return PowerState(on_ac=supply.read_text().strip() == "1")
    return PowerState()


def measure() -> PowerState:
    """Read the power state, or return an all-``None`` one if it cannot be read."""
    try:
        return _windows_state() if sys.platform == "win32" else _posix_state()
    except Exception:  # noqa: BLE001 - an unreadable power state stops nothing
        return PowerState()


def history(tags: Iterable[Optional[str]]) -> Optional[str]:
    """The scheme the finished items ran under, or ``None`` if they disagree.

    Disagreement is not an error to report here -- the run has already lost
    that comparison and saying so on every later run would be noise. It simply
    means there is no baseline left to check the next item against.
    """
    seen = Counter(
        tag.split("/", 1)[-1] for tag in tags if tag and tag != UNKNOWN and "/" in tag
    )
    if len(seen) != 1:
        return None
    return next(iter(seen))


def confirm(
    state: Optional[PowerState] = None,
    *,
    baseline: Optional[str] = None,
    assume_yes: bool = False,
    interactive: Optional[bool] = None,
    stream: Optional[TextIO] = None,
) -> bool:
    """Show the power state and wait for the person before a long run.

    Returns whether to go ahead. Enter means yes, because the answer is yes
    almost every time and a run that needs a typed word to start is a run
    someone automates around. ``n`` stops.

    With no terminal -- a scheduled run, output piped to a file -- there is
    nobody to ask, so it prints the state and carries on rather than blocking
    forever on a pipe that will never deliver a newline.
    """
    out = stream or sys.stdout
    state = measure() if state is None else state
    complaints = state.complaints(baseline)

    print("\nBefore a run that may take days:", file=out)
    print("  [ ] plugged in to mains", file=out)
    print("  [ ] performance profile on (Turbo / High performance)", file=out)
    print("  [ ] nothing else heavy running -- a browser costs RAM the decode needs", file=out)
    print("\nThis machine, right now:", file=out)
    print(state.describe(), file=out)
    for line in complaints:
        print(f"\n  !! {line}", file=out)

    if assume_yes:
        return True
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not interactive:
        print("\n(not a terminal: starting without asking)", file=out)
        return True

    print("", file=out)
    try:
        answer = input("Press Enter to start, or n to stop: ").strip().lower()
    except EOFError:
        return True
    return not answer.startswith("n")


__all__ = [
    "DEFAULT_OVERLAY",
    "OVERLAYS",
    "UNKNOWN",
    "PowerState",
    "confirm",
    "history",
    "measure",
]
