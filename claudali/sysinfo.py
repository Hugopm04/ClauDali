"""How much memory the machine has to spare, measured with the standard library.

The VAE decode policy needs one number -- physical RAM available right now --
to decide whether an untiled fp32 decode on the CPU will fit. psutil would give
it, but a dependency for one reading is not worth it, and importing this module
must stay free for the API server.

Every reading returns ``None`` when it cannot be taken. Callers treat that as
"unknown", never as "plenty": a missing measurement must not pick the path that
can run the machine out of memory.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Optional


class _MemoryStatus(ctypes.Structure):
    """``MEMORYSTATUSEX`` from the Windows API."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows() -> Optional[tuple[float, float]]:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    function = ctypes.windll.kernel32.GlobalMemoryStatusEx  # type: ignore[attr-defined]
    # Without argtypes ctypes guesses at the pointer's width, which fails on 64-bit.
    function.argtypes = [ctypes.POINTER(_MemoryStatus)]
    function.restype = ctypes.c_bool
    if not function(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 1e9, status.ullTotalPhys / 1e9


def _linux() -> Optional[tuple[float, float]]:
    values: dict[str, float] = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[name] = float(parts[0]) * 1024 / 1e9  # reported in kB
    if "MemAvailable" not in values or "MemTotal" not in values:
        return None
    return values["MemAvailable"], values["MemTotal"]


def memory_gb() -> Optional[tuple[float, float]]:
    """``(available, total)`` physical RAM in decimal GB, or None when unmeasurable."""
    try:
        if sys.platform == "win32":
            return _windows()
        if sys.platform.startswith("linux"):
            return _linux()
    except Exception:  # noqa: BLE001 - a reading must never break a render
        return None
    return None


def available_ram_gb() -> Optional[float]:
    """Physical RAM available right now, in decimal GB, or None when unmeasurable."""
    reading = memory_gb()
    return None if reading is None else reading[0]


__all__ = ["available_ram_gb", "memory_gb"]
