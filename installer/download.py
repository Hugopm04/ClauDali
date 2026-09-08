"""Resumable HTTP downloads with byte-level progress.

Written directly against urllib rather than delegating to ``huggingface_hub``
for two reasons: it lets the installer drive its own progress bar from real
byte counts, and it writes files into a plain directory layout instead of the
HF blob cache, whose symlinks need Developer Mode on Windows.

Resume is not a nicety here. A 7 GB checkpoint over a domestic connection will
be interrupted eventually, and restarting from zero each time is how an install
never finishes.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

CHUNK = 1024 * 256
USER_AGENT = "claudali-installer/0.1"

ProgressHook = Callable[[int], None]


class DownloadError(RuntimeError):
    """A download failed after exhausting its retries."""


def _open(url: str, offset: int = 0, timeout: int = 60):
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    on_progress: Optional[ProgressHook] = None,
    attempts: int = 5,
) -> Path:
    """Download ``url`` to ``destination``, resuming a partial file if present.

    ``on_progress`` is called with the number of new bytes written, so a caller
    can drive one bar across many files. Already-complete files are skipped and
    reported through the same hook, which keeps the bar honest when an install
    is re-run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and expected_size and destination.stat().st_size == expected_size:
        if on_progress:
            on_progress(expected_size)
        return destination

    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        offset = part.stat().st_size if part.is_file() else 0
        if expected_size and offset > expected_size:
            # A stale part file larger than the target means the remote content
            # changed; starting over is the only safe option.
            part.unlink()
            offset = 0
        if offset and on_progress and attempt == 0:
            on_progress(offset)

        try:
            with _open(url, offset=offset) as response:
                # A server that ignores Range replies 200 and restarts the body,
                # so the local offset must be discarded to avoid a corrupt file.
                if offset and response.status == 200:
                    offset = 0
                    part.unlink(missing_ok=True)

                mode = "ab" if offset else "wb"
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if on_progress:
                            on_progress(len(chunk))

            size = part.stat().st_size
            if expected_size and size != expected_size:
                raise DownloadError(
                    f"size mismatch for {destination.name}: got {size}, expected {expected_size}"
                )

            part.replace(destination)
            return destination

        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                delay = 2.0 * (2**attempt)
                time.sleep(delay)

    raise DownloadError(f"failed to download {url} after {attempts} attempts: {last_error}")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


__all__ = ["DownloadError", "directory_size", "download_file"]
