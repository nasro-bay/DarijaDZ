"""Atomic file replace with retry.

On Windows, `Path.replace()` can transiently fail with `PermissionError`
(WinError 5) if something else (antivirus, OneDrive sync, an editor/IDE
watching the file) briefly holds a lock on the destination. A short retry
with backoff clears that up without treating it as a real failure.
"""
from __future__ import annotations

import time
from pathlib import Path

MAX_ATTEMPTS = 6
BASE_DELAY_SECONDS = 0.1


def replace_with_retry(src: Path, dst: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(BASE_DELAY_SECONDS * (attempt + 1))
    assert last_error is not None
    raise last_error
