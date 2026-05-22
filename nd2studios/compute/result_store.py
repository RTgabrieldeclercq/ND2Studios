"""Small in-memory keyed cache for job outputs.

The Analysis page uses this to remember the most recent preview
result across param changes, so toggling the overlay button or
moving the viewer slider does not require re-running the
preview job. Phase 6 will sit a Zarr-backed store in front of
this one for committed pipeline outputs that are too large to
hold in RAM.

Concurrency: a single :class:`threading.RLock` guards every
operation. The store is touched from the GUI thread (when the
Analysis page reads it during a render) and from job-runner
threads (only via the public API, which always copies references
out under the lock). Numpy arrays placed in the store are *not*
deep-copied — callers must not mutate them after :meth:`set`.
"""
from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Optional


class ResultStore:
    """Keyed in-memory cache for analysis-job outputs.

    Example
    -------
    >>> store = ResultStore()
    >>> store.set("analysis_preview", some_result)
    >>> store.get("analysis_preview") is some_result
    True
    >>> store.drop("analysis_preview")
    >>> store.get("analysis_preview") is None
    True
    """

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}
        self._lock = RLock()

    def set(self, key: str, value: Any) -> None:
        """Insert or replace the value stored under ``key``."""
        with self._lock:
            self._results[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Return the value stored under ``key``, or ``None``."""
        with self._lock:
            return self._results.get(key)

    def drop(self, key: str) -> None:
        """Forget the value under ``key``. No-op if absent."""
        with self._lock:
            self._results.pop(key, None)

    def clear(self) -> None:
        """Empty the store."""
        with self._lock:
            self._results.clear()

    def keys(self) -> list[str]:
        """Snapshot of currently-cached keys."""
        with self._lock:
            return list(self._results.keys())
