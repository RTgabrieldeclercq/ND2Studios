"""V1.35 Phase 3 — byte-budgeted LRU frame cache with pinning and stats.

Stores normalized 2-D ``(H, W)`` numpy arrays keyed by
``(channel_idx, m, t, z, z_mode)`` with a byte-budget eviction policy.
Three threads touch this cache concurrently:

- the GUI thread (cache hit / pin / unpin / stats read on slider events),
- the :class:`~nd2studios.workers.prefetch_worker.PrefetchManager`
  thread (low-priority neighbor reads),
- the V1.34 :class:`~nd2studios.utils.threading.IOWorker` thread
  (foreground display reads).

Every public method is protected by a single :class:`threading.Lock`,
so the lock is the throughput ceiling — keep critical sections tiny
(``OrderedDict`` ops only; never call user code or compute under it).

V1.35 additions (Phase 3):

- :class:`CacheStats` — hits / misses / evictions / current_bytes /
  peak_bytes, exposed via :attr:`FrameCache.stats`. Read from any
  thread; mutated under the cache lock.
- :meth:`pin` / :meth:`unpin` / :meth:`is_pinned` — the currently-
  displayed key set is exempt from eviction. Prevents the prefetcher
  from kicking the on-screen plane out of the cache when the budget
  is tight.
- :meth:`set_max_bytes` — adjust the budget on the fly. Used by
  `MultiAxisViewer` when a file is opened and `psutil` reports the
  available RAM has changed since startup.
- ``writeable=False`` flag set on every cached array so a downstream
  mutator can't poison the cache for the next hit.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np


@dataclass
class CacheStats:
    """Hit / miss / eviction counters for a :class:`FrameCache`.

    Updated by the cache itself under its lock; readable from any
    thread. Reset to zero only when the cache is rebuilt — :meth:`clear`
    keeps the counters so a session-long hit rate stays meaningful.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_bytes: int = 0
    peak_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Cumulative hit rate in ``[0.0, 1.0]``; 0.0 before any access."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class FrameCache:
    """LRU frame cache with a configurable byte budget and pinning.

    Parameters
    ----------
    max_bytes:
        Maximum total frame data to hold. Defaults to 300 MB for
        backward compatibility with pre-V1.34 callers; production
        code should pass
        :func:`nd2studios.utils.resources.recommended_cache_budget_bytes`.
    """

    def __init__(self, max_bytes: int = 300 * 1024 ** 2) -> None:
        self._max_bytes = int(max_bytes)
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._sizes: dict[tuple, int] = {}
        self._pinned: Set[tuple] = set()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    # ── public API ──────────────────────────────────────────────────────────

    def get(self, key: tuple) -> Optional[np.ndarray]:
        """Return the cached frame for *key*, or ``None`` on a miss.

        Promotes the entry to most-recently-used on a hit and bumps
        :attr:`stats`.
        """
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: tuple, frame: np.ndarray) -> None:
        """Store *frame* under *key*, evicting unpinned LRU entries.

        The stored array is marked ``writeable=False`` so accidental
        in-place mutation by a downstream consumer fails loudly with
        ``ValueError`` instead of silently corrupting future hits.
        Callers that need a mutable copy should explicitly
        ``np.array(cache_hit)``.
        """
        if frame is None:
            return
        # Freeze the handle we hand back on subsequent .get()s. Views
        # are frozen only at the view level; the underlying buffer is
        # untouched — which is fine, because the cache only protects
        # the *handle* it stores, not whatever the producer owns.
        try:
            frame.flags.writeable = False
        except (AttributeError, ValueError):
            # Some numpy variants (read-only views, masked arrays)
            # don't accept the flip; the cache is no worse off.
            pass

        nbytes = int(frame.nbytes)
        with self._lock:
            if key in self._cache:
                self.stats.current_bytes -= self._sizes[key]
                del self._cache[key]
            self._cache[key] = frame
            self._sizes[key] = nbytes
            self.stats.current_bytes += nbytes
            if self.stats.current_bytes > self.stats.peak_bytes:
                self.stats.peak_bytes = self.stats.current_bytes
            self._evict_if_needed_locked()

    def contains(self, key: tuple) -> bool:
        """Return ``True`` if *key* is cached (no LRU promotion, no stats)."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Evict all entries and drop every pin.

        Stats (hits / misses / evictions / peak_bytes) are *not* reset —
        a session-long hit rate stays meaningful across file changes.
        Only ``current_bytes`` resets because there are no bytes left.
        """
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._pinned.clear()
            self.stats.current_bytes = 0

    # ── V1.35 Phase 3 additions ─────────────────────────────────────────────

    def pin(self, key: tuple) -> None:
        """Exempt *key* from eviction while it remains in the cache.

        Used by :class:`~nd2studios.widgets.multi_axis_viewer.MultiAxisViewer`
        to protect the currently-displayed plane from being evicted by
        the prefetcher's neighbor reads on a tight budget. Safe to call
        with a key that is not yet present — eviction simply skips it
        until it appears.
        """
        with self._lock:
            self._pinned.add(key)

    def unpin(self, key: tuple) -> None:
        """Remove *key* from the pinned set; no-op if not pinned."""
        with self._lock:
            self._pinned.discard(key)

    def is_pinned(self, key: tuple) -> bool:
        with self._lock:
            return key in self._pinned

    def set_max_bytes(self, max_bytes: int) -> None:
        """Resize the budget and immediately evict to fit."""
        with self._lock:
            self._max_bytes = int(max_bytes)
            self._evict_if_needed_locked()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        """Current resident bytes; identical to ``stats.current_bytes``."""
        with self._lock:
            return self.stats.current_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    # ── internal ────────────────────────────────────────────────────────────

    def _evict_if_needed_locked(self) -> None:
        """Evict LRU unpinned entries until we are under budget.

        Caller must hold :attr:`_lock`. The walk iterates the
        ``OrderedDict`` from oldest to newest and skips pinned keys;
        if every remaining entry is pinned we stop, accepting that the
        cache is over budget rather than evict a displayed plane. This
        is the trade Phase 3 makes — pinning beats the byte ceiling.
        """
        if self.stats.current_bytes <= self._max_bytes:
            return
        to_remove: list[tuple] = []
        # Iterate the keys in LRU order (oldest first) without
        # mutating during iteration.
        for k in self._cache:
            if self.stats.current_bytes <= self._max_bytes:
                break
            if k in self._pinned:
                continue
            to_remove.append(k)
            self.stats.current_bytes -= self._sizes[k]
            self.stats.evictions += 1
        for k in to_remove:
            del self._cache[k]
            del self._sizes[k]
