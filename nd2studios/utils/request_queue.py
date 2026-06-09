"""Cancel-first single-slot request mailbox (V1.42).

Borrowed pattern from napari's NAP-4 async slicer: when a new
request arrives, any in-flight peer is *immediately* marked
cancelled. Workers cooperate by polling the cancel flag at known
checkpoints and bailing out early. Only the most recent request is
honored — intermediate work is dropped.

Use cases in ND2Studios
-----------------------

* **Slider scrubbing.** A user dragging the T slider fires dozens of
  ``frame_changed`` events per second. Today the existing 5 ms
  debounce coalesces drags but doesn't *abort* a compose that has
  already started. Routing the compose request through a mailbox
  means the in-flight job stops at the next cancel checkpoint and
  the GUI thread isn't blocked when the user releases the slider.
* **Post-process overlay re-render.** ``frame_post_process``
  callbacks (analysis overlays) can take 30+ ms; chained calls
  during fast scrubs queue up. A mailbox-wrapped worker drops
  intermediate requests.
* **Pyramid-level transition.** When zoom changes, the in-flight
  level-N read is no longer relevant; cancel it and start the
  level-N±1 read.

This module ships the utility; consumers wire it in incrementally
so V1.42 doesn't churn every long-running call site at once.

Design notes
------------

The mailbox does **not** start any threads — it owns a token
representing the *current* request. Workers (callable, ``QThread``,
``QRunnable``, or plain function) read the token's ``cancelled``
flag at their own cancel points. This keeps the utility small,
thread-safe, and orthogonal to whatever execution model the caller
prefers.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RequestToken:
    """Handle representing one request through a :class:`SingleSlotMailbox`.

    Workers consult :attr:`cancelled` at known checkpoints and exit
    early when it is ``True``.  The token is cheap to pass to a
    QThread, a callable run on a thread pool, or a generator that
    polls between yields.
    """

    request_id: int
    payload: Any = None
    _cancelled: bool = False
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        """Thread-safe read of the cancel flag."""
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        """Mark the token as cancelled. Idempotent and thread-safe."""
        with self._lock:
            self._cancelled = True


class SingleSlotMailbox:
    """Hold at most one in-flight :class:`RequestToken`.

    Calling :meth:`submit` cancels the previous token (if any) and
    returns a fresh one.  The caller hands the new token to its
    worker; the worker is expected to bail out when ``token.cancelled``
    flips to ``True``.

    The mailbox itself does **not** spawn workers; it is a thin
    coordination primitive.  Pair it with whatever execution model
    your callsite already uses (``QThread`` subclass, function passed
    to a thread pool, generator that polls between yields).

    Example::

        mailbox = SingleSlotMailbox()

        def on_slider_changed(t: int) -> None:
            token = mailbox.submit(payload=t)
            # Hand `token` to a worker — it can be passed to a
            # QThread's __init__, captured in a closure, etc.
            run_in_background(token)

        def run_in_background(token: RequestToken) -> None:
            for step in range(N):
                if token.cancelled:
                    return  # next request is already in flight
                do_step(step)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[RequestToken] = None
        self._next_id: int = 0

    def submit(self, payload: Any = None) -> RequestToken:
        """Cancel any in-flight request, mint a fresh token, return it."""
        with self._lock:
            if self._current is not None:
                self._current.cancel()
            self._next_id += 1
            token = RequestToken(request_id=self._next_id, payload=payload)
            self._current = token
            return token

    def cancel_current(self) -> None:
        """Cancel the in-flight request without minting a new one.

        Useful when the user navigates away from the relevant view
        (closes the dialog, switches tabs) — no replacement work is
        needed, just stop the running job.
        """
        with self._lock:
            if self._current is not None:
                self._current.cancel()
                self._current = None

    def current(self) -> Optional[RequestToken]:
        """Return the in-flight token, or ``None`` if the mailbox is empty."""
        with self._lock:
            return self._current

    def is_current(self, token: RequestToken) -> bool:
        """Return ``True`` if ``token`` is still the active request.

        A worker can call this after finishing its work but before
        publishing results, so a result that arrives *after* a newer
        request was submitted is silently dropped.
        """
        with self._lock:
            return self._current is token


class PriorityLanes:
    """Two-lane variant: a foreground (cancel-first) mailbox plus a
    lower-priority lane for opportunistic prefetch.

    The foreground lane behaves identically to :class:`SingleSlotMailbox`.
    The background lane runs FIFO and *does not* cancel its previous
    request when a new one arrives; instead, callers cancel the
    background lane wholesale when foreground work needs the device.

    Use for: viewport-driven slice reads (foreground) versus
    look-ahead prefetch of neighbouring T frames (background).
    """

    def __init__(self) -> None:
        self.foreground = SingleSlotMailbox()
        # ``background`` is intentionally a list of tokens — newest
        # last.  Consumers iterate front-to-back; on foreground cancel
        # they cancel all background tokens too.
        self._bg_lock = threading.Lock()
        self._background: list[RequestToken] = []
        self._bg_next_id: int = 0

    def submit_foreground(self, payload: Any = None) -> RequestToken:
        """Submit a high-priority request. Cancels any in-flight peer.

        Background tokens are also cancelled so the device isn't
        contested while the new foreground work runs.
        """
        with self._bg_lock:
            for tok in self._background:
                tok.cancel()
            self._background.clear()
        return self.foreground.submit(payload)

    def submit_background(self, payload: Any = None) -> RequestToken:
        """Submit an opportunistic request behind the foreground lane."""
        with self._bg_lock:
            self._bg_next_id += 1
            token = RequestToken(
                request_id=10_000_000 + self._bg_next_id,
                payload=payload,
            )
            self._background.append(token)
            return token

    def cancel_all_background(self) -> None:
        """Cancel every queued background token."""
        with self._bg_lock:
            for tok in self._background:
                tok.cancel()
            self._background.clear()

    def background_tokens(self) -> list[RequestToken]:
        """Snapshot of active background tokens (front = oldest)."""
        with self._bg_lock:
            return list(self._background)
