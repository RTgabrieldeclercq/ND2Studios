"""Thread-pool ``map`` over plane indices.

Addendum Phase 5, Pattern 1. Use for per-plane functions whose hot path
drops into a GIL-releasing C extension — numpy ufuncs, scipy.ndimage,
scikit-image, tifffile, zarr decompression. ND2Studios's analysis
pipelines (tear detection, histogram thresholding, spot detection)
are all of this shape, so this is the right primitive for
parallelising them.

For pure-Python CPU loops the GIL will serialise threads — switch to
:func:`nd2studios.compute.parallel.process_map.process_map_planes` in
that case. The decision rule is in
:mod:`nd2studios.compute.parallel.__init__`.

Cancellation is cooperative: pass ``cancelled_cb`` and the loop will
short-circuit between iterations. In-flight tasks finish their current
plane before unwinding (typically < 1 s for ND2Studios's analysis
shape).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.utils.resources import recommended_worker_count


def thread_map_planes(
    stack: np.ndarray,
    fn: Callable[..., Any],
    indices: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Dict[Any, Any]:
    """Apply ``fn(stack[idx], **kwargs)`` across *indices* on a thread pool.

    Returns ``{idx: result}``. Order is the order *indices* completes
    in; sort or re-index the dict if the caller needs T-order.

    Parameters
    ----------
    stack:
        Source array. ``stack[idx]`` is what each task receives.
    fn:
        Per-plane callable. May be any callable — closures and bound
        methods work, unlike :func:`process_map_planes`.
    indices:
        Iterable of plane keys. Defaults to ``range(len(stack))``.
    kwargs:
        Extra keyword arguments forwarded to every ``fn(...)`` call.
    n_workers:
        Thread count. Defaults to
        :func:`nd2studios.utils.resources.recommended_worker_count`.
    progress_cb:
        Called as ``progress_cb(done, total)`` after each completed
        plane. Receivers should treat invocations as low-frequency
        (one per plane) and forward to a Qt slot via queued connection
        if running under the GUI.
    cancelled_cb:
        Polled between completions. Returning ``True`` short-circuits
        the loop; any planes already running finish but the rest are
        abandoned.
    """
    kwargs = kwargs or {}
    n_workers = n_workers or recommended_worker_count()
    if indices is None:
        indices = list(range(len(stack)))
    if not indices:
        return {}

    total = len(indices)
    results: Dict[Any, Any] = {}

    def _one(idx):
        return idx, fn(stack[idx], **kwargs)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_one, idx): idx for idx in indices}
        done = 0
        for fut in as_completed(futures):
            if cancelled_cb is not None and cancelled_cb():
                for f in futures:
                    f.cancel()
                break
            try:
                idx, value = fut.result()
            except Exception:
                # Re-raise on the calling thread so the analysis job
                # surfaces it via the normal JobResult.error path.
                raise
            results[idx] = value
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
    return results
