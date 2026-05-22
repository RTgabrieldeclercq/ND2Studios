"""Process-pool ``map`` over plane indices with shared memory.

Addendum Phase 5, Improvement 2. Use this when the per-plane
function is **pure-Python CPU-bound** — i.e. a tight numpy / Python
loop that the GIL serialises across threads. For C-extension work
(scipy / skimage / numpy ufuncs that already release the GIL) prefer
:func:`nd2studios.compute.parallel.thread_map.thread_map_planes`,
which avoids the pickle and process-spawn cost.

The worker function must be **importable** (module-level), because
each child process pickles a reference to it. Closures, lambdas, and
inner classes will not work.

Example
-------
::

    # In some module nd2studios.compute.operations.my_op:
    def per_plane(plane, threshold):
        # heavy pure-Python work that benefits from real parallelism
        return ...

    # Caller:
    from nd2studios.compute.parallel import process_map_planes
    results = process_map_planes(
        stack,                          # (T, H, W) ndarray
        "nd2studios.compute.operations.my_op",
        "per_plane",
        indices=list(range(stack.shape[0])),
        kwargs={"threshold": 0.5},
    )
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

from nd2studios.compute.parallel.shared_array import (
    attach_shared,
    shared_ndarray,
)
from nd2studios.utils.resources import recommended_worker_count


def _worker_apply(args):  # pragma: no cover — runs in a subprocess
    """Worker entry point — module-level so it can be pickled."""
    import importlib

    name, shape, dtype_str, idx, fn_module, fn_name, kwargs = args
    module = importlib.import_module(fn_module)
    fn = getattr(module, fn_name)
    arr, shm = attach_shared(name, shape, dtype_str)
    try:
        return idx, fn(arr[idx], **kwargs)
    finally:
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass


def process_map_planes(
    stack: np.ndarray,
    fn_module: str,
    fn_name: str,
    indices: List[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
) -> Dict[Any, Any]:
    """Apply ``fn_module.fn_name(stack[idx], **kwargs)`` across *indices*.

    Each call runs in a child process; *stack* is exposed via
    :func:`shared_ndarray` so no per-task pickle cost is paid. Returns
    a ``{idx: result}`` dict keyed by whatever Python objects the
    caller passed in ``indices`` (typically ``int``\\s along T).

    Parameters
    ----------
    stack:
        Source array. The worker receives ``stack[idx]`` (so for a
        ``(T, H, W)`` stack with integer indices, the worker sees a
        ``(H, W)`` plane).
    fn_module, fn_name:
        Import path of the worker function. ``importlib.import_module``
        must succeed on ``fn_module`` inside the child process — pass a
        fully-qualified package path, not a relative one.
    indices:
        Iterable of plane keys to schedule.
    kwargs:
        Keyword arguments forwarded to the worker on every call.
    n_workers:
        Process count. Defaults to
        :func:`nd2studios.utils.resources.recommended_worker_count`.
    """
    kwargs = kwargs or {}
    n_workers = n_workers or recommended_worker_count()
    if not indices:
        return {}

    chunksize = max(1, len(indices) // (n_workers * 4) or 1)
    with shared_ndarray(stack) as (name, shape, dtype_str):
        args = [
            (name, shape, dtype_str, idx, fn_module, fn_name, kwargs)
            for idx in indices
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_worker_apply, args, chunksize=chunksize))
    return dict(results)
