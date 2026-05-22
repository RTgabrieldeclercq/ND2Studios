"""``multiprocessing.shared_memory`` glue for large numpy arrays.

Addendum Phase 5, Pattern 2. The default :class:`ProcessPoolExecutor`
behavior is to pickle every argument and reconstruct it inside the
worker, which for a 100 MB ND2 channel stack costs hundreds of ms per
task — enough to erase the parallel speedup. ``shared_memory`` lets
the parent allocate one OS-backed buffer, copy the array in once, and
hand workers a tiny ``(name, shape, dtype)`` triple they can attach
to lock-free.

Two surfaces:

- :func:`shared_ndarray` — context manager used by the parent. Allocates,
  copies, yields ``(name, shape, dtype_str)`` for transport into
  worker tasks, then cleans the block up on exit (``close`` + ``unlink``).
- :func:`attach_shared` — used inside the worker. Returns the
  ``ndarray`` view plus the underlying handle so the worker can
  ``close()`` it on exit (workers must **not** ``unlink``; that's the
  parent's job).
"""
from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Iterator, Tuple

import numpy as np


@contextmanager
def shared_ndarray(arr: np.ndarray) -> Iterator[Tuple[str, Tuple[int, ...], str]]:
    """Expose *arr* to worker processes via shared memory.

    Yields ``(name, shape, dtype_str)``. Worker functions reconstruct
    the array with :func:`attach_shared`. On exit the block is closed
    and unlinked so the OS frees the backing memory.

    Use as::

        with shared_ndarray(stack) as (name, shape, dtype_str):
            args = [(name, shape, dtype_str, t, params) for t in range(T)]
            with ProcessPoolExecutor(...) as ex:
                results = list(ex.map(worker_fn, args))

    The yielded ``shape`` is a tuple and ``dtype_str`` is a string
    (e.g. ``'float32'``) so workers can rebuild ``np.dtype`` without
    needing the parent's exact ``np.dtype`` instance.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.ascontiguousarray(arr)
    if not arr.flags.c_contiguous:
        # ``shared_memory`` wants a flat byte image. Materialize a
        # contiguous copy when the caller passed in a strided view.
        arr = np.ascontiguousarray(arr)

    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    try:
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
        yield shm.name, tuple(arr.shape), str(arr.dtype)
    finally:
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            # Already unlinked — another part of the parent or a worker
            # race lost; ignore.
            pass
        except Exception:  # noqa: BLE001
            pass


def attach_shared(
    name: str,
    shape: Tuple[int, ...],
    dtype_str: str,
) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to an existing shared block from inside a worker.

    Returns ``(view, shm_handle)``. The caller **must** keep
    ``shm_handle`` alive while using ``view`` and call
    ``shm_handle.close()`` when done — failing to close leaks a file
    descriptor per task on POSIX. ``unlink`` is the parent's job
    (handled by :func:`shared_ndarray`'s context manager); calling it
    in a worker will tear the block out from under sibling workers.
    """
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return arr, shm
