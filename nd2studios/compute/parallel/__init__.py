"""Parallel-compute utilities — addendum Phase 5 cross-cutting patterns.

The addendum (``08_addendum_parallelism.md``) calls for three parallel
primitives in :mod:`nd2studios.compute`:

* :class:`~nd2studios.compute.parallel.shared_array.shared_ndarray` —
  context manager that exposes a ``numpy`` array to worker processes
  via ``multiprocessing.shared_memory`` without paying the pickle
  cost on every submission.
* :func:`~nd2studios.compute.parallel.process_map.process_map_planes`
  — apply a pure-Python CPU-bound per-plane function across a stack
  using a :class:`concurrent.futures.ProcessPoolExecutor` backed by
  shared memory.
* :func:`~nd2studios.compute.parallel.thread_map.thread_map_planes`
  — apply a per-plane function that drops into a GIL-releasing C
  extension (scipy / scikit-image / numpy) using a
  :class:`concurrent.futures.ThreadPoolExecutor`. The most common
  shape for ND2Studios analysis pipelines.

Choose by the workload, not the convenience: threads are cheaper and
share memory natively, but a pure-Python kernel will be GIL-bound
inside them. The addendum's decision rule is "threads for C-extension
work, processes for pure-Python CPU work".
"""
from __future__ import annotations

from nd2studios.compute.parallel.shared_array import (
    attach_shared,
    shared_ndarray,
)
from nd2studios.compute.parallel.process_map import process_map_planes
from nd2studios.compute.parallel.thread_map import thread_map_planes

__all__ = [
    "attach_shared",
    "process_map_planes",
    "shared_ndarray",
    "thread_map_planes",
]
