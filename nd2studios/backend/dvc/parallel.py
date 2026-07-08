"""
Process-pool fan-out for the DVC local step (Stages 3 & 5).

The IC-GN sweep is embarrassingly parallel across subsets and is the dominant
cost of DVC on a real 3D stack. This module fans the sweep across all physical
cores with a :class:`concurrent.futures.ProcessPoolExecutor`, putting the
reference and (prefiltered) deformed volumes into ``multiprocessing.shared_memory``
**once** (via :mod:`nd2studios.compute.parallel.shared_array`) so workers attach
lock-free instead of re-pickling hundreds of MB per task. The per-subset kernel is
the module-level :func:`nd2studios.backend.dvc.icgn._icgn_subset` (closures can't
pickle into spawned workers — Windows uses spawn).

Trade-off vs. the serial path: workers recompute each subset's reference
SD/Hessian (the serial path caches them across ADMM iterations), but the multicore
win dominates on large grids. ``n_workers <= 1`` keeps the cached serial path.

Pure numpy/scipy — no PySide6.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np

from nd2studios.compute.parallel.shared_array import shared_ndarray, attach_shared
from nd2studios.backend.dvc.mesh import Grid


def default_workers() -> int:
    """A sensible default worker count: physical cores minus one, capped at 8."""
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _icgn_block(args):
    """Worker: IC-GN over a block of subsets. Attaches the shared volumes,
    solves each subset, returns ``[(u, G, zncc, iters) | None, ...]``."""
    # Import inside the worker so the spawned process resolves them fresh.
    from nd2studios.backend.dvc.icgn import (
        _prepare_reference, _icgn_subset, _subset_offsets)
    (ref_meta, def_meta, centers, seeds, u_targets, F_targets,
     subset_size, tol, max_iter, mu, beta) = args
    ref, ref_shm = attach_shared(*ref_meta)
    defm, def_shm = attach_shared(*def_meta)
    try:
        ndim = len(ref_meta[1])
        dx = _subset_offsets(subset_size, ndim)
        half = max(1, int(subset_size) // 2)
        subset_shape = tuple([half * 2 + 1] * ndim)
        out = []
        for k in range(len(centers)):
            seed = seeds[k]
            if not np.all(np.isfinite(seed)):
                out.append(None)
                continue
            c0 = np.rint(centers[k]).astype(np.int64)
            rs = _prepare_reference(ref, c0, dx, subset_shape)
            if rs is None:
                out.append(None)
                continue
            ut = u_targets[k] if u_targets is not None else None
            Ft = F_targets[k] if F_targets is not None else None
            res = _icgn_subset(rs, defm, seed, np.zeros((ndim, ndim)),
                               tol=tol, max_iter=max_iter, mu=mu, beta=beta,
                               u_target=ut, F_target=Ft)
            out.append(res)
        return out
    finally:
        ref_shm.close()
        def_shm.close()


def parallel_local_icgn(
    ref: np.ndarray, defm_pref: np.ndarray, grid: Grid, u0: np.ndarray,
    subset_size: int, *, tol: float, max_iter: int, mu: float, beta: float,
    u_target: Optional[np.ndarray], F_target: Optional[np.ndarray],
    n_workers: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0, progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IC-GN over every subset, fanned across ``n_workers`` processes.

    Same return contract as :func:`nd2studios.backend.dvc.icgn.local_icgn`
    (``u_grid, F_grid, zncc_grid, iters_grid``, NaN for failed subsets).
    """
    ndim = grid.ndim
    gshape = grid.grid_shape
    idx_list = list(np.ndindex(*gshape))
    n = len(idx_list)
    centers = [grid.coords[idx] for idx in idx_list]
    seeds = [u0[(slice(None), *idx)] for idx in idx_list]
    uts = ([u_target[(slice(None), *idx)] for idx in idx_list]
           if u_target is not None else None)
    fts = ([F_target[(slice(None), slice(None), *idx)] for idx in idx_list]
           if F_target is not None else None)

    # Blocks: a few per worker for load balance.
    nb = max(1, int(n_workers) * 4)
    bounds = np.linspace(0, n, nb + 1).astype(int)
    blocks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    u_out = np.full((ndim, *gshape), np.nan)
    F_out = np.full((ndim, ndim, *gshape), np.nan)
    zncc_out = np.full(gshape, np.nan)
    iters_out = np.zeros(gshape, dtype=np.int32)

    ref = np.ascontiguousarray(ref, dtype=np.float32)
    defm_pref = np.ascontiguousarray(defm_pref, dtype=np.float32)
    with shared_ndarray(ref) as ref_meta, shared_ndarray(defm_pref) as def_meta:
        tasks = []
        for (a, b) in blocks:
            tasks.append((
                ref_meta, def_meta, centers[a:b], seeds[a:b],
                (uts[a:b] if uts is not None else None),
                (fts[a:b] if fts is not None else None),
                int(subset_size), float(tol), int(max_iter), float(mu), float(beta)))
        done = 0
        with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
            for (a, b), block_res in zip(blocks, ex.map(_icgn_block, tasks)):
                for j, res in enumerate(block_res):
                    if res is None:
                        continue
                    idx = idx_list[a + j]
                    u, G, zncc, it = res
                    u_out[(slice(None), *idx)] = u
                    F_out[(slice(None), slice(None), *idx)] = G
                    zncc_out[idx] = zncc
                    iters_out[idx] = it
                done += (b - a)
                if progress_cb is not None and n > 0:
                    progress_cb(int(progress_lo + (progress_hi - progress_lo) * done / n))
                if cancelled_cb is not None and cancelled_cb():
                    raise InterruptedError("DVC cancelled")
    return u_out, F_out, zncc_out, iters_out
