"""System-resource detection — backend-pure (no Qt imports).

Single source of truth for "available RAM" / "physical core count" used
by the V1.34 Phase 2 frame cache sizing and (later phases) by the
prefetch budget, analysis worker pool, and GPU/multi-resolution paths.

The 00_README.md guiding this work calls out that ND2Studios under-uses
RAM today (the V1.0 :class:`~nd2studios.backend.frame_cache.FrameCache`
is hard-coded to 300 MB regardless of machine size). The helpers here
let subsystems pick budgets *proportional* to what the user actually
has free, so the lab's workstation gets a multi-GB cache and the laptop
in the field stays well-behaved.

Example
-------
>>> from nd2studios.utils.resources import (
...     detect, recommended_cache_budget_bytes, recommended_worker_count,
... )
>>> sysres = detect()
>>> sysres.available_ram_gb
13.7
>>> recommended_cache_budget_bytes() // (1024**2)
5612
>>> recommended_worker_count()
8
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import psutil


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemResources:
    """Snapshot of host CPU / RAM at the moment :func:`detect` was called.

    Frozen so callers can stash it as a module-level constant without
    risk of mutation. Memory figures are bytes; properties expose the
    matching gigabyte view for log lines and tooltips.
    """

    total_ram_bytes: int
    available_ram_bytes: int
    cpu_count_physical: int
    cpu_count_logical: int

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024 ** 3)

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024 ** 3)


def detect() -> SystemResources:
    """Return a fresh :class:`SystemResources` snapshot.

    ``psutil.virtual_memory().available`` is the OS's estimate of how
    much RAM can be allocated without triggering swap — what we
    actually care about for cache budgeting. ``cpu_count(logical=False)``
    is physical cores; we fall back to 1 if psutil can't tell (some
    containers).
    """
    vm = psutil.virtual_memory()
    return SystemResources(
        total_ram_bytes=int(vm.total),
        available_ram_bytes=int(vm.available),
        cpu_count_physical=int(psutil.cpu_count(logical=False) or 1),
        cpu_count_logical=int(psutil.cpu_count(logical=True) or 1),
    )


def recommended_cache_budget_bytes(reserve_fraction: float = 0.6) -> int:
    """Return a recommended byte budget for in-memory frame caches.

    The default reserves 60% of *available* RAM for the OS, the
    process's own working set, and concurrent workloads (matplotlib
    figures, the Qt pixmap cache, downstream analysis pipelines), and
    hands the remaining 40% to the frame cache. On a workstation with
    32 GB and ~16 GB free that is ~6.4 GB — twenty times the V1.0
    hard-coded 300 MB cap.

    ``reserve_fraction`` is clamped to ``[0.0, 0.95]`` so callers can't
    accidentally hand the cache so little memory that it can't hold a
    single frame. A 4096 × 4096 × 2-byte (uint16) frame is 32 MB; we
    floor the return value at 64 MB so the budget always fits at least
    one such frame.
    """
    rf = max(0.0, min(0.95, float(reserve_fraction)))
    available = detect().available_ram_bytes
    budget = int(available * (1.0 - rf))
    return max(64 * 1024 * 1024, budget)


def recommended_worker_count() -> int:
    """Conservative CPU worker count: physical cores, capped at 8.

    Phase 5 (background analysis pool) and the recipe worker will pick
    up this helper. Cap at 8 because beyond that we hit diminishing
    returns from GIL contention on the CPU-bound numpy / scikit-image
    paths the recipe + analysis pipelines spend time in.
    """
    return min(detect().cpu_count_physical, 8)


def log_system_resources() -> None:
    """Emit a one-line info log summarising host RAM + CPU.

    Called from :func:`nd2studios.__main__.main` after the GPU status
    line so a single grep in the user's console shows the full
    startup environment.  Safe to call repeatedly — psutil reads are
    cheap and ND2Studios does this exactly once per launch.
    """
    res = detect()
    log.info(
        "System resources: RAM %.1f / %.1f GB available, "
        "%d physical / %d logical cores",
        res.available_ram_gb,
        res.total_ram_gb,
        res.cpu_count_physical,
        res.cpu_count_logical,
    )
