"""Pre-flight load strategy selection for ND2/TIFF imports (V1.41).

Estimates the in-RAM footprint of a candidate dataset and picks among
three load strategies based on the host's available memory.  The
worker code (:class:`~nd2studios.workers.load_worker.LoadWorker`) calls
:func:`choose_strategy` between metadata read and the actual decode
pass; the returned :class:`StrategyDecision` is then handed to the
loader (:func:`~nd2studios.backend.materialized_loader.materialize_nd2`)
which branches on it.

Strategies
----------

* :attr:`LoadStrategy.EAGER_FULL` — preallocate ``(M, T, Z, H, W)``
  ndarrays per channel.  Best random access; this is what V1.40 did
  unconditionally.
* :attr:`LoadStrategy.EAGER_REDUCED` — collapse Z at decode time so
  the in-RAM footprint shrinks by the Z factor.  Trades runtime
  z-mode switching for fit; surfaced in the diagnostics panel so
  the user knows why max/min/none aren't switchable until re-import
  with more RAM.
* :attr:`LoadStrategy.LAZY_CACHED` — read frames on demand via
  :class:`~nd2studios.backend.nd2_loader.LazyND2Channel` with a bounded
  LRU cache sized by :func:`recommended_cache_budget_bytes`.  Best
  fit, worst per-frame latency.

A hard cap refuses loads where even the projected footprint exceeds
:data:`HARD_CAP_MULTIPLE` × available RAM, surfacing a clear error
rather than letting Python OOM.

Test seam
---------

Setting the ``ND2_FAKE_RAM_GB`` environment variable to a positive
number overrides ``psutil.virtual_memory().available`` for the
duration of the process.  CI can exercise the low-RAM branches on
fat hosts without juggling cgroups.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from nd2studios.utils.resources import (
    detect,
    recommended_cache_budget_bytes,
    recommended_worker_count,
)

log = logging.getLogger(__name__)


class LoadStrategy(str, Enum):
    EAGER_FULL = "eager_full"
    EAGER_REDUCED = "eager_reduced"
    LAZY_CACHED = "lazy_cached"


@dataclass(frozen=True)
class StrategyDecision:
    """Frozen snapshot of the chosen load strategy + the inputs to it.

    Surfaced in the GUI's diagnostics panel and emitted on
    ``LoadWorker.strategy_chosen`` so the user can see exactly what
    happened and why.
    """

    strategy: LoadStrategy
    estimated_bytes: int               # full (M, T, Z, H, W) footprint
    estimated_bytes_projected: int     # Z-collapsed footprint
    available_bytes: int
    cache_budget_bytes: int            # 0 for EAGER_*
    worker_count: int
    z_collapsed_at_load: bool
    reason: str                         # human-readable diagnostic

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / (1024 ** 3)

    @property
    def estimated_projected_gb(self) -> float:
        return self.estimated_bytes_projected / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024 ** 3)

    @property
    def cache_budget_gb(self) -> float:
        return self.cache_budget_bytes / (1024 ** 3)


# ── Defaults — also exposed on Settings so the Performance dialog can edit them.

EAGER_MAX_FRACTION_DEFAULT = 0.50
"""Default ceiling for eager preallocation: half of available RAM."""

RESERVE_OVERHEAD = 1.20
"""Multiplier applied to the estimated footprint before fit check.

Accounts for transient working-set growth during decode (chunked
reads, dask threading, PIL/imageio buffers) so we don't approve a
load that exactly fits on paper but tips over once decode warms up.
"""

HARD_CAP_MULTIPLE = 2.0
"""Refuse to load files whose projected footprint exceeds this multiple
of available RAM, even via override.  Surfaces as :class:`StrategyError`."""


class StrategyError(RuntimeError):
    """Raised when no load strategy can fit the dataset on the host."""


def _fake_ram_override_bytes() -> Optional[int]:
    """Return the ``ND2_FAKE_RAM_GB`` override, if set, in bytes."""
    raw = os.environ.get("ND2_FAKE_RAM_GB")
    if not raw:
        return None
    try:
        gb = float(raw)
    except ValueError:
        return None
    if gb <= 0:
        return None
    return int(gb * 1024 ** 3)


def _available_bytes() -> int:
    override = _fake_ram_override_bytes()
    if override is not None:
        return override
    return detect().available_ram_bytes


def estimate_footprint(meta, *, project_z: bool) -> int:
    """Return the in-RAM footprint in bytes for ``meta``.

    Reads attributes off the supplied object via :func:`getattr` so
    it works with both :class:`~nd2studios.backend.nd2_loader.ND2Metadata`
    and any duck-typed substitute the multi-file paths construct.
    """
    n_m = max(1, int(getattr(meta, "n_multipoints", 1)))
    n_t = max(1, int(getattr(meta, "n_timepoints", 1)))
    n_z = max(1, int(getattr(meta, "n_zslices", 1)))
    n_c = max(1, int(getattr(meta, "n_channels", 1)))
    h = max(1, int(getattr(meta, "height", 0)))
    w = max(1, int(getattr(meta, "width", 0)))
    dtype = getattr(meta, "dtype", np.dtype("uint16"))
    item = int(np.dtype(dtype).itemsize)
    if project_z:
        return n_m * n_t * h * w * n_c * item
    return n_m * n_t * n_z * h * w * n_c * item


def choose_strategy(
    meta,
    path: Optional[str] = None,
    *,
    override: Optional[LoadStrategy] = None,
    eager_max_fraction: float = EAGER_MAX_FRACTION_DEFAULT,
    reserve_overhead: float = RESERVE_OVERHEAD,
) -> StrategyDecision:
    """Pick a load strategy based on ``meta`` and host resources.

    Parameters
    ----------
    meta : object with ``n_multipoints``, ``n_timepoints``, ``n_zslices``,
        ``n_channels``, ``height``, ``width``, ``dtype`` attributes.
    path : optional source filesystem path.  Used to consult
        :func:`~nd2studios.utils.storage.recommended_io_thread_count`
        so the worker pool sizing reflects storage class, not just
        CPU count.
    override : force a specific strategy regardless of fit.  The hard
        cap still applies so this can't be abused to OOM the app.
    eager_max_fraction, reserve_overhead : exposed for testing /
        Performance dialog overrides.  Defaults come from the module
        constants above.

    Raises
    ------
    StrategyError : the projected footprint exceeds
        :data:`HARD_CAP_MULTIPLE` × available RAM.
    """
    full = estimate_footprint(meta, project_z=False)
    projected = estimate_footprint(meta, project_z=True)
    avail = _available_bytes()

    if projected > avail * HARD_CAP_MULTIPLE:
        raise StrategyError(
            f"Dataset too large to load: projected footprint "
            f"{projected / 1024**3:.2f} GB exceeds {HARD_CAP_MULTIPLE:.1f}× "
            f"available {avail / 1024**3:.2f} GB. "
            f"Close other applications or use a machine with more RAM."
        )

    eager_cap = int(avail * eager_max_fraction)
    full_fits = (full * reserve_overhead) <= eager_cap
    projected_fits = (projected * reserve_overhead) <= eager_cap
    n_z = max(1, int(getattr(meta, "n_zslices", 1)))

    worker_count = recommended_worker_count()
    if path:
        try:
            from nd2studios.utils.storage import recommended_io_thread_count
            io_n = recommended_io_thread_count(path)
            worker_count = max(1, min(worker_count, io_n * 4))
        except Exception:  # noqa: BLE001
            pass

    if override is not None:
        strat = override
        z_collapsed = strat == LoadStrategy.EAGER_REDUCED
        cache_budget = (
            recommended_cache_budget_bytes()
            if strat == LoadStrategy.LAZY_CACHED
            else 0
        )
        reason = f"override={strat.value}"
    elif full_fits:
        strat = LoadStrategy.EAGER_FULL
        z_collapsed = False
        cache_budget = 0
        reason = (
            f"full footprint {full / 1024**3:.2f} GB fits within "
            f"{eager_max_fraction*100:.0f}% of available "
            f"{avail / 1024**3:.2f} GB"
        )
    elif projected_fits and n_z > 1:
        strat = LoadStrategy.EAGER_REDUCED
        z_collapsed = True
        cache_budget = 0
        reason = (
            f"full footprint {full / 1024**3:.2f} GB exceeds eager budget "
            f"{eager_cap / 1024**3:.2f} GB; Z-collapsed "
            f"{projected / 1024**3:.2f} GB fits"
        )
    else:
        strat = LoadStrategy.LAZY_CACHED
        z_collapsed = False
        cache_budget = recommended_cache_budget_bytes()
        reason = (
            f"projected footprint {projected / 1024**3:.2f} GB exceeds eager "
            f"budget {eager_cap / 1024**3:.2f} GB; lazy with "
            f"{cache_budget / 1024**3:.2f} GB cache"
        )

    return StrategyDecision(
        strategy=strat,
        estimated_bytes=full,
        estimated_bytes_projected=projected,
        available_bytes=avail,
        cache_budget_bytes=cache_budget,
        worker_count=worker_count,
        z_collapsed_at_load=z_collapsed,
        reason=reason,
    )


def log_decision(decision: StrategyDecision) -> None:
    """One-line info log summarizing the chosen strategy."""
    log.info(
        "Load strategy: %s — %s (workers=%d, cache=%.2f GB)",
        decision.strategy.value,
        decision.reason,
        decision.worker_count,
        decision.cache_budget_gb,
    )
