"""
Regime decision (spec §0 / §11.1): overlap vs zero-overlap.

The single most important branch in the whole pipeline. Zero-overlap data must
*never* reach a registrar — a cross-correlation peak always exists, so a
registrar would lock onto noise and silently produce garbage.
"""
from __future__ import annotations

from nd2studios.backend.stitch.config import (
    REGIME_AUTO, REGIME_OVERLAP, REGIME_ZERO, StitchConfig,
)
from nd2studios.backend.stitch.dataset import Dataset


def decide_regime(dataset: Dataset, config: StitchConfig) -> str:
    """Return ``"overlap"`` or ``"zero_overlap"`` for this dataset.

    - Explicit ``config.regime`` (non-auto) wins.
    - No stage coordinates, a single tile, or no inferable multi-tile step ⇒
      zero-overlap (nothing to register against).
    - Otherwise compare the inferred overlap fraction against ``zero_overlap_tol``.
    """
    if config.regime == REGIME_OVERLAP:
        return REGIME_OVERLAP
    if config.regime == REGIME_ZERO:
        return REGIME_ZERO
    # auto
    if not dataset.stage_present or dataset.n_tiles < 2:
        return REGIME_ZERO
    if dataset.overlap_frac_x is None and dataset.overlap_frac_y is None:
        return REGIME_ZERO
    return REGIME_OVERLAP if dataset.overlap_frac > config.zero_overlap_tol \
        else REGIME_ZERO
