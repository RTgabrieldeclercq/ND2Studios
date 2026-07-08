"""
``nd2studios.backend.stitch`` — V1.54 regime-aware multipoint stitching.

Public API:

- :class:`StitchConfig` — all knobs.
- :func:`run_stitch` — end-to-end: dataset → regime → positions → composite →
  pyramidal OME-TIFF → QC. Returns :class:`StitchResult`.
- :func:`compute_tile_layout` / :class:`StitchLayout` — coordinate placement used
  by the GUI preview widgets (backward-compatible with the old stitch_exporter).
- :func:`build_dataset`, :func:`decide_regime`, :func:`compute_positions` — the
  individual stages, for tests and headless use.

Backend-pure: no PySide6 imports anywhere in this package.
"""
from __future__ import annotations

from nd2studios.backend.stitch.config import StitchConfig
from nd2studios.backend.stitch.dataset import Dataset, Tile, build_dataset
from nd2studios.backend.stitch.engines import compute_positions, seed_positions
from nd2studios.backend.stitch.pipeline import StitchResult, run_stitch
from nd2studios.backend.stitch.positions import (
    StitchLayout, compute_tile_layout,
)
from nd2studios.backend.stitch.regime import decide_regime

__all__ = [
    "StitchConfig", "StitchResult", "run_stitch",
    "StitchLayout", "compute_tile_layout",
    "Dataset", "Tile", "build_dataset", "decide_regime",
    "compute_positions", "seed_positions",
]
