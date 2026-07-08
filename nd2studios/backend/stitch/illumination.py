"""
Illumination / flat-field correction (spec §6) — optional, off by default.

Uneven per-tile shading is the #1 cause of a visible grid of seams even after
perfect alignment. Three methods:

- ``basic``    — BaSiC (basicpy), the reference low-rank+sparse estimator.
  Optional and gated: basicpy's import chain is fragile (hyperactive /
  gradient_free_optimizers API drift). If it can't be imported we raise a clear
  error only when the user explicitly selects it.
- ``builtin``  — always-available fallback: estimate the flat-field as a heavily
  smoothed (large-σ Gaussian) mean of the tiles, normalized to its own mean, and
  divide it out. Removes low-frequency shading without any extra dependency.
- ``supplied`` — user-provided flat-field (and optional dark-field) images.

Correction is applied per channel, to the raw tiles, *before* compositing, and
always returns the source dtype.
"""
from __future__ import annotations

import importlib.util
from typing import List, Optional

import numpy as np

from nd2studios.backend.stitch.config import (
    ILLUM_BASIC, ILLUM_BUILTIN, ILLUM_NONE, ILLUM_SUPPLIED, StitchConfig,
)


def basicpy_available() -> bool:
    """True only if basicpy can actually be imported (not just installed)."""
    if importlib.util.find_spec("basicpy") is None:
        return False
    try:
        from basicpy import BaSiC  # noqa: F401
        return True
    except Exception:
        return False


class Flatfield:
    """A per-channel (flat, dark) correction that can be applied to tiles."""

    def __init__(self, flat: np.ndarray, dark: Optional[np.ndarray] = None):
        self.flat = np.asarray(flat, dtype=np.float32)
        # Guard against divide-by-zero from a degenerate estimate.
        self.flat = np.where(np.abs(self.flat) < 1e-6, 1.0, self.flat)
        self.dark = None if dark is None else np.asarray(dark, dtype=np.float32)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        if self.dark is not None and self.dark.shape == f.shape:
            f = f - self.dark
        if self.flat.shape == f.shape:
            f = f / self.flat
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
            f = np.clip(np.rint(f), info.min, info.max)
        return f.astype(frame.dtype)


def _builtin_flatfield(tiles: List[np.ndarray]) -> np.ndarray:
    """Large-σ Gaussian of the tile mean, normalized to its own mean."""
    from scipy.ndimage import gaussian_filter
    stack = np.stack([t.astype(np.float32) for t in tiles], axis=0)
    mean_img = stack.mean(axis=0)
    sigma = max(mean_img.shape) / 8.0
    flat = gaussian_filter(mean_img, sigma=sigma)
    m = float(flat.mean())
    if m <= 1e-6:
        return np.ones_like(flat)
    return flat / m


def _basic_flatfield(tiles: List[np.ndarray]) -> "Flatfield":
    from basicpy import BaSiC
    stack = np.stack([t.astype(np.float32) for t in tiles], axis=0)
    basic = BaSiC(get_darkfield=True)
    basic.fit(stack)
    flat = np.asarray(basic.flatfield, dtype=np.float32)
    m = float(flat.mean())
    if m > 1e-6:
        flat = flat / m
    dark = getattr(basic, "darkfield", None)
    dark = None if dark is None else np.asarray(dark, dtype=np.float32)
    return Flatfield(flat, dark)


def estimate_flatfield(tiles: List[np.ndarray],
                       config: StitchConfig) -> Optional[Flatfield]:
    """Build a :class:`Flatfield` for a channel, or ``None`` if disabled.

    ``tiles`` should be the raw reference-timepoint tiles for one channel.
    """
    method = config.illumination_correction
    if method == ILLUM_NONE or not tiles:
        return None

    if method == ILLUM_SUPPLIED:
        import tifffile
        if not config.flatfield_path:
            raise ValueError("illumination_correction='supplied' needs flatfield_path")
        flat = np.asarray(tifffile.imread(config.flatfield_path), dtype=np.float32)
        tile_shape = tiles[0].shape
        if flat.shape != tile_shape:
            raise ValueError(
                f"supplied flatfield shape {flat.shape} != tile shape {tile_shape}; "
                "correction would be silently skipped.")
        m = float(flat.mean())
        if m > 1e-6:
            flat = flat / m
        dark = None
        if config.darkfield_path:
            dark = np.asarray(tifffile.imread(config.darkfield_path), dtype=np.float32)
            if dark.shape != tile_shape:
                raise ValueError(
                    f"supplied darkfield shape {dark.shape} != tile shape {tile_shape}.")
        return Flatfield(flat, dark)

    if method == ILLUM_BASIC:
        if not basicpy_available():
            raise ImportError(
                "BaSiC illumination correction needs a working 'basicpy'. It is "
                "not importable in this environment (basicpy → hyperactive → "
                "gradient_free_optimizers API drift). Use illumination "
                "correction 'builtin' instead, or repair the basicpy install."
            )
        return _basic_flatfield(tiles)

    # builtin (default fallback) — never raises on dependencies.
    return Flatfield(_builtin_flatfield(tiles))
