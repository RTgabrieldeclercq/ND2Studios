"""
``StitchConfig`` — every knob for the V1.54 regime-aware stitching pipeline.

Backend-pure: no PySide6 imports. The GUI (``StitchDialog``) builds one of
these and the worker forwards it to :func:`nd2studios.backend.stitch.run_stitch`.

The **default** orientation (``axis_flip_x=True``, ``axis_flip_y=False``,
``swap_xy=False``) reproduces the exact tile placement the pre-V1.54 stitcher
produced, so previews and coordinate placement are pixel-identical to before —
this is the "how metadata orients the M frames" behavior we deliberately keep.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# Regime the pipeline runs in.
REGIME_AUTO = "auto"
REGIME_OVERLAP = "overlap"
REGIME_ZERO = "zero_overlap"

# Registration / placement engines.
ENGINE_AUTO = "auto"
ENGINE_COORDINATE = "coordinate"
ENGINE_PHASE = "phase_correlation"
ENGINE_M2STITCH = "m2stitch"
ENGINE_ASHLAR = "ashlar"

# Compositor blend modes.
BLEND_FEATHER = "feather"
BLEND_AVERAGE = "average"
BLEND_MAX = "max"
BLEND_NONE = "none"

# Illumination correction methods.
ILLUM_NONE = "none"
ILLUM_BASIC = "basic"        # BaSiC (basicpy) — optional, gated
ILLUM_BUILTIN = "builtin"    # smoothed-flatfield fallback (always available)
ILLUM_SUPPLIED = "supplied"  # user-provided flat/dark reference images


@dataclass
class StitchConfig:
    """All configuration for one stitch run (spec §10)."""

    # ── Regime / engine selection ──
    regime: str = REGIME_AUTO
    engine: str = ENGINE_AUTO
    # A dataset is "zero-overlap" when the inferred overlap fraction on both
    # axes is at or below this tolerance (fraction of tile extent).
    zero_overlap_tol: float = 0.01
    # Explicit overlap fraction override (None → infer from stage coords).
    overlap_frac: Optional[float] = None

    # ── Geometry / metadata ──
    # Pixel size override in µm/px (None → take from volume/metadata).
    pixel_size_um: Optional[float] = None
    # Stage→pixel orientation. Defaults reproduce the historical placement:
    #   col_px = (max_stage_x - stage_x) / px   (flip_x=True)
    #   row_px = (stage_y - min_stage_y) / px    (flip_y=False)
    axis_flip_x: bool = True
    axis_flip_y: bool = False
    swap_xy: bool = False

    # ── Registration (overlap regime) ──
    # Channel index positions are computed on; reused for all C/Z/T (spec §7).
    align_channel: int = 0
    # Pre-filter band-pass sigma for phase correlation (px). 0 disables.
    filter_sigma: float = 1.0
    # Cap (µm) on how far registration may move a tile from its coordinate seed.
    # None → derived from the inferred overlap (½ overlap width) or ½ tile.
    max_shift_um: Optional[float] = None
    # Normalized-cross-correlation acceptance threshold for a pairwise match.
    ncc_threshold: float = 0.3
    # Sub-pixel upsample factor for skimage.phase_cross_correlation.
    upsample_factor: int = 10
    # Re-register every timepoint (stage drift). Off → register once at t0.
    per_timepoint_registration: bool = False

    # ── Compositing ──
    blend: str = BLEND_FEATHER
    # Feather ramp width in px (None → derive from overlap width).
    feather_width_px: Optional[int] = None
    # Canvas fill value for gaps / background.
    fill_value: int = 0

    # ── Illumination correction ──
    illumination_correction: str = ILLUM_NONE
    flatfield_path: Optional[str] = None
    darkfield_path: Optional[str] = None

    # ── Z handling ── (app semantics: 'none' preserves all Z, else project)
    z_mode: str = "max"
    z_index: int = 0

    # ── Output ── (pyramidal, tiled, BigTIFF OME-TIFF)
    output_format: str = "ome_tiff"
    tile_size: int = 256
    pyramid: bool = True
    pyramid_downsample: int = 2
    pyramid_max_levels: int = 6
    # Write a QC sidecar (JSON + PNG) next to the output.
    write_qc: bool = True

    # ── Memory ──
    max_memory_gb: float = 4.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "StitchConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
