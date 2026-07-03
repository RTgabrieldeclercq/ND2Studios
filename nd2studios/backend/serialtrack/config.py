"""
SerialTrack Python — Configuration dataclasses
================================================
    serialtrack/config.py

Replaces MATLAB structs: BeadPara, MPTPara, and trajectory-merge params.

Changes from v1
----------------
- Added TrackingMode.DOUBLE_FRAME  (was missing — MATLAB 'dbf' mode)
- Added TrackingConfig.loc_solver   (was missing — MATLAB locSolver 1 or 2)
- Added TrajectoryConfig dataclass  (was entirely missing — trajectory stitching params)
- Added TrackingConfig.trajectory   (embeds TrajectoryConfig)
- Removed duplicate GlobalSolver from regularization.py — single source of truth here
- Fixed roi_slices() for the case where roi_x/roi_y are None before init
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════

class DetectionMethod(IntEnum):
    """Particle detection strategy."""
    TPT = 1       # Blob → centroid → radial symmetry sub-pixel
    TRACTRAC = 2  # LoG blob → local max → 2nd-order poly sub-pixel


class GlobalSolver(IntEnum):
    """Global step solver for ADMM iterations."""
    MLS = 1              # Moving least-squares fitting
    REGULARIZATION = 2   # Scatter → grid regularization
    ADMM = 3             # Augmented Lagrangian with L-curve


class LocalSolver(IntEnum):
    """Local step solver for particle matching."""
    TOPOLOGY = 1         # Topology-based feature matching
    HISTOGRAM_THEN_TOPOLOGY = 2  # Histogram first, then topology


class TrackingMode(IntEnum):
    """Frame-to-frame vs. all-to-reference tracking."""
    INCREMENTAL = 1
    CUMULATIVE = 2
    DOUBLE_FRAME = 3     # Independent frame-pair mode (was missing)


# ═══════════════════════════════════════════════════════════════
#  Detection config
# ═══════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """Particle detection / localization parameters.

    Replaces MATLAB ``BeadPara`` struct.  All lengths are in pixels.
    Works for both 2-D and 3-D images; dimension is inferred at runtime.
    """
    method: DetectionMethod = DetectionMethod.TRACTRAC
    threshold: float = 0.4
    bead_radius: float = 3.0   # 0 → use regionprops centroid only
    min_size: int = 2          # min blob volume (3D) or area (2D) [px^d]
    max_size: int = 1000
    color: str = "white"       # foreground colour: "white" | "black"
    # Optional PSF deconvolution (Richardson-Lucy)
    psf: Optional[np.ndarray] = None
    deconv_iters: int = 6
    # TPT / radial-symmetry params (3-D only)
    win_size: Tuple[int, ...] = (5, 5, 5)
    dccd: Tuple[float, ...] = (1.0, 1.0, 1.0)
    abc: Tuple[float, ...] = (1.0, 1.0, 1.0)
    rand_noise: float = 1e-7


# ═══════════════════════════════════════════════════════════════
#  Trajectory merge config  (was entirely missing)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryConfig:
    """Parameters for post-processing trajectory segment merging.

    Replaces the MATLAB variables: distThres, extrapMethod,
    minTrajSegLength, maxGapTrajSeqLength.

    Used primarily in incremental tracking mode to stitch together
    short trajectory segments that were split due to detection gaps.
    """
    dist_threshold: float = 1.0
    """Distance threshold to connect split trajectory segments [px]."""

    extrap_method: str = "pchip"
    """Extrapolation scheme: 'pchip' for smooth motion,
    'nearest' for Brownian motion."""

    min_segment_length: int = 10
    """Minimum trajectory segment length (in frames) to attempt
    extrapolation and merging."""

    max_gap_length: int = 0
    """Maximum frame gap allowed between connected segments.
    0 means segments must be adjacent."""

    merge_passes: int = 4
    """Number of merge passes (MATLAB default: 4)."""


# ═══════════════════════════════════════════════════════════════
#  Tracking config
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrackingConfig:
    """Particle linking / tracking parameters.

    Replaces MATLAB ``MPTPara`` struct.
    """
    # --- search & matching ---
    f_o_s: float = 60.0
    n_neighbors_max: int = 25
    n_neighbors_min: int = 1

    # --- local solver (was missing) ---
    loc_solver: LocalSolver = LocalSolver.TOPOLOGY

    # --- global solver ---
    solver: GlobalSolver = GlobalSolver.REGULARIZATION
    smoothness: float = 0.1

    # --- outlier removal (Westerweel universal test) ---
    outlier_threshold: float = 5.0

    # --- ADMM iteration control ---
    max_iter: int = 20
    iter_stop_threshold: float = 1e-2

    # --- strain gauge ---
    strain_n_neighbors: int = 20
    strain_f_o_s: float = 60.0

    # --- prediction / initialisation ---
    use_prev_results: bool = False
    dist_missing: float = 5.0

    # --- mode ---
    mode: TrackingMode = TrackingMode.INCREMENTAL

    # --- trajectory merging (was missing) ---
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    # --- physical scales ---
    xstep: float = 1.0   # length-unit per pixel
    ystep: float = 1.0
    zstep: float = 1.0
    tstep: float = 1.0   # time-unit per frame

    # --- ROI (auto-set from first image) ---
    roi_x: Optional[Tuple[int, int]] = None
    roi_y: Optional[Tuple[int, int]] = None
    roi_z: Optional[Tuple[int, int]] = None   # None ⇒ 2-D
    mask: Optional[np.ndarray] = None

    # ── derived helpers ──

    @property
    def ndim(self) -> int:
        return 2 if self.roi_z is None else 3

    @property
    def steps(self) -> np.ndarray:
        """Physical pixel sizes as array [xstep, ystep(, zstep)]."""
        if self.ndim == 2:
            return np.array([self.xstep, self.ystep])
        return np.array([self.xstep, self.ystep, self.zstep])

    def init_roi_from_image(self, img: np.ndarray) -> None:
        """Set ROI to full image extent and default mask."""
        self.roi_x = (0, img.shape[0])
        self.roi_y = (0, img.shape[1])
        if img.ndim >= 3:
            self.roi_z = (0, img.shape[2])
        else:
            self.roi_z = None
        if self.mask is None:
            self.mask = np.ones(img.shape, dtype=bool)

    def roi_slices(self) -> Tuple[slice, ...]:
        """ROI as a tuple of slices for direct array indexing."""
        sx = slice(*(self.roi_x or (0, None)))
        sy = slice(*(self.roi_y or (0, None)))
        if self.roi_z is not None:
            return (sx, sy, slice(*self.roi_z))
        return (sx, sy)