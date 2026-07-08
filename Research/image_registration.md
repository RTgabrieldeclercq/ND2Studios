# Image Registration — Research + Implementation Knowledge for ND2Studios

**Date:** 2026-07-07
**Author:** ND2Studios (Claude)
**Related version:** V1.56 (proposed)
**Status:** Knowledge/design doc — read before implementing a registration feature.

> Purpose: give Claude Code everything needed to wire image **registration** into
> ND2Studios *as it is built today*. Part 1 is the algorithmic literature review
> (matching the house `Research/` style). Part 2 maps every algorithm to a Python
> implementation. Part 3 is the concrete integration plan against the current
> codebase (real file paths, real base classes, real conventions). Part 4 is the
> recommended approach and validation plan. Follow the mandatory doc workflow in
> `CLAUDE.md` (plan → changelog → architecture → this review) before coding.

---

## 0. What "registration" means here (scope)

Registration = finding a spatial transform `T` that maps a **moving** image onto a
**reference** image so corresponding structures overlap, then **resampling** the
moving image through `T`. In an ND2 microscopy app the concrete jobs are:

1. **Temporal drift correction** — stabilize a `(T, H, W)` time series so a fixed
   structure stays put across frames (stage drift, thermal drift).
2. **Channel co-registration** — remove the small shift/scale/rotation between
   channels caused by chromatic aberration or filter-cube misalignment. Must
   preserve colocalization, so it is inherently *cross-channel*.
3. **Multi-round / multi-session alignment** — align a second acquisition (later
   timepoint, re-stain, different objective) to a first.
4. **Slice-to-slice / volume alignment** — align Z planes, or register a whole
   `(Z, H, W)` volume to another (this shades into the existing **DVC** feature).

Tile/mosaic stitching is a *separate, already-built* feature
(`backend/stitch/`, `Research/multipoint_stitching.md`); it uses the same
phase-correlation primitive but solves a global tile-placement problem. This doc
is about aligning whole images/stacks, not laying out tiles.

---

# PART 1 — Algorithm Literature Review

## 1.1 The two axes that organize every method

**Axis A — the transformation model** (how many degrees of freedom `T` has):

| Model | 2D DOF | Recovers | Typical microscopy use |
|-------|--------|----------|------------------------|
| Translation | 2 | shift (x, y) | stage/thermal drift, single FOV |
| Rigid / Euclidean | 3 | shift + rotation | stage rotation, sample re-mount |
| Similarity | 4 | + isotropic scale | objective/zoom change |
| Affine | 6 | + anisotropic scale + shear | chromatic/optical distortion, cross-modality |
| Projective (homography) | 8 | + perspective | rarely needed for epifluorescence (telecentric optics) |
| Non-rigid / deformable | many | local warping (displacement field / B-spline) | live-tissue deformation, whole-slide, DVC/DIC |

Rule of thumb: **use the simplest model the physics allows.** Epifluorescence on a
motorized stage is almost always translation or rigid; add affine only when you can
justify scale/shear (multi-objective, chromatic). Deformable is for samples that
physically deform.

**Axis B — what is matched to drive the optimization:**

- **Intensity-based** — directly optimize a *similarity metric* between the
  reference and the transformed moving image over the transform parameters. No
  keypoints. Best when images are similar in appearance and displacement is
  modest. (Phase correlation, ECC, mutual-information registration, optical flow.)
- **Feature-based** — detect keypoints (corners/blobs), compute descriptors, match
  them across images, then robustly fit `T` to the matches (RANSAC). Best for
  large displacements, big rotation/scale, or partial overlap, provided the images
  have enough texture. (ORB/SIFT/AKAZE + RANSAC.)

**Similarity metrics** (intensity-based): SSD (identical intensities only), NCC /
ZNCC (invariant to linear intensity changes — the microscopy default), phase
correlation (invariant to illumination, translation-only), **mutual information**
(the *only* good choice across modalities/stains where intensities are unrelated).

## 1.2 Key methods and papers

### (1) Phase correlation — Kuglin & Hines (1975); subpixel: Guizar-Sicairos et al. (2008)
Translation is recovered from the phase of the cross-power spectrum:
`R = (F₁ · conj(F₂)) / |F₁ · conj(F₂)|`, and `argmax(IFFT(R))` is the integer shift.
Subpixel refinement upsamples the DFT only in a small neighborhood of the peak via
a matrix-multiply DFT (Guizar-Sicairos), giving `1/upsample_factor`-pixel accuracy
cheaply. Windowing (Hann) and band-pass/high-pass whitening suppress DC and
edge/spectral-leakage artifacts. **Fast, robust to illumination, translation-only.**
This is the workhorse for drift correction. *(Already used in the repo:
`backend/stitch/register.py`.)*

### (2) Log-polar phase correlation — Reddy & Chatterji (1996)
To recover **rotation + uniform scale** without features: map both images to
log-polar (or Fourier-magnitude log-polar) coordinates, where rotation becomes a
shift along the angle axis and scale becomes a shift along the log-radius axis.
Phase-correlate to get (rotation, scale), then a second phase correlation for
translation. `skimage`'s `warp_polar` + `phase_cross_correlation` implements this
(see the skimage "Using Polar and Log-Polar Transformations for Registration"
example). Good for rigid+scale with no texture.

### (3) Feature-based + RANSAC — Lowe SIFT (2004); Rublee ORB (2011); Fischler & Bolles RANSAC (1981)
Detect keypoints (ORB = FAST corners + BRIEF, free and fast; SIFT = blob scale-space,
now patent-free; AKAZE = nonlinear scale-space, good on low-contrast micrographs),
match descriptors, then fit the transform with **RANSAC** to reject the many wrong
matches. Estimates anything from Euclidean up to homography. **Handles large
displacement/rotation/scale**; needs textured images (struggles on sparse puncta or
near-blank fields).

### (4) ECC — Evangelidis & Psarakis (2008), "Parametric Image Alignment Using ECC"
Intensity-based iterative alignment that maximizes the **Enhanced Correlation
Coefficient** (a zero-mean-normalized correlation, hence invariant to brightness
and contrast). Supports translation / euclidean / affine / homography, converges to
subpixel accuracy, and is a single well-tested OpenCV call
(`cv2.findTransformECC`). Excellent general-purpose 2D registration when a
reasonable initialization exists and displacement is not huge. Good complement to
phase correlation (which can seed it).

### (5) Mutual-information registration — Viola & Wells (1997); Mattes et al. (2003)
Maximize the mutual information of the joint intensity histogram over the transform
parameters. The standard for **multimodal** alignment (e.g. fluorescence↔brightfield,
or two stains whose intensities are not linearly related), where NCC fails. Usually
run inside an iterative optimizer (gradient descent / (1+1)-ES) over a multi-
resolution pyramid. Implemented cleanly in **SimpleITK/elastix**. Overkill for
same-modality data.

### (6) Optical flow (dense, deformable) — Horn–Schunck (1981); TV-L1: Zach et al. (2007); iLK: Le Besnerais & Champagnat (2005)
Estimate a **per-pixel** displacement field `(u, v)` under a brightness-constancy
assumption plus a smoothness regularizer. TV-L1 (total-variation + L1 data term) is
edge-preserving and robust to outliers; iterative Lucas–Kanade (iLK) is faster but
smoother. Recovers small **non-rigid** deformation. In `skimage`:
`optical_flow_tvl1(reference, moving)` and `optical_flow_ilk(reference, moving)`,
both returning `flow` of shape `(ndim, *image_shape)`; apply with
`skimage.transform.warp` over `coords + flow`. (Verified against skimage 0.26 API.)

### (7) B-spline free-form deformation (FFD) — Rueckert et al. (1999)
The gold-standard parametric **deformable** model: a coarse grid of control points
drives a smooth B-spline warp, optimized against MI/NCC on a resolution pyramid.
`SimpleITK`/`elastix` implement this. This is conceptually the same problem the
repo's **DVC/ALDVC** engine already solves with IC-GN subsets + a global
compatibility step — see `Research/aldvc_literature_review.md`. For a deformable
"align these two images" feature you can **reuse the DVC displacement field**
rather than adding elastix.

### (8) pystackreg — Thévenaz et al. (1998) TurboReg/StackReg port
A lightweight, microscopy-popular library that aligns image **stacks** with
translation / rigid-body / scaled-rotation / affine / bilinear models using a
pyramidal intensity optimizer. Nice ergonomics (`sr.register_transform_stack`) but
its capabilities are fully covered by `skimage` + `scipy`, so it is optional.

## 1.3 Method comparison

| Method | Model | Displacement range | Texture needs | Illumination robust | Python |
|--------|-------|--------------------|---------------|---------------------|--------|
| Phase correlation | translation | small–medium | low | yes (phase norm) | skimage |
| Log-polar phase corr | rigid + scale | small–medium | low | yes | skimage |
| Feature + RANSAC | up to homography | **large** | **high** | yes | skimage / cv2 |
| ECC | up to homography | small–medium | medium | yes (ECC) | cv2 |
| Mutual information | any parametric | medium | medium | **multimodal** | SimpleITK |
| Optical flow (TV-L1) | deformable | small | medium | moderate | skimage |
| B-spline FFD | deformable | medium | medium | multimodal | SimpleITK / DVC |

---

# PART 2 — Python implementation of each algorithm

All of these use only libraries **already installed** in ND2Studios (`numpy`,
`scipy`, `scikit-image`, `opencv-python`) except where SimpleITK is called out as an
optional extra. Snippets are backend-pure (no Qt) and match house conventions.

### 2.1 Estimate a translation (phase correlation)
```python
from skimage.registration import phase_cross_correlation
import numpy as np

# shift = vector to move `moving` onto `reference`, axis order (row, col) == (y, x)
shift, error, _phasediff = phase_cross_correlation(
    reference, moving, upsample_factor=20, normalization="phase",
)  # 1/20 px accuracy
```
Notes: `normalization="phase"` for illumination robustness (drift correction);
`normalization=None` (plain cross-correlation) is more robust in very high noise —
the stitch engine deliberately uses `None`. `phase_cross_correlation` works on 3D
arrays too, so a `(Z, H, W)` volume gives a 3D `(dz, dy, dx)` shift directly.
Masked variants are supported via `reference_mask`/`moving_mask` (Padfield masked FFT).

### 2.2 Apply a translation (subpixel)
```python
from scipy.ndimage import shift as nd_shift
aligned = nd_shift(moving, shift=shift, order=1, mode="constant", cval=0.0)
# order=1 bilinear (fast, safe for uint16); order=3 cubic (smoother, can overshoot)
```
Preserve dtype: compute in float, then cast back (`np.clip` before `astype(np.uint16)`).

### 2.3 Estimate rigid/affine from features + RANSAC (large displacement)
```python
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, AffineTransform, warp
import numpy as np

def _keypoints(img):
    orb = ORB(n_keypoints=800, fast_threshold=0.05)
    orb.detect_and_extract(img.astype(np.float32))
    return orb.keypoints, orb.descriptors

kp_r, des_r = _keypoints(reference)
kp_m, des_m = _keypoints(moving)
matches = match_descriptors(des_r, des_m, cross_check=True)
src = kp_m[matches[:, 1]][:, ::-1]   # (x, y)
dst = kp_r[matches[:, 0]][:, ::-1]
model, inliers = ransac(
    (src, dst), AffineTransform, min_samples=3,
    residual_threshold=2, max_trials=2000,
)
aligned = warp(moving, model.inverse, output_shape=reference.shape,
               preserve_range=True).astype(moving.dtype)
```
Use `EuclideanTransform` (rigid) or `SimilarityTransform` when you want to constrain
DOF. **skimage `warp` takes the *inverse* map** (output→input) — pass `model.inverse`.

### 2.4 Estimate with ECC (intensity, subpixel, robust)
```python
import cv2, numpy as np

def ecc_align(reference, moving, motion=cv2.MOTION_EUCLIDEAN, iters=200, eps=1e-6):
    ref = reference.astype(np.float32); mov = moving.astype(np.float32)
    warp_matrix = (np.eye(3, 3, dtype=np.float32)
                   if motion == cv2.MOTION_HOMOGRAPHY
                   else np.eye(2, 3, dtype=np.float32))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    cc, warp_matrix = cv2.findTransformECC(ref, mov, warp_matrix, motion, criteria,
                                           None, 5)  # gaussFiltSize=5
    h, w = reference.shape
    flag = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    if motion == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving, warp_matrix, (w, h), flags=flag)
    else:
        aligned = cv2.warpAffine(moving, warp_matrix, (w, h), flags=flag)
    return warp_matrix, float(cc), aligned
```
Seed ECC from a phase-correlation translation for speed/robustness. `motion` ∈
`MOTION_TRANSLATION | MOTION_EUCLIDEAN | MOTION_AFFINE | MOTION_HOMOGRAPHY`.

### 2.5 Log-polar (rotation + scale, no features)
Recover rotation (and, less reliably, scale) with **no keypoints**, invariant to
translation, by log-polar phase correlation of the **FFT magnitude**. Three details
are essential and are the usual cause of a silent all-zeros result:
(a) window before the FFT to suppress spectral leakage; (b) **`log1p` the magnitude**
so the enormous DC spike does not dominate the correlation; (c) use only the
`[0, 180)` angle range because a real image's Fourier magnitude has 180° symmetry.
*(Verified: recovers ±0.1° on `skimage.data.camera`, translation-invariant.)*
```python
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from skimage.filters import window
import numpy as np

def rotation_scale(reference, moving):
    ref_fs = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(
        reference * window("hann", reference.shape)))))
    mov_fs = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(
        moving * window("hann", moving.shape)))))
    radius = reference.shape[0] // 2
    wr = warp_polar(ref_fs, radius=radius, scaling="log", output_shape=(360, radius))[:180]
    wm = warp_polar(mov_fs, radius=radius, scaling="log", output_shape=(360, radius))[:180]
    (d_angle, d_logr), _err, _p = phase_cross_correlation(
        wr, wm, upsample_factor=10, normalization=None)
    angle = d_angle                       # 180 rows over 180° → 1°/row
    scale = np.exp(d_logr / (radius / np.log(radius)))
    return angle, scale
```
Caveat: rotation recovery is solid; **scale recovery is only approximate** with this
sampling. When scale/shear matter, prefer the feature-based (§2.3) or ECC (§2.4)
paths, optionally seeded by this angle estimate.

### 2.6 Dense deformable (optical flow)
```python
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import numpy as np

flow = optical_flow_tvl1(reference, moving)          # shape (2, H, W): (v_row, v_col)
rr, cc = np.meshgrid(np.arange(reference.shape[0]),
                     np.arange(reference.shape[1]), indexing="ij")
coords = np.array([rr + flow[0], cc + flow[1]])
aligned = warp(moving, coords, order=1, mode="edge", preserve_range=True)
```

### 2.7 Multimodal / B-spline (optional, SimpleITK)
Only if a multimodal or true B-spline deformable need is confirmed. Add SimpleITK as
an **optional lazily-imported extra** (see Part 3.4) — not in `requirements.txt`.
`sitk.ImageRegistrationMethod` with `SetMetricAsMattesMutualInformation`,
`SetOptimizerAsRegularStepGradientDescent`, and a `BSplineTransform` or
`Euler2DTransform`, run on a shrink/​smoothing pyramid.

### 2.8 The one microscopy rule that governs all of the above
**Register once, apply to all.** Estimate the transform on a single **reference
channel** (structural/brightfield, or DAPI/nuclei), then apply that *same* transform
to every other channel (and every Z, and — for channel co-registration — hold it
fixed across T). Registering channels independently destroys colocalization. This is
the identical rule the stitcher uses (`multipoint_stitching.md` §"register once,
apply to all") and it is why a cross-channel feature does **not** fit the per-channel
`EnhancementPlugin` contract cleanly (see Part 3.2 vs 3.3).

---

# PART 3 — Integrating into ND2Studios *as it is today*

## 3.1 What already exists that you must reuse or mirror

- **Data model** (`CLAUDE.md` §Data shapes): single channel = `(T, H, W)` numpy;
  multi-channel = `Dict[str, np.ndarray]` keyed by channel name; Z collapsed at load
  by projection. Full `(Z, H, W)` volumes are reachable off the record via
  `record._raw_volume.get_volume(...)` / `get_frame(..., z_mode="none")`
  (`backend/nd2_volume.py:LazyND2Volume`). The lazy per-channel proxy is
  `backend/nd2_loader.py:LazyND2Channel` (`.materialize()` for a contiguous array).
- **Existing registration primitive**: `backend/stitch/register.py` already has
  `_highpass`, `_hann2d`, `_ncc`, and a `phase_cross_correlation` call. **Reuse
  these low-level helpers** rather than re-implementing windowing/scoring.
- **Physical scale**: `record.pixel_size_um`, `record.z_step_um`; convert shifts to
  µm exactly as the DVC node does (`pipelines_page._run_dvc`):
  `voxel = (z_step_um, px, px) if is_3d else (px, px)`.
- **Three extension paradigms**, each a registry that reuses
  `core/plugin_registry.py:ParamSpec` so the GUI auto-builds the parameter form:
  1. `EnhancementPlugin` (`core/plugin_registry.py`) — `(T,H,W) → (T,H,W)` recipe step.
  2. `AnalysisPipeline` (`core/analysis_registry.py`) — dataset → masks + measurements.
  3. `DVCMethod` (`core/dvc_registry.py`) — ref+deformed volume → displacement/strain **field**.
- **The DVC feature is the exact blueprint** for adding a "third-registry" style
  feature end-to-end: `core/dvc_registry.py` + `backend/dvc/method.py`
  (`@DVCMethod.register class ALDVCMethod`) + a `special:` node in
  `pipeline_graph/registry_adapter.py` + off-thread job and `_run_dvc`/`_finish_dvc`
  handlers + viewer panel in `pages/pipelines_page.py`. Read the ARCHITECTURE.md
  "V1.51 additions (DVC)" section first.

## 3.2 Decision: which paradigm?

| Feature | Cross-channel? | Output | Best home |
|---------|----------------|--------|-----------|
| Single-channel drift correction | no | aligned `(T,H,W)` | **`EnhancementPlugin`** (Option B) |
| Multi-channel drift / channel co-reg / multi-round | **yes** | transforms + aligned stack | **New `RegistrationMethod` registry** (Option A) |
| Deformable align | n/a | displacement field | reuse **DVC** engine |

Recommendation: ship **both** an `EnhancementPlugin` (cheap, immediate value for
drift) **and** a `RegistrationMethod` registry (the general, cross-channel home).
They share one pure estimation module.

## 3.3 Option B — the quick win: a drift-correction `EnhancementPlugin`

Drops straight into the recipe pipeline with **no** node-graph or page changes —
`registry_adapter.enhancement_specs()` enumerates plugins with no hardcoded names,
and `workers/recipe_worker.py` already invokes `execute`. Limitation: it sees one
channel's `(T,H,W)` at a time (no cross-channel), so use it only for per-channel
drift stabilization.

Create `nd2studios/plugins/enhancement/registration.py`:
```python
"""Drift-correction enhancement plugin — phase-correlation frame stabilization."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable

import numpy as np

from nd2studios.core.plugin_registry import EnhancementPlugin, ParamSpec


@EnhancementPlugin.register
class DriftCorrectionPlugin(EnhancementPlugin):
    name = "Drift Correction"
    description = "Stabilize a (T,H,W) series by sub-pixel phase-correlation."

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(name="reference", label="Reference frame",
                      param_type="choice", default="previous",
                      choices=["first", "previous", "mean"],
                      tooltip="Register each frame to the first frame, the "
                              "previous frame (cumulative), or the series mean."),
            ParamSpec(name="upsample", label="Sub-pixel factor",
                      param_type="int", default=20, min_val=1, max_val=100, step=1,
                      tooltip="Registration accuracy = 1/upsample pixels."),
            ParamSpec(name="highpass_sigma", label="High-pass sigma (px)",
                      param_type="float", default=2.0, min_val=0.0, max_val=20.0,
                      step=0.5, tooltip="Band-pass whitening before correlation "
                                        "(0 = off)."),
            ParamSpec(name="interp_order", label="Interpolation order",
                      param_type="int", default=1, min_val=0, max_val=3, step=1,
                      tooltip="0 nearest, 1 bilinear, 3 cubic."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb: Optional[Callable[[int], None]] = None) -> np.ndarray:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift as nd_shift
        # Reuse the stitch engine's windowing/high-pass instead of duplicating:
        from nd2studios.backend.stitch.register import _highpass, _hann2d

        vol = np.asarray(volume)
        T = vol.shape[0]
        ref_mode = str(params.get("reference", "previous"))
        up = int(params.get("upsample", 20))
        sigma = float(params.get("highpass_sigma", 2.0))
        order = int(params.get("interp_order", 1))
        win = _hann2d(vol.shape[1:])

        out = np.empty_like(vol)
        out[0] = vol[0]
        mean_ref = vol.mean(axis=0) if ref_mode == "mean" else None
        cumulative = np.zeros(2, dtype=np.float64)
        for t in range(1, T):
            if ref_mode == "first":
                ref = vol[0]
            elif ref_mode == "mean":
                ref = mean_ref
            else:
                ref = vol[t - 1]
            fa = _highpass(ref, sigma) * win
            fb = _highpass(vol[t], sigma) * win
            sh, _err, _ph = phase_cross_correlation(fa, fb, upsample_factor=up,
                                                     normalization="phase")
            if ref_mode == "previous":
                cumulative += sh
                sh = cumulative
            moved = nd_shift(vol[t].astype(np.float32), shift=sh, order=order,
                             mode="constant", cval=0.0)
            out[t] = np.clip(moved, np.iinfo(vol.dtype).min,
                             np.iinfo(vol.dtype).max).astype(vol.dtype) \
                     if np.issubdtype(vol.dtype, np.integer) else moved
            if progress_cb:
                progress_cb(int(100 * t / max(T - 1, 1)))
        return out
```
Then add the force-import next to the other builtins so the decorator fires at
startup (in `nd2studios/__main__.py`, alongside the existing plugin/registry imports):
```python
import nd2studios.plugins.enhancement.registration  # noqa: F401
```

## 3.4 Option A — the general home: a `RegistrationMethod` registry (mirror DVC)

Use when you need cross-channel "register on reference, apply to all", richer
transform models, or a dedicated result viewer. Mirror the DVC files **byte-for-byte**.

**Step 1 — `nd2studios/core/registration_registry.py`** (copy `core/dvc_registry.py`):
```python
"""Registration method registry — estimate a spatial transform aligning a
`moving` image/stack to a `reference`, then resample. Distinct from enhancement
plugins (per-channel (T,H,W)->(T,H,W)) and DVC (dense field): registration returns
a *parametric transform* plus the aligned data, and is driven by ONE reference
channel then applied to all. No PySide6 import (backend-callable / headless)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from nd2studios.core.plugin_registry import ParamSpec  # reuse, do not duplicate


@dataclass
class RegistrationResult:
    """Output of RegistrationMethod.run.

    - transforms: one (per frame/pair) 3x3 homogeneous matrices in pixels, or a
      list of translation vectors; keep a single, documented convention.
    - shifts_px: (N, d) recovered shifts in pixels (for translation methods).
    - aligned: optional resampled stack (same shape/dtype as the moving input).
    - confidence: (N,) NCC/ECC/inlier-fraction per frame for QC.
    - pixel_size_um: to report shifts in micrometers (mirror DVC.displacement_um).
    """
    model: str = "translation"
    transforms: Optional[np.ndarray] = None
    shifts_px: Optional[np.ndarray] = None
    aligned: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    pixel_size_um: Tuple[float, ...] = ()
    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def shifts_um(self) -> Optional[np.ndarray]:
        if self.shifts_px is None or not self.pixel_size_um:
            return self.shifts_px
        return self.shifts_px * np.asarray(self.pixel_size_um, dtype=np.float64)


class RegistrationMethod(ABC):
    name: str = "Unnamed"
    description: str = ""
    _registry: Dict[str, Type["RegistrationMethod"]] = {}

    @classmethod
    def register(cls, m):
        RegistrationMethod._registry[m.name] = m
        return m

    @classmethod
    def get_methods(cls) -> List[Type["RegistrationMethod"]]:
        return list(RegistrationMethod._registry.values())

    @classmethod
    def get_method(cls, name: str) -> Optional[Type["RegistrationMethod"]]:
        return RegistrationMethod._registry.get(name)

    @abstractmethod
    def get_params(self) -> List[ParamSpec]: ...

    @abstractmethod
    def run(self, reference: np.ndarray, moving: np.ndarray,
            pixel_size_um: Tuple[float, ...], params: Dict[str, Any],
            progress_cb: Optional[Callable[[int], None]] = None,
            cancelled_cb: Optional[Callable[[], bool]] = None,
            ) -> RegistrationResult: ...
```

**Step 2 — the pure estimation engine** `nd2studios/backend/registration/estimate.py`
holds the Part 2 functions (`estimate_translation`, `estimate_affine_features`,
`ecc_align`, `apply_transform`, `apply_shift`), reusing
`backend/stitch/register.py` helpers. Keep it Qt-free with `progress_cb`/`cancelled_cb`.

**Step 3 — the registered method(s)** `nd2studios/backend/registration/method.py`
(mirror `backend/dvc/method.py`):
```python
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from nd2studios.core.registration_registry import RegistrationMethod, RegistrationResult
from nd2studios.core.plugin_registry import ParamSpec
from nd2studios.backend.registration import estimate


@RegistrationMethod.register
class RigidRegistration(RegistrationMethod):
    name = "Rigid / Translation"
    description = "Phase-correlation translation (optionally + ECC euclidean/affine refine)."

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(name="model", label="Transform model", param_type="choice",
                      default="translation",
                      choices=["translation", "euclidean", "affine"]),
            ParamSpec(name="reference", label="Reference frame", param_type="choice",
                      default="first", choices=["first", "previous", "mean"]),
            ParamSpec(name="upsample", label="Sub-pixel factor", param_type="int",
                      default=20, min_val=1, max_val=100, step=1),
            ParamSpec(name="apply_to_all_channels", label="Apply to all channels",
                      param_type="bool", default=True,
                      tooltip="Estimate on the reference channel, apply the same "
                              "transform to every channel (preserve colocalization)."),
        ]

    def run(self, reference, moving, pixel_size_um, params,
            progress_cb=None, cancelled_cb=None) -> RegistrationResult:
        return estimate.register_series(reference, moving, pixel_size_um, params,
                                        progress_cb=progress_cb,
                                        cancelled_cb=cancelled_cb)
```

**Step 4 — force-import** in `nd2studios/__main__.py` next to the DVC import so the
`@RegistrationMethod.register` decorator fires at startup:
```python
import nd2studios.backend.registration.method  # noqa: F401
```

**Step 5 — node-graph wiring** in `nd2studios/pipeline_graph/registry_adapter.py`,
copying the DVC `special:` entries:
- add `SPECIAL_REGISTER_OP_KEY = "special:register"` near `SPECIAL_DVC_OP_KEY`;
- append a `_SPECIAL_OPS` tuple copied from the DVC one — a `HEXAGON` shaped
  `NodeCategory.SPECIAL` op that takes an `IMAGE` channel wire (the 7th tuple
  element is what gives DVC its "rainbow channel wire" — keep it so registration
  can take the reference channel + optional others);
- add a branch to `param_specs_for()` that returns your reference/moving/scope
  params followed by `RegistrationMethod().get_params()` (mirror the DVC branch).

**Step 6 — dispatch + viewer** in `nd2studios/pages/pipelines_page.py`:
- add `_RUN_REGISTER_KEY` near `_RUN_DVC_KEY`;
- add a `_RegisterJob` worker (copy `_DVCJob`) that reads volumes/series off-thread
  via `record._raw_volume` and computes `pixel_size_um` like `_run_dvc` does;
- add `_run_register(node)` / `_finish_register(result)` (copy `_run_dvc`/
  `_finish_dvc`) and wire them into the op dispatch;
- add a small result panel (a shifts-vs-time plot + QC table + "write aligned back
  to channels" action), modeled on `widgets/dvc_panel.py` / `_ensure_dvc_panel`.

> The exact line numbers in `registry_adapter.py` and `pipelines_page.py` drift
> between versions — locate the DVC entries by symbol name (`SPECIAL_DVC_OP_KEY`,
> `_RUN_DVC_KEY`, `_DVCJob`, `_run_dvc`, `_finish_dvc`, `_ensure_dvc_panel`) and
> place the registration equivalents immediately beside each one.

## 3.5 Conventions to honor (from `CLAUDE.md`)
- `from __future__ import annotations` at the top of every module; type hints on
  public functions; NumPy-style docstrings explaining the *why*.
- **Backend purity**: nothing under `backend/`, `plugins/`, `pipeline_graph/` may
  import `PySide6`. Take data in, return data out, accept `progress_cb` /
  `cancelled_cb`.
- All long work runs in a `QThread` worker — never block the GUI.
- Honor `pixel_size_um` for any reported spatial quantity.
- Registries reuse `ParamSpec`; do not duplicate it.
- Preserve dtype through resampling (compute in float, clip, cast back).

---

# PART 4 — Recommended approach, dependencies, validation

## 4.1 Library recommendation (best fit for this codebase)

**Primary engine: the already-installed stack — `scikit-image` + `scipy.ndimage` +
`opencv-python`.** Rationale:
- It covers translation, rigid, similarity, affine, homography, and mild deformable
  (optical flow) with **zero new dependencies** — matching ND2Studios' deliberate
  dependency-minimal philosophy (the stitch and DVC engines are built on exactly
  this stack, and heavy backends are kept optional/gated).
- `skimage.registration.phase_cross_correlation` for translation/drift (already
  proven in `backend/stitch/register.py`); `skimage.feature.ORB` + `skimage.measure.
  ransac` + `skimage.transform` for large-displacement affine; `cv2.findTransformECC`
  for robust subpixel intensity alignment; `scipy.ndimage.shift`/`affine_transform`
  for resampling.

**Optional, lazily-imported extras (never in `requirements.txt`, gated by
`importlib.util.find_spec` with a friendly ImportError — exactly like
cellpose/stardist):**
- **SimpleITK** — only if a confirmed **multimodal (mutual-information)** or true
  **B-spline deformable** need arises.
- **pystackreg** — optional ergonomic stack aligner; not needed since skimage+scipy
  cover its models.

**Do not add** a deformable engine from scratch: the repo's **DVC/ALDVC** engine is
already a deformable-registration solver — reuse its displacement field if
deformable "align" is requested.

## 4.2 Recommended build order
1. `EnhancementPlugin` drift correction (Part 3.3) — immediate value, no GUI work.
2. Pure `backend/registration/estimate.py` engine + tests.
3. `RegistrationMethod` registry + node + viewer (Part 3.4) for cross-channel /
   multi-round alignment.
4. SimpleITK-backed multimodal/deformable method *only if a real dataset needs it.*

## 4.3 Validation strategy (pytest, synthetic ground truth — house pattern)
Mirror `tests/histothresh/` / `tests/spots/` style with a `conftest.py` fixture:
- **Known-shift recovery**: take one image, apply a known sub-pixel shift with
  `scipy.ndimage.shift`, assert recovered shift ≤ ~0.1 px at `upsample_factor=20`.
- **Known affine recovery**: apply a known `AffineTransform`, assert estimated
  matrix ≈ ground truth within tolerance; assert inlier fraction is high.
- **Drift series**: synth a `(T,H,W)` random-walk drift, assert residual drift after
  correction is below threshold; test `first`/`previous`/`mean` reference modes.
- **Register-once-apply-to-all**: two channels with identical drift; assert the
  transform estimated on channel A, applied to channel B, aligns B.
- **dtype/robustness**: `uint16` in → `uint16` out, no overflow; add Poisson noise
  and low-overlap edge cases; assert graceful low-confidence flagging.
- **µm reporting**: assert `shifts_um()` scales `shifts_px` by `pixel_size_um`.
- Note: stale `tests/__pycache__/test_drift_correction.pyc` suggests a prior
  drift-correction feature existed — check `git log`/`git show` for reusable tests.

## 4.4 Pitfalls
- A cross-correlation peak *always* exists — near-blank/low-texture frames produce
  spurious shifts; gate on NCC/ECC confidence and fall back to identity.
- Phase correlation is translation-only and wraps modulo image size — for large
  shifts use `disambiguate=True` or seed a feature/ECC method.
- `skimage.warp` uses the **inverse** map; `scipy.ndimage.affine_transform` maps
  **output→input**; `cv2.warpAffine` uses the **forward** matrix unless
  `WARP_INVERSE_MAP`. Getting the direction wrong silently doubles/negates the
  transform — assert on synthetic data.
- Never register channels independently (breaks colocalization) — estimate on one
  reference channel, apply to all.
- Edge padding (`cval=0`) introduces black borders that bias downstream analysis;
  consider cropping to the common valid region or `mode="nearest"`.

---

# PART 5 — Region-of-interest, feature-based, and late-frame robustness (V1.60, implemented)

## 5.1 Why "later timepoints stop registering" (and the fix)

Symptom: registration is excellent on the first frames of a timelapse and degrades
later. Reproduced on a synthetic 30-frame series (photobleach + moving distractor +
growing noise); mechanisms and fixes:

- **No confidence gating by default.** The NCC/ECC confidence *already* collapses to
  ~0 on a failed late frame, but with `min_confidence=0` the spurious shift was applied
  anyway. **Fix:** *hold the last good transform* when `confidence < min_confidence`
  (absolute modes) or *skip the increment* (`previous`); never snap to identity.
  Default `min_confidence` = 0.2. (Late-frame error 285 px → ~10 px in the repro.)
- **`previous` (cumulative) accumulates error** → late frames drift off (66–89 px once
  SNR drops). **Fix:** a **two-pass `template` reference** — rough-stabilize with a
  cheap `previous` translation pass, average into a template, then register every frame
  to that template (global anchor, no accumulation, robust to a bleached/atypical
  single anchor frame). This is the recommended default.
- **Photobleaching / SNR decay** weakens late correlation. **Fix:** optional per-frame
  `zscore` normalization (estimation only; geometry unaffected).
- **Large-shift wrap.** Phase correlation wraps modulo image size. **Fix:**
  `phase_cross_correlation(..., disambiguate=True)` (and/or the feature model, which has
  no wrap limit).

## 5.2 Region-based registration (does it help? yes)

Estimate the transform on a **sub-region**, apply it **full-frame**. Increases
reliability whenever whole-frame correlation is corrupted:
- live cells crawling/dividing while a **static landmark** (bead, substrate feature,
  pillar, well edge) should define drift — restrict the estimate to the landmark;
- bright debris / dust / vignette hijacking the correlation peak — exclude it;
- only part of the field textured — focus where the signal is.

Mechanism (correctness):
- **Translation + rectangle** → crop both to the box, full phase correlation
  (**subpixel**); a pure shift is invariant to the crop origin → applies full-frame.
- **Translation + freeform** → `phase_cross_correlation(reference_mask, moving_mask)`
  (Padfield) — **integer-pixel** (masked correlation drops subpixel/disambiguate).
- **Euclidean/affine + any ROI** → `cv2.findTransformECC(..., inputMask=...)` in
  full-frame coords (do **not** crop — cropping moves the rotation center).
- Rasterize freeform shapes with the existing `manual_mask.rasterize_shapes`.

## 5.3 Feature-based registration (ORB + RANSAC)

For **large** displacement / rotation / scale / partial overlap (sample re-mount,
multi-round/re-stain, objective change) where phase-correlation wrap and ECC's capture
range fail. ORB keypoints → `match_descriptors(cross_check=True)` → `ransac`
(Euclidean/Similarity/Affine); confidence = inlier fraction; too few inliers → identity
fallback (the graceful failure on sparse/blank fields). Emits the same `(2,3)` affine
warp the ECC path does, so the apply-path is unchanged. Fit **src=reference,
dst=moving** so the model maps reference→moving (matches `apply_warp`'s
`WARP_INVERSE_MAP`) — assert on synthetic data (a known rotation).

## 5.4 Cropping to the common valid region

`apply_shift` zero-pads the border vacated by each frame's shift, so every registered
frame has black borders in a different place and (with subpixel/drift) a slightly
different valid extent. To get frames that are **equal-size, recentred, and border-free**
for downstream analysis/export, crop to the **common valid region** — the intersection
of all frames' non-padded footprints. For **translation** this is an exact axis-aligned
rectangle computed analytically from the shifts (`estimate.common_translation_crop`):
`rows [max(0,max dy) : H+min(0,min dy)]`, `cols` likewise, with a 1 px inset per padded
side for bilinear edge fuzz. For rotation/affine the footprint is a rotated quad and the
intersection is a polygon — the largest inscribed axis-aligned rectangle is a separate
problem (deferred; the app offers common-crop for translation only). The registered
frame stays in **original full-frame coordinates**, so the common-crop rectangle lives in
the same coordinate system as the app's existing preview-crop — composing them is a plain
rectangle intersection, and the crop flows through the one existing crop chokepoint
(`_crop_rect`) to reach analysis / export / viewer uniformly. **Register → crop** (crop
after applying the transform to the full frame), never crop-then-register.

## 5.5 Integration (as built)

All of the above live in the pure engine (`backend/registration/estimate.py`) + method
(`method.py`); the node UI adds a hidden `roi` param + a "Pick ROI…" popup button
(rectangle dialog / freeform draw) and a `crop_to_common` toggle in
`pages/pipelines_page.py`. The ROI spec and the published common-crop are serializable
and reach every consumer through `node.params → method.run → estimate_series` and the
single `_crop_rect()` chokepoint respectively; the register-once/apply-to-all path and
`RegisteredFrameVolume` need no changes. See
`CodeLog/ClaudesPlan/V1.60_registration_roi_and_features.md`.

## References (BibTeX)
```bibtex
@inproceedings{kuglin1975phase,
  author={Kuglin, C. D. and Hines, D. C.},
  title={The phase correlation image alignment method},
  booktitle={Proc. IEEE Int. Conf. Cybernetics and Society}, year={1975}, pages={163--165}}
@article{guizar2008efficient,
  author={Guizar-Sicairos, Manuel and Thurman, Samuel T. and Fienup, James R.},
  title={Efficient subpixel image registration algorithms},
  journal={Optics Letters}, year={2008}, volume={33}, number={2}, pages={156--158},
  doi={10.1364/OL.33.000156}}
@article{reddy1996fft,
  author={Reddy, B. S. and Chatterji, B. N.},
  title={An FFT-based technique for translation, rotation, and scale-invariant image registration},
  journal={IEEE Trans. Image Processing}, year={1996}, volume={5}, number={8}, pages={1266--1271},
  doi={10.1109/83.506761}}
@article{lowe2004sift,
  author={Lowe, David G.}, title={Distinctive image features from scale-invariant keypoints},
  journal={Int. J. Computer Vision}, year={2004}, volume={60}, number={2}, pages={91--110},
  doi={10.1023/B:VISI.0000029664.99615.94}}
@inproceedings{rublee2011orb,
  author={Rublee, Ethan and Rabaud, Vincent and Konolige, Kurt and Bradski, Gary},
  title={ORB: An efficient alternative to SIFT or SURF},
  booktitle={ICCV}, year={2011}, pages={2564--2571}, doi={10.1109/ICCV.2011.6126544}}
@article{fischler1981ransac,
  author={Fischler, Martin A. and Bolles, Robert C.},
  title={Random sample consensus: a paradigm for model fitting},
  journal={Communications of the ACM}, year={1981}, volume={24}, number={6}, pages={381--395},
  doi={10.1145/358669.358692}}
@article{evangelidis2008ecc,
  author={Evangelidis, Georgios D. and Psarakis, Emmanouil Z.},
  title={Parametric image alignment using enhanced correlation coefficient maximization},
  journal={IEEE Trans. PAMI}, year={2008}, volume={30}, number={10}, pages={1858--1865},
  doi={10.1109/TPAMI.2008.113}}
@article{mattes2003mi,
  author={Mattes, David and Haynor, David R. and Vesselle, Hubert and Lewellen, Thomas K. and Eubank, William},
  title={PET-CT image registration in the chest using free-form deformations},
  journal={IEEE Trans. Medical Imaging}, year={2003}, volume={22}, number={1}, pages={120--128},
  doi={10.1109/TMI.2003.809072}}
@article{zach2007tvl1,
  author={Zach, Christopher and Pock, Thomas and Bischof, Horst},
  title={A duality based approach for realtime TV-L1 optical flow},
  booktitle={Pattern Recognition (DAGM)}, year={2007}, pages={214--223},
  doi={10.1007/978-3-540-74936-3_22}}
@article{rueckert1999ffd,
  author={Rueckert, Daniel and Sonoda, L. I. and Hayes, C. and Hill, D. L. G. and Leach, M. O. and Hawkes, D. J.},
  title={Nonrigid registration using free-form deformations},
  journal={IEEE Trans. Medical Imaging}, year={1999}, volume={18}, number={8}, pages={712--721},
  doi={10.1109/42.796284}}
@article{thevenaz1998pyramid,
  author={Th\'evenaz, Philippe and Ruttimann, Urs E. and Unser, Michael},
  title={A pyramid approach to subpixel registration based on intensity},
  journal={IEEE Trans. Image Processing}, year={1998}, volume={7}, number={1}, pages={27--41},
  doi={10.1109/83.650848}}
```
