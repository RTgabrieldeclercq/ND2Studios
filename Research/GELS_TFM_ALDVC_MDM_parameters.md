# Node parameters for `GELS_TFM.tif` — ALDVC + surface MDM

Dataset-specific parameter recommendations for running the ND2Studios DVC / 3D
Mask / surface-MDM nodes on `GELS_TFM.tif` (mCherry hydrogel granules with
embedded/surrounding tracer beads). Grounded in the file's own metadata, the
measured bead statistics, and two papers:

- **ALDVC** — Yang & Franck et al., *"Augmented Lagrangian Digital Volume
  Correlation (ALDVC),"* Exp. Mech. 60:1205–1223 (2020),
  doi:10.1007/s11340-020-00607-3 (the engine behind the DVC node).
- **MDM** — Stout et al., PNAS 113(11):2898 (2016) (the surface deformation-metric
  method the surface/3D-object view implements; see
  `CodeLog/ClaudesPlan/V1.68_dvc_surface_mdm.md`).

> Values map 1:1 onto the actual node fields (`registry_adapter.param_specs_for` +
> `backend/dvc/method.ALDVCMethod.get_params`). "Default" = the app's current
> default; "Recommended" = for *this* dataset.

## 1. Dataset at a glance (measured, not assumed)

| Property | Value | How it was read |
|---|---|---|
| Axes / shape | `TZCYX = (9, 100, 2, 4096, 4096)` | ImageJ hyperstack series |
| Timepoints (T) | **9** | `frames=9` |
| Z-slices | **100** (depth ≈ 16.8 µm) | `slices=100` |
| Channels (C) | **2** | `channels=2` |
| Multipoints (M) | **1** (no P/M axis) | single position |
| Dtype | `uint16` (12-bit range; max 4095) | page tag `BitsPerSample=16` |
| XY pixel size | **0.16776 µm** | `XResolution 2520031060/422768019 ≈ 5.96 px/µm` |
| Z step | **0.16776 µm** | ImageJ `spacing` |
| Voxel | **~0.168 µm isotropic** | XY == Z ⇒ isotropic (ideal for 3D DVC) |
| Lateral FOV | ≈ 687 × 687 µm | 4096 × 0.168 |
| Volume size | **3.35 GB / channel / timepoint**; 60.4 GB total | 4096·4096·100·2 B |

**Channel identity (the single most important setup choice).** Verified by imaging
statistics, not the filename:

- **C1 = beads** → wire into the **DVC** node. Dense discrete puncta (~10,900
  connected components in one plane; 3D nearest-neighbour spacing ≈ 0.5–0.6 µm),
  i.e. a rich speckle texture — exactly what DVC correlates.
- **C0 = mCherry granule body** → wire into the **3D Mask Drawing** node. Diffuse,
  low-contrast rounded blobs; *anti-correlated* with the beads (Pearson ≈ −0.08),
  so the beads sit in the matrix **around/between** the granules, not inside them.

This geometry — object surface from one channel, displacement-bearing tracers in the
*surrounding* medium — is precisely the MDM setup (Stout 2016, Fig 2): the matrix
displacement sampled on the object surface ∂V yields the object's mean deformation.
It is why the surface/UV-wrap analysis is the right tool here.

## 2. Channel wiring

```
channel C0 (mCherry)  ──▶ [3D Mask Drawing]   (defines the granule surface ∂V)
channel C1 (beads)    ──▶ [DVC (ALDVC)]        (displacement field around ∂V)
                                  └──▶ 3D Object / surface-MDM view + C1 context overlay
```

## 3. 3D Mask Drawing node (on C0 — mCherry)

| Field | Recommended | Default | Why |
|---|---|---|---|
| (rainbow channel) | **C0 mCherry** | — | The granule label; beads (C1) don't fill the granule. |
| `mode` | **Threshold seed + edit** | Propagate across Z | Mirrors the MDM recipe (median 5×5×5 → ~50 % of max → largest connected component). C0 is low-contrast, so expect manual cleanup; if seeding is too noisy, fall back to **Propagate across Z** and draw a few planes. |
| `propagate` | **Interpolate between planes** | Interpolate between planes | Signed-distance morph → a smooth surface finer than the drawn planes (paper smooths with Taubin; see below). |
| `apply_all_frames` | **On** for a first pass | On | The MDM `⟨F⟩` is defined on the **reference** surface `∂V0`, so one careful t=0 mask suffices for the metrics. Turn **off** and re-seed key frames only if you want the *displayed* surface to track a visibly deforming granule per frame. |
| `surface_smooth_iterations` (V1.68) | **~10** | 10 | Taubin λ/μ smoothing of the marching-cubes surface (Stout 2016, ref 41). |

Tip: pick **one** granule to validate the workflow, then use the **frame/object
toggle** (§6) to fan out over all granules.

## 4. DVC (ALDVC) node (on C1 — beads)

### Node-level (walk / scope / tractability)

| Field | Recommended | Default | Why |
|---|---|---|---|
| `tracking_mode` | **cumulative** | cumulative | MDM `⟨F⟩` is relative to the undeformed reference; ALDVC's benchmarks used cumulative. Switch to **incremental** only if late frames decorrelate under large accumulating motion. |
| `ref_frame` | **0** | 0 | t=0 is the undeformed reference. |
| `all_multipoints` | **off** | off | Single position (M=1) — moot. |
| `z_start` / `z_end` | **0 / 0 (all 100 Z)** | 0 / 0 | Beads span the stack (only mild depth attenuation was measured); trim `z_end`≈90 only if the deepest planes are too dim. |
| `downsample` | **1** if GPU/compute allows, else **2** | 1 | 1 keeps voxels **isotropic** (0.168 µm) and beads crisp; 2 → 0.335 µm XY, ~4× faster, but XY-only binning makes the cubic subset physically anisotropic (see note). |

### ALDVC engine

| Field | Recommended | Default | Range | Why / source |
|---|---|---|---|---|
| `subset_size` | **32** vox @ downsample 1 (**16** @ downsample 2) | 16 | 4–128 | Keep the physical subset ≈ **5 µm** (32 × 0.168). Dense beads → dozens of beads/subset even near the granule boundary where only the matrix side carries beads. Aligns with ALDVC's 30³-vox test setting. |
| `subset_spacing` | **start = subset (32)**, then densify to **16** | 10 | 1–128 | Non-overlapping first keeps the 4k² grid tractable (~128×128×3 nodes); halve for a finer field once it runs. |
| `search_radius` | **0 (auto = subset)** | 0 | 0–128 | Adequate unless frame-to-frame motion exceeds ~a subset; set explicitly if so. |
| `correlation` | **zncc** | zncc | zncc/phase | Zero-normalized ⇒ robust to the depth-dependent brightness attenuation measured here. |
| `strain_type` | **infinitesimal** to start | infinitesimal | 4 opts | ALDVC's default; switch to **green-lagrange** if granule stretches are large. (MDM `⟨F⟩` handles large deformation via polar decomposition regardless.) |
| `admm_iterations` | **4** (try 5) | 4 | 1–12 | ALDVC converges in **3–5** iterations (paper §Accuracy). |
| `mu` | **1e-3** (raise toward **1e-2** if noisy) | 1e-3 | 1e-6–1 | Paper sets μ ≈ O(10⁻³–10⁻¹) × diagonal of the local Hessian; 1e-3 = light global coupling, faithful to the local solution. |
| `seed_levels` | **3** (→ 4 if large motion) | 3 | 1–5 | Coarse-to-fine FFT integer seed brackets larger displacements. |
| `newFFTSearch` | **off** | off | — | Warm-start each frame from the previous field (smooth 9-frame series); turn on only if a frame loses lock. |
| `n_workers` | **0 (auto)** | 0 | 0–64 | The IC-GN sweep dominates cost on 4k² — use all cores. |
| `use_gpu` | **on if a CUDA GPU is present** | off | — | Routes the FFT seed through CuPy; big win on 4k². |

## 5. Surface / MDM display (the "3D Object" view — V1.68)

| Control | Recommended | Why |
|---|---|---|
| Colour scalar | **u⊥ (normal)**, then |u| and u∥ | u⊥ (divergent red=outward / blue=inward) reads matrix push/pull at the granule surface (Stout Fig 4C). |
| Context-channel overlay | **C1 beads**, volume/MIP **and/or** iso-surface, opacity ~0.3 | The "surrounding factors" here are the matrix beads around the granule (only 2 channels exist). |
| 2D unwrap projection | **Mollweide** (equal-area) | Viewpoint-independent map of the surface field (Stout Fig 4E–G); equirectangular as the alternative. |

## 6. Frame / Object toggle (V1.68) — many granules

The FOV holds **many** granules. On the edge **after** the 3D Mask node, set the
lever to **Objects** to auto-crop the file to each granule's bounding box
(conserving T/Z/C) and run DVC + surface MDM **per granule in parallel**.

- Keep **`mask_out` OFF** (bbox crop, keep the surrounding matrix). The
  displacement-bearing beads are *outside* the granule — hard-masking to the granule
  would delete the very signal the MDM surface integral needs.
- Use **Whole frame** instead if you want one global matrix field.

## 7. Tractability & runtime

Full-res (downsample 1) is heavy: 3.35 GB per bead volume, 8 cumulative correlations,
a ~128×128×3 subset grid at spacing 32. The engine streams one Z-stack at a time so
RAM stays bounded, but expect long CPU runtimes. Practical path:

1. **Validate on one granule** via the object crop (§6) or the preview-crop Run.
2. Turn on **GPU** and **auto workers**.
3. First full pass at **downsample 2, subset 16, spacing 16**; final pass at
   **downsample 1, subset 32** for isotropic, full-resolution metrics.

## 8. Key caveats for this dataset

- **Closed-surface requirement for the MDM scalars.** The imaged Z depth is only
  ≈ 16.8 µm. `⟨F⟩`, `⟨J⟩`, `⟨λᵢ⟩`, `⟨θ⟩` come from a surface integral over a *closed*
  boundary ∂V (divergence theorem). If a granule is taller than the stack and is
  **clipped in Z**, its surface is open and those scalars are biased. Verify each
  granule is fully enclosed in X, Y **and Z**; if not, trust the **surface
  displacement maps (u⊥/u∥/|u|)** — which are always valid — but treat the integral
  metrics as cap-truncated estimates. (Choosing granules that fit within the 100-plane
  Z is the clean fix.)
- **Beads live in the matrix, not the granule.** Subsets falling *inside* a granule
  have no texture and won't correlate (low q-factor) — expected, and harmless for MDM
  (only the surface/matrix nodes matter). This is also why `mask_out` must stay off.
- **Depth attenuation** ⇒ keep `correlation = zncc`; consider trimming the dimmest
  deep Z planes if q-factor falls off there.
- **Anisotropic subset at downsample 2.** XY binning without Z binning makes a cubic
  voxel-count subset physically anisotropic (5.4 µm XY vs 2.7 µm Z at subset 16).
  Prefer downsample 1 (subset 32) for the final, isotropic run.

## 9. Start-here recipe (copy/paste)

```
3D Mask Drawing  (channel C0 mCherry):
    mode = Threshold seed + edit
    propagate = Interpolate between planes
    apply_all_frames = On
    surface_smooth_iterations = 10

DVC (ALDVC)      (channel C1 beads):
    tracking_mode = cumulative      ref_frame = 0      all_multipoints = off
    z_start = 0   z_end = 0 (all)   downsample = 2  (→ 1 for final)
    subset_size = 16 (down=2) / 32 (down=1)
    subset_spacing = 16 (down=2) / 32 (down=1)
    search_radius = 0   correlation = zncc   strain_type = infinitesimal
    admm_iterations = 4   mu = 1e-3   seed_levels = 3
    newFFTSearch = off   n_workers = 0 (auto)   use_gpu = on (if available)

Edge after 3D Mask → scope lever = Objects   (mask_out = off)

3D Object view:  colour = u⊥,  context = C1 (beads),  unwrap = Mollweide
```

## Sources

- ALDVC: Yang et al., Exp. Mech. 60:1205 (2020),
  [doi:10.1007/s11340-020-00607-3](https://doi.org/10.1007/s11340-020-00607-3)
  (attached `s11340-020-00607-3.pdf`) — subset 30³ vx, ADMM 3–5 iters, μ≈O(10⁻³–10⁻¹)·Hessian-diag, tri-cubic, ZNCC, tol 1e-4, cumulative.
- MDM: Stout et al., [PNAS 113(11):2898 (2016)](https://www.pnas.org/doi/10.1073/pnas.1510935113) — marching cubes + Taubin surface, u⊥/u∥, `⟨F⟩` surface integral, Mollweide unwrap.
- Node fields: `nd2studios/pipeline_graph/registry_adapter.py`,
  `nd2studios/backend/dvc/method.py`; surface/MDM view: `CodeLog/ClaudesPlan/V1.68_dvc_surface_mdm.md`.
- Dataset facts: measured directly from `GELS_TFM.tif` (ImageJ metadata + bead statistics).
