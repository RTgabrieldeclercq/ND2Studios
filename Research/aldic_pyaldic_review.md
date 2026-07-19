# Literature Review: 2D Augmented-Lagrangian DIC (AL-DIC) via pyALDIC / `al-dic`

**Date:** 2026-07-14
**Author:** Claude (for G. Declercq)
**Related version:** V1.78

## Problem Statement

ND2Studios measures full-field 3D deformation with its DVC (ALDVC) node, but has
no equivalent for **2D** image pairs — single-plane live-cell / traction / gel
experiments where a Z-stack is unnecessary or unavailable. We want a 2D
Digital Image Correlation (DIC) node that produces a dense in-plane displacement
+ strain field and plugs into the existing pipeline graph (crop / registration /
exclude / channel wiring) and viewer, without reimplementing a solver. The task
specified the open-source **pyALDIC** package as the engine.

## Key Papers

### 1. Yang & Bhattacharya (2019) — "Augmented Lagrangian Digital Image Correlation"
- **Journal:** Experimental Mechanics 59(2):187–205
- **DOI:** 10.1007/s11340-018-00457-0
- **Key contribution:** The AL-DIC method — couples a *local* subset-based
  IC-GN solve with a *global* finite-element compatibility (kinematic
  smoothness) constraint through an Augmented-Lagrangian / ADMM iteration,
  getting local DIC's robustness **and** global DIC's smooth, compatible field
  in a solver that parallelizes over subsets.
- **Method summary:** Alternate (subproblem 1) per-subset IC-GN correlation and
  (subproblem 2) a global FE solve enforcing displacement compatibility, coupled
  by an ADMM penalty `mu` and auto-tuned `beta`, iterated to a tolerance.
- **Relevance to us:** This is the exact algorithm pyALDIC implements and the
  same family as our 3D ALDVC node — so a 2D node mirrors ALDVC conceptually
  and can share its data model and viewer.

### 2. Tong & Yang (2025) — "3D Stereo Adaptive Mesh AL-DIC"
- **Journal:** Experimental Mechanics
- **DOI:** (per pyALDIC README; Tong et al. 2025)
- **Key contribution:** Adaptive **quadtree** mesh refinement for AL-DIC —
  subdivide elements at mask boundaries, ROI edges, brush regions, and
  high-error zones, resolving detail only where needed.
- **Method summary:** A refinement policy marks elements meeting any of five
  criteria and recursively subdivides (with hanging-node / irregular
  constraints) down to a minimum element size.
- **Relevance to us:** Drives the **DIC Mesh Refinement** node — the second
  "mesh drawing" workflow the task asked for.

### 3. Yang, Hazlett, Landauer & Franck (2020) — "Augmented Lagrangian DVC"
- **Journal:** Experimental Mechanics 60:1303–1322
- **DOI:** 10.1007/s11340-020-00607-3
- **Key contribution:** The 3D volumetric counterpart (ALDVC) already shipped in
  ND2Studios as the DVC node.
- **Relevance to us:** Confirms the DVC/DIC family share a result shape
  (gridded displacement → derived strain), justifying reuse of `DVCResult` /
  `DVCPanel` for the 2D case.

## The `al-dic` package (pyALDIC)

- **PyPI:** `al-dic` v0.6.0 (import name `al_dic`). BSD-3-Clause. Python ≥3.10.
  Author Zixiang (Zach) Tong, Jin Yang's group, UT Austin. `jyang526843/pyALDIC`
  is a fork of the canonical `zachtong/pyALDIC`. Software DOI 10.5281/zenodo.19521071.
- **Runtime deps:** numpy, scipy, opencv-python, numba, scikit-image, imageio,
  matplotlib, **PySide6≥6.6**. Numba JIT → first-call compile latency.
- **Headless API used** (never launching its GUI):
  - `al_dic.core.config.dicpara_default(**overrides) -> DICPara` — validated
    (winsize even; winstepsize / winsize_min powers of two; reference_mode ∈
    {incremental, accumulative}; init_guess_mode ∈ {auto, fft, previous,
    seed_propagation}).
  - `al_dic.core.pipeline.run_aldic(para, images, masks, progress_fn, stop_fn,
    compute_strain, mesh, U0, refinement_policy) -> PipelineResult`. `images` =
    list of float64 [0,1] `(H,W)` (first = reference); `masks` = list of `(H,W)`
    (the ROI). `result_disp` has **len(images)−1** `FrameResult`s (U interleaved
    `[u,v,…]`, U_accum, F); `result_strain` per-node `StrainResult`;
    `result_fe_mesh_each_frame` per-frame `DICMesh` (coords `[x,y]`).
  - `al_dic.mesh.refinement.build_refinement_policy(*, refine_inner_boundary,
    refine_outer_boundary, refinement_mask, min_element_size, half_win)`.

## Implementation Notes

### Mesh / result adapter (the one genuinely new algorithm)
pyALDIC returns results on an *adaptive, irregular* FE mesh, whereas the app's
`DVCResult` (and `DVCPanel`) expect a **regular grid** `(Gy, Gx, 2)`. The
adapter (`backend/dic/engine.py::_frame_to_dvcresult`) resamples the nodal
displacement onto a regular grid (pitch = winstepsize over the node bbox) via
`scipy.interpolate.griddata` (linear + nearest fill). **Coordinate convention:**
pyALDIC is `(x, y)` / `[u, v]`; `DVCResult` is `(y, x)` with displacement
`[dy, dx]`, so u→`[...,1]`, v→`[...,0]`. Displacement is kept in **pixels**
(`um2px=1`); `DVCResult.voxel_size_um` does the µm conversion (matching ALDVC).
Strain is left `None` — `DVCPanel.field_bundle_from_result` re-derives it from the
gridded displacement, so all scalar/quiver/strain views work unchanged.

### Numerical considerations
- winsize snapped even; winstepsize / winsize_min snapped to powers of two.
- Whole-series solve per multipoint (`run_aldic` once) so pyALDIC handles
  accumulative vs incremental tracking + seed propagation natively.
- Progress + cancel are threaded through `progress_fn` / `stop_fn` to the app's
  `ProgressReporter` / `CancellationToken`.

### Validation strategy
Headless: stub `al_dic` with a solver imposing a known translation, assert the
adapter recovers it with the correct axis convention and µm scaling
(`scratchpad/dic_smoke.py` — passing: median dy=−2, dx=+3 for an imposed
`u=+3, v=−2`). Live: run over a synthetic shifted pair / a real ND2 with
`pip install al-dic`, confirm the DIC viewer tab renders the field inside the ROI.

## Chosen Approach

**Decision:** Wrap `al-dic` as an *optional, lazily-imported* backend
(`importlib.util.find_spec` gate + friendly `ImportError`; never in
`requirements.txt`), registered as a `DVCMethod` (`name="pyALDIC"`) returning a
2D `DVCResult`, surfaced as a terminal `special:dic` node with a dedicated "DIC"
viewer tab (reusing `DVCPanel`), plus two mesh-drawing nodes (`special:dic_roi`,
`special:dic_refine`).

**Justification:**
- Maximum reuse — the DVC data model + viewer already handle the 2D case, so the
  new surface is a thin engine wrapper + node wiring, not a new solver or panel.
- Optional dependency matches the app's stardist/cellpose convention and avoids
  forcing numba + PySide6≥6.6 pins on the core install.
- Faithful to the user's "just like the repo" request — the ROI/brush editor
  reproduces pyALDIC's ROI-toolbar flow (Add/Cut × rect/polygon/circle + brush).

**Trade-offs accepted:**
- pyALDIC's native adaptive mesh is resampled to a regular grid for display
  (the FE nodes themselves aren't rendered), which is lossless for the field
  values but flattens the mesh topology in the viewer.
- Strain is derived from the gridded displacement rather than pyALDIC's FE
  strain (consistent with how DVCPanel already shows every field).

## References (BibTeX)

```bibtex
@article{yang2019aldic,
  author  = {Yang, Jin and Bhattacharya, Kaushik},
  title   = {Augmented Lagrangian Digital Image Correlation},
  journal = {Experimental Mechanics},
  year    = {2019}, volume = {59}, number = {2}, pages = {187--205},
  doi     = {10.1007/s11340-018-00457-0}
}
@software{tong2026pyaldic,
  author  = {Tong, Zixiang and Yang, Jin},
  title   = {pyALDIC: 2D Augmented-Lagrangian Digital Image Correlation},
  year    = {2026}, doi = {10.5281/zenodo.19521071},
  note    = {PyPI: al-dic}
}
@article{yang2020aldvc,
  author  = {Yang, Jin and Hazlett, L. and Landauer, A. K. and Franck, C.},
  title   = {Augmented Lagrangian Digital Volume Correlation},
  journal = {Experimental Mechanics}, year = {2020},
  volume  = {60}, pages = {1303--1322},
  doi     = {10.1007/s11340-020-00607-3}
}
```
