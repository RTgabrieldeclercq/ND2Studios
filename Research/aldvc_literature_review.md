# Literature Review: Augmented Lagrangian Digital Volume Correlation (ALDVC)

**Date:** 2026-06-18
**Author:** Claude (with A. McGhee)
**Related version:** V1.45

## Problem Statement

ND2Studios can load, enhance, and export ND2 microscopy data, but it cannot
*measure deformation*. Many McGhee Lab experiments (gels, tissues, cells under
mechanical load) acquire a 3D volume (a confocal/two-photon Z-stack) at a
reference state and again at one or more deformed states. The scientific
quantity of interest is the **dense 3D displacement field** `u(x)` between the
reference volume `f` and a deformed volume `g`, and the **strain tensor** field
derived from it.

Digital Volume Correlation (DVC) is the volumetric generalization of Digital
Image Correlation (DIC). Two families exist:

- **Local (subset) DVC** — divide the volume into subsets, register each
  independently. Fast and embarrassingly parallel, but the resulting field is
  noisy and *not kinematically compatible* (neighboring subsets disagree).
- **Global (FE) DVC** — solve one large variational problem for a smooth,
  compatible field. Robust but expensive and sensitive to mesh/regularization.

We want the accuracy of global DVC with the speed/parallelism of local DVC.
The **Augmented Lagrangian DVC (ALDVC)** method achieves exactly this hybrid,
and FranckLab publishes a permissively-licensed MATLAB reference we can port
without copyleft contamination of ND2Studios.

## Key Papers

### 1. Yang, Hazlett, Landauer, Franck (2020) — "Augmented Lagrangian Digital Volume Correlation (ALDVC)"
- **Journal:** Experimental Mechanics, 60, 1205–1223
- **DOI:** 10.1007/s11340-020-00607-3
- **Key contribution:** A hybrid local+global 3D DVC that splits the
  image-correlation problem into a per-subset local registration and a global
  kinematic-compatibility projection, coupled by the Augmented
  Lagrangian / ADMM. Achieves global-DVC accuracy at near-local-DVC cost and is
  trivially parallel in its dominant stage.
- **Method summary:** Minimize the image residual `∫(f(x) − g(x+u))²` subject to
  the compatibility constraint `F = ∇u` (F = deformation gradient). Introduce
  an auxiliary field and dual variables; ADMM alternates between (Subpb1) a
  local 12-DOF inverse-compositional Gauss–Newton (IC-GN) subset solve and
  (Subpb2) a global least-squares projection onto compatible displacements,
  with dual updates between. ~4 outer iterations converge.
- **Relevance to us:** This is the exact method and reference implementation we
  are porting. It is the foundation of our `nd2studios/backend/dvc/` engine.

### 2. Yang, Bhattacharya (2019) — "Augmented Lagrangian Digital Image Correlation"
- **Journal:** Experimental Mechanics, 59, 187–205
- **DOI:** 10.1007/s11340-018-00457-0
- **Key contribution:** The 2D progenitor (AL-DIC) of ALDVC: same
  local+global ADMM split for planar DIC with a 6-DOF (4 deformation-gradient +
  2 displacement) affine subset warp.
- **Method summary:** Identical ADMM structure to ALDVC but on 2D images; the
  global step uses a 2D finite-difference / FE operator.
- **Relevance to us:** Defines the **2D DIC** path the user asked us to support
  alongside 3D DVC. Our engine treats 2D DIC as the dimension-`d=2` instantiation
  (6-DOF warp, quad mesh) of the same algorithm.

### 3. Bar-Kochba, Toyjanova, Andrews, Kim, Franck (2015) — "A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations" (FIDVC)
- **Journal:** Experimental Mechanics, 55, 261–274
- **DOI:** 10.1007/s11340-014-9874-2
- **Key contribution:** The multiscale FFT cross-correlation ("bigxcorr")
  engine used to seed integer displacements robustly under large deformation.
- **Method summary:** Windowed FFT cross-correlation over a grid of subset
  centers with iterative warping/refinement and outlier removal.
- **Relevance to us:** ALDVC's Stage 1 integer-search seed
  (`IntegerSearch3Multigrid`) is derived from FIDVC. We port the windowed
  ZNCC/phase-correlation seed with a 3×3×3 subvoxel peak fit and q-factor
  confidence.

## Method Comparison

| Method | Pros | Cons | Complexity | Accuracy |
|--------|------|------|------------|----------|
| Local (subset) DVC | Embarrassingly parallel, simple | Noisy, incompatible field | Low | Moderate |
| Global (FE) DVC | Smooth, compatible, robust | Large coupled solve, mesh-sensitive, slower | High | High |
| **ALDVC (chosen)** | Local-speed parallelism + global compatibility; auto-regularized | ADMM tuning (μ, β); more moving parts | Medium | High |

## Implementation Notes

### Mathematical formulation

Constrained correlation problem (continuous form):

```
min_{u,F}  ∫_Ω | f(x) − g(x + u(x)) |² dx     subject to   F = ∇u
```

ADMM / Augmented Lagrangian split with scaled duals (v for u, w for F):

- **Subpb1 (local, parallel):** per subset, maximize ZNCC via 12-DOF (3D) /
  6-DOF (2D) affine IC-GN warp `W(P)`, with an added μ-penalty pulling the
  subset displacement toward the current compatible estimate `(û − v)`. The
  warp parameters are 9 deformation-gradient terms + 3 displacements (3D).
  IC-GN updates the warp by **inverse-compositional composition**
  `W ← W ∘ ΔW⁻¹` (not additive), using a **constant reference-image Hessian**
  cached once per subset.
- **Subpb2 (global, one sparse solve):** project the noisy `(u, F)` onto a
  compatible `û` with `F = ∇û`:

  ```
  (β·DᵀD + μ·I) û = β·Dᵀ(F − w) + μ·(u − v)
  ```

  where `D` is the finite-difference gradient operator (a sparse
  `(9·N)×(3·N)` matrix in 3D). `β` is selected per pair by an **L-curve** over
  a fixed `β`-list. **Robustness fix over the reference:** add a small Tikhonov
  term (the MATLAB leaves it commented out) and guard the L-curve `poly2` fit.
- **Dual update:** FD form *accumulates* `w += (F_sub2 − F_sub1)`,
  `v += (u_sub2 − u_sub1)` on non-Neumann indices; FE form *resets* to the
  difference and zeros duals at Dirichlet nodes.
- **Convergence:** `‖ΔU‖/√N < ADMMtol`, hard-capped at ~4 outer iterations;
  `μ` fixed at 1e-3.

Strain: build the deformation gradient `F` (from `∇û`), then the chosen strain
measure — infinitesimal `e = ½(F + Fᵀ) − I`, Green–Lagrange
`E = ½(FᵀF − I)`, Eulerian–Almansi, or Hencky — with optional Gaussian
smoothing and voxel→physical scaling via `diag(voxel_size_um)`, including the
**cross-axis derivative rescaling** required for anisotropic voxels.

### Numerical considerations

- **Stability:** the unregularized global solve in the MATLAB reference runs on a
  near-singular sparse matrix; we add Tikhonov regularization and guard the
  L-curve parabola fit.
- **Convergence:** IC-GN needs the reference-only Hessian cached and warps
  composed (not added); the deformed volume must be spline-prefiltered exactly
  once outside the iteration loop.
- **Performance:** cache the sparse global factorization (matrix constant across
  the L-curve sweep and ADMM iterations — only the RHS changes); cache per-subset
  Hessians across ADMM steps. These are the two dominant levers.
- **Edge cases:** per-subset failures (out-of-bounds warp, singular ΔP) are
  isolated → NaN-then-inpaint, never crashing the whole field.

### MATLAB → Python porting hazards (highest correctness risk)

- **1-based → 0-based** on every coordinate and DOF index.
- **Column-major** MATLAB reshape → `order='F'` for all flat-DOF ↔ `(M,N,L)`
  reshapes; `ndgrid` (x-first) axis order vs numpy `meshgrid`; the swapped
  `(v, u)` interpolation-argument order.
- **Interpolation kernel:** MATLAB `ba_interp3` is Catmull–Rom tricubic; scipy
  `map_coordinates(order=3)` is a prefiltered interpolating B-spline. They give
  small systematic subvoxel differences. We accept this (we validate against a
  Python oracle + synthetic data, not bit-for-bit MATLAB).
- **ZNCC denominator** uses sample variance with `ddof=1` (`N−1`).
- The upstream `main_ALDVC.m` on `master` ships with **unresolved git
  merge-conflict markers** (lines 1 / 1035 / 2068). Port from the **HEAD**
  section (lines 1–1034), not the merged whole.

### Validation strategy

(User-selected: synthetic + Python oracle, no MATLAB.)

1. **Indexing unit tests:** reproduce mesh/DOF layout against hand-computed
   small-grid expectations (centralized in `mesh.py`).
2. **Synthetic deformation tests:** apply a *known* rigid translation, affine
   `F`, and smooth analytic displacement to a synthetic speckle volume; confirm
   recovery to subvoxel accuracy at each stage (integer seed → IC-GN → ADMM →
   strain).
3. **Out-of-band oracle:** install SPAM and/or pyxel in a throwaway environment
   (GPL — used only as an external cross-check, never imported by ND2Studios)
   and compare displacement fields on the same synthetic inputs.

## Chosen Approach

**Decision:** Clean-room port of FranckLab ALDVC to Python on permissive
scipy/scikit-image/numpy primitives, exposed through a new `DVCMethod`
registry, a `DVCWorker`, and a `DVCPage`. FD-operator global step first; FE,
incremental tracking, and GPU as later/optional modes.

**Justification:**
- **License:** every full-featured Python DVC engine (SPAM GPL-3.0, pyxel
  CeCILL, OpenPIV GPL-3.0) is copyleft and would force ND2Studios to GPL. ALDVC
  itself is permissively licensed, so a clean-room derivative is unencumbered.
- **Architecture fit:** the method's dominant stages (FFT seed, IC-GN sweep)
  are embarrassingly parallel and map directly onto the existing
  `compute/parallel` (`shared_ndarray` + `ProcessPoolExecutor`) and
  `compute/gpu` (CuPy) infrastructure that already saturates cores/RAM in
  ND2Studios.
- **Data fit:** full `(M,T,Z,H,W)` volumes are already resident in RAM
  (`MaterializedDataset`); DVC reads them directly without disturbing the
  Z-collapsed `(T,H,W)` recipe/export contract.

**Trade-offs accepted:**
- Not bit-for-bit identical to MATLAB (different cubic interpolation kernel) —
  acceptable given Python-oracle + synthetic validation.
- FD global step (not FE) in V1 — FranckLab-recommended default, faster, no FE
  edge effects; FE deferred.

## References (BibTeX)

```bibtex
@article{yang2020aldvc,
  author  = {Yang, J. and Hazlett, L. and Landauer, A. K. and Franck, C.},
  title   = {Augmented Lagrangian Digital Volume Correlation (ALDVC)},
  journal = {Experimental Mechanics},
  year    = {2020},
  volume  = {60},
  pages   = {1205--1223},
  doi     = {10.1007/s11340-020-00607-3}
}
@article{yang2019aldic,
  author  = {Yang, J. and Bhattacharya, K.},
  title   = {Augmented Lagrangian Digital Image Correlation},
  journal = {Experimental Mechanics},
  year    = {2019},
  volume  = {59},
  pages   = {187--205},
  doi     = {10.1007/s11340-018-00457-0}
}
@article{barkochba2015fidvc,
  author  = {Bar-Kochba, E. and Toyjanova, J. and Andrews, E. and Kim, K.-S. and Franck, C.},
  title   = {A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations},
  journal = {Experimental Mechanics},
  year    = {2015},
  volume  = {55},
  pages   = {261--274},
  doi     = {10.1007/s11340-014-9874-2}
}
```
