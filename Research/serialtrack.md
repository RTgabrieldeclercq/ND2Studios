# Literature Review: SerialTrack particle tracking

**Date:** 2026-06-19
**Author:** Gabriel Declercq
**Related version:** V1.45

## Problem Statement

The "Track Objects" node links segmented objects across T frames into tracks.
Its only linker is a Hungarian **centroid nearest-neighbor** matcher: per frame
it builds a Euclidean centroid-distance cost matrix (gated by a max distance and
a size-change fraction) and solves a global assignment. This is fast and works
well for small, smooth inter-frame motion, but it links on *position alone* — it
has no notion of the spatial arrangement of neighbors. When objects move far,
rotate, or the field deforms a lot between frames, nearest-neighbor assignment
mis-links or drops tracks. We want a second, more robust linker for those
regimes, reusing the centroids the analysis pipelines already produce.

## Key Papers

### 1. Yang et al. (2022) — "SerialTrack: ScalE and Rotation Invariant Augmented Lagrangian Particle Tracking"
- **Journal:** SoftwareX, vol. 19, 101204
- **DOI:** 10.1016/j.softx.2022.101204
- **Key contribution:** A hybrid local–global particle-tracking velocimetry (PTV)
  method that is invariant to local scale and rotation and stays robust under
  large deformation, in 2-D and 3-D, with incremental / cumulative / double-frame
  modes.
- **Method summary:** Each particle is described by a **scale- and
  rotation-invariant topology descriptor** built from the relative geometry of
  its *k* nearest neighbors (radial distances normalized by the first-NN
  distance; angles relative to a local axis). Local matches minimize the
  Euclidean distance between two particles' distance- and angle-features
  simultaneously. The local matches are then projected onto a globally smooth,
  kinematically admissible displacement field via an **Augmented Lagrangian
  (ADMM)** iteration, which guarantees uniqueness and suppresses mismatches. The
  neighbor count *k* decays exponentially across ADMM iterations (degenerating to
  nearest-neighbor at *k*≤2), adapting to both dense and sparse seeding.
- **Relevance to us:** This is exactly the linker we are adding. Its core "simply
  expects a list of centroid coordinates for each frame" — which is what the
  Track Objects node already has (object centroids from segmentation), so we feed
  them to the library's `track_coordinates` path with no image re-detection.

### 2. Patel et al. (2018) — "Rapid, topology-based particle tracking for high-resolution measurements of large complex 3D motion fields"
- **Journal:** Scientific Reports, vol. 8, 5581
- **DOI:** 10.1038/s41598-018-23488-y
- **Key contribution:** Topology-based particle tracking (T-PT) that matches on
  the local neighbor topology rather than absolute position, enabling large 3-D
  displacement fields.
- **Method summary:** Builds per-particle feature vectors from nearest-neighbor
  topology and matches by feature similarity; the conceptual ancestor of
  SerialTrack's descriptor.
- **Relevance to us:** Explains *why* a topology descriptor beats pure
  nearest-neighbor for large motion — the justification for offering SerialTrack
  alongside the centroid linker.

### 3. Westerweel & Scarano (2005) — "Universal outlier detection for PIV data"
- **Journal:** Experiments in Fluids, vol. 39, pp. 1096–1100
- **DOI:** 10.1007/s00348-005-0016-6
- **Key contribution:** The normalized median residual test for rejecting
  spurious vectors in a velocity field.
- **Method summary:** Compares each vector to the median of its neighbors,
  normalized by the median residual; flags outliers above a threshold.
- **Relevance to us:** SerialTrack uses this test (`outliers.remove_outliers`,
  `TrackingConfig.outlier_threshold`) to clean mismatches during linking.

## Method Comparison

| Method | Pros | Cons | Complexity | Accuracy |
|--------|------|------|------------|----------|
| Centroid (Hungarian NN) | Simple, fast, area/gap gates, no extra deps | Links on position only; fails under large/rotational motion | Low | Good for small smooth motion |
| SerialTrack (topology PTV) | Scale/rotation invariant; robust to large deformation; global ADMM regularization | Needs ≥ a few objects for topology; numba JIT warm-up; no area/gap gates | Medium–high | Sub-pixel-class on validated cases; >95% (2-D) tracking ratio |

## Implementation Notes

### Mathematical formulation
The regularized linking objective (paper Eq. A.4) is

```
min_u  ∫ |E(f_n) − E(f_{n+t}(x − u))|²  +  (α/2)|∇û|²_F  dx     s.t.  u = û
```

— a local particle-matching residual plus a smoothness penalty on the global
field `û`, coupled by `u = û`. ADMM alternates a **local** topology-matching
update, a **global** linear smoothing solve `(∇(α/µ)∇·∇ + I) û = u − θ`, and a
dual update `θ ← θ + û − u`. The smoothing ratio `α/µ` is the `smoothness` knob.

### Numerical considerations
- We use SerialTrack only for **linking** (`track_coordinates`), not detection or
  strain. We set `strain_n_neighbors=0` (skip per-frame strain) and
  `use_prev_results=False` (avoid the lazy sklearn POD-GPR warm-start), so the
  only hard third-party dependency on this path is **numba** (plus numpy/scipy).
- `track_b2a` is `(Nb,)`, aligned row-for-row with the coordinates we pass per
  frame, so a track id can be chained deterministically frame→frame (incremental)
  or to the reference frame (cumulative) — no dependence on stitched-trajectory
  geometry.
- First call JIT-compiles the numba kernels (one-time pause; `cache=True`
  persists). Sparse frames (≤2 objects) fall back to nearest-neighbor within the
  field of search — still correct.
- `f_o_s` ("field of search") is in **pixels**; we map the node's "Max distance"
  knob to it (after the µm→px conversion shared with the centroid path).
- The Track Objects node exposes SerialTrack's **full tunable surface** — all
  three global-step solvers (`GlobalSolver.MLS` / `REGULARIZATION` / `ADMM`), the
  local matcher (`LocalSolver`), the neighbor-count range, `smoothness` (α/µ),
  `outlier_threshold`, the ADMM budget (`max_iter`, `iter_stop_threshold`),
  ghost-cull `dist_missing`, and the `use_prev_results` warm start. **ADMM** is the
  augmented-Lagrangian solver (§2.2) with automatic L-curve α — the most faithful
  to the paper, at higher cost; `REGULARIZATION` stays the default. The
  `use_prev_results` POD-GPR stage (frames ≥7) is the only place scikit-learn is
  reached, so it's kept an optional extra with a friendly install prompt.

### Validation strategy
- Headless unit test (`tests/test_serialtrack_tracking.py`): synthetic objects
  translating across frames → assert each object keeps one stable `track_id` and
  the expected `track_length`, in both Incremental and Cumulative modes.
- The reference run in the source repo's `SERIALTRACK_REFERENCE.md` recovers an
  imposed image shift exactly, confirming the displacement/index semantics relied
  on here.

## Chosen Approach

**Decision:** Vendor the headless subset of the existing SerialTrack Python port
into `nd2studios/backend/serialtrack/` and route it as a second `method` on the
Track Objects node, driven via `track_coordinates` on the object centroids.

**Justification:**
- The implementation already exists and is validated; we reuse it rather than
  re-derive a topology matcher.
- The node operates on centroid rows, which is exactly SerialTrack's
  bring-your-own-detector input — minimal integration surface.
- Vendoring matches the project convention of copying shared code, keeps the app
  self-contained, and (headless subset only) preserves backend purity.

**Trade-offs accepted:**
- A new numba dependency (already installed) and a one-time JIT warm-up.
- The SerialTrack path ignores the centroid-only area and frame-gap gates (its
  param surface hides them and shows mode / neighbor count instead).

## References (BibTeX)

```bibtex
@article{yang2022serialtrack,
  author  = {Yang, Jin and Yin, Yue and Landauer, Alexander K. and Buyukozturk, Selda and Zhang, Mei and Summey, Luke and Chen, Alexander and Estrada, Jonathan B. and Franck, Christian},
  title   = {SerialTrack: ScalE and Rotation Invariant Augmented Lagrangian Particle Tracking},
  journal = {SoftwareX},
  year    = {2022},
  volume  = {19},
  pages   = {101204},
  doi     = {10.1016/j.softx.2022.101204}
}

@article{patel2018topology,
  author  = {Patel, Mohak and Leggett, Susan E. and Landauer, Alexander K. and Wong, Ian Y. and Franck, Christian},
  title   = {Rapid, topology-based particle tracking for high-resolution measurements of large complex 3D motion fields},
  journal = {Scientific Reports},
  year    = {2018},
  volume  = {8},
  pages   = {5581},
  doi     = {10.1038/s41598-018-23488-y}
}

@article{westerweel2005universal,
  author  = {Westerweel, Jerry and Scarano, Fulvio},
  title   = {Universal outlier detection for PIV data},
  journal = {Experiments in Fluids},
  year    = {2005},
  volume  = {39},
  pages   = {1096--1100},
  doi     = {10.1007/s00348-005-0016-6}
}
```
