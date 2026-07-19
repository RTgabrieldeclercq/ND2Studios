# Literature Review: Mean Deformation Metrics (MDM) — DVC on a Surface

**Date:** 2026-07-13
**Author:** Claude (with A. McGhee / G. Declercq)
**Related version:** V1.68

## Problem Statement

ND2Studios computes a dense 3-D displacement field `u(x)` with its ALDVC engine
(see `aldvc_literature_review.md`). For a 3-D object drawn across the Z-stack (the
V1.65 "3D Mask Drawing" node → V1.67 DVC-on-object voxel render), we want to
quantify **how the surrounding matrix deforms the object** — contraction,
expansion, rotation, principal stretch directions — *without* assuming a
constitutive law for the matrix, and to display the field **on the object's
surface** the way the reference figures do. V1.67 colored the object's *voxels*;
we want the object's **boundary surface** carrying the displacement, the paper's
full metric suite, and a 2-D cartographic "unwrap" for viewpoint-independent
inspection. Crucially, **DVC itself must not change** — this is a *kinematic
post-processing* of an already-measured field.

## Key Papers

### 1. Stout, Bar-Kochba, Estrada, Toyjanova, Kesari, Reichner, Franck (2016) — "Mean deformation metrics for quantifying 3D cell–matrix interactions without requiring information about matrix material properties"
- **Journal:** PNAS 113(11): 2898–2903
- **DOI:** 10.1073/pnas.1510935113
- **Key contribution:** A **material-property-free** set of *mean* deformation
  metrics for a cell (object) embedded in a 3-D matrix. Takes an
  already-measured displacement field (they use FIDVC) and, via the divergence
  theorem, reduces the **mean deformation gradient over the cell volume** to a
  **surface integral of the displacement over the cell boundary ∂V**. From the
  mean deformation gradient `⟨F⟩` it derives volume change `⟨J⟩`, principal
  stretches/directions (`⟨λ_i⟩`, `⟨N_i⟩` from the polar decomposition), and the
  mean/cumulative rotation (`⟨θ⟩`, `⟨Θ⟩`).
- **Method summary:** Continuum kinematics: `x(X,t) = u(X,t) + X`, deformation
  gradient `F = ∇x = I + ∇u`. Instead of a pointwise `F`, define a mean over the
  cell reference volume `V0` (Eq. 3) and convert the volume integral of `∇u` to a
  surface integral (divergence theorem, Eq. 6): `⟨∇u⟩ = (1/vol V0) ∮_∂V0 u ⊗ n
  dA`. Then `⟨F⟩ = I + ⟨∇u⟩`, `⟨J⟩ = det⟨F⟩`, polar-decompose `⟨F⟩ = ⟨R⟩⟨U⟩`,
  eigen-decompose `⟨U⟩` for stretches + principal directions, and
  `cos⟨θ⟩ = (tr⟨R⟩ − 1)/2` with cumulative `⟨Θ⟩ = ∫|⟨θ⟩|dτ`. The cell surface is
  discretized from a binary mask by **marching cubes** (ref 40) + **Taubin
  smoothing** (ref 41); displacements are interpolated onto the surface and split
  into normal `u⊥` and tangential `u∥` parts; the closed surface is parameterized
  to a sphere and unwrapped to 2-D via the **Mollweide** (equal-area) projection.
- **Relevance to us:** This *is* the method V1.68 implements. It is exactly the
  "adapt the DVC field onto a surface, don't touch DVC" split the user asked for:
  the correlation is never re-solved, only its output is integrated over ∂V. Our
  ALDVC engine is the same family as their FIDVC, so `DVCResult.displacement_field`
  is a drop-in `u`.

### 2. Lorensen & Cline (1987) — "Marching Cubes: A high resolution 3D surface construction algorithm"
- **Journal:** ACM SIGGRAPH Computer Graphics 21(4): 163–169
- **DOI:** 10.1145/37402.37422
- **Key contribution:** The isosurface-extraction algorithm turning a binary /
  scalar volume into a triangle mesh.
- **Method summary:** Per-cube lookup of the 0.5-level crossing → triangles.
- **Relevance to us:** How the drawn `(Z,H,W)` object mask becomes the closed
  surface `∂V` (`skimage.measure.marching_cubes`, a core dependency). The paper
  cites this exact algorithm (their ref 40).

### 3. Taubin (1995) — "A signal processing approach to fair surface design"
- **Journal:** SIGGRAPH '95: 351–358
- **DOI:** 10.1145/218380.218473
- **Key contribution:** **Non-shrinking** mesh smoothing (the λ|μ two-pass
  Laplacian) — smooths a mesh without the volume collapse a plain Laplacian
  causes.
- **Method summary:** Alternate a positive-λ Laplacian step with a
  negative-μ step (`μ < −λ`), which acts as a low-pass filter in the mesh's
  spectral domain, preserving low frequencies (overall shape/volume).
- **Relevance to us:** Smooths the blocky marching-cubes surface so `u⊥`/`u∥` and
  the surface integral are well-conditioned, without shrinking the object (which
  would bias `⟨J⟩`). Implemented in pure numpy in `backend/viz3d/surface.py`.

## Method Comparison

| Method | Pros | Cons | Complexity | Accuracy |
|--------|------|------|------------|----------|
| Voxel field on object (V1.67) | Simple; shows interior | Not a *metric*; viewpoint-dependent; no ⟨F⟩ | Low | n/a (display only) |
| Pointwise strain over the matrix | Full field detail | Noisy; needs a matrix model to interpret force; no single summary | High | Field-level |
| **MDM surface integral (V1.68, chosen)** | Material-property-free; one interpretable ⟨F⟩/⟨J⟩/⟨λ⟩/⟨θ⟩ per object; robust (surface integral averages field noise) | A *mean* only (no intra-object detail); needs a closed surface | Medium | ⟨F⟩ exact for a linear field |

## Implementation Notes

### Mathematical formulation
```
F = ∇x = I + ∇u                                             (Eq. 2)
⟨∇u⟩ = (1/vol V0) ∮_∂V0 u ⊗ n dA                            (Eq. 6, divergence thm)
⟨F⟩  = I + ⟨∇u⟩                                             (Eq. 7)
⟨J⟩  = det⟨F⟩
⟨F⟩  = ⟨R⟩⟨U⟩  (polar);  ⟨λ_i⟩,⟨N_i⟩ = eig(⟨U⟩)
cos⟨θ⟩ = (tr⟨R⟩ − 1)/2                                       (Eq. 8)
⟨Θ⟩   = ∫₀ᵗ |⟨θ(τ)⟩| dτ                                      (Eq. 9)
```
Discrete surface integral over the triangulation:
`⟨∇u⟩_ij = (1/V) Σ_faces (ū_f)_i (n_f)_j A_f`, with `ū_f` the mean of the face's
three vertex displacements, `n_f` the outward unit face normal, `A_f` the face
area, and `V = (1/6) Σ_faces v0·(v1×v2)` (the same triangulation's enclosed
volume, so `V` and `∮` are consistent). Polar decomposition via SVD with a Kabsch
sign correction so `⟨R⟩` is a proper rotation.

### Numerical considerations
- **Outward orientation** is mandatory (the sign of `∮ u⊗n` depends on it). We
  compute the signed volume and flip the face winding if negative, so face
  normals point out and `V > 0`.
- Marching cubes on a mask that touches the array border leaves the surface
  **open**; we pad one background voxel so it is always closed.
- `arccos((tr R − 1)/2)` is sensitive near 0/π; `⟨θ⟩` carries ~1e-6° round-off on
  a pure stretch even though `⟨F⟩` is exact — acceptable (it *is* ~0).
- Physical-µm alignment (from V1.67) makes DVC↔surface robust to XY downsample and
  anisotropic Z; a rare DVC `z_start` crop uses `z_offset_um`.

### Validation strategy
1. **Surface**: marching cubes + Taubin on a synthetic sphere → enclosed volume
   ≈ `(4/3)πr³`; unit vertex/face normals; watertight (every edge shared by two
   faces, `V − E + F = 2`); non-shrinking under 50 Taubin iterations.
2. **MDM on the paper's canonical cases (Fig 1)**: simple stretch
   `λ=[1, 1/1.5, 1.5]`, axial rotation `θ=45°`, simple shear `k` — a **linear**
   field, so the surface integral is **exact**: recovered `⟨F⟩` matches the
   analytic form to ~1e-12–1e-15, `⟨J⟩`/`⟨λ⟩`/`⟨θ⟩` recovered exactly. Any random
   linear field is recovered exactly (subsumes A–C).
3. **On-surface sampling** of a synthetic linear displacement field is exact at
   the vertices (so `⟨F⟩` is exact).
4. **Unwrap**: Mollweide (equal-area) covers ≈ π/4 of the frame; equirectangular
   fills it; round-trips a known lat/lon pattern.
(Tests in `tests/viz3d/test_mdm.py`, `test_surface.py`, `test_object_surface.py`.)

## Chosen Approach

**Decision:** Implement the paper's kinematic MDM as pure-numpy post-processing of
`DVCResult`, in `backend/viz3d/{surface,mdm}.py` + `overlays.dvc_object_surface` /
`unwrap_surface`, rendered on the existing PyVista surface + a matplotlib 2-D
unwrap in the DVC panel. **`backend/dvc/` is neither imported nor modified.**

**Justification:**
- The paper is explicitly material-property-free and purely kinematic — it needs
  only the measured field + a surface, both of which we already have.
- Marching cubes (`skimage`) and the interpolation (`scipy`) are core deps; Taubin
  is ~20 lines of numpy, keeping `backend/viz3d` Qt-free / VTK-free / headless.
- The surface integral averages out per-node DVC noise, giving one interpretable
  number set per object per frame — exactly what the lab wants to compare across
  conditions.

**Trade-offs accepted:**
- MDM is a **mean** — it does not resolve intra-object heterogeneity (by design;
  that is the point of the paper).
- The spherical UV parameterization is star-shaped-convex-friendly (the paper's
  granule-like objects); a strongly non-convex object may fold in the unwrap (a
  *display* artifact only — the 3-D surface and MDM are unaffected). Conformal
  parameterization is deferred.

## References (BibTeX)

```bibtex
@article{stout2016mdm,
  author  = {Stout, D. A. and Bar-Kochba, E. and Estrada, J. B. and Toyjanova, J.
             and Kesari, H. and Reichner, J. S. and Franck, C.},
  title   = {Mean deformation metrics for quantifying {3D} cell--matrix
             interactions without requiring information about matrix material
             properties},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2016},
  volume  = {113},
  number  = {11},
  pages   = {2898--2903},
  doi     = {10.1073/pnas.1510935113}
}
@article{lorensen1987marchingcubes,
  author  = {Lorensen, W. E. and Cline, H. E.},
  title   = {Marching Cubes: A High Resolution {3D} Surface Construction
             Algorithm},
  journal = {ACM SIGGRAPH Computer Graphics},
  year    = {1987},
  volume  = {21},
  number  = {4},
  pages   = {163--169},
  doi     = {10.1145/37402.37422}
}
@inproceedings{taubin1995fair,
  author    = {Taubin, G.},
  title     = {A Signal Processing Approach to Fair Surface Design},
  booktitle = {SIGGRAPH '95},
  year      = {1995},
  pages     = {351--358},
  doi       = {10.1145/218380.218473}
}
```
