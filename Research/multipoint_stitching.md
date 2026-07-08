# Literature Review: Multipoint (M) Image Stitching / Mosaicking

**Date:** 2026-07-07
**Author:** ND2Studios (Claude)
**Related version:** V1.54

## Problem Statement

A Nikon multipoint (M) acquisition records N field-of-view tiles at known stage
positions. To reconstruct a single mosaic we must place each tile on a common
canvas. Two physically distinct regimes exist and demand different math:

- **Overlapping tiles** share a strip of pixels; adjacent tiles can be
  *registered* from image content to sub-pixel accuracy (true stitching).
- **Zero-overlap tiles** share no pixels; image-content registration is
  impossible and would lock onto a spurious cross-correlation peak in noise. The
  only correct placement is from **stage coordinates** (mosaicking / montaging).

Our data mixes both. The prior ND2Studios stitcher only ever did coordinate
placement with overwrite compositing — correct for zero-overlap, but it wastes
the information in overlapping datasets (no registration, no seam blending, no
regime awareness, no QC). This review grounds the V1.54 regime-aware rebuild.

## Key Papers

### 1. Kuglin & Hines (1975) — "The phase correlation image alignment method"
- **Venue:** Proc. IEEE Int. Conf. on Cybernetics and Society, pp. 163–165.
- **Key contribution:** Translation between two images is recovered from the
  phase of their cross-power spectrum; the inverse FFT is a sharp delta at the
  shift.
- **Method summary:** `R = (F₁ · conj(F₂)) / |F₁ · conj(F₂)|`; `argmax(IFFT(R))`
  is the integer shift. Sub-pixel refinement via upsampled DFT (Guizar-Sicairos
  2008) or peak fitting. Windowing (Hann/Blackman) and band-pass whitening
  suppress DC/edge artifacts.
- **Relevance:** The core of our built-in overlap engine
  (`skimage.registration.phase_cross_correlation`, `upsample_factor`) and of
  m2stitch/ASHLAR internally.

### 2. Chalfoun et al. (2017) — "MIST: Accurate and Scalable Microscopy Image Stitching…"
- **Journal:** Scientific Reports 7, 4988. **DOI:** 10.1038/s41598-017-04567-y
- **Key contribution:** A robust grid stitcher: phase-correlation per adjacent
  pair, then a **maximum-spanning-tree global optimization** over trusted edges,
  tolerant of missing tiles and low-texture regions.
- **Method summary:** Estimate pairwise translations, score by normalized cross
  correlation, keep the most confident edges, propagate absolute positions along
  a spanning tree, refine with the stage-coordinate prior bounding the search.
- **Relevance:** `m2stitch` is a clean Python reimplementation of MIST and is our
  primary **grid overlap** engine (`stitch_images(images, rows, cols,
  position_initial_guess=…, ncc_threshold=…)` → absolute `x_pos`/`y_pos`).

### 3. Muhlich et al. (2022) — "Stitching and registering highly multiplexed whole-slide images (ASHLAR)"
- **Journal:** Bioinformatics 38(19):4613–4621. **DOI:** 10.1093/bioinformatics/btac544
- **Key contribution:** Multi-channel/multi-cycle mosaic stitching: register on
  one channel, apply the transform to all; minimum-spanning-tree from the most
  confident edges; writes pyramidal OME-TIFF.
- **Method summary:** `EdgeAligner` (intra-cycle phase correlation with a
  permutation-null confidence threshold and a LoG-like `filter_sigma`),
  `LayerAligner` (inter-cycle), `Mosaic` compositor.
- **Relevance:** Installed as a gated engine. On this Windows box `ashlar.reg`
  hard-imports `jnius`/BioFormats and needs a JDK, so `auto` never selects it;
  its multi-channel + OME-writer role is covered by our compositor + OME writer +
  the register-once-apply-to-all rule (below).

### 4. Peng et al. (2017) — "A BaSiC tool for background and shading correction…"
- **Journal:** Nature Communications 8, 14836. **DOI:** 10.1038/ncomms14836
- **Key contribution:** Low-rank + sparse decomposition estimates a smooth
  flat-field (and dark-field) from an image collection without a calibration
  reference — removes the per-tile shading that otherwise reads as a grid of
  seams.
- **Relevance:** `basicpy` is the reference implementation (optional). Its import
  chain is broken in this env (`hyperactive`/`gradient_free_optimizers` API
  drift), so BaSiC is gated with a friendly error and we ship a built-in
  smoothed-flatfield fallback (large-σ Gaussian of the per-channel tile mean,
  normalized and divided out). Off by default (it changes pixel values).

### 5. Feather / linear alpha blending (Szeliski, *Image Alignment and Stitching*, 2006)
- **Venue:** Foundations and Trends in Computer Graphics and Vision 2(1).
- **Key contribution:** In overlap zones, weight each tile by a distance
  transform from its own border (alpha ramp) and normalize — hides seams under
  uneven illumination far better than max/average.
- **Relevance:** Our default `blend="feather"`; `average`/`max`/`none` offered.

## Method Comparison

| Method | Pros | Cons | Complexity | Accuracy |
|--------|------|------|------------|----------|
| Coordinate placement | Only correct option for zero-overlap; trivial; fast | Alignment limited to stage precision | O(N) | Stage-limited |
| Built-in phase-corr + spanning tree | Always available (skimage/scipy); handles non-grid overlap | We maintain the global-opt; slower than a tuned lib | O(E·FFT) | ~1–2 px |
| m2stitch (MIST) | Robust, missing-tile tolerant, seedable | Regular grid only; not a compositor | O(E·FFT) | Sub-px |
| ashlar | Multi-channel/cycle, writes OME-TIFF | Needs a JDK on Windows; fragile internal API | High | Sub-px |

## Implementation Notes

### Mathematical formulation
- **Coordinate placement (KEPT orientation).** Negate stage XY (lab Nikon sign
  convention), then `x_px = round((sx−min_sx)/px)`, `y_px = round((max_sy−sy)/px)`
  (Y flipped so highest stage-Y → row 0). `axis_flip`/`swap_xy` generalize this.
- **Overlap inference.** `overlap_frac_x = 1 − |Δx_stage_adjacent_cols| /
  (tile_w · px)` (and Y). `overlap_frac ≤ zero_overlap_tol` (≈0.01) ⇒
  zero-overlap regime.
- **Registration.** Per adjacent pair: window + band-pass, phase correlation with
  `upsample_factor`, NCC of the aligned overlap as confidence; reject below
  `ncc_threshold`; propagate absolute positions along a maximum-confidence
  spanning tree seeded/bounded by the coordinate prior (`max_shift`).
- **Register once, apply to all** (spec §7): compute positions on `align_channel`
  and reuse for every C/Z/T to avoid channel misregistration.

### Numerical considerations
- A correlation peak *always* exists → never register zero-overlap data.
- Featureless/blank overlaps → low confidence → fall back to coordinates, flag QC.
- Preserve dtype: accumulate feather weights in float32, cast back to `uint16`.
- Large mosaics: incremental/memmapped OME-TIFF write; never hold the whole
  mosaic in RAM.

### Validation strategy
Synthetic ground truth: cut one large image into a grid at known overlap + coords
with jitter; assert recovered positions ≤ ~1–2 px. Zero-overlap variant: assert
regime selection and exact coordinate placement. Regime unit tests at 0/1/10 %.
Orientation fixtures (flip/swap). OME-TIFF round-trip (shape/dtype/channels/px).

## Chosen Approach

**Decision:** Regime-aware router. Zero-overlap → coordinate placement only.
Overlap → `m2stitch` for inferable regular grids, built-in phase-correlation
otherwise; `ashlar` gated (JDK). Feather blending; optional BaSiC/built-in
flat-field; pyramidal OME-TIFF output; QC report. Keep only the existing
metadata-orientation reads and Nikon sign/flip convention.

**Justification:**
- The zero-overlap branch is a first-class route, not an afterthought — the most
  common way naive stitchers silently produce garbage.
- m2stitch is a maintained MIST reimplementation that seeds from our coordinate
  prior and tolerates missing tiles; the built-in engine guarantees the feature
  works with only the core scientific stack.
- Our compositor + OME writer + register-once make ashlar non-essential, so the
  missing JDK does not block the feature.

**Trade-offs accepted:**
- ashlar unusable without a JDK (documented, gated). BaSiC unusable in this env
  (documented, gated; built-in fallback). Both are optional.
- pandas downgraded 3.0.3 → 2.3.3 to satisfy m2stitch (`pandas<3`); 2.3.3 is
  stable and the app is verified against it.

## References (BibTeX)

```bibtex
@inproceedings{kuglin1975phase,
  author = {Kuglin, C. D. and Hines, D. C.},
  title  = {The phase correlation image alignment method},
  booktitle = {Proc. IEEE Int. Conf. Cybernetics and Society}, year = {1975}, pages = {163--165}
}
@article{chalfoun2017mist,
  author = {Chalfoun, Joe and Majurski, Michael and Blattner, Tim and others},
  title  = {MIST: Accurate and Scalable Microscopy Image Stitching Tool with Stage Modeling and Error Minimization},
  journal = {Scientific Reports}, year = {2017}, volume = {7}, pages = {4988}, doi = {10.1038/s41598-017-04567-y}
}
@article{muhlich2022ashlar,
  author = {Muhlich, Jeremy L. and Chen, Yu-An and Yapp, Clarence and others},
  title  = {Stitching and registering highly multiplexed whole-slide images of tissues with ASHLAR},
  journal = {Bioinformatics}, year = {2022}, volume = {38}, number = {19}, pages = {4613--4621}, doi = {10.1093/bioinformatics/btac544}
}
@article{peng2017basic,
  author = {Peng, Tingying and Thorn, Kurt and Schroeder, Timm and others},
  title  = {A BaSiC tool for background and shading correction of optical microscopy images},
  journal = {Nature Communications}, year = {2017}, volume = {8}, pages = {14836}, doi = {10.1038/ncomms14836}
}
@article{guizar2008efficient,
  author = {Guizar-Sicairos, Manuel and Thurman, Samuel T. and Fienup, James R.},
  title  = {Efficient subpixel image registration algorithms},
  journal = {Optics Letters}, year = {2008}, volume = {33}, number = {2}, pages = {156--158}, doi = {10.1364/OL.33.000156}
}
```
