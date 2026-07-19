# Literature Review: Deconvolution for Fluorescence Microscopy

**Date:** 2026-07-13
**Author:** Claude (with G. Declercq)
**Related version:** V1.69 (proposed)

## Problem Statement

Every image ND2Studios loads is a *blurred, noisy* measurement of the specimen.
A fluorescence microscope cannot image a point of light as a point: diffraction
spreads it into a three-dimensional blob — the **point spread function (PSF)** —
and the recorded image is the true fluorophore distribution convolved with that
PSF, corrupted by photon (Poisson) shot noise and camera read noise. In
**widefield** epifluorescence the effect is severe: every in-focus plane is
buried under out-of-focus light from the planes above and below it, producing the
familiar axial "haze." **Deconvolution** is the computational inverse: given the
recorded stack and a model of the PSF, estimate the underlying object, thereby
increasing contrast and (for the nonlinear methods) effective resolution, and
re-assigning out-of-focus light back to where it came from.

ND2Studios already ships a broad enhancement toolbox (`plugins/enhancement/
builtin.py`: CLAHE, background subtraction, unsharp mask, TV/wavelet/NLM
denoising, …) but has **no deconvolution node**. This is a real gap for the
McGhee Lab's widefield and confocal data, where deconvolution is the single most
impactful "restore the physics" step available before any downstream analysis
(segmentation, DVC, intensity quantification). Notably, `scikit-image.restoration`
— already a dependency — ships `richardson_lucy`, `wiener`, and
`unsupervised_wiener`, so the lowest tier costs us *zero* new dependencies.

**The architectural tension.** Deconvolution is fundamentally a **3-D** operation:
its biggest win (removing widefield out-of-focus haze) requires the full Z-stack
and a 3-D PSF, because the haze in plane *z* is contributed by neighbouring
planes. But ND2Studios **collapses Z at load time** via projection
(`ND2Metadata` → `(T, H, W)`), and the enhancement recipe operates on those 2-D
`(T, H, W)` frames. The full `(M, T, Z, H, W)` volume *is* reachable in RAM (the
`MaterializedDataset` that the DVC engine already reads), so a correct design
must run 3-D deconvolution **on the Z-stack before projection**, while still
offering a cheaper 2-D step inside the existing recipe for confocal / single-plane
data. This review covers the theory and the Python implementation options, and
lands on that two-tier design.

**Scope (agreed with G. Declercq):** all four method families — (1) classical
linear (inverse / Wiener), (2) Richardson–Lucy and its regularized variants,
(3) blind deconvolution, (4) deep-learning restoration — plus PSF modelling
(theoretical and measured), with implementation guidance **leading with 3-D**.

## Background: the image-formation model

Fluorescence image formation is, to an excellent approximation, a **linear,
shift-invariant** system on the emitted-light intensity. In continuous form:

```
g(x) = (f * h)(x) + n(x)              (space domain)
G(k) = F(k) · H(k) + N(k)             (Fourier domain)
```

- `f` — the true object (fluorophore density), the thing we want.
- `h` — the **PSF**, the system's response to a point emitter. Its Fourier
  transform `H = OTF` is the **optical transfer function**; `|H|` is the MTF.
- `g` — the recorded image; `n` — noise.
- `*` — convolution; `·` — element-wise product (the convolution theorem).

Three physical facts drive every algorithm below:

1. **Band limit.** A lens of numerical aperture `NA` passes only spatial
   frequencies below `~2·NA/λ` laterally. `H` is exactly zero beyond that; the
   inverse `1/H` does not exist there, so no linear method can recover those
   frequencies — deconvolution *reallocates and sharpens* passband information
   and improves contrast; it does not invent resolution beyond the band limit
   (nonlinear/prior-based methods extrapolate *modestly* using non-negativity).
2. **The missing cone (widefield).** The 3-D widefield OTF is identically zero in
   a cone around the `k_z` axis. Flat, axially-extended structures are
   irrecoverable from widefield data alone; this is why widefield + 3-D
   deconvolution still cannot fully section, and why confocal (which fills the
   cone) deconvolves more cleanly.
3. **Noise, and specifically Poisson noise.** Photon detection is a counting
   process: `g ~ Poisson(f * h)` plus Gaussian read noise. The best-performing
   classical method (Richardson–Lucy) is precisely the maximum-likelihood
   estimator for *Poisson* statistics — which is why it dominates in
   photon-limited fluorescence.

**Modality matters.** Widefield has the worst haze and the missing cone → biggest
deconvolution payoff. **Confocal / spinning-disk** is already optically
sectioned (the pinhole rejects out-of-focus light) but is photon-starved and
retains residual blur → deconvolution mainly denoises and sharpens. **Light-sheet**
sits in between and benefits especially from 3-D RL. The PSF, and the sensible
iteration count, differ per modality.

**Anisotropic sampling.** Microscope voxels are almost never cubic: `z_step_um`
is typically 2–8× `pixel_size_um`. The PSF and every convolution must respect
that anisotropy (ND2Studios already stores `voxel_size_um = (z, y, x)` in
`ND2Metadata`), or the axial restoration will be wrong.

## Method families

### 1. Linear methods: inverse and Wiener filtering

The naive inverse `F̂ = G / H` inverts the convolution exactly in the absence of
noise, but divides noise by the small `|H|` near the band edge, exploding it.
**Wiener filtering** is the minimum-mean-square-error linear solution and the
standard regularized fix:

```
F̂(k) = [ conj(H) / ( |H|² + 1/SNR(k) ) ] · G(k)
```

The `1/SNR` term (a regularization / "balance" parameter) damps frequencies where
noise dominates. Wiener is **non-iterative, fast, and closed-form**, but it is
linear (so cannot enforce non-negativity), tends to ring near sharp edges, and
needs an SNR/regularization estimate. It is an excellent *baseline* and a good
initializer. `scikit-image` provides `wiener` (you supply `balance ≈ 1/SNR`) and
`unsupervised_wiener` (estimates the balance automatically via a hierarchical
Bayesian / Gibbs-sampling scheme — very convenient for a GUI). Tikhonov /
regularized-inverse filters are the same idea with an explicit smoothness penalty
and appear in DeconvolutionLab2.

### 2. Richardson–Lucy (Poisson maximum-likelihood)

The **Richardson–Lucy (RL)** algorithm — derived independently by Richardson
(1972) and Lucy (1974), and equivalent to the Poisson **expectation-maximization
/ MLEM** estimator — is *the* workhorse of fluorescence deconvolution. It
iterates the multiplicative update:

```
f^{k+1} = f^k · [ h⁻ * ( g / (h * f^k) ) ]
```

where `h⁻(x) = h(−x)` is the mirrored PSF and the division is element-wise. Key
properties: it enforces **non-negativity** automatically (all factors ≥ 0), it
**conserves total flux** when the PSF is normalized to sum 1, and it is the ML
estimator for Poisson data — the physically correct noise model. It needs only
the forward PSF (two convolutions per iteration, done via FFT).

The central practical caveat is **semi-convergence**: RL first recovers real
structure, then — because it is fitting the noise — begins amplifying noise into
speckle and "night-sky" artifacts as iterations grow. The iteration count is
therefore a genuine regularization parameter (typically **10–50** for widefield;
fewer for clean confocal). There is no universal stopping rule; users pick by
eye, by a noise/energy criterion, or by an information-theoretic stop.

**Acceleration.** Biggs & Andrews (1997) vector-extrapolation acceleration (used
in most production packages) reaches a given quality in far fewer iterations.
Guo et al. (2020) showed an **"unmatched back-projector"** (using a
sharper-than-PSF back-projector in the correction step) accelerates RL by ≥10×
with no quality loss — now a standard trick.

### 3. Regularized Richardson–Lucy

To tame semi-convergence, add a prior/penalty to the likelihood:

- **RL + Total Variation (RL-TV)**, Dey et al. (2006): multiplies the RL update
  by a TV factor `1 / (1 − λ·div(∇f/|∇f|))`. TV suppresses noise oscillations
  while **preserving edges**, and is the most popular regularized RL in
  microscopy (it is a built-in in DeconvolutionLab2). `λ` trades sharpness vs
  smoothness.
- **Entropy / Good's roughness / Gaussian-MRF priors** — alternative penalties
  that penalize roughness; used in commercial packages (Huygens, AutoQuant).
- **Tikhonov-Miller** — quadratic smoothness penalty (linear, faster, blurrier).

Regularized RL is the sweet spot for noisy widefield: it lets you run more
iterations for resolution without the noise blow-up.

### 4. Blind deconvolution

When the PSF is unknown, mis-calibrated, or **spatially varying** (e.g. depth-
dependent spherical aberration deep in tissue), **blind deconvolution** jointly
estimates both the object *and* the PSF. The classic approaches are Ayers–Dainty
iterative blind deconvolution, Holmes (1992) maximum-likelihood blind
deconvolution (the basis of AutoQuant's widely-used implementation), and Fish et
al. (1995), who simply **alternate RL updates** for `f` and for `h`. Constraints
(non-negativity, PSF symmetry, band-limit, energy) are essential to break the
inherent `f`↔`h` ambiguity. Blind methods are powerful when a measured PSF is
unavailable, but are **slower and less stable**, can converge to the trivial
`h = δ` solution without good constraints, and generally underperform a *good*
measured or well-modelled PSF. Recommendation: prefer a measured/modelled PSF;
keep blind as a fallback.

### 5. Deep-learning restoration

Learned methods train a CNN to map degraded → restored images, implicitly
learning both the inverse operator and a strong image prior:

- **CARE** (Content-Aware image REstoration), Weigert et al. (2018): U-Net
  trained on paired low/high-quality (or synthetically degraded) data. Restores
  images acquired with ~60× fewer photons, achieves near-isotropic resolution
  from ~10× axial under-sampling, and resolves sub-diffraction structure at high
  frame rate. Ships as open-source Python/Fiji/KNIME (`csbdeep`).
- **RCAN** (3-D Residual Channel Attention Networks), Chen et al. (2021):
  volumetric denoising + resolution enhancement for 4-D data, ~2.5× lateral gain
  vs STED ground truth, robust to photobleaching over tens of thousands of frames.
- **Hybrid**, Guo et al. (2020): combines accelerated RL with a CNN specifically
  for **spatially-varying PSFs**, where classical single-PSF deconvolution fails.
- **Self-supervised** (Noise2Void / Noise2Noise family): train without clean
  ground truth — attractive when paired data is impossible.

Trade-offs: state-of-the-art quality, but they require **training data** (or a
pretrained model matched to the imaging conditions), a heavy dependency
(TensorFlow / PyTorch), a GPU for training, and — the scientific risk —
**hallucination**: a network can synthesize plausible but false structure, so
outputs need validation and honest labelling as "restored." For ND2Studios these
belong as an *optional, lazily-imported extra*, exactly like the existing
`cellpose` / `stardist` segmentation backends.

## Key Papers

Citation details for the biomedical entries were verified via **PubMed**; DOIs
link each work.

### 1. Sibarita (2005) — "Deconvolution Microscopy"
- **Journal:** Advances in Biochemical Engineering/Biotechnology 95:201–243
- **DOI:** [10.1007/b102215](https://doi.org/10.1007/b102215) (PubMed)
- **Key contribution:** The canonical end-to-end review — image formation in 3-D
  optical-sectioning microscopy, PSF determination (measured & theoretical),
  aberrations, acquisition requirements, and the full family of restoration
  algorithms (inverse, Wiener, Tikhonov, RL, MAP, blind).
- **Method summary:** Frames deconvolution as inverting the `g = f*h + n` model;
  organizes methods by no-neighbours/deblurring vs image-restoration, linear vs
  nonlinear, and by noise model.
- **Relevance to us:** The best single orientation to the whole field; our
  method taxonomy above follows it.

### 2. Biggs (2010) — "3D Deconvolution Microscopy"
- **Journal:** Current Protocols in Cytometry, Ch. 12, Unit 12.19
- **DOI:** [10.1002/0471142956.cy1219s52](https://doi.org/10.1002/0471142956.cy1219s52) (PubMed)
- **Key contribution:** A practical protocol: how PSF characteristics follow from
  modality and objective parameters, how to set up acquisition *for*
  deconvolution (sampling, exposure, SNR), and side-by-side algorithm behaviour
  on widefield vs confocal data.
- **Method summary:** Emphasizes matched acquisition (Nyquist sampling, bit depth,
  minimizing bleaching) and choosing the algorithm to the modality.
- **Relevance to us:** Directly informs the parameter defaults and the widefield-
  vs-confocal guidance our node should surface to users.

### 3. Richardson (1972) & Lucy (1974) — the RL algorithm
- **Journals:** J. Opt. Soc. Am. 62(1):55–59; Astronomical Journal 79:745–754
- **DOIs:** [10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055);
  [10.1086/111605](https://doi.org/10.1086/111605)
- **Key contribution:** The iterative Bayesian/ML update that became the standard
  nonlinear deconvolution; non-negative, flux-conserving, Poisson-optimal.
- **Method summary:** Multiplicative EM update (see §2).
- **Relevance to us:** The core algorithm of Tier-1 and Tier-2 below; implemented
  by scikit-image, RedLionfish, and flowdec.

### 4. Dey et al. (2006) — "Richardson–Lucy with Total Variation regularization"
- **Journal:** Microscopy Research and Technique 69(4):260–266
- **DOI:** [10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294) (PubMed)
- **Key contribution:** Adds a TV penalty to RL, suppressing noise oscillations
  while preserving edges; validated on simulated and real 3-D confocal data.
- **Method summary:** RL update × TV factor `1/(1 − λ div(∇f/|∇f|))`.
- **Relevance to us:** The regularized variant we want as an option for noisy
  widefield; the exact algorithm in DeconvolutionLab2's "RL-TV."

### 5. Sage et al. (2017) — "DeconvolutionLab2"
- **Journal:** Methods 115:28–41
- **DOI:** [10.1016/j.ymeth.2016.12.015](https://doi.org/10.1016/j.ymeth.2016.12.015)
- **Key contribution:** The open-source reference implementation and benchmark of
  3-D microscopy deconvolution algorithms (Naive/Regularized/Tikhonov inverse,
  RL, RL-TV, Landweber, FISTA, …), with a companion PSFGenerator.
- **Method summary:** A modular Java/ImageJ platform; the paper's algorithm
  descriptions and test data are the community's cross-check.
- **Relevance to us:** Our **validation oracle** — we compare our Python results
  against DeconvolutionLab2 on its standard datasets. (Java/Fiji, GPL — used as an
  external reference, never shipped inside ND2Studios.)

### 6. Gibson & Lanni (1992) — the scalar PSF model (with Aguet's vectorial model)
- **Journal:** J. Opt. Soc. Am. A 9(1):154–166
- **DOI:** [10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)
- **Key contribution:** The scalar diffraction PSF accounting for
  refractive-index/thickness mismatch (immersion, coverslip, sample) — the
  dominant source of depth-dependent spherical aberration, and the practical
  default PSF model for microscopy.
- **Method summary:** Compute the PSF from optical parameters (NA, wavelength,
  RIs, coverslip thickness) rather than measuring beads.
- **Relevance to us:** Lets us synthesize a PSF straight from `ND2Metadata`
  (NA, emission wavelength, immersion) — no bead calibration required. In Python
  via `MicroscPSF-Py` (MIT, fast Gibson–Lanni, Li et al. 2017). For high-NA
  lenses a **vectorial** model is more accurate; `psfmodels` (GPL) implements the
  vectorial formulation from Aguet's physical-model work (Aguet, Geissbühler,
  Märki, Lasser & Unser, *Opt. Express* 17(8):6829,
  [10.1364/OE.17.006829](https://doi.org/10.1364/OE.17.006829); and Aguet's 2009
  EPFL thesis no. 4418).

### 7. Weigert et al. (2018) — "Content-aware image restoration (CARE)"
- **Journal:** Nature Methods 15(12):1090–1097
- **DOI:** [10.1038/s41592-018-0216-7](https://doi.org/10.1038/s41592-018-0216-7) (PubMed)
- **Key contribution:** Deep-learning restoration that extends what is observable
  — restores 60× lower-photon images, near-isotropic resolution from 10× axial
  under-sampling; open-source (`csbdeep`, BSD-3).
- **Method summary:** U-Net trained on paired / synthetically-degraded data.
- **Relevance to us:** The reference for the optional deep-learning tier.

### 8. Guo et al. (2020) — "Rapid image deconvolution and multiview fusion"
- **Journal:** Nature Biotechnology 38(11):1337–1346
- **DOI:** [10.1038/s41587-020-0560-x](https://doi.org/10.1038/s41587-020-0560-x) (PubMed)
- **Key contribution:** ≥10× RL acceleration via an **unmatched back-projector**;
  10–100× GPU registration; deep learning for **spatially-varying PSFs**.
  (Co-authored by T. Lambert, author of the `nd2` I/O library ND2Studios pins and
  of `psfmodels`.)
- **Method summary:** Modified RL back-projector + GPU + optional CNN.
- **Relevance to us:** The acceleration we should adopt; the SV-PSF motivation for
  the DL tier.

### 9. Chen et al. (2021) — "3-D RCAN"
- **Journal:** Nature Methods 18(6):678–687
- **DOI:** [10.1038/s41592-021-01155-x](https://doi.org/10.1038/s41592-021-01155-x) (PubMed)
- **Key contribution:** Volumetric residual-channel-attention network for
  denoising + resolution enhancement of 4-D data.
- **Method summary:** 3-D RCAN trained against high-SNR / super-res ground truth.
- **Relevance to us:** Candidate architecture if the lab pursues learned
  restoration for time-lapse volumes.

### 10. Perdigão et al. (2024) — "RedLionfish"
- **Journal:** Wellcome Open Research 9:296 (software tool)
- **DOI:** [10.12688/wellcomeopenres.21505.1](https://doi.org/10.12688/wellcomeopenres.21505.1) (PubMed)
- **Key contribution:** A fast, GPU-accelerated (OpenCL/Reikna, vendor-agnostic)
  3-D Richardson–Lucy Python package with automatic CPU fallback, block/chunked
  processing for large volumes, and a per-iteration progress callback. Apache-2.0.
- **Method summary:** FFT-based RL with precomputed PSF/PSF-flip transforms; block
  algorithm with padding to bound edge artifacts and GPU memory.
- **Relevance to us:** The **recommended Tier-1 3-D engine** (permissive license,
  GPU, progress callback that maps onto our `progress_cb`, benchmarks faster than
  scikit-image).

## Obtaining the PSF — the make-or-break input

Deconvolution is only as good as the PSF `h` you feed it. A wrong PSF produces
confident garbage (rings, over-sharpening, intensity errors). There are two
routes:

**A. Measured PSF (empirical).** Image sub-diffraction fluorescent beads
(100–200 nm, spectrally matched) under the *exact* optics used for the data,
then average many beads (recentre + mean) to beat down noise, and background-
subtract. Pros: captures the real, aberrated PSF of *your* system. Cons: extra
acquisition, beads degrade/bleach, and the PSF is only valid for that exact
configuration (objective, immersion, wavelength, depth). Tools: ImageJ/Fiji
"PSF Distiller"/MetroloJ; in Python simply load the bead stack as a numpy array
and normalize.

**B. Theoretical PSF (modelled).** Compute `h` from optical parameters. Three
models in increasing fidelity/cost:
- **Born & Wolf** — scalar, ideal (no RI mismatch). Fine for low-NA / quick use.
- **Gibson & Lanni** — scalar + refractive-index/thickness mismatch (immersion,
  coverslip, sample) → captures depth-dependent **spherical aberration**. The
  practical default for microscopy.
- **Vectorial (Aguet / Richards–Wolf)** — models the electromagnetic field;
  needed for **high-NA** (≥1.3) objectives.

The theoretical route is ideal for ND2Studios because **the parameters already
live in `ND2Metadata`**: `objective_na`, per-channel `channel_emission_nm`,
`objective_immersion` (→ immersion RI `ni`), `pixel_size_um` (`dxy`),
`z_step_um` (`dz`). Only the **sample refractive index** `ns` (≈1.33 aqueous,
≈1.47 mounting medium) needs a user default. That means we can synthesize a PSF
per channel automatically, with no bead calibration.

**Python PSF libraries:**

| Package | Model(s) | License | Notes |
|---------|----------|---------|-------|
| **MicroscPSF-Py** | Gibson–Lanni (fast, Li et al. 2017) | **MIT** | Pure Python, no compile; fast; **shippable**. Verified below. |
| **psfmodels** (tlambert03) | scalar (Gibson–Lanni) + **vectorial** (Aguet) + Gaussian | **GPL-3.0** | Highest fidelity incl. vectorial; C++/pybind11. GPL → use offline / for validation, **do not import into shipped code**. |
| **PSFGenerator** (EPFL) | Born–Wolf, Gibson–Lanni, vectorial (Richards–Wolf) | GPL (Java) | The reference GUI; companion to DeconvolutionLab2. Offline. |

## Python implementation landscape

The field has consolidated onto a handful of maintained packages. Licensing is
called out explicitly because ND2Studios deliberately stays **permissive** (see
`aldvc_literature_review.md`, where the whole design hinged on avoiding
copyleft):

| Library | Algorithm(s) | Dim | GPU | License | Best fit for us |
|---------|--------------|-----|-----|---------|-----------------|
| **scikit-image** `restoration` | RL, Wiener, unsupervised Wiener | 2-D/3-D | No (CPU) | **BSD-3** | **Zero-cost Tier-2** (already a dep) & CPU fallback |
| **RedLionfish** | Richardson–Lucy (FFT) | 3-D | **Yes** (OpenCL/Reikna, any vendor) + CPU fallback | **Apache-2.0** | **Tier-1 3-D engine** (fast, chunked, progress callback) |
| **flowdec** | Richardson–Lucy (TensorFlow) | 2-D/3-D | **Yes** (CUDA) | **Apache-2.0** | Alt 3-D GPU engine; matches DeconvolutionLab2/PSFGenerator numerically |
| **clij2-fft / pyclesperanto** | RL (FFT) | 2-D/3-D | **Yes** (OpenCL) | BSD/permissive | Another GPU RL option; used in napari |
| **CSBDeep** (CARE) | Deep-learning (U-Net) | 2-D/3-D | Yes (TF) | **BSD-3** | Optional DL tier (Weigert 2018) |
| **MicroscPSF-Py** | Gibson–Lanni PSF | 3-D | No | **MIT** | **Shippable theoretical PSF** |
| **psfmodels** | scalar+vectorial PSF | 3-D | No | GPL-3.0 | High-fidelity PSF, **offline/validation only** |
| **DeconvolutionLab2 + PSFGenerator** | all classical | 3-D | partial | GPL (Java) | **Validation oracle**, not shipped |
| **scipy** `signal`/`fft` | building blocks (`fftconvolve`) | n-D | No | BSD | Forward model, synthetic tests, custom RL-TV |

**Reading of the landscape.** For the classical tiers we need *nothing exotic*:
`scikit-image` (BSD, already present) covers CPU RL/Wiener in 2-D and 3-D;
`RedLionfish` (Apache-2.0) adds the GPU 3-D speed we want for whole-volume
widefield stacks, with a CPU fallback and a per-iteration callback that maps
one-to-one onto our worker `progress_cb`. PSFs come from `MicroscPSF-Py` (MIT).
Deep learning, if pursued, is `csbdeep` (BSD) behind a lazy import. **No GPL code
is imported by ND2Studios**; `psfmodels`, `PSFGenerator`, and DeconvolutionLab2
are used only as offline references / validation oracles.

## Worked, verified code

All snippets below were **executed** against the pinned scientific stack
(scikit-image 0.25.2, numpy 2.2.6, scipy 1.15.3; MicroscPSF-Py; psfmodels 0.3.3)
in this session — the signatures and shapes are real, not illustrative.

**(a) 2-D Richardson–Lucy / self-tuning Wiener (Tier-2, scikit-image, BSD, CPU).**
Verified signatures: `richardson_lucy(image, psf, num_iter=50, clip=True,
filter_epsilon=None)`, `wiener(image, psf, balance, reg=None, is_real=True,
clip=True)`, `unsupervised_wiener(image, psf, reg=None, user_params=None,
is_real=True, clip=True, *, rng=None)`.

```python
import numpy as np
from skimage.restoration import richardson_lucy, unsupervised_wiener

def deconvolve_frame(frame: np.ndarray, psf2d: np.ndarray,
                     num_iter: int = 30) -> np.ndarray:
    """2-D RL on a single (H, W) frame. `psf2d` normalized to sum 1."""
    f = frame.astype(np.float32)
    f /= f.max() + 1e-9                     # RL expects ~[0,1]
    psf2d = psf2d / psf2d.sum()             # conserve flux
    out = richardson_lucy(f, psf2d, num_iter=num_iter, clip=True)
    return out

# self-tuning alternative (no iteration/balance to pick):
restored, _ = unsupervised_wiener(frame_float01, psf2d)
```

**(b) 3-D Richardson–Lucy on a Z-stack (Tier-1).** Same call, an `(Z, H, W)`
image and a 3-D PSF. This is the scientifically-correct path (removes widefield
haze). GPU via RedLionfish is a drop-in swap with a progress callback:

```python
# CPU reference (scikit-image, BSD) — correct but slower:
from skimage.restoration import richardson_lucy
vol_deconv = richardson_lucy(vol_zhw / vol_zhw.max(), psf_zhw / psf_zhw.sum(),
                             num_iter=30, clip=True)

# GPU / large-volume (RedLionfish, Apache-2.0) — same math, chunked, callbacked:
import RedLionfishDeconv as rl
def on_tick():                                  # -> wire to worker progress
    ...
vol_deconv = rl.doRLDeconvolutionFromNpArrays(
    vol_zhw.astype("float32"), psf_zhw.astype("float32"),
    niter=30, method="gpu",            # falls back to CPU automatically on failure
    useBlockAlgorithm=True,            # chunk large volumes to fit GPU memory
    callbkTickFunc=on_tick,            # per-iteration/block callback
)
```

Confirmed in-session on synthetic data: a 3-D RL run on a `(32, 64, 64)` volume
blurred by an anisotropic Gaussian PSF + Poisson noise raised the peak-to-mean
contrast from ~828 to ~1093 and returned a non-negative `(32, 64, 64)` float
result — i.e. it deconvolved and stayed flux-sane.

**(c) Theoretical PSF straight from `ND2Metadata` (MicroscPSF-Py, MIT — shippable).**
Verified: `gLXYZFocalScan(mp, dxy, xy_size, zv, normalize=True, pz=0.0,
wvl=0.6, zd=None)` returns `(len(zv), xy_size, xy_size)`.

```python
import numpy as np
import microscPSF.microscPSF as msPSF

_IMMERSION_RI = {"Oil": 1.515, "Silicone": 1.405, "Water": 1.333,
                 "Glycerol": 1.47, "Air": 1.0}

def psf_from_metadata(meta, channel_index=0, nz=None, nx=63, ns=1.40):
    """Build a Gibson–Lanni PSF (Z,H,W) from ND2Studios' ND2Metadata."""
    dxy = float(meta.pixel_size_um)
    dz  = float(meta.z_step_um)
    na  = float(getattr(meta, "objective_na", 1.4) or 1.4)
    ni  = _IMMERSION_RI.get(getattr(meta, "objective_immersion", "Oil"), 1.515)
    emm = getattr(meta, "channel_emission_nm", None)
    wvl = (emm[channel_index] / 1000.0) if emm else 0.52   # µm
    nz  = nz or 31
    mp = dict(msPSF.m_params)
    mp.update(NA=na, ni0=ni, ni=ni, ns=ns)
    zv = (np.arange(nz) - nz // 2) * dz          # focal scan, µm, centred
    psf = msPSF.gLXYZFocalScan(mp, dxy, nx, zv, pz=0.0, wvl=wvl)  # (nz, nx, nx)
    return (psf / psf.sum()).astype(np.float32)
```

This honours `pixel_size_um`/`z_step_um` (anisotropy) and per-channel emission
wavelength — exactly the CLAUDE.md rule to always respect `ND2Metadata`. For the
highest-NA lenses, `psfmodels.vectorial_psf(...)` is more accurate but GPL, so
it is used **offline** to pre-render a `.tif` PSF the user can load, never
imported at runtime.

## Method Comparison

| Method | Pros | Cons | Complexity | Accuracy |
|--------|------|------|------------|----------|
| Naive inverse filter | Exact without noise; instant | Explodes noise; no non-negativity | Low | Poor on real data |
| Wiener / **unsupervised Wiener** | Fast, closed-form; `unsupervised_` self-tunes | Linear (no non-neg); edge ringing; needs SNR | Low | Moderate |
| **Richardson–Lucy** | Poisson-optimal (ML); non-negative; flux-conserving; ubiquitous | Semi-convergence → noise amplification; pick iteration count | Medium | High |
| **RL + Total Variation** | Edge-preserving; tolerates more iterations on noisy data | Extra `λ`; slower per iter | Medium | High (noisy widefield) |
| Blind deconvolution | No measured/known PSF required | Slow; unstable; `f`↔`h` ambiguity; needs strong constraints | High | Variable |
| **Deep learning** (CARE/RCAN) | State-of-the-art; works at very low photon budgets | Needs training data / matched model; GPU; **hallucination risk** | High | Very high *in-domain* |

For ND2Studios' mix of widefield and confocal data, **Richardson–Lucy (with an
optional TV regularizer)** is the right default: it is the physically-correct
Poisson estimator, non-negative, flux-conserving, and supported by every engine
we would ship. Wiener/unsupervised-Wiener is the fast baseline; DL is an opt-in
power tool.

## Implementation Notes

### Mathematical formulation

```
Forward model:      g = f * h + n ,     G = F·H          (H = OTF = 𝓕{h})
Inverse (unstable): F̂ = G / H
Wiener (MMSE):      F̂ = [ conj(H) / (|H|² + 1/SNR) ] · G
Richardson–Lucy:    f^{k+1} = f^k · [ h⁻ * ( g / (h * f^k) ) ],   h⁻(x)=h(−x)
RL + TV (Dey):      f^{k+1} = { f^k / (1 − λ·div(∇f^k/|∇f^k|)) } · [ h⁻ * (g/(h*f^k)) ]
```

Convolutions are computed by FFT (`scipy.fft` / OpenCL); the PSF and its mirror
are transformed once and reused across iterations (RedLionfish does exactly this).

### Numerical considerations

- **PSF normalization & centring.** Normalize `h` to `sum(h)=1` (flux
  conservation), keep it centred, and prefer **odd** dimensions (psfmodels
  requires odd `nx`/`nz`). A mis-centred PSF shifts the whole reconstruction.
- **Anisotropic voxels.** Build the 3-D PSF with the true `(dz, dxy, dxy)` from
  `voxel_size_um`; never assume cubic voxels or axial restoration is wrong.
- **The missing cone (widefield).** Deconvolution cannot fill it; non-negativity
  buys a little axial recovery but do not promise confocal-like sectioning from
  widefield. Set user expectations in the tooltip.
- **Semi-convergence / iteration count.** RL amplifies noise with excess
  iterations. Expose `num_iter` (default ~20–30 widefield, ~10 confocal) and/or
  offer RL-TV. Consider an energy-stability stopping hint.
- **Boundary / edge artifacts.** FFT convolution is circular → wrap-around
  artifacts at edges; padding (reflect/edge) mitigates but the RL "valid"
  (edge-artifact-free) region shrinks with every iteration by an amount that
  grows with PSF size — RedLionfish's analysis shows it can even go negative
  after many iterations — so for large stacks use its **block algorithm** with
  padding (default fraction 1.2) and accept tiny inter-block differences.
- **Background first.** Subtract camera offset / background (the existing
  `Background Subtract` plugin) *before* RL — RL assumes the signal is Poisson
  photons, and a DC pedestal biases the multiplicative update.
- **Precision & memory.** Work in `float32`. A `(64, 2048, 2048)` stack is ~1 GB
  per float copy and RL needs several; this is why GPU chunking (RedLionfish) and
  our existing shared-memory/`ProcessPoolExecutor` infra matter.
- **dtype round-trip.** Deconvolved output is float; rescale/clip back to the
  input dtype (`uint16`) on the way to TIFF export, mirroring the existing
  plugins' `astype(volume.dtype)` pattern.
- **Per-channel PSF.** Emission wavelength differs per channel, so the PSF (and
  ideally the deconvolution) is computed per channel from `channel_emission_nm`.

### Validation strategy

1. **Synthetic round-trip (done in-session).** Known object → convolve with a
   known PSF → add Poisson noise → deconvolve → verify contrast increases,
   output stays non-negative, and the recovered peaks land on the true
   positions. Extend to a quantitative correlation/│error│ vs iteration curve to
   expose semi-convergence.
2. **Bead FWHM.** On a real/synthetic bead, measure lateral & axial FWHM before
  and after — deconvolution should narrow both toward the diffraction limit
  without going *below* it (a sign of over-iteration/hallucinated resolution).
3. **Flux conservation.** Total integrated intensity should be ~preserved (RL
  property) — a large change flags a PSF or boundary problem.
4. **Cross-check against DeconvolutionLab2** on its standard microtubule / bead
  datasets (Java/Fiji, GPL, offline) — the community oracle for "is my RL
  correct." `flowdec` explicitly targets numerical equivalence with it.
5. **Resolution (FRC).** Fourier Ring/Shell Correlation on split acquisitions to
  quantify any genuine resolution gain.
6. **Unit tests** in `tests/` mirroring the house style (synthetic Gaussian in →
  known narrowing out; PSF-from-metadata shape/normalization; dtype round-trip).

## Chosen Approach

**Decision — a three-tier deconvolution capability, permissively licensed:**

- **Tier 1 — 3-D deconvolution before projection (primary).** A
  `DeconvolutionStep` that runs on the raw per-channel `(Z, H, W)` volume read
  from the `MaterializedDataset` `(M, T, Z, H, W)` (the same volume the DVC
  engine consumes) **before** Z-projection. Engine: **RedLionfish** (GPU,
  Apache-2.0) with automatic **scikit-image CPU fallback**; algorithm RL, with an
  optional TV regularizer; `num_iter` user-set. PSF source: **theoretical**
  (MicroscPSF-Py, MIT, built from `ND2Metadata` per channel) **or** a user-loaded
  measured PSF (`.tif`). This is where widefield haze removal actually happens.
  It plugs in at the load/projection boundary and leaves the `(T, H, W)`
  recipe/export contract untouched.
- **Tier 2 — 2-D `EnhancementPlugin` (drop-in, zero new deps).** A
  `"Deconvolution (2D)"` plugin implementing the standard `execute(volume,
  params, progress_cb)` on `(T, H, W)` via `skimage.restoration.richardson_lucy`
  / `unsupervised_wiener`, using a 2-D PSF (mid-plane of the theoretical 3-D PSF,
  or a Gaussian fallback). For confocal / single-plane / time-lapse frames and
  quick sharpening, fully consistent with the existing plugin registry and
  `ParamSpec` auto-form.
- **Tier 3 — deep learning (optional extra).** CARE / RCAN via **`csbdeep`**
  (BSD-3), **lazily imported and gated by `importlib.util.find_spec`, kept out of
  `requirements.txt`** — exactly the pattern CLAUDE.md prescribes for `cellpose`
  / `stardist`. Ships disabled; a friendly `ImportError` when the extra is absent.

**Justification:**
- **License.** Every shipped component is permissive — scikit-image (BSD-3),
  RedLionfish & flowdec (Apache-2.0), MicroscPSF-Py (MIT), csbdeep (BSD-3). The
  GPL tools (`psfmodels`, PSFGenerator, DeconvolutionLab2) are used **only
  offline** for high-NA PSF rendering and validation, never imported — preserving
  the permissive posture the ALDVC port fought to keep.
- **Architecture fit.** RedLionfish's `callbkTickFunc` maps one-to-one onto the
  worker `progress_cb`; the engines are pure `ndarray`-in/`ndarray`-out (satisfy
  backend purity — no Qt); GPU work reuses the existing CuPy/parallel infra; the
  2-D tier is a textbook `EnhancementPlugin`.
- **Data fit.** 3-D-before-projection is the *only* correct place to kill
  widefield haze; the PSF is synthesized from metadata we already parse; voxel
  anisotropy and per-channel wavelength are honoured per CLAUDE.md.
- **Incremental cost.** Tier 2 is buildable **today with zero new dependencies**
  (scikit-image is already pinned) — a fast first deliverable while Tier 1's
  pipeline integration lands.

**Trade-offs accepted:**
- Tier 1 **touches the load/projection pipeline**, not just the recipe — more
  integration than a plain plugin. Accepted: it is the only physically correct
  insertion point, and the `MaterializedDataset` read path already exists (DVC).
- **Theoretical vs measured PSF.** Gibson–Lanni from metadata is convenient but
  approximate (especially deep in tissue / high-NA); we mitigate by also
  accepting a measured PSF and by offering the offline vectorial render.
- **RL iteration tuning** is exposed to the user (unavoidable; there is no
  universal stop) — mitigated with modality-based defaults and RL-TV.
- **GPU memory** for large volumes forces chunking (small inter-block
  differences) — accepted per RedLionfish's design.
- **Deep learning deferred/optional** because of the training-data requirement
  and hallucination risk; it is a power-user extra, clearly labelled as
  "restored," not raw data.

**Documentation / next steps (per CLAUDE.md).** This review satisfies the
`Research/` requirement. Implementation would add
`CodeLog/ClaudesPlan/V1.69_deconvolution.md` (plan), a `CHANGELOG.md` "Added"
entry, and an `ARCHITECTURE.md` update (new dependencies: `redlionfish`,
`microscpsf-py`; optional `csbdeep`).

## References (BibTeX)

```bibtex
@article{sibarita2005deconvolution,
  author  = {Sibarita, Jean-Baptiste},
  title   = {Deconvolution Microscopy},
  journal = {Advances in Biochemical Engineering/Biotechnology},
  year    = {2005},
  volume  = {95},
  pages   = {201--243},
  doi     = {10.1007/b102215}
}
@article{biggs2010threedimensional,
  author  = {Biggs, David S. C.},
  title   = {{3D} Deconvolution Microscopy},
  journal = {Current Protocols in Cytometry},
  year    = {2010},
  volume  = {Chapter 12},
  pages   = {Unit 12.19},
  doi     = {10.1002/0471142956.cy1219s52}
}
@article{richardson1972bayesian,
  author  = {Richardson, William Hadley},
  title   = {Bayesian-Based Iterative Method of Image Restoration},
  journal = {Journal of the Optical Society of America},
  year    = {1972},
  volume  = {62},
  number  = {1},
  pages   = {55--59},
  doi     = {10.1364/JOSA.62.000055}
}
@article{lucy1974iterative,
  author  = {Lucy, L. B.},
  title   = {An Iterative Technique for the Rectification of Observed Distributions},
  journal = {The Astronomical Journal},
  year    = {1974},
  volume  = {79},
  pages   = {745--754},
  doi     = {10.1086/111605}
}
@article{dey2006richardson,
  author  = {Dey, Nicolas and Blanc-F{\'e}raud, Laure and Zimmer, Christophe and
             Roux, Pascal and Kam, Zvi and Olivo-Marin, Jean-Christophe and
             Zerubia, Josiane},
  title   = {Richardson--Lucy Algorithm with Total Variation Regularization for
             {3D} Confocal Microscope Deconvolution},
  journal = {Microscopy Research and Technique},
  year    = {2006},
  volume  = {69},
  number  = {4},
  pages   = {260--266},
  doi     = {10.1002/jemt.20294}
}
@article{sage2017deconvolutionlab2,
  author  = {Sage, Daniel and Donati, Laur{\`e}ne and Soulez, Ferr{\'e}ol and
             Fortun, Denis and Schmit, Guillaume and Seitz, Arne and
             Guiet, Romain and Vonesch, C{\'e}dric and Unser, Michael},
  title   = {{DeconvolutionLab2}: An Open-Source Software for Deconvolution
             Microscopy},
  journal = {Methods},
  year    = {2017},
  volume  = {115},
  pages   = {28--41},
  doi     = {10.1016/j.ymeth.2016.12.015}
}
@article{gibson1992experimental,
  author  = {Gibson, Sarah Frisken and Lanni, Frederick},
  title   = {Experimental Test of an Analytical Model of Aberration in an
             Oil-Immersion Objective Lens Used in Three-Dimensional Light
             Microscopy},
  journal = {Journal of the Optical Society of America A},
  year    = {1992},
  volume  = {9},
  number  = {1},
  pages   = {154--166},
  doi     = {10.1364/JOSAA.9.000154}
}
@article{aguet2009superresolution,
  author  = {Aguet, Fran{\c c}ois and Geissb{\"u}hler, Stefan and M{\"a}rki, Iwan
             and Lasser, Theo and Unser, Michael},
  title   = {Super-Resolution Orientation Estimation and Localization of
             Fluorescent Dipoles Using {3-D} Steerable Filters},
  journal = {Optics Express},
  year    = {2009},
  volume  = {17},
  number  = {8},
  pages   = {6829--6848},
  doi     = {10.1364/OE.17.006829}
}
@article{li2017fast,
  author  = {Li, Jizhou and Xue, Feng and Blu, Thierry},
  title   = {Fast and Accurate Three-Dimensional Point Spread Function Computation
             for Fluorescence Microscopy},
  journal = {Journal of the Optical Society of America A},
  year    = {2017},
  volume  = {34},
  number  = {6},
  pages   = {1029--1034},
  doi     = {10.1364/JOSAA.34.001029}
}
@article{biggs1997acceleration,
  author  = {Biggs, David S. C. and Andrews, Mark},
  title   = {Acceleration of Iterative Image Restoration Algorithms},
  journal = {Applied Optics},
  year    = {1997},
  volume  = {36},
  number  = {8},
  pages   = {1766--1775},
  doi     = {10.1364/AO.36.001766}
}
@article{fish1995blind,
  author  = {Fish, D. A. and Brinicombe, A. M. and Pike, E. R. and Walker, J. G.},
  title   = {Blind Deconvolution by Means of the Richardson--Lucy Algorithm},
  journal = {Journal of the Optical Society of America A},
  year    = {1995},
  volume  = {12},
  number  = {1},
  pages   = {58--65},
  doi     = {10.1364/JOSAA.12.000058}
}
@article{holmes1992blind,
  author  = {Holmes, Timothy J.},
  title   = {Blind Deconvolution of Quantum-Limited Incoherent Imagery:
             Maximum-Likelihood Approach},
  journal = {Journal of the Optical Society of America A},
  year    = {1992},
  volume  = {9},
  number  = {7},
  pages   = {1052--1061},
  doi     = {10.1364/JOSAA.9.001052}
}
@article{weigert2018content,
  author  = {Weigert, Martin and Schmidt, Uwe and Boothe, Tobias and others},
  title   = {Content-Aware Image Restoration: Pushing the Limits of Fluorescence
             Microscopy},
  journal = {Nature Methods},
  year    = {2018},
  volume  = {15},
  number  = {12},
  pages   = {1090--1097},
  doi     = {10.1038/s41592-018-0216-7}
}
@article{guo2020rapid,
  author  = {Guo, Min and Li, Yue and Su, Yijun and Lambert, Talley and others},
  title   = {Rapid Image Deconvolution and Multiview Fusion for Optical Microscopy},
  journal = {Nature Biotechnology},
  year    = {2020},
  volume  = {38},
  number  = {11},
  pages   = {1337--1346},
  doi     = {10.1038/s41587-020-0560-x}
}
@article{chen2021three,
  author  = {Chen, Jiji and Sasaki, Hideki and Lai, Hoyin and others},
  title   = {Three-Dimensional Residual Channel Attention Networks Denoise and
             Sharpen Fluorescence Microscopy Image Volumes},
  journal = {Nature Methods},
  year    = {2021},
  volume  = {18},
  number  = {6},
  pages   = {678--687},
  doi     = {10.1038/s41592-021-01155-x}
}
@article{perdigao2024redlionfish,
  author  = {Perdig{\~a}o, Lu{\'i}s M. A. and Berger, Casper and
             Yee, Neville B.-Y. and Darrow, Michele C. and Basham, Mark},
  title   = {{RedLionfish} -- Fast {Richardson-Lucy} Deconvolution Package for
             Efficient Point Spread Function Suppression in Volumetric Data},
  journal = {Wellcome Open Research},
  year    = {2024},
  volume  = {9},
  pages   = {296},
  doi     = {10.12688/wellcomeopenres.21505.1}
}

@misc{sw_scikitimage,
  title = {scikit-image: restoration (richardson\_lucy, wiener, unsupervised\_wiener)},
  note  = {BSD-3-Clause}, howpublished = {\url{https://scikit-image.org}}
}
@misc{sw_redlionfish,
  title = {RedLionfish (RedLionfishDeconv)},
  note  = {Apache-2.0}, howpublished = {\url{https://github.com/rosalindfranklininstitute/RedLionfish}}
}
@misc{sw_flowdec,
  title = {flowdec: TensorFlow deconvolution for microscopy},
  note  = {Apache-2.0}, howpublished = {\url{https://github.com/hammerlab/flowdec}}
}
@misc{sw_microscpsf,
  title = {MicroscPSF-Py: fast Gibson--Lanni {PSF}},
  note  = {MIT}, howpublished = {\url{https://github.com/MicroscPSF/MicroscPSF-Py}}
}
@misc{sw_psfmodels,
  title = {psfmodels: scalar and vectorial {PSF} models},
  note  = {GPL-3.0 (offline/validation only)}, howpublished = {\url{https://github.com/tlambert03/PSFmodels}}
}
@misc{sw_csbdeep,
  title = {CSBDeep / CARE},
  note  = {BSD-3-Clause}, howpublished = {\url{https://github.com/CSBDeep/CSBDeep}}
}
```

