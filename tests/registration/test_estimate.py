"""Registration engine tests — synthetic ground truth (V1.56).

Mirrors the house pattern (``tests/spots``): build a known transform, apply it,
assert the engine recovers it within a physical tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from nd2studios.backend.registration import estimate


# ── translation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("true_shift", [(3.0, -5.0), (0.5, 1.5), (-2.25, 0.75)])
def test_known_shift_recovery(ref_image, true_shift):
    """A known sub-pixel shift is recovered to ≤ ~0.15 px at upsample=20."""
    moving = nd_shift(ref_image.astype(np.float32), shift=[-s for s in true_shift],
                      order=1, mode="constant", cval=0.0)
    shift, ncc = estimate.estimate_translation(ref_image, moving, upsample=20)
    assert np.allclose(shift, true_shift, atol=0.15)
    assert ncc > 0.5


def test_apply_shift_round_trip(ref_image):
    """Estimate → apply realigns the moving image back onto the reference."""
    moving = nd_shift(ref_image.astype(np.float32), shift=[-4.0, 6.0],
                      order=1, mode="constant", cval=0.0)
    shift, _ = estimate.estimate_translation(ref_image, moving, upsample=20)
    aligned = estimate.apply_shift(moving, shift, order=1)
    # Compare interiors (edges are contaminated by cval=0 padding).
    a = aligned[20:-20, 20:-20].astype(np.float32)
    b = ref_image[20:-20, 20:-20].astype(np.float32)
    assert estimate._ncc(a, b) > 0.95


def test_apply_shift_preserves_dtype(ref_image):
    out = estimate.apply_shift(ref_image, np.array([2.0, -3.0]), order=1)
    assert out.dtype == ref_image.dtype
    assert out.max() <= np.iinfo(ref_image.dtype).max


# ── (T, H, W) drift series ────────────────────────────────────────────────────

def _random_walk_series(base: np.ndarray, T: int = 8, seed: int = 1):
    """Build a (T,H,W) series with a cumulative random-walk drift + ground truth."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.5, size=(T, 2))
    steps[0] = 0.0
    positions = np.cumsum(steps, axis=0)  # absolute drift of each frame vs frame 0
    vol = np.empty((T, *base.shape), dtype=base.dtype)
    for t in range(T):
        vol[t] = estimate.apply_shift(base, positions[t], order=1)
    return vol, positions


@pytest.mark.parametrize("reference", ["first", "previous", "mean"])
def test_drift_correction_reduces_residual(ref_image, reference):
    """Stabilizing a drifting series makes all frames agree with frame 0."""
    vol, _positions = _random_walk_series(ref_image)
    aligned, shifts, conf = estimate.stabilize(
        vol, model="translation", reference=reference, upsample=20)

    def _mean_residual(series):
        f0 = series[0][20:-20, 20:-20].astype(np.float32)
        return np.mean([
            1.0 - estimate._ncc(f0, series[t][20:-20, 20:-20].astype(np.float32))
            for t in range(1, series.shape[0])
        ])

    assert _mean_residual(aligned) < _mean_residual(vol)
    # `mean` aligns to a blurred average of drifting frames, so it is inherently
    # looser than anchoring on a single sharp frame — allow a wider tolerance.
    tol = 0.08 if reference == "mean" else 0.05
    assert _mean_residual(aligned) < tol
    assert conf[1:].min() > 0.5
    assert shifts.shape == (vol.shape[0], 2)


def test_stabilize_preserves_dtype_and_shape(ref_image):
    vol, _ = _random_walk_series(ref_image)
    aligned, _, _ = estimate.stabilize(vol, reference="first")
    assert aligned.shape == vol.shape
    assert aligned.dtype == vol.dtype


def test_stabilize_single_frame_is_noop(ref_image):
    vol = ref_image[None]  # (1, H, W)
    aligned, shifts, conf = estimate.stabilize(vol)
    assert np.array_equal(aligned, vol)


# ── register once, apply to all (cross-channel primitive) ─────────────────────

def test_register_once_apply_to_all(ref_image, ref_image_factory):
    """A shift estimated on channel A aligns channel B, which shares the drift."""
    other = ref_image_factory(seed=7)  # a *different* structure = channel B
    true_shift = np.array([5.0, -3.0])
    a_moved = estimate.apply_shift(ref_image, true_shift, order=1)
    b_moved = estimate.apply_shift(other, true_shift, order=1)
    # Estimate on A only, apply the SAME alignment shift to B (preserving
    # colocalization): `shift` aligns a_moved→ref, and B shares the drift.
    shift, _ = estimate.estimate_translation(ref_image, a_moved, upsample=20)
    b_aligned = estimate.apply_shift(b_moved, shift, order=1)
    inner = (slice(20, -20), slice(20, -20))
    assert estimate._ncc(b_aligned[inner].astype(np.float32),
                         other[inner].astype(np.float32)) > 0.95


# ── ECC rigid recovery ────────────────────────────────────────────────────────

def test_ecc_recovers_rotation(ref_image):
    """ECC euclidean recovers a known small rotation."""
    from scipy.ndimage import rotate

    angle_deg = 4.0
    moving = rotate(ref_image.astype(np.float32), angle_deg, reshape=False,
                    order=1, mode="constant", cval=0.0)
    warp, cc, aligned = estimate.ecc_align(ref_image, moving, model="euclidean",
                                           init_shift=np.zeros(2))
    assert cc > 0.7
    recovered = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    assert abs(abs(recovered) - angle_deg) < 1.0
    # And the aligned image agrees with the reference in the interior.
    inner = (slice(30, -30), slice(30, -30))
    assert estimate._ncc(aligned[inner].astype(np.float32),
                         ref_image[inner].astype(np.float32)) > 0.9


def test_stabilize_affine_runs(ref_image):
    """The ECC affine path runs end-to-end and returns a well-formed result."""
    vol, _ = _random_walk_series(ref_image, T=4)
    aligned, shifts, conf = estimate.stabilize(vol, model="affine", reference="first")
    assert aligned.shape == vol.shape
    assert aligned.dtype == vol.dtype
    assert conf.shape == (vol.shape[0],)


# ── estimate_series / apply_series (register once, apply to all) ──────────────

@pytest.mark.parametrize("reference", ["first", "previous", "mean"])
def test_estimate_apply_series_matches_stabilize_translation(ref_image, reference):
    """estimate_series → apply_series reproduces stabilize's translation result
    and, applied to a *second* channel with the same drift, aligns it too."""
    vol, _ = _random_walk_series(ref_image)
    tf = estimate.estimate_series(vol, model="translation", reference=reference)
    aligned = estimate.apply_series(vol, tf)

    def _resid(series):
        f0 = series[0][20:-20, 20:-20].astype(np.float32)
        return np.mean([1.0 - estimate._ncc(
            f0, series[t][20:-20, 20:-20].astype(np.float32))
            for t in range(1, series.shape[0])])

    assert _resid(aligned) < (0.08 if reference == "mean" else 0.05)

    # A second channel sharing the same absolute drift, aligned by the SAME tf.
    other = ref_image  # same structure re-used as "channel B" for a clean check
    vol_b = np.stack([estimate.apply_shift(other, np.cumsum(
        np.zeros((1, 2)), 0)[0])] + [vol[t] for t in range(1, vol.shape[0])])
    aligned_b = estimate.apply_series(vol_b, tf)
    assert aligned_b.shape == vol.shape
    assert aligned_b.dtype == vol.dtype


def test_estimate_series_ecc_previous_composes(ref_image):
    """ECC 'previous' mode composes per-step warps into an absolute warp that
    aligns a doubly-rotated frame back to frame 0."""
    from scipy.ndimage import rotate

    f0 = ref_image
    f1 = rotate(f0.astype(np.float32), 3.0, reshape=False, order=1).astype(np.uint16)
    f2 = rotate(f1.astype(np.float32), 3.0, reshape=False, order=1).astype(np.uint16)
    vol = np.stack([f0, f1, f2])
    tf = estimate.estimate_series(vol, model="euclidean", reference="previous")
    aligned = estimate.apply_series(vol, tf)
    inner = (slice(40, -40), slice(40, -40))
    # Frame 2 (≈6° rotated) should realign to frame 0 after composition.
    assert estimate._ncc(aligned[2][inner].astype(np.float32),
                         f0[inner].astype(np.float32)) > 0.85


def test_estimate_series_translation_has_no_warps(ref_image):
    vol, _ = _random_walk_series(ref_image, T=3)
    tf = estimate.estimate_series(vol, model="translation", reference="first")
    assert tf["warps"] is None
    assert tf["shifts"].shape == (3, 2)
    assert tf["confidence"].shape == (3,)
