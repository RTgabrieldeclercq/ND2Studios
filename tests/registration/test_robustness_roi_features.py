"""Registration V1.60 tests — late-frame robustness, ROI, and feature model.

Synthetic ground truth (house pattern). Covers the reported "later timepoints stop
registering" bug (Phase 0), region-of-interest estimation (Phase 1), and the
feature-based model (Phase 2).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, rotate, shift as nd_shift

from nd2studios.backend.registration import estimate


def _ncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / (d + 1e-9))


def _textured(H=256, W=256, seed=0, sigma=3.0, amp=3000.0, base=200.0):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma)
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    return f * amp + base


# ── Phase 0: late-frame robustness ────────────────────────────────────────────

def _degraded_series(T=30, seed=0):
    """Cumulative drift + photobleaching + a moving distractor + growing noise —
    the conditions under which whole-frame 'first' registration degrades late."""
    rng = np.random.default_rng(seed)
    base = _textured(seed=seed)
    steps = rng.normal(0, 1.6, (T, 2)); steps[0] = 0
    pos = np.cumsum(steps, 0)
    H, W = base.shape
    yy, xx = np.ogrid[:H, :W]
    vol = np.empty((T, H, W), np.float32)
    for t in range(T):
        f = nd_shift(base, pos[t], order=1, mode="constant", cval=0.0) * np.exp(-t / 12.0)
        f = f + 4000 * np.exp(-(((yy - 40 - 5 * t) ** 2 + (xx - 40 - 5 * t) ** 2) / (2 * 18.0 ** 2)))
        f = f + rng.normal(0, 40 + 6 * t, (H, W))
        vol[t] = f
    return vol.astype(np.uint16), -(pos - pos[0])


def test_template_gating_beats_first_on_degraded_series():
    """`template` + confidence gating keeps late-frame error bounded where the old
    default ('first', no gate) blows up — the reported bug."""
    vol, target = _degraded_series()
    old = estimate.estimate_series(vol, model="translation", reference="first",
                                   min_confidence=0.0)
    new = estimate.estimate_series(vol, model="translation", reference="template",
                                   min_confidence=0.2)
    late_old = np.linalg.norm(old["shifts"][-6:] - target[-6:], axis=1).mean()
    late_new = np.linalg.norm(new["shifts"][-6:] - target[-6:], axis=1).mean()
    assert late_new < late_old * 0.5      # dramatically better late
    assert late_new < late_old - 20.0
    assert new["gated"].any()             # degraded frames are flagged


def test_previous_gating_prevents_explosion():
    """A single catastrophic (blank) frame must not poison the cumulative sum."""
    base = _textured()
    T = 12
    vol = np.stack([nd_shift(base, (0.8 * t, -0.5 * t), order=1) for t in range(T)])
    vol = vol.astype(np.uint16)
    vol[6] = np.zeros_like(vol[6])        # one dead frame (confidence collapses)
    tf = estimate.estimate_series(vol, model="translation", reference="previous",
                                  min_confidence=0.2)
    assert tf["gated"][6]                 # flagged
    # frames after the dead one are still ~on-trajectory (no huge jump)
    assert np.linalg.norm(tf["shifts"][7] - tf["shifts"][5]) < 5.0


def test_gated_key_present_and_holds_last():
    base = _textured()
    vol = np.stack([base, nd_shift(base, (3.0, -2.0), order=1),
                    np.zeros_like(base)]).astype(np.uint16)  # 3rd frame blank
    tf = estimate.estimate_series(vol, model="translation", reference="first",
                                  min_confidence=0.3)
    assert "gated" in tf and tf["gated"].shape == (3,)
    assert tf["gated"][2]
    # held the previous good absolute shift rather than snapping to zero
    assert np.allclose(tf["shifts"][2], tf["shifts"][1], atol=1e-6)


def test_template_and_zscore_run(ref_image):
    vol = np.stack([nd_shift(ref_image.astype(np.float32), (t * 0.7, -t * 0.4),
                             order=1).astype(np.uint16) for t in range(6)])
    tf = estimate.estimate_series(vol, reference="template", normalize="zscore")
    assert tf["reference"] == "template" and tf["shifts"].shape == (6, 2)


# ── Phase 1: region of interest ───────────────────────────────────────────────

def test_roi_to_mask_helper():
    mask, bbox = estimate.roi_to_mask({"kind": "rect", "x": 10, "y": 20, "w": 30, "h": 40},
                                      (100, 100))
    assert bbox == (20, 60, 10, 40) and mask[20:60, 10:40].all() and not mask[0, 0]
    mask2, bbox2 = estimate.roi_to_mask(
        {"kind": "shapes", "shapes": [{"type": "rect", "vertices": [[5, 5], [25, 25]]}]},
        (64, 64))
    assert bbox2 is None and mask2 is not None and mask2.any()
    assert estimate.roi_to_mask(None, (10, 10)) == (None, None)


def _two_region_series(T=8, seed=1):
    """Two textured regions moving DIFFERENTLY: a small ROI box moves by ``sA``
    while the dominant rest of the frame moves by ``sB``. Whole-frame correlation
    tracks the dominant rest; an ROI on the box should recover the box's own
    motion (``sA``). ``mode="wrap"`` avoids border artifacts so the rest truly
    dominates. Returns ``(vol, roi, expected_roi_shift_last)``."""
    regionA = _textured(seed=seed)
    regionB = _textured(seed=seed + 7)
    box = (90, 150, 90, 150)  # y0,y1,x0,x1 — ~5% of a 256² frame
    sA = np.array([1.2, -0.8])
    sB = np.array([-2.5, 2.0])
    vol = np.empty((T, *regionA.shape), np.float32)
    for t in range(T):
        fB = nd_shift(regionB, sB * t, order=1, mode="wrap")
        fA = nd_shift(regionA, sA * t, order=1, mode="wrap")
        f = fB.copy()
        f[box[0]:box[1], box[2]:box[3]] = fA[box[0]:box[1], box[2]:box[3]]
        vol[t] = f
    roi = {"kind": "rect", "x": box[2], "y": box[0],
           "w": box[3] - box[2], "h": box[1] - box[0]}
    # recovered shift aligns moving→reference, i.e. the negative of the content motion
    expected_last = -sA * (T - 1)
    return vol.astype(np.uint16), roi, expected_last


def test_roi_rect_beats_wholeframe():
    vol, roi, expected_last = _two_region_series()
    full = estimate.estimate_series(vol, model="translation", reference="first")
    roied = estimate.estimate_series(vol, model="translation", reference="first", roi=roi)
    # ROI recovers the box's own motion...
    assert np.allclose(roied["shifts"][-1], expected_last, atol=1.5)
    # ...while the whole-frame estimate (tracking the dominant rest) disagrees.
    assert np.linalg.norm(full["shifts"][-1] - roied["shifts"][-1]) > 8.0


def test_roi_rect_subpixel(ref_image):
    true = np.array([2.3, -1.7])
    moving = nd_shift(ref_image.astype(np.float32), -true, order=1)
    vol = np.stack([ref_image, moving.astype(np.uint16)])
    roi = {"kind": "rect", "x": 30, "y": 30, "w": 190, "h": 190}
    tf = estimate.estimate_series(vol, model="translation", reference="first", roi=roi)
    assert np.allclose(tf["shifts"][1], true, atol=0.2)


def test_roi_freeform_mask_runs(ref_image):
    true = np.array([4.0, 3.0])
    moving = nd_shift(ref_image.astype(np.float32), -true, order=1)
    vol = np.stack([ref_image, moving.astype(np.uint16)])
    roi = {"kind": "shapes",
           "shapes": [{"type": "rect", "vertices": [[20, 20], [230, 230]]}]}
    tf = estimate.estimate_series(vol, model="translation", reference="first", roi=roi)
    # masked path is integer-pixel — recover within ~1 px
    assert np.allclose(tf["shifts"][1], true, atol=1.5)


# ── Phase 2: feature-based model ──────────────────────────────────────────────

def test_feature_recovers_large_rotation(ref_image):
    f0 = ref_image
    f1 = nd_shift(rotate(ref_image.astype(np.float32), 12.0, reshape=False, order=1),
                  [25, -18], order=1).astype(np.uint16)
    vol = np.stack([f0, f1])
    feat = estimate.estimate_series(vol, model="feature", reference="first",
                                    feature_transform="euclidean", min_inliers=8)
    trans = estimate.estimate_series(vol, model="translation", reference="first")
    inr = (slice(40, -40), slice(40, -40))
    af = estimate.apply_series(vol, feat)[1][inr]
    at = estimate.apply_series(vol, trans)[1][inr]
    assert feat["confidence"][1] > 0.3
    assert _ncc(af, f0[inr]) > 0.85            # feature aligns the big rotation
    assert _ncc(af, f0[inr]) > _ncc(at, f0[inr]) + 0.3   # translation cannot


def test_feature_warp_convention(ref_image):
    """warp must map reference→moving so apply_warp (WARP_INVERSE_MAP) aligns."""
    moving = rotate(ref_image.astype(np.float32), 6.0, reshape=False, order=1).astype(np.uint16)
    warp, conf = estimate.estimate_features(ref_image, moving, transform="euclidean")
    assert conf > 0.3
    aligned = estimate.apply_warp(moving, warp, output_shape=ref_image.shape)
    inr = (slice(40, -40), slice(40, -40))
    assert _ncc(aligned[inr], ref_image[inr]) > 0.9


# ── RegisteredFrameVolume playback cache (V1.60 perf) ─────────────────────────

def test_registered_frame_volume_caches_warps(ref_image):
    """The lazy display volume must warp each (c,m,t) once and serve repeats from
    cache — otherwise playback recomputes the shift every tick and crawls."""
    from nd2studios.pipeline_graph.executor import RegisteredFrameVolume

    class _Base:
        channel_names = ["GFP"]
        n_multipoints = 1
        n_timepoints = 3
        n_zslices = 1
        height, width = ref_image.shape
        dtype = ref_image.dtype
        pixel_size_um = 1.0
        z_step_um = 1.0

        def __init__(self):
            self.calls = 0

        def get_frame(self, c=0, m=0, t=0, z=0, z_mode="max", **kw):
            self.calls += 1
            return ref_image

    tf = {"model": "translation", "reference": "first",
          "shifts": np.array([[0.0, 0.0], [3.0, -2.0], [1.0, 4.0]]),
          "warps": None, "confidence": np.ones(3)}
    base = _Base()
    vol = RegisteredFrameVolume(base, {0: tf})
    f1 = vol.get_frame(0, m=0, t=1)
    f2 = vol.get_frame(0, m=0, t=1)          # same plane → served from cache
    assert base.calls == 1                    # base read (and warp) happened once
    assert f2 is f1                           # exact cached object
    vol.get_frame(0, m=0, t=2)                # a different plane → one more read
    assert base.calls == 2

    # An M with no transform reads straight through (no caching needed).
    base2 = _Base()
    vol2 = RegisteredFrameVolume(base2, {})   # no transform for M0
    vol2.get_frame(0, m=0, t=0)
    vol2.get_frame(0, m=0, t=0)
    assert base2.calls == 2                    # pass-through, not cached


# ── common-region crop (translation) ──────────────────────────────────────────

def test_common_translation_crop_matches_valid_region(ref_image):
    """The crop rect equals the intersection of per-frame valid regions, and the
    cropped registered frames have NO zero-padding borders left."""
    base = ref_image
    # known integer shifts (so we can check exact borders); mixed signs
    shifts = np.array([[0, 0], [5, -3], [-4, 6], [2, 2]], dtype=float)
    vol = np.stack([estimate.apply_shift(base, s, order=1) for s in shifts])
    H, W = base.shape
    crop = estimate.common_translation_crop(shifts, (H, W), inset_edges=False)
    # analytic intersection: rows [max(0,dy) .. H+min(0,dy)], cols likewise
    y0 = int(np.ceil(max(0, shifts[:, 0].max())))
    y1 = int(np.floor(min(H, H + shifts[:, 0].min())))
    x0 = int(np.ceil(max(0, shifts[:, 1].max())))
    x1 = int(np.floor(min(W, W + shifts[:, 1].min())))
    assert crop == (y0, y1, x0, x1)
    # every frame's cropped region is fully non-zero (no padding leaked in)
    yy0, yy1, xx0, xx1 = crop
    for f in vol:
        sub = f[yy0:yy1, xx0:xx1]
        assert (sub > 0).all()


def test_common_translation_crop_none_when_no_overlap():
    # a shift larger than the frame leaves no common region
    shifts = np.array([[0, 0], [200, 0]], dtype=float)
    assert estimate.common_translation_crop(shifts, (128, 128)) is None


def test_common_translation_crop_full_when_no_drift():
    shifts = np.zeros((5, 2))
    assert estimate.common_translation_crop(shifts, (64, 96), inset_edges=False) == (0, 64, 0, 96)


def test_common_crop_all_frames_equal_size_and_recentred():
    """End-to-end with the faithful stage-drift model — each raw frame is a FULL
    HxW window of a larger scene at a drifting offset (no pre-existing padding).
    After registration + common crop: all frames equal size, no leaked borders,
    aligned content agrees with frame 0."""
    H = W = 200
    margin = 18
    scene = _textured(H + 2 * margin, W + 2 * margin, seed=5)
    rng = np.random.default_rng(3)
    off = np.cumsum(rng.normal(0, 3.0, (8, 2)), 0)
    off = np.clip(np.round(off).astype(int), -margin, margin)
    off[0] = 0
    vol = np.stack([scene[margin + oy:margin + oy + H, margin + ox:margin + ox + W]
                    for oy, ox in off]).astype(np.uint16)
    tf = estimate.estimate_series(vol, model="translation", reference="first")
    aligned = estimate.apply_series(vol, tf)
    crop = estimate.common_translation_crop(tf["shifts"], (H, W))
    assert crop is not None
    y0, y1, x0, x1 = crop
    cropped = aligned[:, y0:y1, x0:x1]
    assert cropped.shape[0] == vol.shape[0] and cropped.shape[1:] == (y1 - y0, x1 - x0)
    assert (cropped > 0).all()                    # no drift-border padding leaked in
    f0 = cropped[0].astype(np.float32)
    for t in range(1, cropped.shape[0]):
        assert _ncc(cropped[t].astype(np.float32), f0) > 0.97   # co-registered


def test_feature_blank_field_fallback():
    blank = np.zeros((128, 128), np.uint16)
    warp, conf = estimate.estimate_features(blank, blank, transform="affine")
    assert conf == 0.0 and np.allclose(warp, np.eye(2, 3))
    # and through the series driver: blank pair → identity, gated
    vol = np.stack([blank, blank])
    tf = estimate.estimate_series(vol, model="feature", reference="first",
                                  min_confidence=0.2)
    assert tf["gated"][1] and np.allclose(tf["warps"][1], np.eye(2, 3))
