"""
Tests for the V1.54 regime-aware stitching pipeline (spec §12).

Synthetic ground truth: cut a large textured image into a known grid at known
stage coordinates (with jitter), run the pipeline, and assert recovered
positions / regime / output. Runnable directly (``py tests/test_stitch_pipeline.py``)
or under pytest.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import tifffile

from nd2studios.backend.stitch import StitchConfig, run_stitch, build_dataset, decide_regime
from nd2studios.backend.stitch.config import (
    ENGINE_PHASE, REGIME_AUTO, ENGINE_AUTO, ENGINE_ASHLAR, BLEND_FEATHER,
)
from nd2studios.backend.stitch.engines import compute_positions, seed_positions
from nd2studios.backend.stitch.compositor import normalize_positions
from nd2studios.backend import tiff_loader


PX = 0.5  # µm/px


class SyntheticVolume:
    """Minimal LazyND2Volume-compatible stub over pre-cut tiles."""

    def __init__(self, tiles_by_c, tile_h, tile_w, channel_names, dtype=np.uint16,
                 n_timepoints=1):
        # tiles_by_c: list over channels of {m: (H,W) array}
        self._tiles = tiles_by_c
        self.height = tile_h
        self.width = tile_w
        self.channel_names = list(channel_names)
        self.n_channels = len(channel_names)
        self.n_timepoints = int(n_timepoints)
        self.n_zslices = 1
        self.n_multipoints = len(tiles_by_c[0])
        self.dtype = np.dtype(dtype)
        self.pixel_size_um = PX

    def get_frame(self, c=0, m=0, t=0, z=0, z_mode="none", z_start=None, z_end=None):
        return self._tiles[c][m]


def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(2000, 500, size=(h, w))
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=1.5)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _build_grid(overlap_frac=0.2, jitter_px=0.0, n_channels=1, seed=0):
    """Return (volume, stage_xy_um, gt_positions) for a 2x3 grid.

    gt_positions: dict m -> (row_px, col_px) ground-truth top-left (0-origin).
    """
    th = tw = 256
    step = int(round(tw * (1.0 - overlap_frac)))
    rows, cols = 2, 3
    base_h = rows * step + th + 8
    base_w = cols * step + tw + 8
    bases = [_texture(base_h, base_w, seed=seed + 100 * ci) for ci in range(n_channels)]

    rng = np.random.default_rng(seed + 7)
    tiles_by_c = [dict() for _ in range(n_channels)]
    gt = {}
    stage_xy = []
    X0, Y0 = 5000.0, 3000.0  # arbitrary stage origin (µm)
    m = 0
    for r in range(rows):
        for c in range(cols):
            y, x = r * step, c * step
            gt[m] = (float(y), float(x))
            for ci in range(n_channels):
                tiles_by_c[ci][m] = bases[ci][y:y + th, x:x + tw].copy()
            # Inverse of the default orientation (flip_x=True, flip_y=False):
            #   col_px = (max_stage_x - stage_x)/px  → stage_x = X0 - col_px*px
            #   row_px = (stage_y - min_stage_y)/px  → stage_y = Y0 + row_px*px
            jx = rng.uniform(-jitter_px, jitter_px) * PX
            jy = rng.uniform(-jitter_px, jitter_px) * PX
            stage_x = X0 - x * PX + jx
            stage_y = Y0 + y * PX + jy
            stage_xy.append((stage_x, stage_y))
            m += 1
    vol = SyntheticVolume(tiles_by_c, th, tw,
                          [f"Ch{i}" for i in range(n_channels)])
    return vol, stage_xy, gt


def _max_pos_error(recovered, gt):
    rec = normalize_positions(recovered)
    errs = []
    for m in gt:
        ry, rx = rec[m]
        gy, gx = gt[m]
        errs.append(max(abs(ry - gy), abs(rx - gx)))
    return max(errs)


# ── regime selection ──

def test_regime_zero_overlap_abutting():
    vol, stage, gt = _build_grid(overlap_frac=0.0)
    ds = build_dataset(vol, stage, list(range(len(stage))), StitchConfig())
    assert decide_regime(ds, StitchConfig()) == "zero_overlap"


def test_regime_overlap_10pct():
    vol, stage, gt = _build_grid(overlap_frac=0.10)
    ds = build_dataset(vol, stage, list(range(len(stage))), StitchConfig())
    assert ds.overlap_frac > 0.05
    assert decide_regime(ds, StitchConfig()) == "overlap"


def _single_row(overlap_frac, jitter_px=2.0, n=5, seed=11):
    """Single-row scan: real X overlap, constant Y with jitter."""
    th = tw = 256
    step_px = int(round(tw * (1.0 - overlap_frac)))
    rng = np.random.default_rng(seed)
    stage = []
    X0, Y0 = 4000.0, 2000.0
    for c in range(n):
        jx = rng.uniform(-jitter_px, jitter_px) * PX
        jy = rng.uniform(-jitter_px, jitter_px) * PX
        stage.append((X0 - c * step_px * PX + jx, Y0 + jy))
    vol = SyntheticVolume([{m: np.zeros((th, tw), np.uint16) for m in range(n)}],
                          th, tw, ["Ch0"])
    return vol, stage


def test_single_row_jitter_not_spurious_overlap():
    # Constant Y with jitter must NOT become a ~99% overlap → single grid row.
    vol, stage = _single_row(overlap_frac=0.2)
    ds = build_dataset(vol, stage, list(range(len(stage))), StitchConfig())
    assert ds.grid_shape[0] == 1, f"expected 1 row, got {ds.grid_shape}"
    assert ds.overlap_frac_y is None
    assert abs(ds.overlap_frac_x - 0.2) < 0.05
    assert decide_regime(ds, StitchConfig()) == "overlap"


def test_single_row_abutting_is_zero_overlap():
    vol, stage = _single_row(overlap_frac=0.0)
    ds = build_dataset(vol, stage, list(range(len(stage))), StitchConfig())
    assert ds.grid_shape[0] == 1
    assert decide_regime(ds, StitchConfig()) == "zero_overlap"


def test_regime_gap_is_zero_overlap():
    # A negative "overlap" (gap between tiles) must classify as zero-overlap.
    th = tw = 256
    stage = []
    X0, Y0 = 0.0, 0.0
    step_px = 300  # > tile → gap
    for r in range(2):
        for c in range(3):
            stage.append((X0 - c * step_px * PX, Y0 + r * step_px * PX))
    vol = SyntheticVolume([{m: np.zeros((th, tw), np.uint16) for m in range(6)}],
                          th, tw, ["Ch0"])
    ds = build_dataset(vol, stage, list(range(6)), StitchConfig())
    assert decide_regime(ds, StitchConfig()) == "zero_overlap"


# ── coordinate placement (zero-overlap) is exact ──

def test_zero_overlap_exact_placement():
    vol, stage, gt = _build_grid(overlap_frac=0.0)
    seed = seed_positions(build_dataset(vol, stage, list(range(len(stage))), StitchConfig()))
    err = _max_pos_error(seed, gt)
    assert err < 1.0, f"coordinate placement off by {err}px"


# ── overlap registration recovers positions despite jitter ──

def _register_and_check(engine):
    vol, stage, gt = _build_grid(overlap_frac=0.25, jitter_px=2.0, seed=1)
    cfg = StitchConfig(engine=engine, ncc_threshold=0.2)
    ds = build_dataset(vol, stage, list(range(len(stage))), cfg)
    regime = decide_regime(ds, cfg)
    assert regime == "overlap"
    ref = {t.index: vol.get_frame(c=0, m=t.index) for t in ds.tiles}
    pos, conf, info = compute_positions(ds, ref, cfg, regime)
    err = _max_pos_error(pos, gt)
    return err, info


def test_phase_correlation_registration():
    err, info = _register_and_check(ENGINE_PHASE)
    assert err <= 3.0, f"phase-corr recovered positions off by {err}px ({info})"


def test_auto_engine_registration():
    err, info = _register_and_check(ENGINE_AUTO)
    assert err <= 3.0, f"auto engine recovered positions off by {err}px ({info})"


# ── OME-TIFF round-trip ──

def test_ome_roundtrip(tmpdir=None):
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.2, n_channels=2, seed=2)
    out = os.path.join(d, "mosaic.ome.tif")
    cfg = StitchConfig(engine=ENGINE_AUTO, blend=BLEND_FEATHER, write_qc=True)
    res = run_stitch(vol, stage, list(range(len(stage))), [0, 1], cfg, out)
    assert os.path.exists(res.out_path)

    with tifffile.TiffFile(res.out_path) as tif:
        assert tif.is_ome
        s = tif.series[0]
        dmap = dict(zip(s.axes, s.shape))
        assert dmap.get("C") == 2
        assert str(s.dtype) == "uint16"

    ome = tiff_loader.read_ome_tiff_metadata(res.out_path)
    assert ome["n_channels"] == 2
    assert abs((ome["pixel_size_um"] or 0) - PX) < 1e-3

    view = tiff_loader._SingleFileTIFFView(res.out_path)
    try:
        assert view.n_channels == 2
        f0 = view.get_frame(c=0, m=0, t=0, z=0, z_mode="none")
        f1 = view.get_frame(c=1, m=0, t=0, z=0, z_mode="none")
        assert f0.shape == (res.canvas_h, res.canvas_w)
        assert f0.dtype == np.uint16
        # Channels differ (different textures) → not identical.
        assert not np.array_equal(f0, f1)
    finally:
        view.close()

    # The real app reload path: single-file TIFF → LazyMultiFileTIFFVolume.
    comp = tiff_loader.LazyMultiFileTIFFVolume([res.out_path], chain_axis="Z")
    try:
        assert comp.n_channels == 2
        chans = comp.all_channels_as_lazy(m=0)
        assert len(chans) == 2
        arr = np.asarray(next(iter(chans.values()))[0])
        assert arr.shape == (res.canvas_h, res.canvas_w)
        assert int(arr.max()) > 0  # non-empty
    finally:
        comp.close()


def test_single_tile_degenerate(tmpdir=None):
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    tile = _texture(256, 256, seed=9)
    vol = SyntheticVolume([{0: tile}], 256, 256, ["Ch0"])
    out = os.path.join(d, "single.ome.tif")
    res = run_stitch(vol, [(1000.0, 1000.0)], [0], [0], StitchConfig(), out)
    assert res.canvas_h == 256 and res.canvas_w == 256


def test_dtype_preserved(tmpdir=None):
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.2, seed=3)
    out = os.path.join(d, "dtype.ome.tif")
    res = run_stitch(vol, stage, list(range(len(stage))), [0], StitchConfig(), out)
    with tifffile.TiffFile(res.out_path) as tif:
        assert str(tif.series[0].dtype) == "uint16"


def test_memmap_large_mosaic_path(tmpdir=None):
    """Force the disk-backed (memmap) assembly branch with a tiny RAM budget."""
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.2, seed=4)
    out = os.path.join(d, "memmap.ome.tif")
    cfg = StitchConfig(max_memory_gb=1e-9)  # anything > 0 bytes → memmap
    res = run_stitch(vol, stage, list(range(len(stage))), [0], cfg, out)
    assert os.path.exists(res.out_path)
    # Temp scratch file must be cleaned up.
    assert not os.path.exists(out + ".stitch_tmp.dat")
    with tifffile.TiffFile(res.out_path) as tif:
        assert tif.is_ome


def test_builtin_illumination(tmpdir=None):
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.2, seed=5)
    out = os.path.join(d, "illum.ome.tif")
    cfg = StitchConfig(illumination_correction="builtin")
    res = run_stitch(vol, stage, list(range(len(stage))), [0], cfg, out)
    assert os.path.exists(res.out_path)


def test_per_timepoint_registration(tmpdir=None):
    """Per-T re-registration must run and keep a single consistent canvas."""
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.25, jitter_px=1.5, seed=6)
    vol.n_timepoints = 3  # reuse the same tiles for every T (no drift)
    out = os.path.join(d, "pertime.ome.tif")
    cfg = StitchConfig(per_timepoint_registration=True, engine=ENGINE_PHASE)
    res = run_stitch(vol, stage, list(range(len(stage))), [0], cfg, out)
    with tifffile.TiffFile(res.out_path) as tif:
        dmap = dict(zip(tif.series[0].axes, tif.series[0].shape))
        assert dmap.get("T") == 3
        assert dmap.get("Y") == res.canvas_h and dmap.get("X") == res.canvas_w


def test_compositor_negative_position():
    from nd2studios.backend.stitch.compositor import composite_frame
    a = np.full((4, 4), 10, np.uint16)
    b = np.full((4, 4), 20, np.uint16)
    # b placed at (-2,-2): only its bottom-right 2x2 lands at the canvas origin.
    canvas = composite_frame([a, b], [(0.0, 0.0), (-2.0, -2.0)], 4, 4,
                             np.uint16, blend="none", fill_value=0)
    assert canvas.shape == (4, 4)
    assert canvas[0, 0] == 20 and canvas[1, 1] == 20  # b overwrote here
    assert canvas[3, 3] == 10                          # a only, beyond b


def test_ome_multiz_single_channel_reload(tmpdir=None):
    d = str(tmpdir) if tmpdir is not None else tempfile.mkdtemp()
    vol, stage, gt = _build_grid(overlap_frac=0.2, seed=8)
    vol.n_zslices = 3  # single channel, 3 Z planes (stub returns same plane)
    out = os.path.join(d, "multiz.ome.tif")
    cfg = StitchConfig(z_mode="none")  # preserve Z
    res = run_stitch(vol, stage, list(range(len(stage))), [0], cfg, out)
    with tifffile.TiffFile(res.out_path) as tif:
        dmap = dict(zip(tif.series[0].axes, tif.series[0].shape))
        assert dmap.get("Z") == 3
    comp = tiff_loader.LazyMultiFileTIFFVolume([res.out_path], chain_axis="Z")
    try:
        assert comp.n_zslices == 3
        f = comp.get_frame(c=0, m=0, t=0, z=1, z_mode="none")
        assert f.shape == (res.canvas_h, res.canvas_w) and int(f.max()) > 0
    finally:
        comp.close()


def test_ashlar_registration_or_gated():
    """ashlar recovers jittered positions when a JDK is available (jdk4py);
    otherwise it must raise a clear, actionable error."""
    from nd2studios.backend.stitch.ashlar_engine import ashlar_available
    if ashlar_available():
        err, info = _register_and_check(ENGINE_ASHLAR)
        assert err <= 3.0, f"ashlar recovered positions off by {err}px ({info})"
        assert info.get("engine") == "ashlar"
    else:
        vol, stage, gt = _build_grid(overlap_frac=0.25, jitter_px=2.0, seed=1)
        cfg = StitchConfig(engine=ENGINE_ASHLAR)
        ds = build_dataset(vol, stage, list(range(len(stage))), cfg)
        ref = {t.index: vol.get_frame(c=0, m=t.index) for t in ds.tiles}
        raised = False
        try:
            compute_positions(ds, ref, cfg, decide_regime(ds, cfg))
        except ImportError as exc:
            raised = True
            assert "ashlar" in str(exc).lower() or "java" in str(exc).lower()
        assert raised


if __name__ == "__main__":
    import traceback
    tmp = tempfile.mkdtemp()
    tests = [
        test_regime_zero_overlap_abutting,
        test_regime_overlap_10pct,
        test_single_row_jitter_not_spurious_overlap,
        test_single_row_abutting_is_zero_overlap,
        test_regime_gap_is_zero_overlap,
        test_zero_overlap_exact_placement,
        test_phase_correlation_registration,
        test_auto_engine_registration,
        lambda: test_ome_roundtrip(tmp),
        lambda: test_single_tile_degenerate(tmp),
        lambda: test_dtype_preserved(tmp),
        lambda: test_memmap_large_mosaic_path(tmp),
        lambda: test_builtin_illumination(tmp),
        lambda: test_per_timepoint_registration(tmp),
        test_compositor_negative_position,
        lambda: test_ome_multiz_single_channel_reload(tmp),
        test_ashlar_registration_or_gated,
    ]
    names = [
        "regime_zero_overlap_abutting", "regime_overlap_10pct",
        "single_row_jitter_not_spurious_overlap", "single_row_abutting_is_zero_overlap",
        "regime_gap_is_zero_overlap", "zero_overlap_exact_placement",
        "phase_correlation_registration", "auto_engine_registration",
        "ome_roundtrip", "single_tile_degenerate", "dtype_preserved",
        "memmap_large_mosaic_path", "builtin_illumination",
        "per_timepoint_registration", "compositor_negative_position",
        "ome_multiz_single_channel_reload", "ashlar_registration_or_gated",
    ]
    n_pass = 0
    for name, fn in zip(names, tests):
        try:
            fn()
            print(f"PASS  {name}")
            n_pass += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{n_pass}/{len(tests)} passed")
