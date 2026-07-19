"""Headless tests for StreamingDataset (fake source; no files, no display)."""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.streaming_dataset import (
    LazyVolumeSource, StreamingDataset,
)


class FakeSource:
    """Deterministic read source: plane value encodes (c, t, z)."""

    def __init__(self, M=1, T=4, Z=6, C=2, H=8, W=8):
        self.filepath = "fake"
        self.dtype = np.dtype(np.uint16)
        self.height, self.width = H, W
        self.n_zslices, self.n_channels = Z, C
        self.n_timepoints, self.n_multipoints = T, M
        self.channel_names = [f"c{i}" for i in range(C)]
        self.pixel_size_um, self.z_step_um = 0.5, 2.0
        self.reads = 0

    def read_volume(self, c, m, t, z0, z1):
        self.reads += 1
        arr = np.empty((z1 - z0, self.height, self.width), np.uint16)
        for i, z in enumerate(range(z0, z1)):
            arr[i] = np.uint16(c * 1000 + t * 100 + z)
        return arr

    def reopen(self):
        return FakeSource(self.n_multipoints, self.n_timepoints, self.n_zslices,
                          self.n_channels, self.height, self.width)

    def close(self):
        pass


def _ds(**kw):
    return StreamingDataset(FakeSource(**{k: kw.pop(k) for k in list(kw)
                                          if k in ("M", "T", "Z", "C", "H", "W")}),
                            prefetch=False, cache_budget_bytes=kw.get("cache", 1 << 20))


def test_shape_and_attrs():
    ds = StreamingDataset(FakeSource(M=1, T=4, Z=6, C=2, H=8, W=8), prefetch=False)
    assert ds.shape == (1, 4, 6, 8, 8)
    assert ds.channel_names == ["c0", "c1"]
    assert ds.pixel_size_um == 0.5 and ds.z_step_um == 2.0
    assert ds.is_lazy is True


def test_get_frame_none_and_cache():
    src = FakeSource()
    ds = StreamingDataset(src, prefetch=False)
    f = ds.get_frame(1, m=0, t=2, z=3, z_mode="none")
    assert f.shape == (8, 8)
    assert int(f[0, 0]) == 1 * 1000 + 2 * 100 + 3        # c=1,t=2,z=3
    reads_after_first = src.reads
    ds.get_frame(1, m=0, t=2, z=3, z_mode="none")         # cache hit
    assert src.reads == reads_after_first                 # no new read


def test_max_projection_blocked_matches_full():
    src = FakeSource(Z=6)
    ds = StreamingDataset(src, prefetch=False)
    ds._z_block = 2                                        # force blocking
    f = ds.get_frame(0, m=0, t=1, z_mode="max", z_start=0, z_end=6)
    # value = c*1000+t*100+z, max over z in [0,6) → z=5
    assert int(f[0, 0]) == 0 * 1000 + 1 * 100 + 5
    fmean = ds.get_frame(0, m=0, t=1, z_mode="mean", z_start=0, z_end=6)
    assert int(round(float(fmean[0, 0]))) == int(round(100 + np.mean(range(6))))


def test_get_volume_shape():
    ds = StreamingDataset(FakeSource(Z=6), prefetch=False)
    v = ds.get_volume(0, m=0, t=0, z_start=1, z_end=4)
    assert v.shape == (3, 8, 8)


def test_to_lazy_channel_proxy():
    ds = StreamingDataset(FakeSource(T=4, Z=6, C=2), prefetch=False)
    ch = ds.to_lazy_channel(1, m=0, z_mode="max")
    assert ch.shape == (4, 8, 8)
    frame0 = np.asarray(ch[0])
    assert frame0.shape == (8, 8)
    assert int(frame0[0, 0]) == 1000 + 0 + 5              # c=1,t=0,max z=5
    full = ch.materialize()
    assert full.shape == (4, 8, 8)
    cropped = ch.crop(0, 4, 0, 4)
    assert cropped.shape == (4, 4, 4)


def test_all_channels_as_lazy_keys():
    ds = StreamingDataset(FakeSource(C=3), prefetch=False)
    chans = ds.all_channels_as_lazy(m=0, z_mode="max")
    assert list(chans.keys()) == ["c0", "c1", "c2"]
    assert chans["c1"].shape == (4, 8, 8)


def test_lru_eviction_bounds_resident_bytes():
    src = FakeSource(T=50, Z=1, C=1, H=64, W=64)            # 8 KB/frame
    budget = 40 * 1024                                     # ~5 frames
    ds = StreamingDataset(src, prefetch=False, cache_budget_bytes=budget)
    for t in range(50):
        ds.get_frame(0, m=0, t=t, z=0, z_mode="none")
    assert ds.nbytes() <= budget                           # bounded
    assert ds.nbytes() < 50 * 64 * 64 * 2                   # far below full


def test_subset_returns_materialized():
    from nd2studios.backend.materialized_dataset import MaterializedDataset
    ds = StreamingDataset(FakeSource(M=1, T=4, Z=6, C=2), prefetch=False)
    sub = ds.subset(t=[0, 2], z=[1, 2, 3])
    assert isinstance(sub, MaterializedDataset)
    assert sub.n_timepoints == 2 and sub.n_zslices == 3
    assert sub.get_frame(0, m=0, t=0, z=0, z_mode="none").shape == (8, 8)


def test_reopen_and_close():
    ds = StreamingDataset(FakeSource(), prefetch=False)
    ds2 = ds.reopen()
    assert ds2.shape == ds.shape
    ds2.close()
    ds.close()


def test_prefetch_enabled_does_not_crash():
    ds = StreamingDataset(FakeSource(T=5), prefetch=True)
    for t in range(5):
        ds.get_frame(0, m=0, t=t, z=0, z_mode="none")
    import time
    time.sleep(0.2)
    ds.close()                                             # stops prefetch thread


def test_lazy_volume_source_wraps_get_volume():
    class Vol:
        filepath = "v"; dtype = np.dtype(np.uint16)
        height = width = 8; n_zslices = 4; n_channels = 1
        n_timepoints = 3; n_multipoints = 1; channel_names = ["a"]
        pixel_size_um = 1.0; z_step_um = 1.0
        def get_volume(self, c, m=0, t=0, z_start=None, z_end=None):
            z0 = z_start or 0; z1 = z_end or 4
            return np.full((z1 - z0, 8, 8), t + 1, np.uint16)
        def reopen(self): return self
        def close(self): pass
    ds = StreamingDataset(LazyVolumeSource(Vol()), prefetch=False)
    f = ds.get_frame(0, m=0, t=2, z_mode="max", z_start=0, z_end=4)
    assert int(f[0, 0]) == 3                                # t=2 → value 3


def test_lazy_volume_source_get_frame_fallback():
    """A composite exposing only get_frame (no get_volume) still streams."""
    class FrameOnly:
        filepath = "v"; dtype = np.dtype(np.uint16)
        height = width = 8; n_zslices = 4; n_channels = 1
        n_timepoints = 2; n_multipoints = 1; channel_names = ["a"]
        pixel_size_um = z_step_um = 1.0
        def get_frame(self, c, m=0, t=0, z=0, z_mode="none",
                      z_start=None, z_end=None):
            return np.full((8, 8), 10 * t + z, np.uint16)
        def reopen(self): return self
        def close(self): pass
    src = LazyVolumeSource(FrameOnly())
    assert src._has_get_volume is False
    blk = src.read_volume(0, 0, 1, 0, 4)                    # stack z0..3 at t=1
    assert blk.shape == (4, 8, 8)
    assert [int(blk[z, 0, 0]) for z in range(4)] == [10, 11, 12, 13]


def test_preview_projection_is_bounded_and_exact_stays_exact():
    src = FakeSource(T=2, Z=12, C=1)
    ds = StreamingDataset(src, prefetch=False, preview_planes=4, preview_default=True)
    # Deep max-proj over 12 Z with a 4-plane preview budget → subsampled.
    reads_before = src.reads
    prev = ds.get_frame(0, m=0, t=0, z_mode="max", z_start=0, z_end=12)
    prev_reads = src.reads - reads_before
    # 4 evenly-spaced single-plane reads (bounded), not the full 12.
    assert prev_reads <= 4
    # value = c*1000 + t*100 + z → max over sampled planes ≤ max over all (z=11)
    assert int(prev[0, 0]) <= 0 + 0 + 11
    # The recipe/export path (to_lazy_channel) is EXACT — max over all 12 → z=11.
    ch = ds.to_lazy_channel(0, m=0, z_mode="max")
    assert int(np.asarray(ch[0])[0, 0]) == 11
    # preview=False on get_frame is also exact.
    exact = ds.get_frame(0, m=0, t=0, z_mode="max", z_start=0, z_end=12, preview=False)
    assert int(exact[0, 0]) == 11
