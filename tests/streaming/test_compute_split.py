"""Tests for the stream-for-view / materialize-for-analysis split (V1.66)."""
from __future__ import annotations

import os

import numpy as np

# Deterministic "available RAM" for the fit checks (16 GB).
os.environ["ND2_FAKE_RAM_GB"] = "16"

from nd2studios.utils.resource_strategy import (  # noqa: E402
    fits_resident, materialize_channels_if_fits, should_stream_analysis,
)


class FakeStream:
    """Lazy/streaming volume stand-in with footprint attributes."""
    is_lazy = True

    def __init__(self, T, Z, C, H, W, M=1):
        self.n_multipoints, self.n_timepoints, self.n_zslices = M, T, Z
        self.n_channels, self.height, self.width = C, H, W
        self.dtype = np.dtype(np.uint16)


class Proxy:
    """Lazy channel proxy: materialize() returns the backing array."""
    def __init__(self, arr):
        self._a = np.asarray(arr)
        self.shape = self._a.shape
        self.dtype = self._a.dtype

    def materialize(self):
        return self._a


def _small():
    return FakeStream(T=5, Z=1, C=1, H=512, W=512)      # ~2.6 MB → fits in 16 GB


def _huge():
    return FakeStream(T=100, Z=100, C=2, H=4096, W=4096)  # ~137 GB → does not fit


# ── fits_resident ─────────────────────────────────────────────────────────────
def test_fits_resident_small_true_huge_false():
    assert fits_resident(_small()) is True
    assert fits_resident(_huge()) is False
    assert fits_resident(None) is False


def test_fits_resident_memory_pressure_blocks():
    class Band:
        def current_band(self):
            return 2               # CRITICAL → never materialize
    assert fits_resident(_small(), monitor=Band()) is False


# ── should_stream_analysis (the smart split decision) ─────────────────────────
def test_should_stream_small_streamed_dataset_is_in_ram():
    # A streamed dataset that fits → materialize for analysis (don't stream).
    assert should_stream_analysis(_small()) is False


def test_should_stream_huge_streamed_dataset_streams():
    assert should_stream_analysis(_huge()) is True


def test_should_stream_respects_force():
    assert should_stream_analysis(_huge(), force=False) is False
    assert should_stream_analysis(_small(), force=True) is True


# ── materialize_channels_if_fits ──────────────────────────────────────────────
def test_materialize_when_fits_reads_proxies():
    ch = {"a": Proxy(np.arange(5 * 4 * 4).reshape(5, 4, 4).astype(np.uint16)),
          "b": np.ones((5, 4, 4), np.uint16)}
    out = materialize_channels_if_fits(ch, _small())
    assert isinstance(out["a"], np.ndarray)          # proxy → materialized
    assert out["a"].shape == (5, 4, 4)
    assert isinstance(out["b"], np.ndarray)           # ndarray passthrough
    assert out["b"] is ch["b"]


def test_no_materialize_when_too_big_keeps_lazy():
    proxy = Proxy(np.zeros((5, 4, 4), np.uint16))
    ch = {"a": proxy}
    out = materialize_channels_if_fits(ch, _huge())
    assert out["a"] is proxy                          # unchanged → streams


def test_materialize_empty_or_no_volume_is_safe():
    assert materialize_channels_if_fits({}, _small()) == {}
    proxy = Proxy(np.zeros((2, 2, 2), np.uint16))
    out = materialize_channels_if_fits({"a": proxy}, None)
    assert out["a"] is proxy                          # None volume → no fit → lazy
