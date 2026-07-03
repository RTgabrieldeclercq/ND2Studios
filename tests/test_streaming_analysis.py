"""V1.46 — streaming, resource-aware analysis regression tests.

Covers the adaptive gate, the incremental LabelStackWriter (zarr + memmap),
pipeline eager-vs-streamed parity, and compute_measurements lazy-vs-eager
parity, so the streaming path stays byte-identical to the in-RAM path.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.backend.results_engine import compute_measurements
from nd2studios.utils.resource_strategy import (
    LoadStrategy, choose_strategy, should_stream_analysis,
)
import nd2studios.pipeline.storage as storage
from nd2studios.pipeline.storage import LabelStackWriter


class _LazySrc:
    """Minimal lazy reader: ``.shape`` + per-frame ``[t]`` (like LazyND2Channel)."""

    def __init__(self, arr: np.ndarray) -> None:
        self._a = arr
        self.shape = arr.shape
        self.ndim = arr.ndim

    def __getitem__(self, t):
        return self._a[t]


def _materialized(shape=(1, 1, 1, 4, 4)):
    return MaterializedDataset(
        filepath="x", channels={"C0": np.zeros(shape, np.uint16)},
        channel_names=["C0"], dtype=np.dtype("uint16"),
        n_multipoints=shape[0], n_timepoints=shape[1], n_channels=1,
        n_zslices=shape[2], height=shape[3], width=shape[4],
    )


# ── 1. adaptive gate ──────────────────────────────────────────────────
def test_gate_materialized_is_eager():
    assert should_stream_analysis(_materialized()) is False


def test_gate_lazy_streams():
    class FakeLazy:
        is_lazy = True
    assert should_stream_analysis(FakeLazy()) is True


def test_gate_lazy_strategy_decision():
    meta = type("M", (), dict(n_timepoints=2, n_zslices=1, n_channels=1,
                              n_multipoints=1, height=8, width=8,
                              dtype=np.dtype("uint16")))()
    dec = choose_strategy(meta, "x.nd2", override=LoadStrategy.LAZY_CACHED)
    assert should_stream_analysis(_materialized(), decision=dec) is True


def test_gate_memory_pressure_overrides_eager():
    class Band:
        def current_band(self):
            return 2  # CRITICAL
    assert should_stream_analysis(_materialized(), monitor=Band()) is True


def test_gate_force_keeps_in_ram():
    class FakeLazy:
        is_lazy = True
    assert should_stream_analysis(FakeLazy(), force=False) is False


# ── 2. LabelStackWriter round-trip (both backends) ────────────────────
@pytest.mark.parametrize("force_npy", [False, True])
def test_label_stack_writer_roundtrip(tmp_path, monkeypatch, force_npy):
    if force_npy:
        monkeypatch.setattr(storage, "HAS_ZARR", False)
    w = LabelStackWriter(tmp_path / "labels_C0", (4, 5, 6))
    for t in (0, 2, 1):  # out of order; frame 3 left unwritten (background)
        w.write_frame(t, np.full((5, 6), t + 1, dtype=np.int32))
    reader = w.close()
    assert tuple(reader.shape) == (4, 5, 6)
    assert int(np.asarray(reader[1]).max()) == 2
    assert int(np.asarray(reader[3]).max()) == 0  # unwritten → background
    # open_label_stack gives a lazy reader; read_label_stack a full array
    assert int(np.asarray(storage.open_label_stack(w.path)[2]).max()) == 3
    assert storage.read_label_stack(w.path).shape == (4, 5, 6)


# ── 3. pipeline eager vs streamed parity ──────────────────────────────
def test_histogram_pipeline_parity(tmp_path):
    from nd2studios.backend.analysis.histogram_threshold_pipeline import (
        HistogramThresholdPipeline,
    )
    rng = np.random.default_rng(0)
    data = (rng.random((5, 24, 28)) * 1000).astype(np.uint16)
    pipe = HistogramThresholdPipeline()
    params = {p.name: p.default for p in pipe.get_params()}
    params["channel_name"] = "C0"
    meta = {"pixel_size_um": 0.5}

    eager = pipe.run({"C0": data}, meta, dict(params))

    def sink_factory(name, shape):
        return LabelStackWriter(tmp_path / f"labels_{name}", shape)
    sp = dict(params)
    sp["_stream_labels"] = True
    sp["_label_sink_factory"] = sink_factory
    streamed = pipe.run({"C0": _LazySrc(data)}, meta, sp)

    assert np.array_equal(
        np.asarray(eager.label_masks["C0"]),
        np.asarray(streamed.label_masks["C0"]),
    )
    assert eager.measurements == streamed.measurements
    assert eager.summary == streamed.summary


# ── 4. compute_measurements lazy vs eager parity ──────────────────────
def test_compute_measurements_lazy_parity():
    masks = np.zeros((3, 20, 20), np.int32)
    masks[0, 2:6, 2:6] = 1
    masks[0, 10:14, 10:15] = 2
    masks[1, 3:7, 3:7] = 1
    rng = np.random.default_rng(1)
    ch = (rng.random((3, 20, 20)) * 500).astype(np.uint16)
    meta = {"pixel_size_um": 0.65, "z_step_um": 1.0, "n_zslices": 1}

    eager = compute_measurements({"C0": masks}, {"C0": ch}, meta)
    lazy = compute_measurements({"C0": _LazySrc(masks)}, {"C0": _LazySrc(ch)}, meta)
    assert eager == lazy
