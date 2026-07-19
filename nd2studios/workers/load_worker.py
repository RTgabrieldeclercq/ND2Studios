"""
LoadWorker — open an ND2 (or TIFF) file in a background thread and
eagerly materialize every (M, T) plane into RAM with Z-projection
applied.

V1.41 replaces the LazyND2Volume / LazyND2Channel + IOWorker + cache +
prefetch hot path with a single parallel decode at file open. The
GUI viewer reads from in-RAM ndarrays after this point — slider
scrubbing is pure dict / ndarray indexing, sub-millisecond per tick.

Returns:
    {
        "metadata": dict (extended ND2 metadata or TIFF-derived equivalent),
        "channels": {channel_name: np.ndarray of shape (T, H, W) at fixed M=0},
        "channel_names": list[str],
        "frame_timestamps_s": np.ndarray | None,
        "volume": MaterializedDataset  # exposes the LazyND2Volume API
        "source_type": "nd2" | "nd2_multi" | "tiff" | "tiff_multi",
    }
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Signal

from nd2studios.utils.resource_strategy import (
    LoadStrategy, StrategyDecision, choose_strategy, log_decision,
)
from nd2studios.workers.base_worker import BaseWorker


def _forced_strategy_override() -> "Optional[LoadStrategy]":
    """Read the user's "Load strategy override" preference (Performance dialog).

    Returns the forced :class:`LoadStrategy`, or ``None`` for Auto. This is what
    makes "Force Lazy (cached)" actually take effect — without it the loader
    always materialized eagerly regardless of the setting (V1.44 fix).
    """
    from nd2studios.core.settings import Settings
    val = (getattr(Settings, "FORCED_LOAD_STRATEGY", "") or "").strip()
    if not val:
        return None
    try:
        return LoadStrategy(val)
    except ValueError:
        return None


def _preview_planes() -> int:
    """Deep-Z display preview budget (0 = exact). V1.66; Performance dialog."""
    try:
        from nd2studios.core.settings import Settings
        return max(0, int(getattr(Settings, "DISPLAY_PREVIEW_Z_PLANES", 0) or 0))
    except Exception:  # noqa: BLE001
        return 0


def _meta_like_from_dataset_meta(ext_meta: Dict[str, Any]) -> Any:
    """Adapt the dict-shaped extended metadata into a duck-typed object
    that :func:`choose_strategy` can read footprint fields off of.

    The TIFF / multi-file paths build their metadata as dicts; rather
    than reshape them into :class:`ND2Metadata`, we wrap with
    :class:`types.SimpleNamespace` so ``getattr(meta, ...)`` lookups
    succeed identically.
    """
    import types
    dtype_raw = ext_meta.get("dtype", "uint16")
    if isinstance(dtype_raw, str):
        dtype = np.dtype(dtype_raw)
    else:
        dtype = np.dtype(dtype_raw)
    return types.SimpleNamespace(
        n_multipoints=ext_meta.get("n_multipoints", 1),
        n_timepoints=ext_meta.get("n_timepoints", 1),
        n_zslices=ext_meta.get("n_zslices", 1),
        n_channels=ext_meta.get("n_channels", 1),
        height=ext_meta.get("height", 0),
        width=ext_meta.get("width", 0),
        dtype=dtype,
    )


class LoadWorker(BaseWorker):
    """Open a file (or set of files) and emit channels + rich metadata.

    Accepts either a single ``filepath`` or a list of ``filepaths``.
    When more than one path is supplied the worker takes the V1.28
    multi-file path: it builds a composite lazy volume that chains the
    files along ``chain_axis`` (one of ``T`` / ``M`` / ``Z`` / ``C``).
    The format is inferred from the file extensions — all files in a
    single import must share a format.

    V1.41 emits :attr:`strategy_chosen` after metadata is read so the
    GUI can show which :class:`~nd2studios.utils.resource_strategy.LoadStrategy`
    will run.  The decision is also returned under the ``strategy``
    key on the finished payload so callers that don't connect the
    signal can still inspect it.  Loads that don't fit raise
    :class:`~nd2studios.utils.resource_strategy.StrategyError` which
    surfaces through :attr:`BaseWorker.error` with a clear message
    rather than allowing Python to OOM mid-decode.
    """

    strategy_chosen = Signal(object)

    def __init__(
        self,
        filepath: Optional[str] = None,
        z_projection: str = "max",
        z_start: int = 0,
        z_end: Optional[int] = None,
        t_start: int = 0,
        t_end: Optional[int] = None,
        t_stride: int = 1,
        parent=None,
        filepaths: Optional[List[str]] = None,
        chain_axis: str = "Z",
        chain_mapping: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        if filepaths:
            self.filepaths: List[str] = list(filepaths)
            self.filepath: str = self.filepaths[0]
        elif filepath:
            self.filepaths = [filepath]
            self.filepath = filepath
        else:
            raise ValueError("LoadWorker needs filepath or filepaths")
        self.z_projection = z_projection
        self.z_start = z_start
        self.z_end = z_end
        self.t_start = t_start
        self.t_end = t_end
        self.t_stride = t_stride
        self.chain_axis = chain_axis
        self.chain_mapping = chain_mapping

    def run_task(self) -> Dict[str, Any]:
        if len(self.filepaths) > 1:
            exts = {os.path.splitext(p)[1].lower() for p in self.filepaths}
            if exts == {".nd2"}:
                return self._load_nd2_multi()
            if exts.issubset({".tif", ".tiff"}):
                return self._load_tiff_multi()
            raise ValueError(
                "Multi-file import requires all files to be the same "
                f"format (got extensions: {sorted(exts)})"
            )
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == ".nd2":
            return self._load_nd2()
        elif ext in (".tif", ".tiff"):
            return self._load_tiff()
        raise ValueError(f"Unsupported file extension: {ext}")

    # ── ND2 path ──
    def _load_nd2(self) -> Dict[str, Any]:
        from nd2studios.backend.nd2_loader import (
            read_nd2_metadata, read_or_cache_nd2_metadata,
        )
        from nd2studios.backend.materialized_loader import materialize_nd2

        self.set_status("Reading metadata…")
        meta = read_nd2_metadata(self.filepath)
        # V1.42 — sidecar cache. First open writes a small JSON next
        # to the ND2; subsequent opens skip the multi-second chunk
        # scan when the source file's mtime is unchanged.
        ext_meta = read_or_cache_nd2_metadata(self.filepath)
        self.set_progress(15)

        # V1.41 pre-flight: decide a load strategy from the host RAM
        # budget. Raises StrategyError (→ BaseWorker.error signal) when
        # even the projected footprint can't fit; that surfaces to the
        # user as a modal with actionable wording rather than an OOM.
        # V1.44: honor the user's forced "Load strategy override".
        decision = choose_strategy(meta, self.filepath,
                                   override=_forced_strategy_override())
        log_decision(decision)
        self.strategy_chosen.emit(decision)
        self.set_status(
            f"Plan: {decision.strategy.value} ({decision.estimated_gb:.2f} GB "
            f"into {decision.available_gb:.1f} GB available)"
        )

        n_channels = meta.n_channels
        channel_names: List[str] = list(ext_meta.get("channel_names")
                                        or meta.channel_names
                                        or [f"Ch{i}" for i in range(n_channels)])

        def _on_pct(pct: int) -> None:
            self.set_progress(25 + int(70 * pct / 100))

        if decision.strategy == LoadStrategy.LAZY_CACHED:
            # V1.44: actually load lazily — read frames on demand via the
            # numpy-protocol proxy instead of materializing every plane into
            # RAM. Previously the decision was computed but ignored, so
            # "Force Lazy" still eagerly materialized (and could OOM).
            # V1.66 — stream via StreamingDataset (bounded LRU cache + neighbour
            # prefetch + blocked Z-projection) instead of holding the raw lazy
            # volume, which had no working cache (configure_cache never existed
            # on LazyND2Volume) so every scrub re-decoded through dask.
            from nd2studios.backend.nd2_volume import LazyND2Volume
            from nd2studios.backend.streaming_dataset import StreamingDataset
            self.set_status("Opening volume (streaming, on-demand reads)…")
            lazy = LazyND2Volume(self.filepath)
            budget = int(getattr(decision, "cache_budget_bytes", 0) or 0) \
                or (256 * 1024 * 1024)
            dataset = StreamingDataset.from_volume(
                lazy, cache_budget_bytes=budget, z_mode=self.z_projection,
                preview_planes=_preview_planes(),
                preview_default=_preview_planes() > 0)
            self.set_progress(95)
        else:
            # Eager parallel decode into a single in-RAM dict. After this
            # returns, every (m, t, c) plane is one ndarray index away.
            self.set_status("Materializing volume into RAM…")
            dataset = materialize_nd2(
                self.filepath,
                z_mode=self.z_projection,
                progress_cb=_on_pct,
                cancel_cb=lambda: bool(self.cancelled),
            )
        if self.cancelled:
            return {}

        # Per-channel (T, H, W) views at the currently-selected M=0,
        # for the recipe pipeline which works on a 2D timeseries.
        channels: Dict[str, Any] = dataset.all_channels_as_lazy(m=0)

        ts = ext_meta.get("frame_timestamps_s") or []
        ts_array = np.asarray(ts, dtype=np.float64) if ts else None

        self.set_progress(100)
        self.set_status("Done.")
        return {
            "metadata": ext_meta,
            "channels": channels,
            "channel_names": list(dataset.channel_names),
            "frame_timestamps_s": ts_array,
            "volume": dataset,
            "source_type": "nd2",
            "strategy": decision,
        }

    # ── ND2 multi-file path (V1.28 — chain along T/M/Z/C) ──
    def _load_nd2_multi(self) -> Dict[str, Any]:
        from nd2studios.backend.nd2_loader import (
            read_nd2_metadata_extended_multi,
        )
        from nd2studios.backend.nd2_volume import LazyMultiFileND2Volume
        from nd2studios.backend.materialized_loader import materialize_from_volume

        n_files = len(self.filepaths)
        axis = self.chain_axis
        mapping = self.chain_mapping
        self.set_status(
            f"Combining metadata across {n_files} ND2 files (chain {axis})…"
        )
        ext_meta = read_nd2_metadata_extended_multi(
            self.filepaths, axis, chain_mapping=mapping,
        )
        self.set_progress(15)

        # Pre-flight resource decision before opening the composite
        # volume. The composite's child file handles allocate non-
        # trivial OS resources, so we'd rather refuse early than open
        # them only to discover the data won't fit in RAM.
        meta_adapter = _meta_like_from_dataset_meta(ext_meta)
        decision = choose_strategy(meta_adapter, self.filepaths[0],
                                   override=_forced_strategy_override())
        log_decision(decision)
        self.strategy_chosen.emit(decision)
        self.set_status(
            f"Plan: {decision.strategy.value} "
            f"({decision.estimated_gb:.2f} GB into "
            f"{decision.available_gb:.1f} GB available)"
        )

        self.set_status(f"Opening composite volume (chain {axis})…")
        composite = LazyMultiFileND2Volume(
            self.filepaths, axis, chain_mapping=mapping,
        )
        self.set_progress(25)

        def _on_pct(pct: int) -> None:
            self.set_progress(25 + int(70 * pct / 100))

        if decision.strategy == LoadStrategy.LAZY_CACHED:
            # V1.66 — stream the composite on demand (LazyVolumeSource stacks
            # per-plane get_frame reads across files); keep it open for reads.
            from nd2studios.backend.streaming_dataset import StreamingDataset
            self.set_status(
                f"Opening {n_files}-file composite (streaming, on-demand)…")
            budget = int(getattr(decision, "cache_budget_bytes", 0) or 0) \
                or (256 * 1024 * 1024)
            dataset = StreamingDataset.from_volume(
                composite, cache_budget_bytes=budget, z_mode=self.z_projection,
                preview_planes=_preview_planes(),
                preview_default=_preview_planes() > 0)
        else:
            self.set_status(
                f"Materializing {n_files}-file composite volume into RAM…")
            dataset = materialize_from_volume(
                composite,
                z_mode=self.z_projection,
                progress_cb=_on_pct,
                cancel_cb=lambda: bool(self.cancelled),
            )
            # Composite child handles were only kept open for the decode.
            try:
                close = getattr(composite, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        if self.cancelled:
            return {}

        channels: Dict[str, Any] = dataset.all_channels_as_lazy(m=0)

        ts = ext_meta.get("frame_timestamps_s") or []
        ts_array = np.asarray(ts, dtype=np.float64) if ts else None

        self.set_progress(100)
        self.set_status(f"Done — {n_files} files chained on {axis}.")
        return {
            "metadata": ext_meta,
            "channels": channels,
            "channel_names": list(dataset.channel_names),
            "frame_timestamps_s": ts_array,
            "volume": dataset,
            "source_type": "nd2_multi",
            "strategy": decision,
        }

    # ── TIFF multi-file path (V1.28) ──
    def _load_tiff_multi(self) -> Dict[str, Any]:
        from nd2studios.backend.tiff_loader import (
            LazyMultiFileTIFFVolume, read_tiff_meta_fast,
        )
        from nd2studios.backend.materialized_loader import materialize_from_volume

        n_files = len(self.filepaths)
        axis = self.chain_axis
        self.set_status(
            f"Combining metadata across {n_files} TIFF files (chain {axis})…"
        )
        # File 0 metadata as a base for the import payload.
        sorted_paths = sorted(self.filepaths, key=lambda p: os.path.basename(p))
        f0 = read_tiff_meta_fast(sorted_paths[0])
        del f0  # noqa: F841 — only sanity-probed; metadata comes from composite
        self.set_progress(15)

        self.set_status(f"Opening composite TIFF volume (chain {axis})…")
        composite = LazyMultiFileTIFFVolume(
            self.filepaths, axis, chain_mapping=self.chain_mapping,
        )
        self.set_progress(25)

        # Pre-flight resource decision once the composite reports its
        # dimensions. Same modal-fail-fast behavior as the ND2 paths.
        composite_meta = _meta_like_from_dataset_meta({
            "n_multipoints": getattr(composite, "n_multipoints", 1),
            "n_timepoints": getattr(composite, "n_timepoints", 1),
            "n_zslices": getattr(composite, "n_zslices", 1),
            "n_channels": getattr(composite, "n_channels", 1),
            "height": getattr(composite, "height", 0),
            "width": getattr(composite, "width", 0),
            "dtype": str(getattr(composite, "dtype", "uint16")),
        })
        decision = choose_strategy(composite_meta, self.filepaths[0],
                                   override=_forced_strategy_override())
        log_decision(decision)
        self.strategy_chosen.emit(decision)
        self.set_status(
            f"Plan: {decision.strategy.value} "
            f"({decision.estimated_gb:.2f} GB into "
            f"{decision.available_gb:.1f} GB available)"
        )

        def _on_pct(pct: int) -> None:
            self.set_progress(25 + int(70 * pct / 100))

        if decision.strategy == LoadStrategy.LAZY_CACHED:
            # V1.66 — stream the multi-file TIFF composite on demand.
            from nd2studios.backend.streaming_dataset import StreamingDataset
            self.set_status(
                f"Opening {n_files}-file composite TIFF (streaming)…")
            budget = int(getattr(decision, "cache_budget_bytes", 0) or 0) \
                or (256 * 1024 * 1024)
            dataset = StreamingDataset.from_volume(
                composite, cache_budget_bytes=budget, z_mode=self.z_projection,
                preview_planes=_preview_planes(),
                preview_default=_preview_planes() > 0)
        else:
            self.set_status(
                f"Materializing {n_files}-file composite TIFF into RAM…")
            dataset = materialize_from_volume(
                composite,
                z_mode=self.z_projection,
                progress_cb=_on_pct,
                cancel_cb=lambda: bool(self.cancelled),
            )
            try:
                close = getattr(composite, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        if self.cancelled:
            return {}

        channels: Dict[str, Any] = dataset.all_channels_as_lazy(m=0)
        channel_names = list(dataset.channel_names)

        meta = {
            "filepath": dataset.filepath,
            "source_filepaths": list(self.filepaths),
            "chain_axis": axis,
            "dtype": str(dataset.dtype),
            "height": dataset.height,
            "width": dataset.width,
            "n_timepoints": dataset.n_timepoints,
            "n_channels": dataset.n_channels,
            "n_zslices": dataset.n_zslices,
            "n_multipoints": dataset.n_multipoints,
            "pixel_size_um": dataset.pixel_size_um,
            "z_step_um": dataset.z_step_um,
            "channel_names": channel_names,
            "channel_exposure_ms": [None] * len(channel_names),
            "channel_emission_nm": [None] * len(channel_names),
            "channel_excitation_nm": [None] * len(channel_names),
            "frame_timestamps_s": [],
            "stage_xy_um": [],
            "stage_z_um": [],
            "objective_name": "",
            "objective_magnification": None,
            "objective_na": None,
            "binning_x": None,
            "binning_y": None,
            "camera_name": "",
            "microscope_name": "",
            "loops": [],
        }

        self.set_progress(100)
        self.set_status(f"Done — {n_files} TIFFs chained on {axis}.")
        return {
            "metadata": meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": None,
            "volume": dataset,
            "source_type": "tiff_multi",
            "strategy": decision,
        }

    # ── TIFF path ──
    def _load_tiff(self) -> Dict[str, Any]:
        """Single-file TIFF load — eager parallel materialize into RAM."""
        from nd2studios.backend.tiff_loader import LazyMultiFileTIFFVolume
        from nd2studios.backend.materialized_loader import materialize_from_volume

        self.set_status("Inspecting TIFF…")
        composite = LazyMultiFileTIFFVolume([self.filepath], chain_axis="Z")
        self.set_progress(20)

        # Pre-flight resource decision before materializing.
        composite_meta = _meta_like_from_dataset_meta({
            "n_multipoints": getattr(composite, "n_multipoints", 1),
            "n_timepoints": getattr(composite, "n_timepoints", 1),
            "n_zslices": getattr(composite, "n_zslices", 1),
            "n_channels": getattr(composite, "n_channels", 1),
            "height": getattr(composite, "height", 0),
            "width": getattr(composite, "width", 0),
            "dtype": str(getattr(composite, "dtype", "uint16")),
        })
        decision = choose_strategy(composite_meta, self.filepath,
                                   override=_forced_strategy_override())
        log_decision(decision)
        self.strategy_chosen.emit(decision)
        self.set_status(
            f"Plan: {decision.strategy.value} "
            f"({decision.estimated_gb:.2f} GB into "
            f"{decision.available_gb:.1f} GB available)"
        )

        def _on_pct(pct: int) -> None:
            self.set_progress(20 + int(75 * pct / 100))

        if decision.strategy == LoadStrategy.LAZY_CACHED:
            # V1.66 — stream a large TIFF via a memory-map (opens instantly,
            # only touched frames are read) instead of decoding every plane
            # into RAM. Uncompressed/contiguous (Big)TIFFs mmap directly;
            # compressed/tiled fall back to per-page reads inside the source.
            from nd2studios.backend.streaming_dataset import StreamingDataset
            self.set_status("Opening TIFF (streaming, memory-mapped)…")
            budget = int(getattr(decision, "cache_budget_bytes", 0) or 0) \
                or (256 * 1024 * 1024)
            dataset = StreamingDataset.from_tiff(
                self.filepath,
                channel_names=list(getattr(composite, "channel_names", []) or []),
                pixel_size_um=float(getattr(composite, "pixel_size_um", 1.0)),
                z_step_um=float(getattr(composite, "z_step_um", 1.0)),
                cache_budget_bytes=budget,
                z_mode=self.z_projection,
                preview_planes=_preview_planes(),
                preview_default=_preview_planes() > 0,
            )
            self.set_progress(95)
        else:
            self.set_status("Materializing TIFF into RAM…")
            dataset = materialize_from_volume(
                composite,
                z_mode=self.z_projection,
                progress_cb=_on_pct,
                cancel_cb=lambda: bool(self.cancelled),
            )
        try:
            close = getattr(composite, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        if self.cancelled:
            return {}

        channels: Dict[str, Any] = dataset.all_channels_as_lazy(m=0)
        channel_names = list(dataset.channel_names)

        meta = {
            "filepath": dataset.filepath,
            "dtype": str(dataset.dtype),
            "height": dataset.height,
            "width": dataset.width,
            "n_timepoints": dataset.n_timepoints,
            "n_channels": dataset.n_channels,
            "n_zslices": dataset.n_zslices,
            "n_multipoints": dataset.n_multipoints,
            "pixel_size_um": dataset.pixel_size_um,
            "z_step_um": dataset.z_step_um,
            "channel_names": channel_names,
            "channel_exposure_ms": [None] * len(channel_names),
            "channel_emission_nm": [None] * len(channel_names),
            "channel_excitation_nm": [None] * len(channel_names),
            "frame_timestamps_s": [],
            "stage_xy_um": [],
            "stage_z_um": [],
            "objective_name": "",
            "objective_magnification": None,
            "objective_na": None,
            "binning_x": None,
            "binning_y": None,
            "camera_name": "",
            "microscope_name": "",
            "loops": [],
        }
        self.set_progress(100)
        self.set_status("Done.")
        return {
            "metadata": meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": None,
            "volume": dataset,
            "source_type": "tiff",
            "strategy": decision,
        }
