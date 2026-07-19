"""StreamingDataset — NIS-Elements-style on-demand loader (V1.66).

The V1.41 ``MaterializedDataset`` reads the *entire* ``(M,T,Z,H,W)`` volume into
RAM at file open. On a large file (e.g. a 56 GB BigTIFF) that means minutes of
decode and tens of GB of RAM before a single pixel appears — the GUI freezes.

``StreamingDataset`` mirrors how Nikon NIS-Elements opens large images: the file
is **memory-mapped / opened lazily** (metadata only — milliseconds, a few MB),
frames are **streamed on demand**, and a bounded **LRU cache + neighbour
prefetch** keep scrubbing smooth without ever materializing the whole volume.
Deep Z-projections are computed in **Z-blocks** so transient RAM stays bounded,
and the display path can use a fast **Z-subsampled preview** for an instant
first frame (the recipe / export / DVC paths always read exact).

It implements the exact public surface the viewer/pipeline already speak
(``get_frame`` / ``get_volume`` / ``to_lazy_channel`` / ``all_channels_as_lazy`` /
``channel_array`` / ``subset`` / ``reopen`` / ``close`` / ``shape`` / ``n_*`` /
``channel_names`` / ``pixel_size_um`` / ``z_step_um`` / ``dtype`` / ``z_mode``),
so it is a drop-in alternate for ``MaterializedDataset`` — no viewer changes.

Two read sources are provided:

* :class:`TiffMemmapSource` — ``tifffile.memmap`` for an uncompressed/contiguous
  (Big)TIFF (true OS-paged mmap; the fast path for the McGhee Lab's TFM stacks),
  with a per-page fallback for compressed/tiled TIFFs.
* :class:`LazyVolumeSource` — wraps any existing lazy volume exposing
  ``get_volume`` *or* ``get_frame`` (``LazyND2Volume`` /
  ``LazyMultiFileND2Volume`` / ``LazyMultiFileTIFFVolume`` / TIFF views), so ND2
  and multi-file imports stream through the same cache.

Backend-pure: no Qt imports (``tifffile`` / ``numpy`` only).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_VALID_Z_MODES = ("max", "mean", "min", "none")

# Default resident cache budget if the caller does not supply one (256 MB).
_DEFAULT_CACHE_BYTES = 256 * 1024 * 1024

# Hard cap on the resident frame cache regardless of the requested budget:
# recommended_cache_budget_bytes() can return tens of GB on a big-RAM host, and
# a cache that large defeats streaming's low-RAM benefit (and risks OOM).
_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024

# Target transient RAM for a deep Z-projection (read the stack in blocks).
_PROJECTION_BLOCK_BYTES = 256 * 1024 * 1024


# ── bounded LRU frame cache ──────────────────────────────────────────────────
class _LRUByteCache:
    """Thread-safe, byte-bounded LRU cache of decoded ``(H,W)`` frames."""

    def __init__(self, max_bytes: int):
        self._max = max(1, int(max_bytes))
        self._d: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key) -> Optional[np.ndarray]:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        n = int(arr.nbytes)
        with self._lock:
            if key in self._d:
                self._bytes -= int(self._d[key].nbytes)
                del self._d[key]
            self._d[key] = arr
            self._bytes += n
            while self._bytes > self._max and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                self._bytes -= int(old.nbytes)

    def contains(self, key) -> bool:
        with self._lock:
            return key in self._d

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def resident_bytes(self) -> int:
        with self._lock:
            return self._bytes


# ── read sources ─────────────────────────────────────────────────────────────
class TiffMemmapSource:
    """Read planes from a (Big)TIFF via ``tifffile.memmap`` (OS-paged).

    Parses the series axis order (e.g. ``TZCYX``) so ``read_volume`` can slice
    the right Z range for a ``(channel, multipoint, timepoint)``. Falls back to
    per-page ``asarray`` reads when the file is compressed/tiled (not memmappable).
    """

    def __init__(self, filepath: str,
                 channel_names: Optional[List[str]] = None,
                 pixel_size_um: float = 1.0,
                 z_step_um: float = 1.0):
        import tifffile

        self.filepath = str(filepath)
        self._tf = tifffile.TiffFile(self.filepath)
        series = self._tf.series[0]
        self._axes = str(series.axes).upper()
        self._shape = tuple(int(s) for s in series.shape)
        self.dtype = np.dtype(series.dtype)

        def _extent(letter: Optional[str], default: int = 1) -> int:
            if letter and letter in self._axes:
                return self._shape[self._axes.index(letter)]
            return default

        self.height = _extent("Y", 0)
        self.width = _extent("X", 0)
        self.n_zslices = _extent("Z", 1)
        self.n_channels = _extent("C", 1)
        self.n_timepoints = _extent("T", 1)
        # Multipoint: any leading axis that is not one of T/Z/C/Y/X/S (rare in TIFF).
        self._m_axis = next(
            (ax for ax in self._axes if ax not in ("T", "Z", "C", "Y", "X", "S")),
            None,
        )
        self.n_multipoints = _extent(self._m_axis, 1)

        self.channel_names = list(channel_names or []) or [
            f"Ch{i}" for i in range(self.n_channels)
        ]
        if len(self.channel_names) < self.n_channels:
            self.channel_names.extend(
                f"Ch{i}" for i in range(len(self.channel_names), self.n_channels))
        self.pixel_size_um = float(pixel_size_um)
        self.z_step_um = float(z_step_um)

        # Try a true memory-map (fast path). On failure keep the page reader.
        self._mm = None
        try:
            self._mm = tifffile.memmap(self.filepath, mode="r")
        except Exception:  # noqa: BLE001 — compressed/tiled → per-page fallback
            self._mm = None

        # Per-page fallback bookkeeping: non-spatial axes and their extents.
        self._nonspatial = [ax for ax in self._axes if ax not in ("Y", "X")]
        self._nonspatial_shape = [
            self._shape[self._axes.index(ax)] for ax in self._nonspatial
        ]

    def read_volume(self, c: int, m: int, t: int, z0: int, z1: int) -> np.ndarray:
        """Return ``(Z', H, W)`` for ``[z0, z1)`` at ``(c, m, t)``."""
        z0 = int(z0)
        z1 = max(z0 + 1, int(z1))
        if self._mm is not None:
            idx: List[Any] = []
            for ax in self._axes:
                if ax in ("Y", "X"):
                    idx.append(slice(None))
                elif ax == "Z":
                    idx.append(slice(z0, z1))
                elif ax == "T":
                    idx.append(int(t))
                elif ax == "C":
                    idx.append(int(c))
                elif ax == self._m_axis:
                    idx.append(int(m))
                else:
                    idx.append(0)
            block = np.asarray(self._mm[tuple(idx)])
        else:
            block = self._read_pages(c, m, t, z0, z1)
        if "Z" not in self._axes:
            block = block[None, ...]
        if block.ndim == 2:
            block = block[None, ...]
        elif block.ndim > 3:
            block = block.reshape((-1, self.height, self.width))
        return block

    def _read_pages(self, c: int, m: int, t: int, z0: int, z1: int) -> np.ndarray:
        coord = {"T": int(t), "C": int(c), "Z": 0}
        if self._m_axis:
            coord[self._m_axis] = int(m)
        planes = []
        for zi in range(z0, z1):
            coord["Z"] = zi
            multi = [coord.get(ax, 0) for ax in self._nonspatial]
            page = int(np.ravel_multi_index(multi, self._nonspatial_shape)) \
                if self._nonspatial else 0
            planes.append(np.asarray(self._tf.pages[page].asarray()))
        return np.stack(planes, axis=0) if planes else np.empty(
            (0, self.height, self.width), self.dtype)

    def reopen(self) -> "TiffMemmapSource":
        return TiffMemmapSource(self.filepath, self.channel_names,
                                self.pixel_size_um, self.z_step_um)

    def close(self) -> None:
        self._mm = None
        try:
            self._tf.close()
        except Exception:  # noqa: BLE001
            pass


class LazyVolumeSource:
    """Wrap any lazy volume exposing ``get_volume`` or ``get_frame``.

    Single-file lazy volumes (``LazyND2Volume``) expose ``get_volume``; the
    multi-file composites (``LazyMultiFileND2Volume`` /
    ``LazyMultiFileTIFFVolume``) expose only ``get_frame`` (they route each plane
    to its owning file). ``read_volume`` uses ``get_volume`` when available and
    otherwise stacks per-plane ``get_frame`` reads, so both stream identically.
    """

    def __init__(self, volume: Any):
        self._volume = volume
        self.filepath = str(getattr(volume, "filepath", "") or "")
        self.dtype = np.dtype(getattr(volume, "dtype", np.uint16))
        self.height = int(getattr(volume, "height", 0))
        self.width = int(getattr(volume, "width", 0))
        self.n_zslices = int(getattr(volume, "n_zslices", 1))
        self.n_channels = int(getattr(volume, "n_channels", 1))
        self.n_timepoints = int(getattr(volume, "n_timepoints", 1))
        self.n_multipoints = int(getattr(volume, "n_multipoints", 1))
        self.channel_names = list(getattr(volume, "channel_names", []) or [])
        if len(self.channel_names) < self.n_channels:
            self.channel_names.extend(
                f"Ch{i}" for i in range(len(self.channel_names), self.n_channels))
        self.pixel_size_um = float(getattr(volume, "pixel_size_um", 1.0))
        self.z_step_um = float(getattr(volume, "z_step_um", 1.0))
        self._has_get_volume = callable(getattr(volume, "get_volume", None))

    def read_volume(self, c: int, m: int, t: int, z0: int, z1: int) -> np.ndarray:
        if self._has_get_volume:
            block = np.asarray(self._volume.get_volume(
                int(c), m=int(m), t=int(t), z_start=int(z0), z_end=int(z1)))
        else:
            # Multi-file composites expose get_frame but not get_volume — stack
            # the per-plane reads so they stream through the same machinery.
            planes = [
                np.asarray(self._volume.get_frame(
                    c=int(c), m=int(m), t=int(t), z=int(z), z_mode="none"))
                for z in range(int(z0), int(z1))
            ]
            block = (np.stack(planes, axis=0) if planes
                     else np.empty((0, self.height, self.width), self.dtype))
        if block.ndim == 2:
            block = block[None, ...]
        return block

    def reopen(self) -> "LazyVolumeSource":
        reopen = getattr(self._volume, "reopen", None)
        return LazyVolumeSource(reopen() if callable(reopen) else self._volume)

    def close(self) -> None:
        close = getattr(self._volume, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


# ── prefetch ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _FrameReq:
    c: int
    m: int
    t: int
    z: int
    z_mode: str
    z0: int
    z1: int
    key: tuple
    preview: bool = False


class _Prefetcher:
    """Single background thread that warms the cache with neighbour frames.

    Uses its own ``reopen()``'d source (nd2/tifffile handles are not thread-safe;
    a numpy memmap is, but reopening is cheap and uniformly safe). Latest requests
    win; already-cached / in-flight keys are skipped so the queue stays bounded.
    """

    def __init__(self, dataset: "StreamingDataset"):
        self._ds = dataset
        self._ex = ThreadPoolExecutor(max_workers=1,
                                      thread_name_prefix="stream-prefetch")
        self._src = None
        self._inflight: set = set()
        self._lock = threading.Lock()
        self._stopped = False

    def submit(self, req: _FrameReq) -> None:
        if self._stopped:
            return
        with self._lock:
            if req.key in self._inflight or self._ds._cache.contains(req.key):
                return
            self._inflight.add(req.key)
        try:
            self._ex.submit(self._run, req)
        except RuntimeError:  # pool shutting down
            with self._lock:
                self._inflight.discard(req.key)

    def _run(self, req: _FrameReq) -> None:
        try:
            if self._src is None:
                self._src = self._ds._source.reopen()
            frame = self._ds._read_key(self._src, req)
            self._ds._cache.put(req.key, frame)
        except Exception:  # noqa: BLE001 — prefetch must never raise
            pass
        finally:
            with self._lock:
                self._inflight.discard(req.key)

    def stop(self) -> None:
        self._stopped = True
        self._ex.shutdown(wait=False)
        if self._src is not None:
            self._src.close()
            self._src = None


# ── the dataset ──────────────────────────────────────────────────────────────
class StreamingDataset:
    """MaterializedDataset-compatible, streaming + cached (never fully resident)."""

    is_lazy = True  # routes analysis/results to the streaming path (V1.46)

    def __init__(self, source: Any,
                 cache_budget_bytes: int = _DEFAULT_CACHE_BYTES,
                 prefetch: bool = True,
                 z_mode: str = "max",
                 preview_planes: int = 0,
                 preview_default: bool = False):
        self._source = source
        self.filepath = str(getattr(source, "filepath", "") or "")
        self.dtype = np.dtype(getattr(source, "dtype", np.uint16))
        self.height = int(getattr(source, "height", 0))
        self.width = int(getattr(source, "width", 0))
        self.n_zslices = int(getattr(source, "n_zslices", 1))
        self.n_channels = int(getattr(source, "n_channels", 1))
        self.n_timepoints = int(getattr(source, "n_timepoints", 1))
        self.n_multipoints = int(getattr(source, "n_multipoints", 1))
        self.channel_names = list(getattr(source, "channel_names", []) or [])
        self.pixel_size_um = float(getattr(source, "pixel_size_um", 1.0))
        self.z_step_um = float(getattr(source, "z_step_um", 1.0))
        self.z_mode = z_mode

        # Blocked-projection budget: cap transient RAM for a deep max/mean/min
        # projection by reading the Z stack in blocks of this many planes and
        # reducing incrementally (never holds the full stack in heap).
        frame_bytes = max(1, self.height * self.width * int(self.dtype.itemsize))
        self._z_block = max(1, _PROJECTION_BLOCK_BYTES // frame_bytes)
        self._prefetch_max_z = self._z_block   # don't background deep projections

        # Deep-Z fast display: when enabled, get_frame (the viewer path) projects
        # a bounded number of evenly-spaced Z planes for an instant first frame;
        # to_lazy_channel / get_volume stay exact (recipe / export / DVC).
        self._preview_planes = max(0, int(preview_planes))
        self._preview_default = bool(preview_default)

        self._cache = _LRUByteCache(min(int(cache_budget_bytes), _MAX_CACHE_BYTES))
        self._prefetcher = _Prefetcher(self) if prefetch else None

    # -- construction helpers --
    @classmethod
    def from_tiff(cls, filepath: str, *,
                  channel_names: Optional[List[str]] = None,
                  pixel_size_um: float = 1.0, z_step_um: float = 1.0,
                  cache_budget_bytes: int = _DEFAULT_CACHE_BYTES,
                  prefetch: bool = True, z_mode: str = "max",
                  preview_planes: int = 0,
                  preview_default: bool = False) -> "StreamingDataset":
        src = TiffMemmapSource(filepath, channel_names, pixel_size_um, z_step_um)
        return cls(src, cache_budget_bytes=cache_budget_bytes,
                   prefetch=prefetch, z_mode=z_mode,
                   preview_planes=preview_planes, preview_default=preview_default)

    @classmethod
    def from_volume(cls, volume: Any, *,
                    cache_budget_bytes: int = _DEFAULT_CACHE_BYTES,
                    prefetch: bool = True, z_mode: str = "max",
                    preview_planes: int = 0,
                    preview_default: bool = False) -> "StreamingDataset":
        return cls(LazyVolumeSource(volume), cache_budget_bytes=cache_budget_bytes,
                   prefetch=prefetch, z_mode=z_mode,
                   preview_planes=preview_planes, preview_default=preview_default)

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        return (self.n_multipoints, self.n_timepoints, self.n_zslices,
                self.height, self.width)

    # -- key + read helpers --
    def _make_req(self, c, m, t, z, z_mode, z_start, z_end, preview) -> _FrameReq:
        c, m, t = int(c), int(m), int(t)
        if z_mode == "none" or self.n_zslices <= 1:
            zi = max(0, min(int(z), self.n_zslices - 1))
            key = (c, m, t, "none", zi)
            return _FrameReq(c, m, t, zi, "none", zi, zi + 1, key, False)
        z0 = 0 if z_start is None else max(0, int(z_start))
        z1 = self.n_zslices if z_end is None else min(int(z_end), self.n_zslices)
        z1 = max(z0 + 1, z1)
        use_preview = (bool(preview) and self._preview_planes > 0
                       and (z1 - z0) > self._preview_planes)
        key = (c, m, t, z_mode, z0, z1, use_preview)
        return _FrameReq(c, m, t, int(z), z_mode, z0, z1, key, use_preview)

    def _read_key(self, source: Any, req: _FrameReq) -> np.ndarray:
        return self._read_projection(source, req.c, req.m, req.t,
                                     req.z0, req.z1, req.z_mode, req.preview)

    def _read_projection(self, source: Any, c: int, m: int, t: int,
                         z0: int, z1: int, z_mode: str,
                         preview: bool = False) -> np.ndarray:
        """Read + project ``[z0, z1)`` → a RAM-resident ``(H, W)``.

        Exact by default (read in Z-blocks so transient RAM stays bounded — a
        deep projection never materializes the whole stack). When ``preview`` is
        set, a deep projection is approximated over at most ``self._preview_planes``
        evenly-spaced Z planes for an instant display frame; the recipe / export /
        DVC paths never pass ``preview``, so their data is always exact."""
        if z_mode == "none" or (z1 - z0) <= 1:
            block = source.read_volume(c, m, t, z0, z0 + 1)
            return np.array(block[0], copy=True)   # force off the memmap

        if preview and self._preview_planes > 0 and (z1 - z0) > self._preview_planes:
            z_list = np.unique(np.linspace(z0, z1 - 1, self._preview_planes).astype(int))
            acc = None
            count = 0
            for z in z_list:
                plane = np.asarray(source.read_volume(c, m, t, int(z), int(z) + 1))[0]
                if z_mode == "max":
                    acc = plane.copy() if acc is None else np.maximum(acc, plane)
                elif z_mode == "min":
                    acc = plane.copy() if acc is None else np.minimum(acc, plane)
                else:  # mean
                    p = plane.astype(np.float64)
                    acc = p if acc is None else acc + p
                    count += 1
            if z_mode == "mean":
                acc = (acc / max(1, count)).astype(self.dtype)
            return np.ascontiguousarray(acc)

        acc = None
        count = 0
        for zs in range(z0, z1, self._z_block):
            ze = min(zs + self._z_block, z1)
            chunk = np.asarray(source.read_volume(c, m, t, zs, ze))
            if z_mode == "max":
                part = chunk.max(axis=0)
                acc = part.copy() if acc is None else np.maximum(acc, part)
            elif z_mode == "min":
                part = chunk.min(axis=0)
                acc = part.copy() if acc is None else np.minimum(acc, part)
            else:  # mean
                part = chunk.sum(axis=0, dtype=np.float64)
                acc = part if acc is None else acc + part
                count += (ze - zs)
        if z_mode == "mean":
            acc = (acc / max(1, count)).astype(self.dtype)
        return np.ascontiguousarray(acc)

    def _prefetch_neighbours(self, req: _FrameReq) -> None:
        if self._prefetcher is None:
            return
        # Don't warm exact deep projections in the background — too large. A
        # bounded preview projection is cheap enough to prefetch.
        if (not req.preview and req.z_mode != "none"
                and (req.z1 - req.z0) > self._prefetch_max_z):
            return
        for dt in (1, -1):
            tt = req.t + dt
            if 0 <= tt < self.n_timepoints:
                self._prefetcher.submit(_make_neighbour(req, tt))

    # -- frame access (MaterializedDataset API) --
    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None,
                  preview: Optional[bool] = None) -> np.ndarray:
        if z_mode not in _VALID_Z_MODES:
            raise ValueError(f"unknown z_mode {z_mode!r}")
        eff_preview = self._preview_default if preview is None else bool(preview)
        req = self._make_req(c, m, t, z, z_mode, z_start, z_end, eff_preview)
        hit = self._cache.get(req.key)
        if hit is not None:
            self._prefetch_neighbours(req)
            return hit
        frame = self._read_key(self._source, req)
        self._cache.put(req.key, frame)
        self._prefetch_neighbours(req)
        return frame

    def get_volume(self, c: int, m: int = 0, t: int = 0,
                   z_start: Optional[int] = None,
                   z_end: Optional[int] = None) -> np.ndarray:
        z0 = 0 if z_start is None else max(0, int(z_start))
        z1 = self.n_zslices if z_end is None else min(int(z_end), self.n_zslices)
        z1 = max(z0 + 1, z1)
        return np.asarray(self._source.read_volume(int(c), int(m), int(t), z0, z1))

    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max", z_index: int = 0,
                        z_start: int = 0, z_end: Optional[int] = None,
                        t_start: int = 0, t_end: Optional[int] = None,
                        t_stride: int = 1) -> "_StreamChannel":
        t_end = self.n_timepoints if t_end is None else int(t_end)
        z_end = self.n_zslices if z_end is None else int(z_end)
        return _StreamChannel(self, int(c), int(m), z_mode, int(z_index),
                              int(z_start), int(z_end),
                              int(t_start), int(t_end), max(1, int(t_stride)))

    def all_channels_as_lazy(self, m: int = 0, z_mode: str = "max",
                             z_index: int = 0) -> "OrderedDict[str, _StreamChannel]":
        out: "OrderedDict[str, _StreamChannel]" = OrderedDict()
        for c, name in enumerate(self.channel_names):
            out[name] = self.to_lazy_channel(c, m=m, z_mode=z_mode, z_index=z_index)
        return out

    def channel_array(self, name: str, m: Optional[int] = None) -> np.ndarray:
        """Stream one channel into ``(T,Z,H,W)`` (m given) or ``(M,T,Z,H,W)``.

        Deliberate materialization — bounded to the requested slab. Prefer
        ``to_lazy_channel`` / ``get_frame`` on the hot path.
        """
        c = self.channel_names.index(name)
        m_range = [int(m)] if m is not None else list(range(self.n_multipoints))
        out = np.empty((len(m_range), self.n_timepoints, self.n_zslices,
                        self.height, self.width), dtype=self.dtype)
        for mi, mm in enumerate(m_range):
            for t in range(self.n_timepoints):
                out[mi, t] = self._source.read_volume(c, mm, t, 0, self.n_zslices)
        return out[0] if m is not None else out

    def subset(self, *, m=None, t=None, z=None, progress_cb=None):
        """Materialize the selected M/T/Z indices into a MaterializedDataset."""
        from nd2studios.backend.materialized_dataset import MaterializedDataset

        m_idx = _sel(m, self.n_multipoints)
        t_idx = _sel(t, self.n_timepoints)
        z_idx = _sel(z, self.n_zslices)
        channels: Dict[str, np.ndarray] = {}
        n = max(1, self.n_channels)
        for ci, name in enumerate(self.channel_names):
            arr = np.empty((len(m_idx), len(t_idx), len(z_idx),
                            self.height, self.width), dtype=self.dtype)
            for mi, mm in enumerate(m_idx):
                for ti, tt in enumerate(t_idx):
                    full = self._source.read_volume(ci, mm, tt, 0, self.n_zslices)
                    arr[mi, ti] = full[z_idx]
            channels[name] = arr
            if progress_cb is not None:
                progress_cb(int((ci + 1) / n * 100))
        return MaterializedDataset(
            filepath=self.filepath, channels=channels,
            channel_names=list(self.channel_names), dtype=self.dtype,
            pixel_size_um=self.pixel_size_um, z_step_um=self.z_step_um,
            n_multipoints=len(m_idx), n_timepoints=len(t_idx),
            n_channels=self.n_channels, height=self.height, width=self.width,
            z_mode=self.z_mode, n_zslices=len(z_idx),
            extra={"streamed_subset_from": self.filepath},
        )

    def reopen(self) -> "StreamingDataset":
        return StreamingDataset(self._source.reopen(),
                                cache_budget_bytes=self._cache._max,
                                prefetch=self._prefetcher is not None,
                                z_mode=self.z_mode,
                                preview_planes=self._preview_planes,
                                preview_default=self._preview_default)

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.stop()
        self._cache.clear()
        try:
            self._source.close()
        except Exception:  # noqa: BLE001
            pass

    def nbytes(self) -> int:
        """Resident bytes (cache only) — NOT the full volume footprint."""
        return self._cache.resident_bytes()


def _make_neighbour(req: _FrameReq, tt: int) -> _FrameReq:
    if req.z_mode == "none":
        key = (req.c, req.m, tt, "none", req.z)
    else:
        key = (req.c, req.m, tt, req.z_mode, req.z0, req.z1, req.preview)
    return _FrameReq(req.c, req.m, tt, req.z, req.z_mode,
                     req.z0, req.z1, key, req.preview)


def _sel(indices, n: int) -> List[int]:
    if indices is None:
        return list(range(n))
    out = sorted({int(i) for i in indices if 0 <= int(i) < n})
    return out or list(range(n))


class _StreamChannel:
    """``(T,H,W)`` lazy proxy over a :class:`StreamingDataset` channel.

    Mirrors ``MaterializedDataset``'s ``_ChannelView`` / ``MultiFileLazyChannel``
    numpy-protocol surface (``__getitem__`` / ``__array__`` / ``materialize`` /
    ``crop`` / ``shape`` / ``dtype``) so the recipe & export pipelines stay
    format-agnostic. Frames are pulled through the parent's cache and are always
    **exact** (``preview=False``) — the recipe/export must never approximate Z.
    """

    def __init__(self, ds: StreamingDataset, c: int, m: int, z_mode: str,
                 z_index: int, z_start: int, z_end: int,
                 t_start: int, t_end: int, t_stride: int):
        self._ds = ds
        self._c = c
        self._m = m
        self._z_mode = z_mode
        self._z_index = z_index
        self._z_start = z_start
        self._z_end = z_end
        self._t0 = t_start
        self._t1 = t_end
        self._t_stride = t_stride
        self._crop: Optional[Tuple[int, int, int, int]] = None
        self._n = max(0, (t_end - t_start + t_stride - 1) // t_stride)
        self.dtype = ds.dtype
        self.ndim = 3

    @property
    def shape(self) -> Tuple[int, int, int]:
        if self._crop is not None:
            y0, y1, x0, x1 = self._crop
            return (self._n, y1 - y0, x1 - x0)
        return (self._n, self._ds.height, self._ds.width)

    def __len__(self) -> int:
        return self._n

    def _read(self, t_local: int) -> np.ndarray:
        t_abs = self._t0 + int(t_local) * self._t_stride
        frame = self._ds.get_frame(
            self._c, m=self._m, t=t_abs, z=self._z_index,
            z_mode=self._z_mode, z_start=self._z_start, z_end=self._z_end,
            preview=False)
        if self._crop is not None:
            y0, y1, x0, x1 = self._crop
            frame = frame[y0:y1, x0:x1]
        return frame

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key = key[0]
        spatial = key[1:] if len(key) > 1 else ()
        if isinstance(t_key, (int, np.integer)):
            fr = self._read(int(t_key))
            return fr[spatial] if spatial else fr
        if isinstance(t_key, slice):
            idxs = range(*t_key.indices(self._n))
        else:
            idxs = list(t_key)
        frames = [self._read(int(i)) for i in idxs]
        if not frames:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        stacked = np.stack(frames, axis=0)
        return stacked[(slice(None),) + spatial] if spatial else stacked

    def __array__(self, dtype=None) -> np.ndarray:
        full = np.stack([self._read(t) for t in range(self._n)], axis=0)
        return full.astype(dtype) if dtype is not None else full

    def materialize(self) -> np.ndarray:
        return np.asarray(self)

    def copy(self) -> "_StreamChannel":
        return self

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> "_StreamChannel":
        view = _StreamChannel(self._ds, self._c, self._m, self._z_mode,
                              self._z_index, self._z_start, self._z_end,
                              self._t0, self._t1, self._t_stride)
        view._crop = (int(y0), int(y1), int(x0), int(x1))
        return view

    def close(self) -> None:
        pass


__all__ = [
    "StreamingDataset",
    "TiffMemmapSource",
    "LazyVolumeSource",
]
