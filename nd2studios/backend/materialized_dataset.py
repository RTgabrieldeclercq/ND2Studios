"""
MaterializedDataset — all pixel data resident in RAM, keyed by channel.

Replaces the LazyND2Volume + LazyND2Channel + FrameCache + IOWorker +
PrefetchManager stack used through V1.40 for the viewer hot path. After
the eager parallel materialization at file open, channel access is
direct ndarray indexing — no Dask, no disk, no projection, no cache
misses, no GIL contention on the GUI thread.

Z is collapsed at load time using the user-chosen mode (max / mean /
min); the resulting per-channel shape is (M, T, H, W). The class
preserves the LazyND2Volume API the viewer + downstream pipelines
already speak (get_frame, to_lazy_channel, all_channels_as_lazy,
reopen, shape) so callers do not need to be rewritten — they just get
faster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, OrderedDict as _OD, Tuple

import numpy as np


class _ChannelView(np.ndarray):
    """np.ndarray view with a no-op ``materialize()`` for V1.40 callers.

    Several pages (analysis, results, batch, recipe) historically called
    ``volume.to_lazy_channel(...).materialize()`` because the returned
    object was a :class:`LazyND2Channel` proxy that had to be eagerly
    read before use. After V1.41 the data is already in RAM, so this
    subclass just exposes a ``materialize()`` that hands back the array.
    Subclassing :class:`numpy.ndarray` via ``ndarray.view`` keeps the
    full numpy protocol (indexing, shape, dtype, asarray, ...) intact.
    """

    def materialize(self) -> np.ndarray:
        return np.asarray(self)


def _as_channel_view(arr: np.ndarray) -> "_ChannelView":
    return arr.view(_ChannelView)


@dataclass
class MaterializedDataset:
    """In-RAM dataset; channels keyed by name, each shape (M, T, H, W).

    The MaterializedDataset replaces LazyND2Volume on the viewer hot
    path. Z is already collapsed (the per-channel array has no Z axis).
    ``z_mode`` is recorded for metadata / export; ``get_frame`` ignores
    its z_mode argument since the projection is baked in.

    Attributes
    ----------
    filepath : str
        Path of the source file (or first file of a multi-file set).
    channels : Dict[str, np.ndarray]
        channel_name -> array of shape (M, T, H, W).
    channel_names : List[str]
        Ordered channel names.
    dtype : np.dtype
        Per-pixel dtype.
    pixel_size_um, z_step_um : float
    n_multipoints, n_timepoints, n_channels, height, width : int
    n_zslices : int
        Always 1 after materialization (Z is collapsed).
    z_mode : str
        "max" | "mean" | "min" — projection applied at load.
    """

    filepath: str
    channels: Dict[str, np.ndarray]
    channel_names: List[str]
    dtype: np.dtype
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    n_multipoints: int = 1
    n_timepoints: int = 1
    n_channels: int = 1
    height: int = 0
    width: int = 0
    z_mode: str = "max"
    n_zslices: int = 1  # always 1 post-materialization
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """LazyND2Volume-compatible (M, T, Z, H, W). Z is always 1."""
        return (self.n_multipoints, self.n_timepoints, 1, self.height, self.width)

    # ── frame access ──
    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        """Return a single (H, W) frame.

        The z, z_mode, z_start, z_end arguments are accepted for API
        compatibility with LazyND2Volume but ignored — Z projection
        was applied at materialization time. The mode used at load is
        available as ``self.z_mode``.
        """
        del z, z_mode, z_start, z_end  # ignored — Z already collapsed
        name = self.channel_names[int(c)]
        return self.channels[name][int(m), int(t)]

    def channel_array(self, name: str, m: Optional[int] = None) -> np.ndarray:
        """Return (T, H, W) for one channel at fixed M, or (M, T, H, W) if m is None."""
        arr = self.channels[name]
        return arr if m is None else arr[int(m)]

    # ── LazyND2Volume compatibility shims ──
    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max",
                        z_index: int = 0,
                        z_start: int = 0,
                        z_end: Optional[int] = None,
                        t_start: int = 0,
                        t_end: Optional[int] = None,
                        t_stride: int = 1) -> "_ChannelView":
        """Return a (T, H, W) ndarray view of channel ``c`` at M=m.

        Originally returned a LazyND2Channel proxy. After materialization
        the slice is a zero-copy view of the in-RAM array. We cast it
        to a :class:`_ChannelView` so legacy callers that do
        ``...to_lazy_channel(...).materialize()`` (analysis, results,
        batch, recipe workers) keep working unchanged — the view's
        ``materialize()`` is a no-op that returns the array.
        """
        del z_mode, z_index, z_start, z_end  # ignored — Z collapsed
        name = self.channel_names[int(c)]
        view = self.channels[name][int(m)]                  # (T, H, W)
        if t_end is None:
            t_end = view.shape[0]
        if t_stride > 1 or t_start != 0 or t_end != view.shape[0]:
            view = view[int(t_start):int(t_end):max(1, int(t_stride))]
        return _as_channel_view(view)

    def all_channels_as_lazy(self, m: int = 0,
                             z_mode: str = "max",
                             z_index: int = 0) -> "_OD[str, _ChannelView]":
        """Return an OrderedDict of channel_name -> (T, H, W) view at M=m.

        Each view is a :class:`_ChannelView` (ndarray subclass) so
        legacy callers that expect ``.materialize()`` on the per-channel
        proxy keep working.
        """
        from collections import OrderedDict
        del z_mode, z_index  # ignored
        out: "_OD[str, _ChannelView]" = OrderedDict()
        for name in self.channel_names:
            out[name] = _as_channel_view(self.channels[name][int(m)])
        return out

    def reopen(self) -> "MaterializedDataset":
        """Materialized data is already in RAM; reopen is a no-op identity."""
        return self

    def close(self) -> None:
        """No file handles to release."""
        return None

    def nbytes(self) -> int:
        """Total bytes held in the channels dict."""
        return sum(int(arr.nbytes) for arr in self.channels.values())


__all__ = ["MaterializedDataset"]
