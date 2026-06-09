"""
MaterializedDataset — all pixel data resident in RAM, keyed by channel.

Replaces the LazyND2Volume + LazyND2Channel + FrameCache + IOWorker +
PrefetchManager stack used through V1.40 for the viewer hot path. After
the eager parallel materialization at file open, channel access is
direct ndarray indexing — no Dask, no disk, no cache misses, no GIL
contention on the GUI thread.

Each channel is stored as ``(M, T, Z, H, W)`` so the user can switch Z
projection modes (max / mean / min / none) dynamically without reloading.
``get_frame`` and ``to_lazy_channel`` project or slice the Z axis on the
fly from in-RAM data — still sub-millisecond since no disk I/O is needed.

The class preserves the LazyND2Volume API the viewer + downstream
pipelines already speak (get_frame, to_lazy_channel, all_channels_as_lazy,
reopen, shape) so callers do not need to be rewritten.
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
    """In-RAM dataset; channels keyed by name, each shape (M, T, Z, H, W).

    The MaterializedDataset replaces LazyND2Volume on the viewer hot
    path. Z is stored in full so the user can switch projection modes
    (max / mean / min / none) dynamically without reloading. All
    projection and Z-slice selection happens on the fly from in-RAM
    data — no disk reads after the initial load.

    ``z_mode`` records the projection mode that was active when the file
    was loaded (used as the initial display default). The actual mode
    applied to any given frame access is determined by the caller via
    the ``z_mode`` argument to ``get_frame`` / ``to_lazy_channel``.

    Attributes
    ----------
    filepath : str
        Path of the source file (or first file of a multi-file set).
    channels : Dict[str, np.ndarray]
        channel_name -> array of shape (M, T, Z, H, W).
    channel_names : List[str]
        Ordered channel names.
    dtype : np.dtype
        Per-pixel dtype.
    pixel_size_um, z_step_um : float
    n_multipoints, n_timepoints, n_channels, n_zslices, height, width : int
    z_mode : str
        Projection mode active at load time ("max" | "mean" | "min" | "none").
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
    n_zslices: int = 1
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """LazyND2Volume-compatible (M, T, Z, H, W)."""
        return (self.n_multipoints, self.n_timepoints, self.n_zslices,
                self.height, self.width)

    # ── frame access ──
    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        """Return a single (H, W) frame.

        Parameters
        ----------
        c, m, t : int — channel / multipoint / timepoint indices.
        z : int — Z slice index (used when ``z_mode == 'none'``).
        z_mode : 'none' | 'max' | 'mean' | 'min' — Z handling.
        z_start, z_end : int — optional range for projection.
        """
        name = self.channel_names[int(c)]
        arr = self.channels[name][int(m), int(t)]  # (Z, H, W)
        n_z = arr.shape[0]

        if z_mode == "none" or n_z <= 1:
            zi = max(0, min(int(z), n_z - 1))
            return arr[zi]  # (H, W)

        z_s = int(z_start) if z_start is not None else 0
        z_e = int(z_end) if z_end is not None else n_z
        z_s = max(0, min(z_s, n_z))
        z_e = max(z_s, min(z_e, n_z))
        stack = arr[z_s:z_e]  # (Z', H, W)
        if z_mode == "max":
            return stack.max(axis=0)
        if z_mode == "min":
            return stack.min(axis=0)
        # mean
        return stack.mean(axis=0).astype(self.dtype)

    def channel_array(self, name: str, m: Optional[int] = None) -> np.ndarray:
        """Return (T, Z, H, W) for one channel at fixed M, or (M, T, Z, H, W) if m is None."""
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

        Z projection or slice selection is applied on the fly from the
        in-RAM ``(M, T, Z, H, W)`` array. Legacy callers that call
        ``.materialize()`` on the result keep working unchanged — the
        view's ``materialize()`` is a no-op that returns the array.
        """
        name = self.channel_names[int(c)]
        vol = self.channels[name][int(m)]   # (T, Z, H, W)
        n_z = vol.shape[1]

        if t_end is None:
            t_end = vol.shape[0]
        if t_stride > 1 or t_start != 0 or t_end != vol.shape[0]:
            vol = vol[int(t_start):int(t_end):max(1, int(t_stride))]  # (T', Z, H, W)

        if n_z <= 1:
            result = vol[:, 0]  # (T, H, W)
        elif z_mode == "none":
            z_i = max(0, min(int(z_index), n_z - 1))
            result = vol[:, z_i]  # (T, H, W)
        else:
            z_e = int(z_end) if z_end is not None else n_z
            z_s = int(z_start)
            stack = vol[:, z_s:z_e]   # (T, Z', H, W)
            if z_mode == "max":
                result = stack.max(axis=1)
            elif z_mode == "min":
                result = stack.min(axis=1)
            else:  # mean
                result = stack.mean(axis=1).astype(self.dtype)

        return _as_channel_view(result)

    def all_channels_as_lazy(self, m: int = 0,
                             z_mode: str = "max",
                             z_index: int = 0) -> "_OD[str, _ChannelView]":
        """Return an OrderedDict of channel_name -> (T, H, W) view at M=m.

        Each view applies z_mode / z_index against the stored (T, Z, H, W)
        block. Legacy callers that expect ``.materialize()`` keep working.
        """
        from collections import OrderedDict
        out: "_OD[str, _ChannelView]" = OrderedDict()
        for i, name in enumerate(self.channel_names):
            out[name] = self.to_lazy_channel(i, m=m, z_mode=z_mode,
                                             z_index=z_index)
        return out

    def subset(self, *, m=None, t=None, z=None,
               progress_cb=None) -> "MaterializedDataset":
        """Return a NEW dataset containing only the selected M / T / Z indices.

        ``m`` / ``t`` / ``z`` are iterables of indices (``None`` = keep the
        whole axis). The source dataset is never mutated — advanced indexing
        copies — so this is the "crop to selection that does not touch the
        original file" operation (V1.43). ``progress_cb`` (0–100) is invoked
        per channel as the copies complete.
        """
        m_idx = (sorted({int(i) for i in m if 0 <= int(i) < self.n_multipoints})
                 if m is not None else list(range(self.n_multipoints)))
        t_idx = (sorted({int(i) for i in t if 0 <= int(i) < self.n_timepoints})
                 if t is not None else list(range(self.n_timepoints)))
        z_idx = (sorted({int(i) for i in z if 0 <= int(i) < self.n_zslices})
                 if z is not None else list(range(self.n_zslices)))
        # Never collapse an axis to empty — fall back to "keep all".
        if not m_idx:
            m_idx = list(range(self.n_multipoints))
        if not t_idx:
            t_idx = list(range(self.n_timepoints))
        if not z_idx:
            z_idx = list(range(self.n_zslices))

        sel = np.ix_(m_idx, t_idx, z_idx)
        new_channels: Dict[str, np.ndarray] = {}
        n = max(1, len(self.channels))
        for i, (name, arr) in enumerate(self.channels.items()):
            # arr is (M, T, Z, H, W); np.ix_ indexes the leading 3 axes and
            # preserves H, W. Advanced indexing returns a fresh contiguous copy.
            new_channels[name] = arr[sel]
            if progress_cb is not None:
                progress_cb(int((i + 1) / n * 100))

        return MaterializedDataset(
            filepath=self.filepath,
            channels=new_channels,
            channel_names=list(self.channel_names),
            dtype=self.dtype,
            pixel_size_um=self.pixel_size_um,
            z_step_um=self.z_step_um,
            n_multipoints=len(m_idx),
            n_timepoints=len(t_idx),
            n_channels=self.n_channels,
            height=self.height,
            width=self.width,
            z_mode=self.z_mode,
            n_zslices=len(z_idx),
            extra={**self.extra, "cropped_from": self.filepath,
                   "crop_m": m_idx, "crop_t": t_idx, "crop_z": z_idx},
        )

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
