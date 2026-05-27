"""
Eager parallel ND2 / multi-file / TIFF materializer.

The viewer hot path used to read planes through Dask + IOWorker +
FrameCache + PrefetchManager, paying tens to hundreds of milliseconds
per slider tick. We replace that with a single parallel decode at file
open: read everything, apply the user's Z-projection mode once, and
hand the GUI a plain ``Dict[channel_name, np.ndarray(M, T, H, W)]``.

The decode itself uses Dask's threaded scheduler — chunks are
decompressed in parallel by the ``nd2`` / ``tifffile`` C extensions
which release the GIL. The result is one np.ndarray per channel
preallocated and filled in place, never copied.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.utils.resources import recommended_worker_count


# Z-projection modes the materializer accepts. "none" is only valid for
# files with a single Z slice; if n_z > 1 with mode="none" we raise to
# avoid silently dropping data.
_VALID_Z_MODES = ("max", "mean", "min", "none")


def _project_z(stack: np.ndarray, z_mode: str, dtype: np.dtype) -> np.ndarray:
    """Project a (Z, H, W) stack into (H, W) using the chosen mode."""
    if stack.ndim == 2:
        return stack
    if stack.ndim != 3:
        stack = np.squeeze(stack)
        if stack.ndim != 3:
            return stack.reshape(stack.shape[-2:]) if stack.size else stack
    if z_mode == "max":
        return stack.max(axis=0)
    if z_mode == "min":
        return stack.min(axis=0)
    if z_mode == "mean":
        return stack.mean(axis=0).astype(dtype)
    # z_mode == "none" → take z=0 (caller guards against n_z > 1)
    return stack[0]


def materialize_nd2(
    filepath: str,
    z_mode: str = "max",
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> MaterializedDataset:
    """Load an entire ND2 file into RAM with Z-projection applied.

    Parameters
    ----------
    filepath : str
        Path to the .nd2 file.
    z_mode : str
        One of "max" | "mean" | "min" | "none". "none" is only allowed
        for files with a single Z slice.
    progress_cb : callable, optional
        Called with int percent (0..100) as work proceeds.
    cancel_cb : callable, optional
        Called periodically; if it returns True the materialization
        raises ``RuntimeError("cancelled")``.

    Returns
    -------
    MaterializedDataset
    """
    import nd2

    if z_mode not in _VALID_Z_MODES:
        raise ValueError(f"unknown z_mode {z_mode!r}; "
                         f"expected one of {_VALID_Z_MODES}")

    with nd2.ND2File(filepath) as f:
        sizes = dict(f.sizes)
        dim_order = list(sizes.keys())
        n_t = sizes.get("T", 1)
        n_z = sizes.get("Z", 1)
        n_c = sizes.get("C", 1)
        n_m = sizes.get("P", sizes.get("M", 1))
        h = sizes.get("Y", 0)
        w = sizes.get("X", 0)
        dtype = np.dtype(f.dtype)

        if n_z > 1 and z_mode == "none":
            raise ValueError(
                f"file has {n_z} Z-slices but z_mode='none' was requested. "
                "Pick 'max', 'mean', or 'min' to collapse Z at load."
            )

        try:
            channel_names: List[str] = [
                str(getattr(getattr(c, "channel", c), "name", f"Ch{i}"))
                for i, c in enumerate(f.metadata.channels)
            ]
        except Exception:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        if len(channel_names) < n_c:
            channel_names.extend(
                f"Ch{i}" for i in range(len(channel_names), n_c)
            )

        try:
            vox = f.voxel_size()
            pixel_size_um = float(getattr(vox, "x", 1.0))
            z_step_um = float(getattr(vox, "z", 1.0))
        except Exception:
            pixel_size_um = 1.0
            z_step_um = 1.0

        out_dtype = np.float32 if z_mode == "mean" else dtype
        channels: Dict[str, np.ndarray] = {
            name: np.empty((n_m, n_t, h, w), dtype=out_dtype)
            for name in channel_names
        }

        dask_arr = f.to_dask()  # decode source — never indexed on the GUI thread

        def _build_index(c: int, m: int, t: int) -> Tuple:
            idx = []
            for d in dim_order:
                if d == "T":
                    idx.append(int(t))
                elif d == "C":
                    idx.append(int(c))
                elif d == "Z":
                    idx.append(slice(0, n_z) if n_z > 1 else 0)
                elif d in ("P", "M"):
                    idx.append(int(m))
                elif d in ("Y", "X"):
                    idx.append(slice(None))
                else:
                    idx.append(0)
            return tuple(idx)

        def decode_one(c: int, m: int, t: int) -> None:
            stack = np.asarray(dask_arr[_build_index(c, m, t)])
            projected = _project_z(stack, z_mode, out_dtype)
            channels[channel_names[c]][m, t] = projected

        work: List[Tuple[int, int, int]] = [
            (c, m, t)
            for c in range(n_c)
            for m in range(n_m)
            for t in range(n_t)
        ]
        total = len(work)
        if total == 0:
            return MaterializedDataset(
                filepath=filepath, channels=channels,
                channel_names=channel_names, dtype=out_dtype,
                pixel_size_um=pixel_size_um, z_step_um=z_step_um,
                n_multipoints=n_m, n_timepoints=n_t, n_channels=n_c,
                height=h, width=w, z_mode=z_mode,
            )

        n_workers = recommended_worker_count()
        done = 0
        last_pct = -1
        # Dask + nd2 release the GIL during decode, so threads parallelize
        # decompression effectively. We use a thread pool here rather than
        # dask.compute(scheduler='threads') so we can interleave progress
        # and cancellation checks.
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(decode_one, c, m, t): (c, m, t)
                       for (c, m, t) in work}
            try:
                for fut in as_completed(futures):
                    if cancel_cb is not None and cancel_cb():
                        # Cancel remaining work; running tasks finish their
                        # current frame but no new ones start.
                        for f2 in futures:
                            f2.cancel()
                        raise RuntimeError("cancelled")
                    fut.result()  # surface any decode errors
                    done += 1
                    if progress_cb is not None:
                        pct = int(100 * done / total)
                        if pct != last_pct:
                            progress_cb(pct)
                            last_pct = pct
            finally:
                # Wake any still-blocked workers — they'll exit once their
                # current decode returns. ThreadPoolExecutor's context
                # manager waits for them.
                pass

    return MaterializedDataset(
        filepath=filepath,
        channels=channels,
        channel_names=channel_names,
        dtype=out_dtype,
        pixel_size_um=pixel_size_um,
        z_step_um=z_step_um,
        n_multipoints=n_m,
        n_timepoints=n_t,
        n_channels=n_c,
        height=h,
        width=w,
        z_mode=z_mode,
    )


def materialize_from_volume(
    volume: Any,
    z_mode: str = "max",
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> MaterializedDataset:
    """Materialize any lazy volume (multi-file ND2, TIFF) via its get_frame API.

    Works for any object that exposes the LazyND2Volume surface:
    ``n_multipoints``, ``n_timepoints``, ``n_zslices``, ``n_channels``,
    ``height``, ``width``, ``dtype``, ``channel_names``, ``pixel_size_um``,
    ``z_step_um``, ``reopen()``, and
    ``get_frame(c, m, t, z, z_mode, z_start, z_end)``.

    Uses a thread pool with per-thread reopen()'d volumes since the
    underlying readers (nd2, tifffile) are not handle-thread-safe.
    """
    if z_mode not in _VALID_Z_MODES:
        raise ValueError(f"unknown z_mode {z_mode!r}; "
                         f"expected one of {_VALID_Z_MODES}")

    n_m = int(getattr(volume, "n_multipoints", 1))
    n_t = int(getattr(volume, "n_timepoints", 1))
    n_z = int(getattr(volume, "n_zslices", 1))
    n_c = int(getattr(volume, "n_channels", 1))
    h = int(getattr(volume, "height", 0))
    w = int(getattr(volume, "width", 0))
    dtype = np.dtype(getattr(volume, "dtype", np.uint16))
    channel_names: List[str] = list(getattr(volume, "channel_names", []))
    if len(channel_names) < n_c:
        channel_names.extend(
            f"Ch{i}" for i in range(len(channel_names), n_c)
        )
    pixel_size_um = float(getattr(volume, "pixel_size_um", 1.0))
    z_step_um = float(getattr(volume, "z_step_um", 1.0))
    filepath = str(getattr(volume, "filepath", "") or "")

    if n_z > 1 and z_mode == "none":
        raise ValueError(
            f"file has {n_z} Z-slices but z_mode='none' was requested. "
            "Pick 'max', 'mean', or 'min' to collapse Z at load."
        )

    out_dtype = np.float32 if z_mode == "mean" else dtype
    channels: Dict[str, np.ndarray] = {
        name: np.empty((n_m, n_t, h, w), dtype=out_dtype)
        for name in channel_names
    }

    # One reader per worker (each gets its own reopen()'d handle).
    n_workers = recommended_worker_count()
    import threading
    tls = threading.local()

    def _reader():
        r = getattr(tls, "reader", None)
        if r is None:
            r = volume.reopen()
            tls.reader = r
        return r

    def decode_one(c: int, m: int, t: int) -> None:
        r = _reader()
        if n_z > 1 and z_mode != "none":
            plane = r.get_frame(c=c, m=m, t=t, z=0, z_mode=z_mode,
                                z_start=0, z_end=n_z)
        else:
            plane = r.get_frame(c=c, m=m, t=t, z=0, z_mode="none")
        plane = np.asarray(plane)
        if plane.ndim != 2:
            plane = np.squeeze(plane)
            if plane.ndim == 3 and plane.shape[-1] in (3, 4):
                # RGB/RGBA frame — extract the color component that matches
                # this channel index.  When the volume reports n_channels=3
                # for an RGB TIFF (set by read_tiff_meta_fast), c=0/1/2
                # maps to R/G/B, preserving per-object color in the viewer.
                comp = min(int(c), plane.shape[-1] - 1)
                plane = plane[..., comp]
            elif plane.ndim != 2:
                plane = plane.reshape(h, w)
        if plane.dtype != out_dtype:
            plane = plane.astype(out_dtype)
        channels[channel_names[c]][m, t] = plane

    work: List[Tuple[int, int, int]] = [
        (c, m, t)
        for c in range(n_c)
        for m in range(n_m)
        for t in range(n_t)
    ]
    total = len(work)

    done = 0
    last_pct = -1
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(decode_one, c, m, t): (c, m, t)
                   for (c, m, t) in work}
        for fut in as_completed(futures):
            if cancel_cb is not None and cancel_cb():
                for f2 in futures:
                    f2.cancel()
                raise RuntimeError("cancelled")
            fut.result()
            done += 1
            if progress_cb is not None and total:
                pct = int(100 * done / total)
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

    return MaterializedDataset(
        filepath=filepath,
        channels=channels,
        channel_names=channel_names,
        dtype=out_dtype,
        pixel_size_um=pixel_size_um,
        z_step_um=z_step_um,
        n_multipoints=n_m,
        n_timepoints=n_t,
        n_channels=n_c,
        height=h,
        width=w,
        z_mode=z_mode,
    )


__all__ = ["materialize_nd2", "materialize_from_volume"]
