"""
Pyramid stage (V1.39 Phase 7) — multi-resolution display tier.

A pyramid is a stack of downsampled mirrors of the source volume.
ND2Studios stores levels ``1..N`` (level 0 is the source ND2 itself,
read by :class:`~nd2studios.backend.nd2_volume.LazyND2Volume`) so the
on-disk overhead is bounded to ≈ 1/4 + 1/16 + 1/64 + ... ≈ 33 % of
source size rather than the 133 % a full mirror would consume.

The pyramid lives in the V1.38 :class:`~nd2studios.pipeline.session.Session`
workspace::

    ~/.nd2studios/workspace/sessions/<hash>/
    └── pyramid/
        └── pyramid.zarr/
            ├── 1/        # (M, T, Z, C, H/2,  W/2)   chunked (1,1,1,1,H/2,W/2)
            ├── 2/        # (M, T, Z, C, H/4,  W/4)
            ├── 3/        # (M, T, Z, C, H/8,  W/8)
            └── 4/        # (M, T, Z, C, H/16, W/16)

Builds run in the background as a Phase 5 :class:`AnalysisJob`
(:class:`BuildPyramidJob`). The viewer's
:class:`PyramidReader` is the read-side interface — it mirrors the
``LazyND2Volume.get_frame(c, m, t, z, z_mode)`` signature with an
extra ``level`` slot, so the read site dispatches with one branch.

Zarr is *required* for pyramids. NPZ does not support sub-chunk
indexing, so without zarr there is no way to stream a single plane
without materialising the entire stack. When zarr is missing,
:meth:`PyramidStage.build` raises :class:`PyramidUnavailable` which
the caller surfaces as a one-line status note ("Pyramid unavailable:
zarr not installed") and the viewer pins ``_active_level = 0``.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.pipeline.session import Session, StageRecord
from nd2studios.pipeline.stage import PipelineStage
from nd2studios.pipeline.storage import HAS_ZARR
from nd2studios.utils.resources import recommended_worker_count

log = logging.getLogger(__name__)

_STAGE_NAME = "pyramid"
_PYRAMID_SUBDIR = "pyramid"
_PYRAMID_ZARR = "pyramid.zarr"
_MAX_LEVEL = 6  # cap pyramid depth even on enormous images


class _AtomicCounter:
    """Lock-guarded integer counter for cross-thread progress accounting.

    Addendum Phase 7 helper: the parallel pyramid build needs to know
    how many planes have finished across N worker threads so the
    ``progress_cb`` hook reports a monotonic 0–100. A bare ``int +=``
    is non-atomic under CPython once the GIL is released inside the
    worker's C-extension hot path, so we wrap it.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self, start: int = 0) -> None:
        self._value = int(start)
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class PyramidUnavailable(RuntimeError):
    """Raised when the pyramid cannot be built or opened.

    Reasons: ``zarr`` package missing, write-protected workspace, or
    source volume too small to benefit from a pyramid (≤ 256² at
    level 0). Callers should catch this and degrade gracefully —
    pyramids are an optional speed-up, never a correctness gate.
    """


def default_pyramid_levels(height: int, width: int) -> int:
    """Pick a sensible level count from the source dimensions.

    A useful pyramid has ``log2(max_dim / 256)`` levels — beyond that
    the bottom level is < 256 px on a side and brings no display
    benefit. Clamped to ``[2, 6]`` so even tiny stacks get a
    half-resolution mirror and gigantic ones don't bloat the disk.
    """
    if height <= 0 or width <= 0:
        return 2
    longest = max(int(height), int(width))
    raw = int(math.floor(math.log2(longest / 256.0))) if longest > 256 else 2
    return max(2, min(_MAX_LEVEL, raw))


def _downscale_pow2(plane: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a 2D plane by an integer power-of-two factor.

    Operates on numpy only — pyramid building is decoder-bound, not
    filter-bound, so the GPU path is intentionally not used here.
    For a 2048² uint16 frame the mean-pool runs in ≈ 20 ms on CPU
    vs. ≈ 8 ms on GPU; the host↔device transfer would erase the win.
    """
    if factor <= 1:
        return np.ascontiguousarray(plane)
    h, w = plane.shape[-2:]
    # Crop to a multiple of factor before reshape — pyramids tolerate
    # a one-pixel boundary loss far better than padding artefacts.
    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor
    cropped = plane[..., :h_trim, :w_trim].astype(np.float32, copy=False)
    reshaped = cropped.reshape(h_trim // factor, factor, w_trim // factor, factor)
    return reshaped.mean(axis=(1, 3)).astype(plane.dtype, copy=False)


class PyramidStage(PipelineStage):
    """Persistent multi-resolution pyramid for one source file.

    Usage::

        stage = PyramidStage(session)
        if not stage.is_committed():
            stage.build(volume, progress_cb=..., cancel_cb=...)
        reader = stage.reader()  # returns None if not built / no zarr

    The stage is Qt-free; long builds run inside
    :class:`~nd2studios.compute.pipeline_jobs.BuildPyramidJob` which
    wraps the call into the Phase 5 :class:`JobRunner`.
    """

    name = _STAGE_NAME

    def __init__(
        self,
        session: Session,
        *,
        n_levels: Optional[int] = None,
        downscale: int = 2,
    ):
        super().__init__(session)
        self._n_levels_override = n_levels
        self._downscale = max(2, int(downscale))

    # ── build ────────────────────────────────────────────────────────

    def build(
        self,
        volume,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> StageRecord:
        """Build the pyramid from ``volume``. Idempotent.

        ``volume`` must expose ``get_frame(c, m, t, z, z_mode='none')``
        — both :class:`LazyND2Volume` and
        :class:`LazyMultiFileND2Volume` do. ``progress_cb`` receives
        integer percentages in ``[0, 100]``; ``cancel_cb`` returning
        True aborts the build (a partial pyramid is left on disk; the
        manifest is not updated, so the next build retries cleanly).
        """
        if not HAS_ZARR:
            raise PyramidUnavailable(
                "zarr is not installed — pyramids require zarr for "
                "sub-chunk reads. Install with `pip install zarr`."
            )

        n_m = max(1, int(getattr(volume, "n_multipoints", 1)))
        n_t = max(1, int(getattr(volume, "n_timepoints", 1)))
        n_z = max(1, int(getattr(volume, "n_zslices", 1)))
        n_c = max(1, int(getattr(volume, "n_channels", 1)))
        h = int(getattr(volume, "height", 0))
        w = int(getattr(volume, "width", 0))
        if h <= 0 or w <= 0:
            raise PyramidUnavailable("source volume has zero spatial dims")
        if max(h, w) <= 256:
            raise PyramidUnavailable("source is already ≤ 256 px; no pyramid needed")

        n_levels = (
            self._n_levels_override
            if self._n_levels_override is not None
            else default_pyramid_levels(h, w)
        )
        n_levels = max(2, min(_MAX_LEVEL, int(n_levels)))

        import zarr  # type: ignore[import]  # HAS_ZARR guarded above

        out_dir = self._pyramid_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        store_path = out_dir / _PYRAMID_ZARR
        # Wipe any half-written prior build before we start.
        if store_path.exists():
            try:
                shutil.rmtree(store_path)
            except OSError:
                log.warning("could not remove stale pyramid at %s", store_path)

        root = zarr.open(str(store_path), mode="w")

        # Pre-compute the per-level shapes & create arrays up front so
        # a cancellation mid-build leaves recognisable empty levels
        # rather than a missing array entry. The chunk shape is
        # ``(1, 1, 1, 1, H_L, W_L)`` so a viewer read for one
        # ``(c, m, t, z)`` is exactly one block.
        dtype = np.dtype(getattr(volume, "dtype", np.uint16))
        level_shapes: List[Tuple[int, int]] = []
        for level in range(1, n_levels + 1):
            factor = self._downscale ** level
            h_l, w_l = h // factor, w // factor
            if h_l < 32 or w_l < 32:
                break
            level_shapes.append((h_l, w_l))
            shape = (n_m, n_t, n_z, n_c, h_l, w_l)
            chunks = (1, 1, 1, 1, h_l, w_l)
            root.create_dataset(
                str(level),
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                overwrite=True,
            )

        if not level_shapes:
            raise PyramidUnavailable(
                f"source {h}x{w} too small for any pyramid level"
            )

        # Total plane reads: one per (m, t, z, c) per level. We could
        # reuse a downsampled-from-finer-level read to save reads, but
        # that complicates cancellation; one fresh read per level is
        # simple and idempotent.
        total_planes = n_m * n_t * n_z * n_c * len(level_shapes)

        # Addendum Phase 7: the per-plane work splits cleanly into
        # ``reader.get_frame`` (GIL-releasing nd2/tifffile decode) +
        # ``_downscale_pow2`` (numpy mean-pool, releases GIL) + a
        # zarr write into a unique ``(m, t, z, c)`` chunk (releases
        # GIL, writes go to distinct files in a DirectoryStore so
        # they don't contend). All three release the GIL, so a
        # :class:`ThreadPoolExecutor` scales near-linearly with cores
        # for the build. Each worker thread holds its own
        # ``volume.reopen()`` handle — the ``nd2`` SDK's per-file
        # state is not thread-safe, so reusing the caller's handle
        # would race.
        n_workers = max(1, recommended_worker_count())
        worker_done = _AtomicCounter()
        worker_cancel = threading.Event()
        # We open one reader per thread on first use and stash it in
        # ``threading.local``. We also park them in ``readers_registry``
        # so we can close them deterministically once the level
        # finishes — ``ThreadPoolExecutor`` doesn't expose per-thread
        # finalisers.
        thread_local = threading.local()
        readers_registry: List[Any] = []
        registry_lock = threading.Lock()

        def _get_reader():
            reader = getattr(thread_local, "reader", None)
            if reader is None:
                reader = volume.reopen()
                thread_local.reader = reader
                with registry_lock:
                    readers_registry.append(reader)
            return reader

        def _process_one(args):
            m, t, z, c, factor, arr_ref = args
            if worker_cancel.is_set():
                return None
            reader = _get_reader()
            plane = reader.get_frame(c=c, m=m, t=t, z=z, z_mode="none")
            plane = np.asarray(plane)
            if plane.ndim != 2:
                plane = np.squeeze(plane)
                if plane.ndim != 2:
                    return None
            down = _downscale_pow2(plane, factor)
            # Distinct (m, t, z, c) → distinct zarr chunk → no contention.
            arr_ref[m, t, z, c, : down.shape[0], : down.shape[1]] = down
            return None

        try:
            for level_idx, (h_l, w_l) in enumerate(level_shapes, start=1):
                factor = self._downscale ** level_idx
                arr = root[str(level_idx)]
                tasks = [
                    (m, t, z, c, factor, arr)
                    for m in range(n_m)
                    for t in range(n_t)
                    for z in range(n_z)
                    for c in range(n_c)
                ]
                with ThreadPoolExecutor(
                    max_workers=n_workers,
                    thread_name_prefix=f"PyramidL{level_idx}",
                ) as ex:
                    futures = [ex.submit(_process_one, args) for args in tasks]
                    for fut in as_completed(futures):
                        if cancel_cb is not None and cancel_cb():
                            worker_cancel.set()
                            for f in futures:
                                f.cancel()
                            log.info(
                                "pyramid build cancelled at level %d",
                                level_idx,
                            )
                            return self._partial_record(
                                out_dir, store_path, level_idx,
                            )
                        # Propagate any exception from the worker.
                        fut.result()
                        done = worker_done.increment()
                        if progress_cb is not None and done % 4 == 0:
                            progress_cb(int(done / total_planes * 100))
        finally:
            # Close every per-thread reader we opened — ``volume.reopen()``
            # returns a fresh ND2/TIFF file handle and Python's GC
            # won't run inside the worker threads after the executor
            # shuts down.
            with registry_lock:
                for reader in readers_registry:
                    close = getattr(reader, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: BLE001
                            pass
                readers_registry.clear()

        # Final progress nudge so the bar always lands on 100.
        if progress_cb is not None:
            progress_cb(100)

        # Record the build in the manifest. ``parameters`` captures
        # what we'd need to detect a stale pyramid (e.g. source dims
        # changed) on a future open.
        record = StageRecord(
            name=self.name,
            parameters={
                "n_levels": len(level_shapes),
                "downscale": self._downscale,
                "source_height": h,
                "source_width": w,
                "level_shapes": [list(s) for s in level_shapes],
            },
            artifacts={
                _PYRAMID_ZARR: f"{_PYRAMID_SUBDIR}/{_PYRAMID_ZARR}",
            },
        )
        self._session.record_stage(record)
        return record

    def _partial_record(
        self,
        out_dir: Path,
        store_path: Path,
        completed_level: int,
    ) -> StageRecord:
        """Return a non-committed StageRecord for a cancelled build.

        We deliberately do not write this into the manifest — a
        cancelled build is treated as "no pyramid". The on-disk
        partial Zarr is left behind so the next build can either
        overwrite it cheaply or, in the future, resume.
        """
        return StageRecord(
            name=self.name,
            committed_at=None,
            parameters={"partial_level": completed_level},
            artifacts={},
        )

    # ── read ─────────────────────────────────────────────────────────

    def commit(self) -> StageRecord:
        """Re-stamp the existing record (idempotent).

        :meth:`build` already calls :meth:`Session.record_stage`, so
        this hook only matters when callers want to mark an
        externally-built pyramid as committed. We just refresh the
        manifest timestamp in that case.
        """
        existing = self._session.get_stage(self.name)
        if existing is None:
            existing = StageRecord(name=self.name)
        self._session.record_stage(existing)
        return existing

    def reader(self) -> Optional["PyramidReader"]:
        """Return a :class:`PyramidReader` if the pyramid is on disk.

        Returns ``None`` when the stage was never committed, when the
        zarr store is missing, or when the ``zarr`` package is
        unavailable in this process.
        """
        if not HAS_ZARR:
            return None
        rec = self._session.get_stage(self.name)
        if rec is None or rec.committed_at is None:
            return None
        rel = rec.artifacts.get(_PYRAMID_ZARR)
        if rel is None:
            return None
        store_path = self._session.session_dir / rel
        if not store_path.exists():
            return None
        try:
            return PyramidReader(store_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("pyramid reader open failed: %s", exc)
            return None

    # ── helpers ──────────────────────────────────────────────────────

    def _pyramid_dir(self) -> Path:
        return self._session.session_dir / _PYRAMID_SUBDIR


class PyramidReader:
    """Read-side interface to a built pyramid.

    Construction opens the Zarr store (cheap; metadata-only). Reads
    return numpy ``(H_L, W_L)`` planes. Levels are indexed
    1-based to match the on-disk layout — level 0 is the source ND2,
    not represented here.
    """

    def __init__(self, store_path: Path):
        import zarr  # type: ignore[import]

        self._path = Path(store_path)
        self._root = zarr.open(str(self._path), mode="r")
        # Discover available levels by scanning keys; some may be
        # missing if a downscale floor cut the stack short.
        self._levels: List[int] = sorted(
            int(k) for k in self._root.array_keys()
        )
        if not self._levels:
            raise PyramidUnavailable(f"empty pyramid at {self._path}")

    @property
    def levels(self) -> List[int]:
        """Available level indices (smallest = finest, e.g. ``[1, 2, 3, 4]``)."""
        return list(self._levels)

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    def shape_at(self, level: int) -> Tuple[int, ...]:
        """Return the ``(M, T, Z, C, H, W)`` shape of ``level``."""
        return tuple(self._root[str(level)].shape)

    def get_frame(
        self,
        level: int,
        c: int,
        m: int = 0,
        t: int = 0,
        z: int = 0,
        z_mode: str = "none",
        z_start: Optional[int] = None,
        z_end: Optional[int] = None,
    ) -> np.ndarray:
        """Read a single ``(H_L, W_L)`` plane from ``level``.

        Mirrors :meth:`LazyND2Volume.get_frame` so the viewer's read
        site can dispatch between the two with one extra argument.
        ``z_mode != 'none'`` projects across the existing per-Z slabs
        stored at this level (max / mean / min). The pyramid keeps
        Z separate (not pre-projected) so the projection mode can be
        changed at the viewer without rebuilding.
        """
        arr = self._root[str(int(level))]
        # arr shape: (n_m, n_t, n_z, n_c, H_L, W_L)
        if z_mode == "none" or arr.shape[2] <= 1:
            plane = arr[int(m), int(t), int(z), int(c), :, :]
            return np.asarray(plane)
        z0 = 0 if z_start is None else max(0, int(z_start))
        z1 = arr.shape[2] if z_end is None else min(arr.shape[2], int(z_end))
        if z1 <= z0:
            return np.asarray(arr[int(m), int(t), int(z), int(c), :, :])
        stack = np.asarray(arr[int(m), int(t), z0:z1, int(c), :, :])
        if z_mode == "max":
            return stack.max(axis=0)
        if z_mode == "min":
            return stack.min(axis=0)
        if z_mode == "mean":
            return stack.mean(axis=0).astype(stack.dtype, copy=False)
        # Unknown mode → first slice.
        return stack[0]

    def pick_level_for_viewport(
        self,
        viewport_screen_px: int,
        image_pixels_visible: int,
        *,
        source_width: Optional[int] = None,
    ) -> int:
        """Pick the smallest level whose width still exceeds the viewport.

        ``viewport_screen_px``: width of the visible image area in
        screen pixels (i.e. the ``QGraphicsView`` viewport width).

        ``image_pixels_visible``: width of the ``ViewBox`` rect in
        image pixels at level 0 (how many source pixels the viewport
        currently spans). For a fit-to-window display this equals
        the full image width.

        Returns 0 for "use level 0 (read from source ND2)" and
        ``>= 1`` for "use this pyramid level".

        The choice rule: we want ``viewport_screen_px ≤
        image_pixels_visible / 2^L`` so each screen pixel is backed
        by at least one source pixel. The largest such ``L`` is the
        most efficient level that doesn't lose visual quality.
        """
        if viewport_screen_px <= 0 or image_pixels_visible <= 0:
            return 0
        ratio = image_pixels_visible / float(viewport_screen_px)
        if ratio <= 1.0:
            return 0
        # Largest L with 2^L <= ratio  →  L = floor(log2(ratio))
        best = int(math.floor(math.log2(ratio)))
        if best <= 0:
            return 0
        # Clamp to available levels (best is 0-based offset from
        # source; our levels start at 1).
        if best > self._levels[-1]:
            return self._levels[-1]
        # Find the largest level we actually have that is ≤ best.
        candidate = 0
        for lvl in self._levels:
            if lvl <= best:
                candidate = lvl
            else:
                break
        return candidate
