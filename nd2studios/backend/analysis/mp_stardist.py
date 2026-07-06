"""Multiprocessing StarDist runner — many-core CPU throughput (V1.46).

The default StarDist node segments frames strictly sequentially (``n_workers=1``
in :mod:`nd2studios.backend.analysis.plane_runner`) because TensorFlow/StarDist
share a single, non-thread-safe model and TF is pinned to one op-thread. That
leaves a many-core CPU workstation mostly idle. This module runs StarDist across
**separate processes** — each with its own model + TF (still 1 op-thread each) —
so N cores segment N frames concurrently, bounded by RAM.

Design:

* **The parent owns the lazy source and the label sink.** Frames are read one at
  a time in the parent (``read_plane``) and shipped to workers as plain
  ndarrays, so the lazy ND2/disk reader is never opened in a child process and
  the parent stays the single writer of the streaming sink / in-RAM stack.
* **All per-frame CPU runs in the worker** (normalize → predict → filter/relabel
  → regionprops), keeping the parent off the hot path.
* **Bounded in-flight window** (~2×workers frames): frames are not all submitted
  up front, so parent RAM stays bounded even for a huge stack — the same
  streaming guarantee the thread path gives.

Backend-pure: no Qt. Uses the ``spawn`` start method (Windows default; also
safest for TensorFlow on POSIX), so ``_init_worker`` / ``_segment_frame_worker``
are module-level and picklable. ``run.py`` guards on
``if __name__ == "__main__"``, so spawned children never re-launch the GUI.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.analysis.plane_runner import PlaneRunOutput
from nd2studios.backend.analysis.source_utils import filter_and_relabel, read_plane

# Per-process worker state, populated once by ``_init_worker`` in each child.
_STATE: Dict[str, Any] = {}


def _init_worker(model_name: str, disable_gpu: bool, seg_params: Dict[str, Any]) -> None:
    """Load the StarDist model once per worker process (pool initializer).

    Each process gets its own singleton model + TF init, so there is no shared,
    non-thread-safe state across workers. TF threading is pinned to one op-thread
    inside ``get_stardist_model`` exactly as the sequential path does.
    """
    from nd2studios.backend.celltracker.segmentation import get_stardist_model

    _STATE["model"] = get_stardist_model(model_name, disable_gpu=disable_gpu)
    _STATE["p"] = seg_params


def _segment_frame_worker(
    t: int, frame: np.ndarray
) -> Tuple[int, np.ndarray, List[Dict[str, Any]]]:
    """Segment one frame in a worker process; return ``(t, mask, rows)``.

    Mirrors ``StarDistSegmentationPipeline._per_frame`` exactly (same normalize,
    predict, area-filter, and regionprops), so MP output is identical to the
    sequential path — only *where* the work runs changes.
    """
    from skimage.measure import regionprops

    from nd2studios.backend.celltracker.segmentation import segment_frame

    p = _STATE["p"]
    frame = np.asarray(frame, dtype=np.float32)

    mask, _ = segment_frame(
        frame,
        model=_STATE["model"],
        prob_thresh=p["prob_thresh"],
        nms_thresh=p["nms_thresh"],
        scale=p["scale"],
        model_name=p["model_name"],
        disable_gpu=p["disable_gpu"],
    )
    mask = np.asarray(mask, dtype=np.int32)
    mask = filter_and_relabel(mask, p["min_area"], p["max_area"])

    pixel_size_um: float = p["pixel_size_um"]
    rows: List[Dict[str, Any]] = []
    for region in regionprops(mask, intensity_image=frame):
        cy, cx = region.centroid
        rows.append(
            {
                "frame": t,
                "label_id": int(region.label),
                "area_px": float(region.area),
                "area_um2": float(region.area) * pixel_size_um ** 2,
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "mean_intensity": float(region.mean_intensity),
            }
        )
    return t, mask, rows


def run_stardist_multiprocess(
    *,
    source: Any,
    n_frames: int,
    height: int,
    width: int,
    model_name: str,
    disable_gpu: bool,
    seg_params: Dict[str, Any],
    n_workers: int,
    primary_writer=None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    frame_cb: Optional[Callable[[int, np.ndarray], None]] = None,
) -> PlaneRunOutput:
    """Segment ``n_frames`` across ``n_workers`` processes.

    A bounded in-flight window (~2×workers frames) keeps parent RAM bounded even
    for a huge stack: we prime the window, then submit the next frame each time
    one completes. Results are written by the parent (streaming ``primary_writer``
    when provided, else an in-RAM ``(T, H, W)`` array) so ordering is handled
    here — worker completions may arrive out of order.

    ``seg_params`` supplies the StarDist knobs (``prob_thresh``, ``nms_thresh``,
    ``scale``, ``min_area``, ``max_area``, ``pixel_size_um``); ``model_name`` and
    ``disable_gpu`` are added here before it is shipped to the workers.
    """
    n_frames = max(0, int(n_frames))
    stream = primary_writer is not None
    label_stack = None if stream else np.zeros((n_frames, height, width), np.int32)
    measurements: List[Dict[str, Any]] = []

    # Freeze the exact knobs the worker needs (picklable scalars only — never the
    # injected _frame_cb / _label_sink_factory, which stay in the parent).
    seg_params = dict(seg_params)
    seg_params["model_name"] = model_name
    seg_params["disable_gpu"] = disable_gpu

    def _finish(result: Tuple[int, np.ndarray, List[Dict[str, Any]]]) -> None:
        t, mask, rows = result
        if stream:
            primary_writer.write_frame(t, mask)
        else:
            label_stack[t] = mask
        if rows:
            measurements.extend(rows)
        if frame_cb is not None:
            try:
                frame_cb(t, mask)
            except Exception:  # noqa: BLE001 — live UI must never break a run
                pass

    if n_frames > 0:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        window = max(2 * n_workers, n_workers + 1)
        done = 0
        next_t = 0

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model_name, disable_gpu, seg_params),
        ) as ex:
            inflight = set()
            # Prime the in-flight window.
            while next_t < n_frames and len(inflight) < window:
                frame = np.asarray(read_plane(source, next_t))
                inflight.add(ex.submit(_segment_frame_worker, next_t, frame))
                next_t += 1

            while inflight:
                if cancelled_cb is not None and cancelled_cb():
                    for fut in inflight:
                        fut.cancel()
                    # Exiting the `with` block shuts the pool down; already-running
                    # workers (≤ n_workers frames) finish, the rest are cancelled.
                    break
                completed, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in completed:
                    _finish(fut.result())
                    done += 1
                    if progress_cb is not None:
                        progress_cb(int(done / n_frames * 100))
                    # Backfill the window with the next unsubmitted frame.
                    if next_t < n_frames:
                        frame = np.asarray(read_plane(source, next_t))
                        inflight.add(ex.submit(_segment_frame_worker, next_t, frame))
                        next_t += 1

    primary = primary_writer.close() if stream else label_stack
    return PlaneRunOutput(primary=primary, measurements=measurements, secondary=None)
