"""Per-frame analysis driver with adaptive eager/streaming output (V1.46).

Centralizes the "iterate frames → produce a label frame + measurement rows"
loop that every per-frame analysis pipeline (histogram threshold, tear
detection, spot detection, nuclei segmentation) shares, and routes the label
output to either:

* an **in-RAM** ``(T, H, W)`` array (eager — fast, used when the dataset fits
  RAM), or
* a **disk-backed streaming sink** (:class:`~nd2studios.pipeline.storage.LabelStackWriter`)
  written one frame at a time, so a memory-constrained machine never holds the
  whole mask stack — the Fiji/NIS-Elements "virtual stack + streaming sink".

The per-frame callable receives the frame **index** ``t`` (not just the plane)
so it can read auxiliary channels at the same index (e.g. tear detection's
counterstain) and stamp ``"frame": t`` onto measurement rows. The frame source
is read inside the callable, so a lazy reader streams from disk on demand.

Backend-pure: no Qt imports.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from nd2studios.utils.resources import recommended_worker_count

# A per-frame result: (labels (H,W) int32, measurement rows, optional secondary
# mask (H,W) int32). ``secondary`` is None for pipelines without one.
FrameResult = Tuple[np.ndarray, List[dict], Optional[np.ndarray]]


@dataclass
class PlaneRunOutput:
    """Result of :func:`run_planes_to_labels`.

    ``primary`` / ``secondary`` are an in-RAM ndarray (eager) or a disk-backed
    lazy reader (streamed); both support ``[t]`` indexing and ``.shape``.
    """
    primary: Any
    measurements: List[dict] = field(default_factory=list)
    secondary: Any = None


def run_planes_to_labels(
    *,
    n_frames: int,
    height: int,
    width: int,
    per_frame_fn: Callable[[int], FrameResult],
    primary_writer=None,
    secondary_writer=None,
    has_secondary: bool = False,
    n_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    frame_cb: Optional[Callable[[int, np.ndarray], None]] = None,
) -> PlaneRunOutput:
    """Map ``per_frame_fn(t)`` across ``range(n_frames)`` on a thread pool.

    Parameters
    ----------
    per_frame_fn:
        ``per_frame_fn(t) -> (labels, rows, secondary_or_None)``. Reads the
        frame(s) it needs itself (so a lazy source streams from disk), returns
        the ``(H, W)`` int32 label frame, a list of measurement-row dicts, and
        an optional ``(H, W)`` secondary mask.
    primary_writer:
        A :class:`LabelStackWriter` to stream the primary masks to disk. When
        ``None`` the masks are collected into an in-RAM ``(T, H, W)`` array.
    secondary_writer:
        Optional streaming sink for secondary masks (mirrors ``primary_writer``).
    has_secondary:
        When eager (no ``secondary_writer``) and the pipeline emits secondary
        masks, pre-allocate the in-RAM secondary array.
    progress_cb:
        ``progress_cb(percent_0_100)``.
    cancelled_cb:
        Polled between completions; ``True`` short-circuits. Frames not yet
        processed stay at the fill value (0 = background).
    frame_cb:
        ``frame_cb(t, labels)`` called as each frame completes — for live
        per-frame overlay streaming to the GUI. Frames may arrive out of order
        when ``n_workers > 1``.
    """
    n_frames = max(0, int(n_frames))
    stream = primary_writer is not None
    label_stack = None if stream else np.zeros((n_frames, height, width), np.int32)
    sec_stack = (None if (stream or not has_secondary)
                 else np.zeros((n_frames, height, width), np.int32))
    measurements: List[dict] = []

    if n_frames > 0:
        n_workers = n_workers or recommended_worker_count()

        def _one(t: int):
            return t, per_frame_fn(t)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_one, t): t for t in range(n_frames)}
            done = 0
            for fut in as_completed(futures):
                if cancelled_cb is not None and cancelled_cb():
                    for f in futures:
                        f.cancel()
                    break
                t, (labels, rows, secondary) = fut.result()
                if stream:
                    primary_writer.write_frame(t, labels)
                    if secondary is not None and secondary_writer is not None:
                        secondary_writer.write_frame(t, secondary)
                else:
                    label_stack[t] = labels
                    if secondary is not None and sec_stack is not None:
                        sec_stack[t] = secondary
                if rows:
                    measurements.extend(rows)
                if frame_cb is not None:
                    try:
                        frame_cb(t, labels)
                    except Exception:  # noqa: BLE001 — live UI must never break a run
                        pass
                done += 1
                if progress_cb is not None:
                    progress_cb(int(done / n_frames * 100))

    primary = primary_writer.close() if stream else label_stack
    if stream:
        secondary = secondary_writer.close() if secondary_writer is not None else None
    else:
        secondary = sec_stack
    return PlaneRunOutput(primary=primary, measurements=measurements,
                          secondary=secondary)


def make_label_writers(params: dict, channel: str, shape: Tuple[int, int, int],
                       *, secondary_name: Optional[str] = None):
    """Build streaming label writers from the reserved pipeline params, or
    ``(None, None)`` for the eager in-RAM path.

    Pipelines call this once at the top of ``run()``. The caller (page/worker)
    injects ``params["_stream_labels"]`` (bool) and
    ``params["_label_sink_factory"]`` (``factory(name, shape) -> LabelStackWriter``).
    Direct/headless callers omit them and get the eager path.
    """
    if not params.get("_stream_labels"):
        return None, None
    factory = params.get("_label_sink_factory")
    if factory is None:
        return None, None
    primary = factory(channel, shape)
    secondary = factory(secondary_name, shape) if secondary_name else None
    return primary, secondary


def make_frame_cb(params: dict):
    """Return the per-frame streaming callback injected by the job, or ``None``.

    The page/worker injects ``params["_frame_cb"]`` (``Callable[[int, ndarray],
    None]``) so each finished frame's labels stream to the GUI for a live overlay.
    Direct/headless callers omit it and get ``None`` (no streaming).
    """
    return params.get("_frame_cb")
