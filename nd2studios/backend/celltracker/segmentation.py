"""StarDist nuclear segmentation — vendored from CellTracker.

Copied from ``Cell-Tracker/CellTracker/backend/segmentation.py``. StarDist
(Schmidt et al., MICCAI 2018) is CellTracker's only segmentation method; here it
backs ND2Studios' optional "StarDist Segmentation" analysis node. Pure
numpy + (lazy) stardist/tensorflow/csbdeep — no Qt, so the backend-purity rule
holds. ``segment_timeseries`` is kept for parity but the analysis node iterates
frames itself via ``plane_runner`` (so it streams + reports progress/cancel).

TensorFlow is an **optional** dependency, imported lazily on first model load.
TF op-parallelism is pinned to a single inter/intra-op thread (``_ensure_tf_threading``),
exactly as CellTracker's original repository does — far faster than TF's
multi-threaded default for StarDist's tiled inference (multi-threaded oversubscribes
the cores on the many small tile ops); the GPU flag is honored by setting
``CUDA_VISIBLE_DEVICES`` **before** the first TF import.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# StarDist model singleton — loaded once on first use.
_stardist_model = None
_stardist_model_name = None
_tf_configured = False


def _ensure_tf_threading(disable_gpu: bool = False) -> None:
    """Configure TF threading on first use. Must happen before any TF ops.

    Op-parallelism is pinned to a **single** inter/intra-op thread — this is exactly
    how CellTracker's original repository runs StarDist, and it is dramatically
    faster here than TF's multi-threaded default: StarDist tiles a large frame into
    many small ``predict_instances`` passes, and multi-threaded intra-op on those
    small ops oversubscribes the cores and thrashes (removing the cap measured ~10×
    slower). The setting only takes effect before TF is initialised, so it is applied
    on the first model load, before any inference.

    When *disable_gpu* is True, ``CUDA_VISIBLE_DEVICES`` is cleared **before** the
    first TensorFlow import so inference stays on CPU (toggling it after import has
    no effect).
    """
    global _tf_configured
    if _tf_configured:
        return
    import os
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf_configured = True


def _patch_windows_symlink() -> None:
    """On Windows, os.symlink requires elevated privileges or Developer Mode.
    Fall back to copying the directory tree so csbdeep model loading still works."""
    import sys
    if sys.platform != "win32":
        return
    import os
    _orig = os.symlink

    def _symlink_or_copy(src, dst, target_is_directory=False, *, dir_fd=None):
        try:
            _orig(src, dst, target_is_directory, dir_fd=dir_fd)
        except OSError:
            import shutil
            from pathlib import Path
            dst_path = Path(str(dst))
            src_path = (dst_path.parent / src) if not Path(str(src)).is_absolute() else Path(str(src))
            if src_path.is_dir():
                shutil.copytree(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

    os.symlink = _symlink_or_copy


def get_stardist_model(model_name: str = "2D_versatile_fluo", disable_gpu: bool = False):
    """Get or load the StarDist model (singleton)."""
    global _stardist_model, _stardist_model_name

    if _stardist_model is not None and _stardist_model_name == model_name:
        return _stardist_model

    _patch_windows_symlink()
    _ensure_tf_threading(disable_gpu=disable_gpu)
    from stardist.models import StarDist2D
    _stardist_model = StarDist2D.from_pretrained(model_name)
    _stardist_model_name = model_name
    return _stardist_model


def segment_frame(
    image: np.ndarray,
    model=None,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    scale: Optional[float] = None,
    model_name: str = "2D_versatile_fluo",
    disable_gpu: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Segment nuclei in a single 2D frame using StarDist.

    Parameters
    ----------
    image : 2D array, any dtype
    model : StarDist2D instance, or None to use singleton
    prob_thresh : float, object probability threshold
    nms_thresh : float, non-maximum suppression threshold
    scale : float or None, rescale factor
    model_name : str, pretrained model name
    disable_gpu : bool, clear CUDA_VISIBLE_DEVICES before the first TF import

    Returns
    -------
    labels : 2D int array, each nucleus has a unique ID
    details : dict with prob, coord, etc.
    """
    from csbdeep.utils import normalize

    if model is None:
        model = get_stardist_model(model_name, disable_gpu=disable_gpu)

    img_norm = normalize(image, 1, 99.8)

    # Tiling for large images
    h, w = img_norm.shape
    n_tiles = None
    if max(h, w) > 1024:
        n_tiles = (max(1, h // 512), max(1, w // 512))

    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        scale=scale,
        n_tiles=n_tiles,
    )

    return labels, details


def segment_timeseries(
    timeseries: np.ndarray,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    scale: Optional[float] = None,
    model_name: str = "2D_versatile_fluo",
    progress_cb=None,
    frame_cb=None,
    disable_gpu: bool = False,
) -> np.ndarray:
    """
    Segment all frames of a timeseries.

    Parameters
    ----------
    timeseries : np.ndarray (T, H, W)
    progress_cb : callable(int), progress 0-100
    frame_cb : callable(int, np.ndarray, int), called per frame with
               (frame_idx, label_array, n_cells)

    Returns
    -------
    label_stack : np.ndarray (T, H, W) int32
    """
    T, H, W = timeseries.shape
    label_stack = np.zeros((T, H, W), dtype=np.int32)

    model = get_stardist_model(model_name, disable_gpu=disable_gpu)

    for t in range(T):
        labels, _ = segment_frame(
            timeseries[t],
            model=model,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            scale=scale,
            model_name=model_name,
        )
        label_stack[t] = labels

        n_cells = int(labels.max())
        if frame_cb:
            frame_cb(t, labels, n_cells)
        if progress_cb:
            progress_cb(int((t + 1) / T * 100))

    return label_stack
