"""
Pure-backend measurement engine for the Results tab (V1.22).

Computes extended per-object measurements from AnalysisResult label masks
plus the source channel arrays.  No Qt imports — callable from workers and
headless scripts.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops

from nd2studios.backend.exporters.composite_exporter import (
    CHANNEL_COLORS, _composite_frame,
)

# Visually distinct colors cycled by label ID for per-object mask overlays.
_LABEL_PALETTE: List[Tuple[int, int, int]] = [
    (240, 60,  60),   # red
    (60,  210, 60),   # green
    (60,  100, 240),  # blue
    (240, 200, 50),   # yellow
    (200, 60,  200),  # magenta
    (50,  200, 200),  # cyan
    (240, 130, 50),   # orange
    (150, 60,  240),  # purple
    (60,  240, 150),  # mint
    (240, 60,  150),  # pink
    (110, 200, 60),   # lime
    (60,  150, 240),  # sky blue
]


def _label_boundaries(frame_mask: np.ndarray) -> np.ndarray:
    """Boolean boundary mask for an integer label frame (outline-overlay mode).

    ``mode="inner"`` keeps boundary pixels object-side so per-label callers can
    attribute them by intersecting with ``(mask == label_id)`` and paint each
    object's own outline.
    """
    from skimage.segmentation import find_boundaries
    return find_boundaries(frame_mask, mode="inner")


# ── Core measurement computation ─────────────────────────────────────────────

def compute_measurements(
    label_masks: Dict[str, np.ndarray],
    channels: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    m_index: int = 0,
    volumetric_voxel_counts: Optional[Dict[str, Dict[Tuple[int, int], int]]] = None,
    metrics: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Compute extended per-object measurements from label masks.

    Args:
        label_masks: {seg_channel_name: (T, H, W) int32}  from AnalysisResult
        channels:    {channel_name: (T, H, W) array}       processed or raw
        metadata:    nd2_metadata dict  (pixel_size_um, z_step_um, n_zslices,
                     stage_xy_um, …)
        m_index:     multipoint index used for absolute stage coordinates
        volumetric_voxel_counts: optional per-channel per-(frame, label_id)
            voxel count from the pipeline. When supplied, ``volume_um3``
            comes from the true 3D voxel count; otherwise it falls back to
            the uniform-Z assumption ``area_um2 x n_zslices x z_step_um``.

    Returns:
        List of measurement dicts, one dict per detected object per frame.
        Columns: segmentation_channel, frame, label_id, area_px, area_um2,
        delta_area_px, delta_area_um2, volume_um3, delta_volume_um3,
        centroid_y/x_px, centroid_y/x_um, [centroid_y/x_stage_um],
        perimeter, circularity, eccentricity, solidity, bbox_*,
        mean/std_intensity_{ch}.
    """
    pixel_size: float = float(metadata.get("pixel_size_um") or 1.0)
    z_step: float = float(metadata.get("z_step_um") or 1.0)
    n_zslices: int = max(1, int(metadata.get("n_zslices") or 1))
    stage_xy = metadata.get("stage_xy_um") or []
    stage_pos: Optional[Tuple[float, float]] = (
        (float(stage_xy[m_index][0]), float(stage_xy[m_index][1]))
        if (stage_xy and m_index < len(stage_xy))
        else None
    )
    voxel_counts = volumetric_voxel_counts or {}

    # Selective computation (V1.45 Phase 2): when ``metrics`` is given, skip the
    # expensive shape props and per-channel intensity unless requested, and prune
    # the output columns to the selection (identity columns are always kept). The
    # cheap regionprops fields (area / centroid / bbox) are computed regardless
    # and pruned at the end. ``metrics`` uses the column keys plus the group
    # tokens ``mean_intensity`` / ``std_intensity`` for the per-channel stats.
    want = set(metrics) if metrics is not None else None
    want_shape = want is None or bool(
        want & {"perimeter", "circularity", "eccentricity", "solidity"})
    want_intensity = want is None or bool(want & {"mean_intensity", "std_intensity"})
    want_mean = want is None or "mean_intensity" in want
    want_std = want is None or "std_intensity" in want

    # V1.46 — do NOT materialize all channels up front. ``channels`` may be
    # lazy readers (LazyND2Channel / memmap / zarr); we read one frame per
    # channel per timepoint into ``frame_cache`` below, so peak RAM is bounded
    # to a single frame per channel rather than the whole dataset.

    rows: List[Dict[str, Any]] = []

    for seg_channel, masks in label_masks.items():
        if getattr(masks, "ndim", 0) != 3:
            continue
        T, H, W = masks.shape

        for t in range(T):
            mask_frame = np.asarray(masks[t], dtype=np.int32)
            if mask_frame.max() == 0:
                continue

            # Read each channel's frame t exactly once (lazy-friendly).
            frame_cache: Dict[str, Optional[np.ndarray]] = {
                name: _get_frame(channels, name, t) for name in channels
            }

            primary_img = frame_cache.get(seg_channel)
            props = regionprops(mask_frame, intensity_image=primary_img)

            # Per-channel per-label intensity stats, computed once per channel in a
            # single labelled pass (scipy.ndimage), instead of the old full-frame
            # boolean scan ``frame_data[mask_frame == prop.label]`` *per object*.
            # The per-object scan is O(n_objects × frame_pixels) and dominated
            # runtime on large, dense frames (e.g. a 4096² frame with thousands of
            # nuclei took ~50 s/frame, stalling measurement — and the tracking that
            # depends on it — for many minutes). ``{ch: (mean_by_label,
            # std_by_label)}``; populated only for the requested intensity stats.
            intensity_stats: Dict[str, Tuple[Dict[int, float], Dict[int, float]]] = {}
            if want_intensity and props:
                from scipy import ndimage as _ndi
                label_ids = np.array([p.label for p in props], dtype=np.int64)
                for ch_name, frame_data in frame_cache.items():
                    if frame_data is None or frame_data.shape != mask_frame.shape:
                        continue
                    fdata = np.asarray(frame_data)
                    mean_map: Dict[int, float] = {}
                    std_map: Dict[int, float] = {}
                    if want_mean:
                        means = np.atleast_1d(
                            _ndi.mean(fdata, labels=mask_frame, index=label_ids))
                        mean_map = dict(zip(label_ids.tolist(), means.tolist()))
                    if want_std:
                        stds = np.atleast_1d(
                            _ndi.standard_deviation(
                                fdata, labels=mask_frame, index=label_ids))
                        std_map = dict(zip(label_ids.tolist(), stds.tolist()))
                    intensity_stats[ch_name] = (mean_map, std_map)

            for prop in props:
                cy_px, cx_px = prop.centroid

                # Volume from real voxel count if the pipeline supplied one
                # (e.g., Manual Mask with per-Z shapes); else apply the
                # uniform-Z assumption: area × n_z × z_step. Z step is in µm
                # so the result is in µm³.
                ch_voxels = voxel_counts.get(seg_channel) or {}
                voxel_count = ch_voxels.get((int(t), int(prop.label)))
                if voxel_count is not None:
                    volume_um3 = float(voxel_count) * (pixel_size ** 2) * z_step
                else:
                    volume_um3 = float(prop.area) * (pixel_size ** 2) * n_zslices * z_step

                row: Dict[str, Any] = {
                    "segmentation_channel": seg_channel,
                    "frame": int(t),
                    "label_id": int(prop.label),
                    "area_px": int(prop.area),
                    "area_um2": round(prop.area * pixel_size ** 2, 4),
                    "volume_um3": round(volume_um3, 4),
                    "centroid_y_px": round(cy_px, 3),
                    "centroid_x_px": round(cx_px, 3),
                    "centroid_y_um": round(cy_px * pixel_size, 4),
                    "centroid_x_um": round(cx_px * pixel_size, 4),
                    "bbox_min_row": int(prop.bbox[0]),
                    "bbox_min_col": int(prop.bbox[1]),
                    "bbox_max_row": int(prop.bbox[2]),
                    "bbox_max_col": int(prop.bbox[3]),
                }

                # Shape props are the costlier regionprops — skip when unselected.
                if want_shape:
                    perim = float(prop.perimeter)
                    row["perimeter"] = round(perim, 3)
                    row["circularity"] = round(
                        4.0 * 3.141592653589793 * float(prop.area) / (perim ** 2)
                        if perim > 0 else 0.0, 4)
                    row["eccentricity"] = round(float(prop.eccentricity), 4)
                    row["solidity"] = (
                        round(float(prop.solidity), 4)
                        if prop.solidity is not None else None)

                # Absolute stage coordinates (offset from frame centre)
                if stage_pos is not None:
                    sx, sy = stage_pos
                    row["centroid_x_stage_um"] = round(
                        sx + (cx_px - W / 2.0) * pixel_size, 4
                    )
                    row["centroid_y_stage_um"] = round(
                        sy + (cy_px - H / 2.0) * pixel_size, 4
                    )

                # Per-channel intensity stats — looked up from the per-frame
                # vectorised maps computed above (no per-object frame scan).
                for ch_name, (mean_map, std_map) in intensity_stats.items():
                    safe = ch_name.replace(" ", "_")
                    if want_mean:
                        mv = mean_map.get(prop.label)
                        if mv is not None and np.isfinite(mv):
                            row[f"mean_intensity_{safe}"] = round(float(mv), 4)
                    if want_std:
                        sv = std_map.get(prop.label)
                        if sv is not None and np.isfinite(sv):
                            row[f"std_intensity_{safe}"] = round(float(sv), 4)

                rows.append(row)

    # Per-track per-frame ΔArea — applied to every pipeline so any mask source
    # (manual, threshold, nuclei, spots, …) gets it for free. Sort by
    # (segmentation_channel, label_id, frame) so consecutive rows belong to
    # the same "track" and we can diff in one pass.
    #
    # Caveat: for instance-segmentation pipelines (nuclei, spots) label_id is
    # not stable across frames — regionprops assigns fresh ids per frame.
    # ΔArea for those reflects "label k between consecutive frames" which is
    # only meaningful after tracking. For single-object pipelines and the
    # manual mask (where label_id is user-controlled) the value is exact.
    # TODO V1.27+: add delta_volume_um3 when Z-stacks land.
    rows.sort(key=lambda r: (r["segmentation_channel"], int(r["label_id"]), int(r["frame"])))
    prev_key = None
    prev_area_px: Optional[int] = None
    prev_area_um2: Optional[float] = None
    prev_volume_um3: Optional[float] = None
    for r in rows:
        key = (r["segmentation_channel"], int(r["label_id"]))
        if key != prev_key or prev_area_px is None:
            r["delta_area_px"] = None
            r["delta_area_um2"] = None
            r["delta_volume_um3"] = None
        else:
            r["delta_area_px"] = int(r["area_px"]) - prev_area_px
            r["delta_area_um2"] = round(float(r["area_um2"]) - prev_area_um2, 4)
            r["delta_volume_um3"] = round(
                float(r["volume_um3"]) - (prev_volume_um3 or 0.0), 4
            )
        prev_key = key
        prev_area_px = int(r["area_px"])
        prev_area_um2 = float(r["area_um2"])
        prev_volume_um3 = float(r["volume_um3"])

    # Prune to the selected metrics (identity columns always kept; the
    # ``mean_intensity`` / ``std_intensity`` tokens keep all per-channel columns).
    if want is not None:
        keep_exact = {"segmentation_channel", "frame", "label_id"} | (
            want - {"mean_intensity", "std_intensity"})
        for r in rows:
            for k in list(r.keys()):
                if k in keep_exact:
                    continue
                if k.startswith("mean_intensity") and want_mean:
                    continue
                if k.startswith("std_intensity") and want_std:
                    continue
                del r[k]

    return rows


# ── Image export with label overlay ──────────────────────────────────────────

def export_overlay_frames(
    channels: Dict[str, np.ndarray],
    label_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_dir: str,
    fmt: str = "tiff",
    channel_display: Optional[Dict[str, Any]] = None,
    pixel_size_um: float = 0.0,
    show_scale_bar: bool = True,
    scale_bar_um: float = 50.0,
    show_channel_labels: bool = True,
    mask_alpha: float = 0.5,
    frame_timestamps: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    basename: str = "",
    outline: bool = False,
) -> List[str]:
    """Export per-frame composite images with per-object colored label masks.

    When *outline* is True, only each object's boundary pixels are painted, so
    cells render as outlines (matching the boundary-outline overlay mode).

    For each T frame: builds a uint8 RGB composite from *channels*, overlays
    all label masks using per-object distinct colors at *mask_alpha* opacity,
    then burns in a scale bar and channel labels if requested.

    Args:
        channels:            {name: (T, H, W)} arrays
        label_masks:         {seg_channel: (T, H, W) int32}
        metadata:            nd2_metadata dict
        output_dir:          directory to write files into (must exist)
        fmt:                 "tiff" or "jpg"
        channel_display:     {name: {color, lut_lo, lut_hi}} — viewer state
        pixel_size_um:       µm per pixel for scale bar; ≤0 disables bar
        show_scale_bar:      draw scale bar when pixel_size_um > 0
        scale_bar_um:        physical length of the scale bar in µm
        show_channel_labels: draw per-channel color swatches + names
        mask_alpha:          opacity of the label overlay (0–1)
        frame_timestamps:    per-frame timestamps in seconds (optional)
        progress_cb:         called with 0–100 progress values

    Returns:
        List of written file paths.
    """
    import imageio

    mat_channels = {k: _materialise(v) for k, v in channels.items()}

    T = 1
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            T = arr.shape[0]
            break

    H = W = 0
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            _, H, W = arr.shape
            break

    ext = "tiff" if fmt.lower() in ("tif", "tiff") else "jpg"

    default_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]

    written: List[str] = []
    for t in range(T):
        # ── RGB composite from enabled channels ──
        ch_list = list(mat_channels.keys())
        ch_colors: Dict[str, Tuple[int, int, int]] = {}
        ch_enabled: Dict[str, bool] = {}
        frame_dict: Dict[str, np.ndarray] = {}
        for ci, ch_name in enumerate(ch_list):
            arr = _get_frame(mat_channels, ch_name, t)
            disp = (channel_display or {}).get(ch_name, {})
            ch_enabled[ch_name] = bool(disp.get("enabled", True))
            color_name: str = disp.get("color") or ""
            ch_colors[ch_name] = (
                CHANNEL_COLORS.get(color_name)
                or default_colors[ci % len(default_colors)]
            )
            if arr is not None:
                frame_dict[ch_name] = arr
        if frame_dict:
            rgb = _composite_frame(frame_dict, ch_colors, ch_enabled, lut_settings=None)
        else:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Per-object colored mask overlay ──
        if label_masks:
            overlay = rgb.astype(np.float32)
            for masks in label_masks.values():
                if masks.ndim != 3 or t >= masks.shape[0]:
                    continue
                frame_mask = np.asarray(masks[t], dtype=np.int32)
                boundaries = _label_boundaries(frame_mask) if outline else None
                label_ids = np.unique(frame_mask)
                for lid in label_ids:
                    if lid == 0:
                        continue
                    obj_pixels = frame_mask == lid
                    if boundaries is not None:
                        obj_pixels = obj_pixels & boundaries
                    color = _LABEL_PALETTE[(int(lid) - 1) % len(_LABEL_PALETTE)]
                    for c in range(3):
                        overlay[obj_pixels, c] = (
                            overlay[obj_pixels, c] * (1.0 - mask_alpha)
                            + color[c] * mask_alpha
                        )
            rgb = np.clip(overlay, 0, 255).astype(np.uint8)

        # ── Scale bar + channel labels via PIL ──
        needs_overlay = (
            (show_scale_bar and pixel_size_um > 0) or show_channel_labels
        )
        if needs_overlay:
            rgb = _draw_image_overlays(
                rgb,
                t_index=t,
                pixel_size_um=pixel_size_um,
                show_scale_bar=show_scale_bar,
                scale_bar_um=scale_bar_um,
                show_channel_labels=show_channel_labels,
                channel_names=[n for n in ch_list if ch_enabled.get(n, True)],
                channel_colors=ch_colors,
                frame_timestamps=frame_timestamps,
            )

        prefix = f"{basename}_overlay" if basename else "overlay"
        fname = os.path.join(output_dir, f"{prefix}_{t:04d}.{ext}")
        if ext == "tiff":
            imageio.imwrite(fname, rgb)
        else:
            imageio.imwrite(fname, rgb, quality=92)
        written.append(fname)

        if progress_cb is not None:
            progress_cb(int((t + 1) / T * 100))

    return written


def export_label_masks_tiff(
    label_masks: Dict[str, np.ndarray],
    output_dir: str,
    basename: str = "",
) -> List[str]:
    """Export each label mask stack as an int32 TIFF (one file per channel)."""
    import tifffile

    written: List[str] = []
    for ch_name, masks in label_masks.items():
        safe = ch_name.replace(" ", "_").replace("/", "_")
        prefix = f"{basename}_masks" if basename else "labels"
        fpath = os.path.join(output_dir, f"{prefix}_{safe}.tif")
        arr = np.asarray(masks, dtype=np.int32)
        # Plain TIFF, not ``imagej=True``: the ImageJ hyperstack format rejects
        # 32-bit integer data ("data type 'l'"). A standard multipage int32 TIFF
        # preserves the exact label IDs and still opens in ImageJ as a 32-bit
        # stack.
        tifffile.imwrite(fpath, arr)
        written.append(fpath)
    return written


# ── Organized (folding) export ────────────────────────────────────────────────
# A single "frame organization" choice decides which axes (M / T / Z) FOLD into a
# file (become its pages/stack) vs SPLIT into separate files. Z is collapsed at
# load (analysis masks are per-(M, T)), so Z-fold options coincide with their
# non-Z siblings until per-Z masks land — the formula is general regardless.
_ORG_FOLD: Dict[str, frozenset] = {
    "per_frame": frozenset(),
    "fold_M": frozenset({"M"}),     # M folds into T → one file per T
    "fold_T": frozenset({"T"}),     # T folds into M → one file per M
    "fold_Z": frozenset({"Z"}),
    "fold_MZ": frozenset({"M", "Z"}),
    "fold_TZ": frozenset({"T", "Z"}),
    "fold_MT": frozenset({"M", "T"}),
    "fold_all": frozenset({"M", "T", "Z"}),
}

# User-facing labels → fold keys (ordered). Imported by the registry adapter
# (param choices) and the Pipelines page (label → key lookup).
EXPORT_ORG_MAP: Dict[str, str] = {
    "Per frame (split M, T, Z)": "per_frame",
    "M folds into T (one file per T)": "fold_M",
    "T folds into M (one file per M)": "fold_T",
    "Z folds (one file per M, T)": "fold_Z",
    "M and Z fold into T (one file per T)": "fold_MZ",
    "T and Z fold into M (one file per M)": "fold_TZ",
    "M and T fold (one file per Z)": "fold_MT",
    "All frames merged (one file)": "fold_all",
}
EXPORT_ORG_DEFAULT = "T folds into M (one file per M)"


def _label_to_rgb(mask2d: np.ndarray, outline: bool = False) -> np.ndarray:
    """Color a 2-D int label image with the per-object palette (for PNG/JPG).

    When *outline* is True, only boundary pixels are colored (cells as outlines).
    """
    mask2d = np.asarray(mask2d, dtype=np.int32)
    out = np.zeros(mask2d.shape + (3,), dtype=np.uint8)
    boundaries = _label_boundaries(mask2d) if outline else None
    for lid in np.unique(mask2d):
        if lid == 0:
            continue
        px = (mask2d == lid)
        if boundaries is not None:
            px = px & boundaries
        out[px] = _LABEL_PALETTE[(int(lid) - 1) % len(_LABEL_PALETTE)]
    return out


def _render_export_rgb(chans_t, masks_t, channel_display, mask_alpha,
                       pixel_size_um, show_scale_bar, scale_bar_um,
                       show_channel_labels, outline: bool = False) -> np.ndarray:
    """One (H, W, 3) composite of ``chans_t`` with ``masks_t`` overlaid — mirrors
    a single frame of :func:`export_overlay_frames`. *outline* paints only object
    boundary pixels (cells as outlines)."""
    default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                      (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    ch_list = list(chans_t.keys())
    ch_colors: Dict[str, Tuple[int, int, int]] = {}
    ch_enabled: Dict[str, bool] = {}
    frame_dict: Dict[str, np.ndarray] = {}
    H = W = 0
    for ci, ch in enumerate(ch_list):
        a = chans_t.get(ch)
        disp = (channel_display or {}).get(ch, {})
        ch_enabled[ch] = bool(disp.get("enabled", True))
        ch_colors[ch] = (CHANNEL_COLORS.get(disp.get("color") or "")
                         or default_colors[ci % len(default_colors)])
        if a is not None:
            frame_dict[ch] = a
            H, W = a.shape[-2], a.shape[-1]
    rgb = (_composite_frame(frame_dict, ch_colors, ch_enabled, lut_settings=None)
           if frame_dict else np.zeros((H, W, 3), dtype=np.uint8))
    if masks_t:
        ov = rgb.astype(np.float32)
        for fm in masks_t.values():
            fm = np.asarray(fm, dtype=np.int32)
            boundaries = _label_boundaries(fm) if outline else None
            for lid in np.unique(fm):
                if lid == 0:
                    continue
                px = fm == lid
                if boundaries is not None:
                    px = px & boundaries
                col = _LABEL_PALETTE[(int(lid) - 1) % len(_LABEL_PALETTE)]
                for c in range(3):
                    ov[px, c] = ov[px, c] * (1.0 - mask_alpha) + col[c] * mask_alpha
        rgb = np.clip(ov, 0, 255).astype(np.uint8)
    if (show_scale_bar and pixel_size_um > 0) or show_channel_labels:
        rgb = _draw_image_overlays(
            rgb, t_index=0, pixel_size_um=pixel_size_um,
            show_scale_bar=show_scale_bar, scale_bar_um=scale_bar_um,
            show_channel_labels=show_channel_labels,
            channel_names=[n for n in ch_list if ch_enabled.get(n, True)],
            channel_colors=ch_colors, frame_timestamps=None)
    return rgb


def export_organized(
    frames_by_m: Dict[int, Dict[str, np.ndarray]],
    masks_by_m: Dict[int, Dict[str, np.ndarray]],
    output_dir: str,
    *,
    fmt: str = "tiff",
    organization: str = "fold_T",
    channel_display: Optional[Dict[str, Any]] = None,
    pixel_size_um: float = 0.0,
    mask_alpha: float = 0.5,
    show_scale_bar: bool = True,
    scale_bar_um: float = 50.0,
    show_channel_labels: bool = True,
    overlay: bool = True,
    basename: str = "pipeline",
    progress_cb: Optional[Callable[[int], None]] = None,
    outline: bool = False,
) -> List[str]:
    """Write per-object overlays (or label masks) organized by ``organization``.

    ``frames_by_m`` = ``{m: {channel: (T, H, W)}}`` processed channels;
    ``masks_by_m`` = ``{m: {seg_channel: (T, H, W) int}}`` label masks. The
    organization key (see :data:`_ORG_FOLD`) chooses which of M / T / Z fold into
    each file (its pages) vs split into separate files. TIFF writes a multi-page
    stack per group; PNG / JPG write a single file (one page) or a per-group
    subfolder of numbered frames. ``overlay`` renders RGB composites; otherwise
    label images (int32 for TIFF, colored for PNG/JPG). Returns written paths.
    """
    import imageio
    import tifffile
    from itertools import product

    fold = _ORG_FOLD.get(organization, frozenset())
    fmtl = (fmt or "tiff").lower()
    ext = ("tif" if fmtl in ("tif", "tiff")
           else "jpg" if fmtl in ("jpg", "jpeg") else "png")
    is_tiff = ext == "tif"

    frames_by_m = {int(m): {k: _materialise(v) for k, v in (d or {}).items()}
                   for m, d in (frames_by_m or {}).items()}
    masks_by_m = {int(m): {k: np.asarray(v) for k, v in (d or {}).items()}
                  for m, d in (masks_by_m or {}).items()}
    ms = sorted(set(frames_by_m) | set(masks_by_m))
    if not ms:
        return []
    T = 1
    for d in list(masks_by_m.values()) + list(frames_by_m.values()):
        for a in d.values():
            if getattr(a, "ndim", 0) == 3:
                T = max(T, a.shape[0])
    axes = {"M": ms, "T": list(range(T)), "Z": [0]}

    def _frame_at(m: int, t: int):
        chans_t = {ch: _get_frame(frames_by_m.get(m, {}), ch, t)
                   for ch in frames_by_m.get(m, {})}
        masks_t = {}
        for seg, arr in masks_by_m.get(m, {}).items():
            if arr.ndim == 3 and t < arr.shape[0]:
                masks_t[seg] = arr[t]
            elif arr.ndim == 2:
                masks_t[seg] = arr
        if overlay:
            return _render_export_rgb(
                chans_t, masks_t, channel_display, mask_alpha, pixel_size_um,
                show_scale_bar, scale_bar_um, show_channel_labels, outline=outline)
        seg = next(iter(masks_t), None)
        if seg is None:
            return None
        arr = np.asarray(masks_t[seg], dtype=np.int32)
        return arr if is_tiff else _label_to_rgb(arr, outline=outline)

    split = [a for a in ("M", "T", "Z") if a not in fold]
    foldax = [a for a in ("M", "T", "Z") if a in fold]
    split_combos = list(product(*[axes[a] for a in split])) or [()]
    fold_combos = list(product(*[axes[a] for a in foldax])) or [()]

    written: List[str] = []
    total = max(1, len(split_combos))
    for gi, sc in enumerate(split_combos):
        sv = dict(zip(split, sc))
        pages = []
        for fc in fold_combos:
            fv = dict(zip(foldax, fc))
            m = int(sv.get("M", fv.get("M", ms[0])))
            t = int(sv.get("T", fv.get("T", 0)))
            img = _frame_at(m, t)
            if img is not None:
                pages.append(img)
        if not pages:
            continue
        tag = "".join(f"_{a}{int(v) + 1:03d}" for a, v in zip(split, sc)) or "_all"
        base = f"{basename}{tag}"
        if is_tiff:
            fp = os.path.join(output_dir, f"{base}.tif")
            tifffile.imwrite(fp, np.stack(pages, axis=0))
            written.append(fp)
        elif len(pages) == 1:
            fp = os.path.join(output_dir, f"{base}.{ext}")
            imageio.imwrite(fp, pages[0], **({"quality": 92} if ext == "jpg" else {}))
            written.append(fp)
        else:
            sub = os.path.join(output_dir, base)
            os.makedirs(sub, exist_ok=True)
            for i, pg in enumerate(pages):
                fp = os.path.join(sub, f"frame_{i:04d}.{ext}")
                imageio.imwrite(fp, pg, **({"quality": 92} if ext == "jpg" else {}))
                written.append(fp)
        if progress_cb is not None:
            progress_cb(int((gi + 1) / total * 100))
    return written


def export_label_masks_as_overlay(
    channels: Dict[str, np.ndarray],
    label_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_dir: str,
    fmt: str = "tiff",
    channel_display: Optional[Dict[str, Any]] = None,
    pixel_size_um: float = 0.0,
    show_scale_bar: bool = True,
    scale_bar_um: float = 50.0,
    show_channel_labels: bool = True,
    mask_alpha: float = 0.5,
    progress_cb: Optional[Callable[[int], None]] = None,
    basename: str = "",
    outline: bool = False,
) -> List[str]:
    """Export label masks as colored per-object overlays on top of image data.

    Each output frame shows the raw/processed image as the background with
    label masks rendered using per-object distinct colors at *mask_alpha*
    opacity — matching the image-viewer overlay appearance. *outline* paints
    only object boundary pixels (cells as outlines).

    Returns:
        List of written file paths (one per T frame per segmentation channel).
    """
    import imageio

    mat_channels = {k: _materialise(v) for k, v in channels.items()}

    T = 1
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            T = arr.shape[0]
            break

    H = W = 0
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            _, H, W = arr.shape
            break

    ext = "tiff" if fmt.lower() in ("tif", "tiff") else "jpg"
    default_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]

    total_frames = T * max(1, len(label_masks))
    done = 0
    written: List[str] = []

    for seg_ch_name, masks in label_masks.items():
        safe_seg = seg_ch_name.replace(" ", "_").replace("/", "_")
        for t in range(T):
            # ── RGB composite background ──
            ch_list = list(mat_channels.keys())
            ch_colors: Dict[str, Tuple[int, int, int]] = {}
            ch_enabled: Dict[str, bool] = {}
            frame_dict: Dict[str, np.ndarray] = {}
            for ci, ch_name in enumerate(ch_list):
                arr = _get_frame(mat_channels, ch_name, t)
                disp = (channel_display or {}).get(ch_name, {})
                ch_enabled[ch_name] = bool(disp.get("enabled", True))
                color_name: str = disp.get("color") or ""
                ch_colors[ch_name] = (
                    CHANNEL_COLORS.get(color_name)
                    or default_colors[ci % len(default_colors)]
                )
                if arr is not None:
                    frame_dict[ch_name] = arr
            if frame_dict:
                rgb = _composite_frame(frame_dict, ch_colors, ch_enabled, lut_settings=None)
            else:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)

            # ── Per-object colored mask for this segmentation channel ──
            if masks.ndim == 3 and t < masks.shape[0]:
                frame_mask = np.asarray(masks[t], dtype=np.int32)
                overlay = rgb.astype(np.float32)
                boundaries = _label_boundaries(frame_mask) if outline else None
                label_ids = np.unique(frame_mask)
                for lid in label_ids:
                    if lid == 0:
                        continue
                    obj_pixels = frame_mask == lid
                    if boundaries is not None:
                        obj_pixels = obj_pixels & boundaries
                    color = _LABEL_PALETTE[(int(lid) - 1) % len(_LABEL_PALETTE)]
                    for c in range(3):
                        overlay[obj_pixels, c] = (
                            overlay[obj_pixels, c] * (1.0 - mask_alpha)
                            + color[c] * mask_alpha
                        )
                rgb = np.clip(overlay, 0, 255).astype(np.uint8)

            # ── Scale bar + channel labels ──
            needs_overlay = (
                (show_scale_bar and pixel_size_um > 0) or show_channel_labels
            )
            if needs_overlay:
                rgb = _draw_image_overlays(
                    rgb,
                    t_index=t,
                    pixel_size_um=pixel_size_um,
                    show_scale_bar=show_scale_bar,
                    scale_bar_um=scale_bar_um,
                    show_channel_labels=show_channel_labels,
                    channel_names=[n for n in ch_list if ch_enabled.get(n, True)],
                    channel_colors=ch_colors,
                    frame_timestamps=None,
                )

            prefix = f"{basename}_masks" if basename else "overlay"
            fname = os.path.join(output_dir, f"{prefix}_{safe_seg}_{t:04d}.{ext}")
            if ext == "tiff":
                imageio.imwrite(fname, rgb)
            else:
                imageio.imwrite(fname, rgb, quality=92)
            written.append(fname)

            done += 1
            if progress_cb is not None:
                progress_cb(int(done / total_frames * 100))

    return written


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_image_overlays(
    rgb: np.ndarray,
    t_index: int,
    pixel_size_um: float,
    show_scale_bar: bool,
    scale_bar_um: float,
    show_channel_labels: bool,
    channel_names: List[str],
    channel_colors: Dict[str, Tuple[int, int, int]],
    frame_timestamps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Burn scale bar and channel labels into an RGB frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return rgb

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    margin = 24

    def _font(size: int):
        candidates = [
            "Arial.ttf",
            "DejaVuSans.ttf",
            "Helvetica.ttc",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    if show_scale_bar and pixel_size_um > 0:
        bar_px = int(round(scale_bar_um / pixel_size_um))
        if 0 < bar_px < w:
            thickness = 6
            x1 = w - margin
            x0 = x1 - bar_px
            y0 = h - margin - thickness
            y1 = h - margin
            draw.rectangle([x0, y0, x1, y1], fill="white")
            label = f"{scale_bar_um:g} µm"
            font = _font(14)
            try:
                tw = draw.textlength(label, font=font)
            except AttributeError:
                tw = len(label) * 8
            tx = (x0 + x1) // 2 - int(tw / 2)
            draw.text((tx, y0 - 18), label, fill="white", font=font)

    if show_channel_labels and channel_names:
        font = _font(13)
        line_h = 20
        x = margin
        y = margin
        for name in channel_names:
            color = channel_colors.get(name, (255, 255, 255))
            sw = 14
            draw.rectangle([x, y, x + sw, y + sw], fill=color)
            draw.text((x + sw + 5, y), name, fill="white", font=font)
            y += line_h

    return np.asarray(img)


def _materialise(data: Any) -> Optional[np.ndarray]:
    if data is None:
        return None
    if hasattr(data, "materialize") and callable(data.materialize):
        return data.materialize()
    try:
        return np.asarray(data)
    except Exception:
        return None


def _get_frame(
    channels: Dict[str, Optional[np.ndarray]],
    name: str,
    t: int,
) -> Optional[np.ndarray]:
    arr = channels.get(name)
    if arr is None:
        return None
    try:
        if arr.ndim == 3:
            return arr[t].astype(np.float32, copy=False)
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
    except (IndexError, TypeError):
        pass
    return None


