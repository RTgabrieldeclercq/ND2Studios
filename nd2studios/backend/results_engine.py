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


# ── Core measurement computation ─────────────────────────────────────────────

def compute_measurements(
    label_masks: Dict[str, np.ndarray],
    channels: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    m_index: int = 0,
    volumetric_voxel_counts: Optional[Dict[str, Dict[Tuple[int, int], int]]] = None,
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

    # Materialise any lazy proxies so we can index freely.
    mat_channels: Dict[str, Optional[np.ndarray]] = {
        k: _materialise(v) for k, v in channels.items()
    }

    rows: List[Dict[str, Any]] = []

    for seg_channel, masks in label_masks.items():
        if masks.ndim != 3:
            continue
        T, H, W = masks.shape

        for t in range(T):
            mask_frame = np.asarray(masks[t], dtype=np.int32)
            if mask_frame.max() == 0:
                continue

            primary_img = _get_frame(mat_channels, seg_channel, t)
            props = regionprops(mask_frame, intensity_image=primary_img)

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
                    "perimeter": round(float(prop.perimeter), 3),
                    "circularity": round(
                        4.0 * 3.141592653589793 * float(prop.area)
                        / (float(prop.perimeter) ** 2)
                        if prop.perimeter > 0 else 0.0,
                        4,
                    ),
                    "eccentricity": round(float(prop.eccentricity), 4),
                    "solidity": (
                        round(float(prop.solidity), 4)
                        if prop.solidity is not None else None
                    ),
                    "bbox_min_row": int(prop.bbox[0]),
                    "bbox_min_col": int(prop.bbox[1]),
                    "bbox_max_row": int(prop.bbox[2]),
                    "bbox_max_col": int(prop.bbox[3]),
                }

                # Absolute stage coordinates (offset from frame centre)
                if stage_pos is not None:
                    sx, sy = stage_pos
                    row["centroid_x_stage_um"] = round(
                        sx + (cx_px - W / 2.0) * pixel_size, 4
                    )
                    row["centroid_y_stage_um"] = round(
                        sy + (cy_px - H / 2.0) * pixel_size, 4
                    )

                # Per-channel intensity stats
                for ch_name, ch_arr in mat_channels.items():
                    frame_data = _get_frame(mat_channels, ch_name, t)
                    if frame_data is None or frame_data.shape != mask_frame.shape:
                        continue
                    pixels = frame_data[mask_frame == prop.label]
                    if pixels.size:
                        safe = ch_name.replace(" ", "_")
                        row[f"mean_intensity_{safe}"] = round(float(pixels.mean()), 4)
                        row[f"std_intensity_{safe}"] = round(float(pixels.std()), 4)

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
) -> List[str]:
    """Export per-frame composite images with per-object colored label masks.

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
                label_ids = np.unique(frame_mask)
                for lid in label_ids:
                    if lid == 0:
                        continue
                    obj_pixels = frame_mask == lid
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

        fname = os.path.join(output_dir, f"frame_{t:04d}.{ext}")
        if ext == "tiff":
            imageio.imwrite(fname, rgb)
        else:
            imageio.imwrite(fname, rgb, quality=92)
        written.append(fname)

        if progress_cb is not None and (t % 4 == 0 or t == T - 1):
            progress_cb(int((t + 1) / T * 100))

    return written


def export_label_masks_tiff(
    label_masks: Dict[str, np.ndarray],
    output_dir: str,
) -> List[str]:
    """Export each label mask stack as an int32 TIFF (one file per channel)."""
    import tifffile

    written: List[str] = []
    for ch_name, masks in label_masks.items():
        safe = ch_name.replace(" ", "_").replace("/", "_")
        fpath = os.path.join(output_dir, f"labels_{safe}.tif")
        arr = np.asarray(masks, dtype=np.int32)
        tifffile.imwrite(fpath, arr, imagej=True)
        written.append(fpath)
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
) -> List[str]:
    """Export label masks as colored per-object overlays on top of image data.

    Each output frame shows the raw/processed image as the background with
    label masks rendered using per-object distinct colors at *mask_alpha*
    opacity — matching the image-viewer overlay appearance.

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
                label_ids = np.unique(frame_mask)
                for lid in label_ids:
                    if lid == 0:
                        continue
                    obj_pixels = frame_mask == lid
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

            fname = os.path.join(output_dir, f"overlay_{safe_seg}_frame_{t:04d}.{ext}")
            if ext == "tiff":
                imageio.imwrite(fname, rgb)
            else:
                imageio.imwrite(fname, rgb, quality=92)
            written.append(fname)

            done += 1
            if progress_cb is not None and (done % 4 == 0 or done == total_frames):
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


