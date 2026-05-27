"""
Tracked objects exporter.

Crops each tracked object at 3× its bounding-box size (matching the validation
dialog), optionally composites image channels and draws the mask highlight,
then tiles multiple objects side-by-side per output file and writes TIFF or PNG.

Output files are named:  tracked_objects_M000.tif / tracked_objects_M001.tif …
(or …_M000_T000.png / …_M001_T001.png for multi-frame PNG series).

Called from ExportPage._on_export_tracked_objects.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_CROP_FACTOR = 3.0
_HIGHLIGHT = np.array([255, 200, 50], dtype=np.float32)
_HIGHLIGHT_ALPHA = 0.55


def export_tracked_objects(
    measurements: List[Dict[str, Any]],
    label_masks: Dict[str, np.ndarray],
    channels: Dict[str, np.ndarray],
    channel_display: Dict[str, Dict],
    output_dir: str,
    objects_per_m: int = 1,
    with_image: bool = True,
    with_mask_overlay: bool = True,
    fmt: str = "tiff",
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[str]:
    """Export tracked objects as cropped TIFF or PNG tiles.

    Parameters
    ----------
    measurements:
        Rows from ResultsPage._measurements where track_id is not None.
    label_masks:
        {segmentation_channel: (T, H, W) int32} label arrays.
    channels:
        {channel_name: (T, H, W)} materialized image data.
    channel_display:
        Per-channel LUT/color config from exp.channel_display.
    output_dir:
        Directory to write files into (must exist).
    objects_per_m:
        Number of tracked objects tiled horizontally per output file.
    with_image:
        Composite image channels into the crop background.
    with_mask_overlay:
        Draw the yellow-orange mask highlight over the object pixels.
    fmt:
        "tiff" or "png".
    progress_cb:
        Optional 0–100 integer progress callback.

    Returns
    -------
    List of absolute file paths written.
    """
    import tifffile

    # Group rows by track_id.
    track_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in measurements:
        tid = r.get("track_id")
        if tid is not None:
            track_rows[int(tid)].append(r)

    # Stable output order: tracks sorted by their earliest frame.
    sorted_tracks = sorted(
        track_rows.items(),
        key=lambda kv: min(int(r.get("frame", 0)) for r in kv[1]),
    )
    n_tracks = len(sorted_tracks)
    if n_tracks == 0:
        return []

    # Determine full-frame dimensions from the first available array.
    H, W = 0, 0
    if channels:
        first_ch = np.asarray(next(iter(channels.values())))
        if first_ch.ndim >= 2:
            H, W = first_ch.shape[-2], first_ch.shape[-1]
    if (H == 0 or W == 0) and label_masks:
        first_mask = next(iter(label_masks.values()))
        H, W = first_mask.shape[1], first_mask.shape[2]

    # Render every (track, frame) as an RGB crop array.
    rendered_tracks: List[List[np.ndarray]] = []
    for ti, (tid, rows) in enumerate(sorted_tracks):
        frames_sorted = sorted(rows, key=lambda r: int(r.get("frame", 0)))
        crop_frames = _render_track(
            frames_sorted, label_masks, channels, channel_display,
            H, W, with_image, with_mask_overlay,
        )
        rendered_tracks.append(crop_frames)
        if progress_cb:
            progress_cb(int(50 * (ti + 1) / n_tracks))

    # Pad all crops to the same size for uniform tiling.
    all_h = [f.shape[0] for crops in rendered_tracks for f in crops]
    all_w = [f.shape[1] for crops in rendered_tracks for f in crops]
    pad_h = max(all_h) if all_h else 64
    pad_w = max(all_w) if all_w else 64

    def _pad(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == pad_h and w == pad_w:
            return img
        out = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        out[:h, :w] = img
        return out

    # Assemble output files: groups of objects_per_m tracks per file.
    objs_per = max(1, int(objects_per_m))
    rendered_groups = [
        rendered_tracks[i: i + objs_per]
        for i in range(0, n_tracks, objs_per)
    ]

    ext = ".tif" if fmt in ("tiff", "tif") else ".png"
    paths: List[str] = []

    for gi, group_crops in enumerate(rendered_groups):
        if progress_cb:
            progress_cb(50 + int(50 * gi / len(rendered_groups)))

        max_t = max(len(crops) for crops in group_crops) if group_crops else 1
        frames_out: List[np.ndarray] = []
        for t in range(max_t):
            tiles = []
            for crops in group_crops:
                tiles.append(_pad(crops[t]) if t < len(crops)
                             else np.zeros((pad_h, pad_w, 3), dtype=np.uint8))
            frames_out.append(np.concatenate(tiles, axis=1))

        fname = f"tracked_objects_M{gi:03d}{ext}"
        fpath = os.path.join(output_dir, fname)

        if fmt in ("tiff", "tif"):
            stack = np.stack(frames_out, axis=0)  # (T, H, W*n, 3)
            tifffile.imwrite(fpath, stack, photometric="rgb")
            paths.append(fpath)
        else:
            import imageio
            if len(frames_out) == 1:
                imageio.imwrite(fpath, frames_out[0])
                paths.append(fpath)
            else:
                base = os.path.splitext(fpath)[0]
                for ti, frame in enumerate(frames_out):
                    frame_path = f"{base}_T{ti:03d}.png"
                    imageio.imwrite(frame_path, frame)
                    paths.append(frame_path)

    if progress_cb:
        progress_cb(100)
    return paths


def _render_track(
    rows: List[Dict[str, Any]],
    label_masks: Dict[str, np.ndarray],
    channels: Dict[str, np.ndarray],
    channel_display: Dict[str, Dict],
    H: int,
    W: int,
    with_image: bool,
    with_mask_overlay: bool,
) -> List[np.ndarray]:
    """Render all frames for one track as a list of (crop_H, crop_W, 3) uint8 arrays."""
    from nd2studios.backend.exporters.composite_exporter import _composite_frame
    from nd2studios.widgets.image_viewer import CHANNEL_COLORS

    # Stable crop: union of all per-frame bboxes, expanded to _CROP_FACTOR × object size.
    min_row = min(int(r.get("bbox_min_row", 0) or 0) for r in rows)
    max_row = max(int(r.get("bbox_max_row", H) or H) for r in rows)
    min_col = min(int(r.get("bbox_min_col", 0) or 0) for r in rows)
    max_col = max(int(r.get("bbox_max_col", W) or W) for r in rows)

    center_r = (min_row + max_row) / 2.0
    center_c = (min_col + max_col) / 2.0
    bh = max(max_row - min_row, 1)
    bw = max(max_col - min_col, 1)
    half = _CROP_FACTOR / 2.0

    r0 = max(0, int(center_r - bh * half))
    r1 = min(H, int(center_r + bh * half))
    c0 = max(0, int(center_c - bw * half))
    c1 = min(W, int(center_c + bw * half))

    if r1 - r0 < 4 or c1 - c0 < 4:
        r0, r1, c0, c1 = 0, H, 0, W

    frame_index = {int(r.get("frame", 0)): r for r in rows}
    output: List[np.ndarray] = []

    for t, row in sorted(frame_index.items()):
        if with_image and channels:
            frame_dict: Dict[str, np.ndarray] = {}
            colors: Dict[str, Any] = {}
            enabled: Dict[str, bool] = {}
            lut_settings: Dict[str, Any] = {}
            for ch_name, ch_data in channels.items():
                arr = np.asarray(ch_data)
                if arr.ndim == 3 and t < arr.shape[0]:
                    frame_dict[ch_name] = arr[t, r0:r1, c0:c1]
                cd = channel_display.get(ch_name, {})
                colors[ch_name] = CHANNEL_COLORS.get(
                    cd.get("color", "gray"), (255, 255, 255))
                enabled[ch_name] = bool(cd.get("enabled", True))
                lut_settings[ch_name] = (
                    float(cd.get("lut_lo", 0.0)),
                    float(cd.get("lut_hi", 1.0)),
                    float(cd.get("lut_gamma", 1.0)),
                )
            rgb = _composite_frame(frame_dict, colors, enabled, lut_settings)
        else:
            rgb = np.zeros((r1 - r0, c1 - c0, 3), dtype=np.uint8)

        if with_mask_overlay:
            seg_ch = str(row.get("segmentation_channel", ""))
            label_id = int(row.get("label_id", row.get("label", 0)) or 0)
            if seg_ch in label_masks:
                mask_vol = label_masks[seg_ch]
                if t < mask_vol.shape[0]:
                    mask_crop = mask_vol[t, r0:r1, c0:c1] == label_id
                    if mask_crop.any():
                        rgb_f = rgb.astype(np.float32)
                        rgb_f[mask_crop] = (
                            rgb_f[mask_crop] * (1.0 - _HIGHLIGHT_ALPHA)
                            + _HIGHLIGHT * _HIGHLIGHT_ALPHA
                        )
                        rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        output.append(rgb)

    return output
