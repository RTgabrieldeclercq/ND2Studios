"""
Split-by-axis exporter (V1.72).

Writes the current (cropped) dataset out as **multiple files split along the
M / T / Z / C axes**. Each axis is either:

* **split** — one output file per index on that axis, or
* **keep**  — every index of that axis bundled inside each file.

The set of *split* axes defines the output-file grid (their cartesian product);
each file spans the full range of the *keep* axes. This one model expresses all
the groupings the user asked for — "one TIFF per Z (all T)", "one file per M
(all T & Z)", "one per T", and every "…etc." combination.

Three output formats can be produced in a single run (independent flags):

* **TIFF hyperstack** — an ImageJ ``TZCYX`` ``.tif`` per group. Kept T/Z/C become
  the hyperstack axes. Because ImageJ hyperstacks have no M axis, when M is a
  *keep* axis with more than one position the M positions are flattened into the
  front of the T (frames) axis, M-major (lossless; the GUI summary states the
  exact page count).
* **PNG image sequence** — one composited RGB PNG per ``(m, t, z)`` frame in the
  group, named with every varying index.
* **Movie (MP4/GIF)** — one clip per group; the timeline is the flattened
  ``(m, t)`` frames (M-major). A kept Z with >1 planes is max-projected for the
  2-D movie frame.

This module reuses the existing writers (:func:`export_tiff_hyperstack`,
:func:`export_movie`) and the shared compositor (:func:`_composite_frame` /
:func:`_draw_overlays`) so split output is byte-compatible with the single-view
exports. It must not import PySide6 (backend purity — CLAUDE.md).
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Spec ──────────────────────────────────────────────────────────────────

@dataclass
class SplitExportSpec:
    """Description of one split-by-axis export job.

    ``split_*`` — True to split that axis into separate files, False to keep it
    bundled inside each file. ``write_*`` — which output format(s) to produce.
    """

    output_dir: str
    basename: str

    split_m: bool = True
    split_t: bool = False
    split_z: bool = False
    split_c: bool = False

    write_tiff: bool = True
    write_png: bool = False
    write_movie: bool = False

    # 'none' | 'max' | 'mean' | 'min'. When not 'none' (or single Z) the Z axis
    # is projected to one plane, so ``split_z`` is a no-op (one Z group).
    z_mode: str = "none"
    bit_depth: str = "passthrough"   # TIFF only: passthrough | uint16 | uint8

    # Contrast / LUT mode applied to the output pixels:
    #   'auto'   — auto-scale each channel independently (0.5–99.5 percentile).
    #   'manual' — use the viewer's per-channel LUT window (lo, hi, gamma).
    #   'full'   — map the full data range (0 … dtype max) with no stretch.
    # Governs the composite (PNG / movie) contrast, and — when a rescaling
    # bit_depth (uint8 / uint16) is chosen — the TIFF rescale bounds too.
    # A ``passthrough`` TIFF always stores raw quantitative data (LUT ignored).
    lut_mode: str = "manual"


# ── Axis-group math (shared by the engine and the dialog's live summary) ────

def _axis_groups(n: int, split: bool) -> List[List[int]]:
    """Group index lists for one axis.

    ``split`` → ``[[0], [1], …, [n-1]]`` (one group per index).
    ``keep``  → ``[[0, 1, …, n-1]]`` (one group with every index).
    """
    n = max(1, int(n))
    if split:
        return [[i] for i in range(n)]
    return [list(range(n))]


def _width(n: int) -> int:
    """Digit count of the largest 1-based index for ``n`` items (>= 1)."""
    return max(1, len(str(max(1, int(n)))))


def _sanitize(name: str) -> str:
    """Make ``name`` safe for a filename fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-") or "C"


def _effective_dims(
    volume: Any, enabled: Dict[str, bool], z_mode: str,
) -> Tuple[int, int, int, bool, List[Tuple[int, str]]]:
    """Return ``(n_m, n_t, n_z_eff, iterate_z, enabled_channels)``.

    ``n_z_eff`` collapses to 1 (and ``iterate_z`` is False) when a projection is
    active or the file has a single Z slice. ``enabled_channels`` is the ordered
    ``[(channel_index, name), …]`` of channels the user left enabled.
    """
    n_m = int(getattr(volume, "n_multipoints", 1))
    n_t = int(getattr(volume, "n_timepoints", 1))
    n_z_raw = int(getattr(volume, "n_zslices", 1))
    iterate_z = (z_mode == "none" and n_z_raw > 1)
    n_z_eff = n_z_raw if iterate_z else 1
    enabled_channels = [
        (c, name) for c, name in enumerate(volume.channel_names)
        if enabled.get(name, True)
    ]
    return n_m, n_t, n_z_eff, iterate_z, enabled_channels


def _dtype_max(volume: Any, sample: Optional[np.ndarray] = None) -> Optional[float]:
    """Full-scale value for the volume's integer dtype, else None (float data)."""
    dt = getattr(volume, "dtype", None)
    if dt is None and sample is not None:
        dt = sample.dtype
    if dt is None:
        return None
    dt = np.dtype(dt)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return None


def _effective_contrast(
    lut_mode: str,
    enabled_names: List[str],
    viewer_lut: Optional[Dict[str, Tuple[float, float, float]]],
    dtype_max: Optional[float],
) -> Tuple[
    Optional[Dict[str, Tuple[float, float, float]]],
    Optional[Dict[str, Tuple[float, float]]],
]:
    """Resolve the LUT mode into ``(composite_lut, tiff_bounds)``.

    ``composite_lut`` (``{name: (lo, hi, gamma)}`` or None) feeds the PNG /
    movie compositor; ``tiff_bounds`` (``{name: (lo, hi)}`` or None) feeds the
    TIFF rescale. ``None`` on either means "let the writer auto-stretch each
    channel by percentile".
    """
    if lut_mode == "manual":
        vl = viewer_lut or {}
        composite = dict(vl) if vl else None
        tiff = {n: (lo, hi) for n, (lo, hi, _g) in vl.items()} or None
        return composite, tiff
    if lut_mode == "full" and dtype_max is not None:
        composite = {n: (0.0, dtype_max, 1.0) for n in enabled_names}
        tiff = {n: (0.0, dtype_max) for n in enabled_names}
        return composite, tiff
    # 'auto' (and 'full' on float data, which has no fixed range) → percentile.
    return None, None


def plan_split_export(
    n_m: int, n_t: int, n_z_eff: int, n_c_enabled: int,
    spec_like: Dict[str, bool],
) -> Dict[str, Any]:
    """Pure count/description helper for the dialog's live summary.

    ``spec_like`` carries the boolean flags ``split_m/t/z/c`` and
    ``write_tiff/png/movie``. Returns a dict of per-format file counts, the
    per-file page shape, and a one-line human description.
    """
    sm = bool(spec_like.get("split_m"))
    st = bool(spec_like.get("split_t"))
    sz = bool(spec_like.get("split_z"))
    sc = bool(spec_like.get("split_c"))

    n_c_enabled = max(1, int(n_c_enabled))
    m_groups = len(_axis_groups(n_m, sm))
    t_groups = len(_axis_groups(n_t, st))
    z_groups = len(_axis_groups(n_z_eff, sz))
    c_groups = len(_axis_groups(n_c_enabled, sc))
    n_groups = m_groups * t_groups * z_groups * c_groups

    # Kept-axis extents inside one file.
    kept_m = 1 if sm else n_m
    kept_t = 1 if st else n_t
    kept_z = 1 if sz else n_z_eff
    kept_c = 1 if sc else n_c_enabled
    pages_per_tiff = kept_m * kept_t * kept_z * kept_c
    frames_per_group = kept_m * kept_t * kept_z          # PNG frames per group

    out: Dict[str, Any] = {
        "n_groups": n_groups,
        "tiff_files": n_groups if spec_like.get("write_tiff") else 0,
        "png_files": (n_groups * frames_per_group) if spec_like.get("write_png") else 0,
        "movie_files": n_groups if spec_like.get("write_movie") else 0,
        "kept_t": kept_t, "kept_z": kept_z, "kept_c": kept_c, "kept_m": kept_m,
        "pages_per_tiff": pages_per_tiff,
    }

    bits: List[str] = []
    if spec_like.get("write_tiff"):
        shape = f"{kept_m * kept_t}×{kept_z}×{kept_c} (T×Z×C)"
        bits.append(f"TIFF: {out['tiff_files']} file(s), {shape} pages each")
    if spec_like.get("write_png"):
        bits.append(f"PNG: {out['png_files']} frame(s)")
    if spec_like.get("write_movie"):
        note = " · 1 frame/clip (T is split)" if st else ""
        bits.append(f"Movie: {out['movie_files']} clip(s){note}")
    out["description"] = "   ·   ".join(bits) if bits else "No output format selected."
    return out


# ── Engine ──────────────────────────────────────────────────────────────────

def export_split(
    volume: Any,
    spec: SplitExportSpec,
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]] = None,
    image_adjustments: Optional[Any] = None,
    movie_options: Optional[Any] = None,
    pixel_size_um: float = 1.0,
    frame_timestamps_s: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Run the split export and return the list of written file paths."""
    from nd2studios.backend.exporters.movie_exporter import MovieOptions

    os.makedirs(spec.output_dir, exist_ok=True)
    opts = movie_options or MovieOptions()

    n_m, n_t, n_z_eff, iterate_z, enabled_channels = _effective_dims(
        volume, enabled, spec.z_mode
    )
    if not enabled_channels:
        raise ValueError("split export: no enabled channels")

    # Normalize split flags against the real dimensions: an axis with a single
    # index (or Z collapsed by a projection) can't be split, so drop the flag
    # to avoid a spurious "_Z1" / "_M1" suffix and a redundant one-group loop.
    spec = dataclasses.replace(
        spec,
        split_m=spec.split_m and n_m > 1,
        split_t=spec.split_t and n_t > 1,
        split_z=spec.split_z and iterate_z,
        split_c=spec.split_c and len(enabled_channels) > 1,
    )

    cx, cy, cw, ch = crop_rect if crop_rect else (0, 0, 0, 0)
    do_crop = bool(crop_rect) and cw > 0 and ch > 0
    frame_ts = (
        np.asarray(frame_timestamps_s)
        if frame_timestamps_s is not None else None
    )

    # Resolve the LUT / contrast mode once: composite_lut drives PNG + movie,
    # tiff_bounds drives the TIFF rescale (uint8/uint16 only).
    enabled_names = [name for _, name in enabled_channels]
    composite_lut, tiff_bounds = _effective_contrast(
        spec.lut_mode, enabled_names, lut_settings, _dtype_max(volume),
    )

    def read_plane(c: int, m: int, t: int, z: int) -> np.ndarray:
        """One (H, W) plane for (c, m, t, z), with projection + XY crop applied."""
        if iterate_z:
            frame = volume.get_frame(c=c, m=m, t=t, z=z, z_mode="none")
        else:
            # z_mode is 'none' (single Z) or a projection collapsing all Z.
            frame = volume.get_frame(c=c, m=m, t=t, z=0, z_mode=spec.z_mode)
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.squeeze()
        if do_crop:
            frame = frame[cy:cy + ch, cx:cx + cw]
        return frame

    # Build the output-file grid: cartesian product of each axis's groups.
    m_groups = _axis_groups(n_m, spec.split_m)
    t_groups = _axis_groups(n_t, spec.split_t)
    z_groups = _axis_groups(n_z_eff, spec.split_z)
    c_index_groups = _axis_groups(len(enabled_channels), spec.split_c)

    formats = [f for f, on in (
        ("TIFF", spec.write_tiff), ("PNG", spec.write_png),
        ("Movie", spec.write_movie),
    ) if on]
    if not formats:
        raise ValueError("split export: no output format selected")

    groups = list(itertools.product(
        m_groups, t_groups, z_groups, c_index_groups))
    total_units = max(1, len(groups) * len(formats))
    unit = 0
    written: List[str] = []

    for gi, (m_list, t_list, z_list, ci_list) in enumerate(groups):
        c_list = [enabled_channels[i] for i in ci_list]  # [(c_idx, name), …]
        group_suffix = _group_suffix(spec, n_m, n_t, n_z_eff,
                                     m_list, t_list, z_list, c_list)

        for fmt in formats:
            if progress_cb is not None:
                progress_cb(int(unit / total_units * 100))
            label = f"Group {gi + 1}/{len(groups)}"
            if group_suffix:
                label += f" · {group_suffix.lstrip('_')}"
            if status_cb is not None:
                status_cb(f"{label} · {fmt}…")

            if fmt == "TIFF":
                written.append(_write_tiff_group(
                    read_plane, spec, group_suffix, m_list, t_list, z_list,
                    c_list, pixel_size_um, tiff_bounds))
            elif fmt == "PNG":
                written.extend(_write_png_group(
                    read_plane, spec, group_suffix, n_m, n_t, n_z_eff,
                    m_list, t_list, z_list, c_list, colors, enabled,
                    composite_lut, image_adjustments, opts, pixel_size_um,
                    frame_ts))
            elif fmt == "Movie":
                path = _write_movie_group(
                    read_plane, spec, group_suffix, m_list, t_list, z_list,
                    c_list, colors, enabled, composite_lut, image_adjustments,
                    opts, pixel_size_um, frame_ts)
                if path:
                    written.append(path)
            unit += 1

    if progress_cb is not None:
        progress_cb(100)
    return written


# ── Filename helpers ─────────────────────────────────────────────────────────

def _group_suffix(
    spec: SplitExportSpec, n_m: int, n_t: int, n_z: int,
    m_list: List[int], t_list: List[int], z_list: List[int],
    c_list: List[Tuple[int, str]],
) -> str:
    """Filename fragment naming the *split* axes of this group (e.g. ``_M02_Z03``)."""
    parts: List[str] = []
    if spec.split_m:
        parts.append(f"M{m_list[0] + 1:0{_width(n_m)}d}")
    if spec.split_t:
        parts.append(f"T{t_list[0] + 1:0{_width(n_t)}d}")
    if spec.split_z:
        parts.append(f"Z{z_list[0] + 1:0{_width(n_z)}d}")
    if spec.split_c:
        parts.append(f"C-{_sanitize(c_list[0][1])}")
    return ("_" + "_".join(parts)) if parts else ""


# ── Per-group writers ────────────────────────────────────────────────────────

def _write_tiff_group(
    read_plane: Callable[[int, int, int, int], np.ndarray],
    spec: SplitExportSpec, group_suffix: str,
    m_list: List[int], t_list: List[int], z_list: List[int],
    c_list: List[Tuple[int, str]], pixel_size_um: float,
    lut_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    """Write one ImageJ ``TZCYX`` hyperstack for this group.

    Kept M is flattened into the front of the T (frames) axis, M-major, so the
    output T extent is ``len(m_list) * len(t_list)``. Z extent is ``len(z_list)``.
    ``lut_bounds`` (manual / full-range window) is applied only for a rescaling
    bit depth; a ``passthrough`` TIFF keeps raw quantitative data.
    """
    from nd2studios.backend.exporters.tiff_exporter import export_tiff_hyperstack

    channels: "Dict[str, np.ndarray]" = {}
    enabled_out: Dict[str, bool] = {}
    for c_idx, name in c_list:
        # (T', Z', H, W) with T' = len(m_list)*len(t_list), M-major.
        stack_t: List[np.ndarray] = []
        for m in m_list:
            for t in t_list:
                planes = [read_plane(c_idx, m, t, z) for z in z_list]
                stack_t.append(np.stack(planes, axis=0))     # (Z', H, W)
        channels[name] = np.stack(stack_t, axis=0)           # (T', Z', H, W)
        enabled_out[name] = True

    path = os.path.join(spec.output_dir, f"{spec.basename}{group_suffix}.tif")
    return export_tiff_hyperstack(
        channels, enabled_out, path,
        bit_depth=spec.bit_depth, pixel_size_um=pixel_size_um,
        lut_bounds=lut_bounds,
    )


def _write_png_group(
    read_plane: Callable[[int, int, int, int], np.ndarray],
    spec: SplitExportSpec, group_suffix: str,
    n_m: int, n_t: int, n_z: int,
    m_list: List[int], t_list: List[int], z_list: List[int],
    c_list: List[Tuple[int, str]],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]],
    image_adjustments: Optional[Any],
    opts: Any, pixel_size_um: float, frame_ts: Optional[np.ndarray],
) -> List[str]:
    """Write one composited RGB PNG per ``(m, t, z)`` frame in this group."""
    from PIL import Image

    from nd2studios.backend.exporters.composite_exporter import _composite_frame
    from nd2studios.backend.exporters.movie_exporter import _draw_overlays

    overlays_enabled = (
        opts.show_scale_bar or opts.show_timestamp or opts.show_channel_labels
    )
    names = [name for _, name in c_list]
    paths: List[str] = []
    for m in m_list:
        for t in t_list:
            for z in z_list:
                frame_dict = {name: read_plane(c_idx, m, t, z)
                              for c_idx, name in c_list}
                rgb = _composite_frame(
                    frame_dict, colors, enabled, lut_settings or None,
                    image_adjustments=image_adjustments,
                )
                if overlays_enabled:
                    rgb = _draw_overlays(
                        rgb, t_index=t, opts=opts, pixel_size_um=pixel_size_um,
                        channel_colors=colors, channel_enabled=enabled,
                        channel_names=names, frame_timestamps=frame_ts,
                    )
                # Name with every varying index (split axes via group_suffix,
                # kept axes appended here) so files never collide.
                frame_suffix = _frame_suffix(
                    spec, n_m, n_t, n_z, m, t, z)
                fname = f"{spec.basename}{group_suffix}{frame_suffix}.png"
                out_path = os.path.join(spec.output_dir, fname)
                Image.fromarray(rgb).save(out_path)
                paths.append(out_path)
    return paths


def _frame_suffix(
    spec: SplitExportSpec, n_m: int, n_t: int, n_z: int,
    m: int, t: int, z: int,
) -> str:
    """Per-frame fragment for the *kept* axes with >1 index (PNG naming)."""
    parts: List[str] = []
    if not spec.split_m and n_m > 1:
        parts.append(f"M{m + 1:0{_width(n_m)}d}")
    if not spec.split_t and n_t > 1:
        parts.append(f"T{t + 1:0{_width(n_t)}d}")
    if not spec.split_z and n_z > 1:
        parts.append(f"Z{z + 1:0{_width(n_z)}d}")
    return ("_" + "_".join(parts)) if parts else ""


def _write_movie_group(
    read_plane: Callable[[int, int, int, int], np.ndarray],
    spec: SplitExportSpec, group_suffix: str,
    m_list: List[int], t_list: List[int], z_list: List[int],
    c_list: List[Tuple[int, str]],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]],
    image_adjustments: Optional[Any],
    opts: Any, pixel_size_um: float, frame_ts: Optional[np.ndarray],
) -> Optional[str]:
    """Write one MP4/GIF for this group; timeline = flattened (m, t), M-major.

    A kept Z with >1 planes is max-projected into the single 2-D movie frame
    (movies are 2-D). Returns the written path, or None if there were no frames.
    """
    from nd2studios.backend.exporters.movie_exporter import export_movie

    channels: "Dict[str, np.ndarray]" = {}
    enabled_out: Dict[str, bool] = {}
    for c_idx, name in c_list:
        frames: List[np.ndarray] = []
        for m in m_list:
            for t in t_list:
                if len(z_list) == 1:
                    plane = read_plane(c_idx, m, t, z_list[0])
                else:
                    planes = [read_plane(c_idx, m, t, z) for z in z_list]
                    plane = np.max(np.stack(planes, axis=0), axis=0)
                frames.append(plane)
        if not frames:
            return None
        channels[name] = np.stack(frames, axis=0)            # (Nframes, H, W)
        enabled_out[name] = True

    codec = (getattr(opts, "codec", None) or "mp4").lower()
    ext = ".gif" if codec == "gif" else ".mp4"
    path = os.path.join(spec.output_dir, f"{spec.basename}{group_suffix}_movie{ext}")

    # Timestamps only align to the timeline when M is not flattened (single m).
    ts = None
    if frame_ts is not None and len(m_list) == 1:
        idx = np.asarray(t_list, dtype=int)
        if idx.max(initial=-1) < len(frame_ts):
            ts = frame_ts[idx]

    export_movie(
        channels, colors, enabled_out, path, options=opts,
        pixel_size_um=pixel_size_um, frame_timestamps_s=ts,
        lut_settings=lut_settings or None, image_adjustments=image_adjustments,
    )
    return path
