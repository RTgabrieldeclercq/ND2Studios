"""
``run_stitch`` — top-level orchestration for the V1.54 stitching pipeline.

    build_dataset (keep metadata orientation)
      → decide_regime
      → compute_positions  (register once on align channel, or coordinate-only)
      → (optional) illumination correction
      → composite every (T, Z, C) frame, reusing ONE position set
      → write pyramidal OME-TIFF
      → write QC report

Register-once-apply-to-all (spec §7): positions come from a single reference
channel/timepoint and are reused for every channel, Z, and T unless
``per_timepoint_registration`` is set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.compositor import (
    canvas_size, composite_frame, normalize_positions,
)
from nd2studios.backend.stitch.config import ILLUM_NONE, StitchConfig
from nd2studios.backend.stitch.dataset import build_dataset
from nd2studios.backend.stitch.engines import compute_positions
from nd2studios.backend.stitch.illumination import estimate_flatfield
from nd2studios.backend.stitch.qc import write_qc
from nd2studios.backend.stitch.regime import decide_regime
from nd2studios.backend.stitch.writer import write_ome_tiff


@dataclass
class StitchResult:
    out_path: str
    regime: str
    engine: str
    canvas_h: int
    canvas_w: int
    n_tiles: int
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    qc: Dict[str, str] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)


def _ref_z_mode(config: StitchConfig) -> str:
    """A robust 2D read mode for the registration/illumination reference."""
    return "max" if config.z_mode == "none" else config.z_mode


def _read_ref_frames(volume, m_indices: List[int], c_abs: int,
                     t_ref: int, config: StitchConfig) -> Dict[int, np.ndarray]:
    frames: Dict[int, np.ndarray] = {}
    for m in m_indices:
        f = volume.get_frame(c=c_abs, m=m, t=t_ref, z=config.z_index,
                             z_mode=_ref_z_mode(config))
        if f.ndim != 2:
            f = f.squeeze()
        frames[m] = f
    return frames


def run_stitch(volume,
               stage_xy_um: List[Tuple[float, float]],
               m_indices: List[int],
               channel_indices: List[int],
               config: StitchConfig,
               out_path: str,
               meta: Optional[Dict] = None,
               progress_cb: Optional[Callable[[int], None]] = None) -> StitchResult:
    """Stitch selected M tiles and write a pyramidal OME-TIFF."""
    def prog(p: int) -> None:
        if progress_cb is not None:
            progress_cb(max(0, min(100, int(p))))

    prog(1)
    dataset = build_dataset(volume, stage_xy_um, m_indices, config)
    m_indices = dataset.m_indices
    regime = decide_regime(dataset, config)

    n_c_vol = int(getattr(volume, "n_channels", 1) or 1)
    align_c = max(0, min(int(config.align_channel), n_c_vol - 1))
    n_t = int(getattr(volume, "n_timepoints", 1) or 1)
    t_ref = n_t // 2

    # ── positions (register once) ──
    ref_frames = None
    if regime != "zero_overlap" and dataset.n_tiles >= 2:
        ref_frames = _read_ref_frames(volume, m_indices, align_c, t_ref, config)
    positions, confidences, info = compute_positions(
        dataset, ref_frames, config, regime)
    prog(15)

    # Capture the origin used to normalize the reference positions so that any
    # per-timepoint re-registration (below) can be placed on the SAME canvas
    # (its positions live in the same coordinate-seed frame).
    if positions:
        ref_origin = (min(p[0] for p in positions.values()),
                      min(p[1] for p in positions.values()))
    else:
        ref_origin = (0.0, 0.0)
    positions = normalize_positions(positions)
    canvas_h, canvas_w = canvas_size(positions, dataset.tile_h, dataset.tile_w)

    # ── illumination flat-fields (per output channel, from t_ref tiles) ──
    flat_by_c: Dict[int, object] = {}
    if config.illumination_correction != ILLUM_NONE:
        for c_abs in channel_indices:
            tiles = list(_read_ref_frames(volume, m_indices, c_abs, t_ref, config).values())
            ff = estimate_flatfield(tiles, config)
            if ff is not None:
                flat_by_c[c_abs] = ff
    prog(20)

    # ── output geometry ──
    n_z = int(getattr(volume, "n_zslices", 1) or 1)
    preserve_z = (config.z_mode == "none" and n_z > 1)
    n_z_out = n_z if preserve_z else 1
    n_c_out = len(channel_indices)
    dtype = np.dtype(getattr(volume, "dtype", np.uint16))

    est_bytes = n_t * n_z_out * n_c_out * canvas_h * canvas_w * dtype.itemsize
    use_memmap = est_bytes > config.max_memory_gb * 1e9
    tmp_path = ""
    if use_memmap:
        tmp_path = out_path + ".stitch_tmp.dat"
        data = np.memmap(tmp_path, dtype=dtype, mode="w+",
                         shape=(n_t, n_z_out, n_c_out, canvas_h, canvas_w))
    else:
        data = np.zeros((n_t, n_z_out, n_c_out, canvas_h, canvas_w), dtype=dtype)

    total_planes = max(1, n_t * n_z_out * n_c_out)
    done = 0
    per_t_positions = positions
    try:
        for t in range(n_t):
            if (config.per_timepoint_registration and regime != "zero_overlap"
                    and t != t_ref and dataset.n_tiles >= 2):
                rf = _read_ref_frames(volume, m_indices, align_c, t, config)
                p_t, _c, _i = compute_positions(dataset, rf, config, regime)
                # Shift by the SAME reference origin (not an independent
                # per-t normalization) so every timepoint lands on one canvas.
                per_t_positions = {m: (y - ref_origin[0], x - ref_origin[1])
                                   for m, (y, x) in p_t.items()}
            else:
                per_t_positions = positions

            for z_out in range(n_z_out):
                z_req = z_out if preserve_z else config.z_index
                z_req_mode = "none" if preserve_z else config.z_mode
                for c_idx, c_abs in enumerate(channel_indices):
                    tiles: List[np.ndarray] = []
                    pos_list: List[Tuple[float, float]] = []
                    for m in m_indices:
                        if m not in per_t_positions:
                            continue
                        f = volume.get_frame(c=c_abs, m=m, t=t, z=z_req,
                                             z_mode=z_req_mode)
                        if f.ndim != 2:
                            f = f.squeeze()
                        ff = flat_by_c.get(c_abs)
                        if ff is not None:
                            f = ff.apply(f)
                        tiles.append(f)
                        pos_list.append(per_t_positions[m])
                    canvas = composite_frame(
                        tiles, pos_list, canvas_h, canvas_w, dtype,
                        blend=config.blend, feather_width_px=config.feather_width_px,
                        fill_value=config.fill_value)
                    data[t, z_out, c_idx] = canvas
                    done += 1
                    prog(20 + int(done / total_planes * 70))
        if use_memmap:
            data.flush()

        # ── write OME-TIFF ──
        px = config.pixel_size_um or float(getattr(volume, "pixel_size_um", 1.0) or 1.0)
        ch_names_all = list(getattr(volume, "channel_names", []) or [])
        ch_names = [ch_names_all[c] if c < len(ch_names_all) else f"Ch{c}"
                    for c in channel_indices]
        final_path = write_ome_tiff(data, out_path, px, ch_names, config)
        prog(95)
    finally:
        if use_memmap:
            # Explicitly close the mmap before deleting the temp file — on
            # Windows an unreleased handle makes os.remove raise PermissionError.
            try:
                mm = getattr(data, "_mmap", None)
                if mm is not None:
                    mm.close()
            except Exception:
                pass
            try:
                del data
            except Exception:
                pass
            import gc
            gc.collect()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    result = StitchResult(
        out_path=final_path, regime=regime, engine=str(info.get("engine", "?")),
        canvas_h=canvas_h, canvas_w=canvas_w, n_tiles=dataset.n_tiles,
        positions=positions, info=info,
    )
    if config.write_qc:
        result.qc = write_qc(final_path, dataset, positions, confidences, regime, info)
    prog(100)
    return result
