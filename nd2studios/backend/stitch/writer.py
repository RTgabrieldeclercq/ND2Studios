"""
Pyramidal, tiled, BigTIFF OME-TIFF writer (spec §8).

Writes a ``(T, Z, C, H, W)`` mosaic as a pyramidal OME-TIFF preserving dtype,
channel names, and physical pixel size. ``data`` may be an in-RAM ndarray or a
disk-backed ``np.memmap`` — pyramid levels are built plane-by-plane so peak RAM
stays small even for multi-GB mosaics.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import tifffile

from nd2studios.backend.stitch.config import StitchConfig


def _pyramid_factors(h: int, w: int, config: StitchConfig) -> List[int]:
    """Cumulative downsample factors for pyramid levels beyond level 0."""
    factors: List[int] = []
    if not config.pyramid:
        return factors
    step = max(2, int(config.pyramid_downsample))
    f = step
    for _ in range(max(0, int(config.pyramid_max_levels) - 1)):
        if h // f < config.tile_size or w // f < config.tile_size:
            break
        factors.append(f)
        f *= step
    return factors


def _downscale_level(src: np.ndarray, factor: int, dtype: np.dtype) -> np.ndarray:
    """Downscale a (T,Z,C,H,W) source by ``factor`` via local-mean blocks."""
    from skimage.transform import downscale_local_mean
    T, Z, C, H, W = src.shape
    Hn, Wn = max(1, H // factor), max(1, W // factor)
    out = np.zeros((T, Z, C, Hn, Wn), dtype=dtype)
    for t in range(T):
        for z in range(Z):
            for c in range(C):
                plane = np.asarray(src[t, z, c], dtype=np.float64)
                ds = downscale_local_mean(plane, (factor, factor))[:Hn, :Wn]
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo(dtype)
                    ds = np.clip(np.rint(ds), info.min, info.max)
                out[t, z, c] = ds.astype(dtype)
    return out


def write_ome_tiff(data: np.ndarray,
                   out_path: str,
                   pixel_size_um: Optional[float],
                   channel_names: List[str],
                   config: StitchConfig) -> str:
    """Write ``data`` (T,Z,C,H,W) as a pyramidal OME-TIFF. Returns the path."""
    if data.ndim != 5:
        raise ValueError(f"expected (T,Z,C,H,W), got shape {data.shape}")
    T, Z, C, H, W = data.shape
    dtype = data.dtype

    if not out_path.lower().endswith((".ome.tif", ".ome.tiff")):
        if out_path.lower().endswith((".tif", ".tiff")):
            out_path = out_path.rsplit(".", 1)[0] + ".ome.tif"
        else:
            out_path += ".ome.tif"

    tile = (int(config.tile_size), int(config.tile_size))
    factors = _pyramid_factors(H, W, config)

    # BigTIFF decision must count the pyramid sub-resolutions too (each level
    # ≈ 1/factor² of level 0), else a mosaic just under 4 GiB at level 0
    # overflows the classic-TIFF 32-bit offset once the pyramid is appended.
    nbytes = int(data.size) * dtype.itemsize
    pyramid_bytes = sum(nbytes / (f * f) for f in factors)
    bigtiff = (nbytes + pyramid_bytes) > 3_500_000_000

    metadata = {"axes": "TZCYX"}
    if channel_names:
        metadata["Channel"] = {"Name": list(channel_names)}
    if pixel_size_um and pixel_size_um > 0:
        metadata["PhysicalSizeX"] = float(pixel_size_um)
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeY"] = float(pixel_size_um)
        metadata["PhysicalSizeYUnit"] = "µm"

    options = dict(photometric="minisblack", tile=tile)
    # Also emit the resolution tag (px/µm) so non-ome-types readers (ImageJ /
    # Fiji) still recover the physical pixel size.
    if pixel_size_um and pixel_size_um > 0:
        options["resolution"] = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        options["resolutionunit"] = "MICROMETER"

    with tifffile.TiffWriter(out_path, bigtiff=bigtiff, ome=True) as tif:
        tif.write(data, subifds=len(factors), metadata=metadata, **options)
        for factor in factors:
            level = _downscale_level(data, factor, dtype)
            tif.write(level, subfiletype=1, **options)
    return out_path
