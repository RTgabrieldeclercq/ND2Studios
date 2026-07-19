"""viz3d.prep — turn a volume source into ``(Z, H, W)`` uint8 render blocks.

Pure numpy: no Qt, no PyVista/VTK. Consumes any duck-typed volume that exposes
the ``MaterializedDataset`` / ``LazyND2Volume`` surface — namely
``get_volume(c, m, t, z_start, z_end) -> (Z, H, W)`` plus the ``n_zslices`` /
``channel_names`` / ``pixel_size_um`` / ``z_step_um`` attributes.

The uint8 contrast mapping in :func:`to_uint8` is a byte-for-byte mirror of
``nd2studios/widgets/lut_histogram.apply_lut`` (that module imports PySide6, so
we cannot import it here without tainting the backend). Keeping the formula
identical means the 3-D view matches the 2-D composite the user already tuned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Mirror of ``widgets/image_viewer.CHANNEL_COLORS`` (kept local so the backend
# stays Qt-free — ``widgets/image_viewer`` imports PySide6).
CHANNEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gray": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 100, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
}


@dataclass(frozen=True)
class Spacing:
    """Physical voxel spacing in micrometers, in VTK ``ImageData`` axis order.

    A ``(Z, H, W)`` numpy block fed into a ``pyvista.ImageData`` has axis order
    ``(z, y, x)``, so spacing is ``(dz, dy, dx)`` — ``dy == dx == pixel_size_um``
    and ``dz == z_step_um``. Confocal ``dz`` is routinely several × the XY pixel,
    so honoring this is the difference between a correct render and a squashed one.
    """

    dz: float
    dy: float
    dx: float

    @property
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.dz, self.dy, self.dx)

    def normalized(self) -> Tuple[float, float, float]:
        """Same axis ratios, rescaled so the smallest positive axis == 1.0.

        Used for camera framing when absolute µm scale is unknown or extreme;
        preserves anisotropy while keeping the scene a sane size.
        """
        vals = [float(self.dz), float(self.dy), float(self.dx)]
        positive = [v for v in vals if v > 0]
        m = min(positive) if positive else 1.0
        if m <= 0:
            m = 1.0
        return (vals[0] / m, vals[1] / m, vals[2] / m)


def spacing_from_volume(volume: Any) -> Spacing:
    """Read :class:`Spacing` off a volume, defaulting missing/invalid to 1.0."""
    dz = float(getattr(volume, "z_step_um", 1.0) or 1.0)
    dxy = float(getattr(volume, "pixel_size_um", 1.0) or 1.0)
    dz = dz if dz > 0 else 1.0
    dxy = dxy if dxy > 0 else 1.0
    return Spacing(dz=dz, dy=dxy, dx=dxy)


def to_uint8(vol: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """Contrast-map ``vol`` to uint8 [0, 255].

    Identical formula to ``widgets/lut_histogram.apply_lut`` (mirrored here so
    the backend stays Qt-free). Works for any ndim; for a ``(Z, H, W)`` stack
    the result equals applying ``apply_lut`` plane-by-plane.
    """
    f = np.asarray(vol, dtype=np.float32)
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    f = (f - lo) / (hi - lo)
    np.clip(f, 0.0, 1.0, out=f)
    if abs(float(gamma) - 1.0) > 1e-3:
        f = f ** (1.0 / max(float(gamma), 0.05))
    return (f * 255.0).astype(np.uint8)


def auto_contrast(vol: np.ndarray, lo_pct: float = 1.0,
                  hi_pct: float = 99.0) -> Tuple[float, float]:
    """Return ``(lo, hi)`` percentile bounds for a volume (NaN-safe)."""
    a = np.asarray(vol)
    a = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    if a.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def color_rgb_float(color: Any) -> Tuple[float, float, float]:
    """Normalize a color name or 0–255 RGB triple to a 0–1 float triple."""
    if isinstance(color, str):
        rgb = CHANNEL_COLORS.get(color, (255, 255, 255))
    elif isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
        rgb = (int(color[0]), int(color[1]), int(color[2]))
    else:
        rgb = (255, 255, 255)
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def clamp_z_range(n_z: int, z_start: Optional[int],
                  z_end: Optional[int]) -> Tuple[int, int]:
    """Clamp a half-open ``[z_start, z_end)`` range to ``[0, n_z]`` (never empty)."""
    n_z = int(n_z)
    if n_z <= 0:
        return 0, 0
    zs = 0 if z_start is None else max(0, min(int(z_start), n_z - 1))
    ze = n_z if z_end is None else max(zs + 1, min(int(z_end), n_z))
    return zs, ze


def channel_volume(volume: Any, c: int, m: int = 0, t: int = 0,
                   z_start: Optional[int] = None,
                   z_end: Optional[int] = None) -> np.ndarray:
    """Return the raw ``(Z, H, W)`` block for one channel via ``get_volume``.

    A single-Z acquisition returns ``(1, H, W)`` (the 3-D viewer degrades to a
    textured plane in that case).
    """
    n_z = int(getattr(volume, "n_zslices", 1) or 1)
    zs, ze = clamp_z_range(n_z, z_start, z_end)
    block = np.asarray(volume.get_volume(int(c), m=int(m), t=int(t),
                                         z_start=zs, z_end=ze))
    if block.ndim == 2:
        block = block[None, ...]
    return block


@dataclass
class ChannelVolume:
    """One channel ready for a ``pyvista.ImageData`` actor."""

    name: str
    data: np.ndarray                       # (Z, H, W) uint8
    color: Tuple[float, float, float]      # 0–1 RGB
    lut: Tuple[float, float, float]        # (lo, hi, gamma) actually applied


def build_channel_volumes(volume: Any,
                          channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                          m: int = 0, t: int = 0,
                          z_start: Optional[int] = None,
                          z_end: Optional[int] = None,
                          only_enabled: bool = True,
                          progress_cb: Optional[Callable[[int], None]] = None,
                          ) -> List[ChannelVolume]:
    """Build LUT-mapped uint8 ``(Z, H, W)`` volumes for the enabled channels.

    ``channel_display`` mirrors ``record.channel_display`` — per name
    ``{enabled, color, lut_lo, lut_hi, lut_gamma}``. When a channel has no
    ``lut_lo``/``lut_hi`` a 1–99 percentile auto-contrast is used. When
    ``channel_display`` is empty, every channel is included with auto-contrast.
    """
    names = list(getattr(volume, "channel_names", []) or [])
    cd = channel_display or {}

    selected: List[Tuple[int, str, Dict[str, Any]]] = []
    for i, name in enumerate(names):
        conf = dict(cd.get(name, {}))
        if only_enabled and cd and not conf.get("enabled", True):
            continue
        selected.append((i, name, conf))

    out: List[ChannelVolume] = []
    total = max(1, len(selected))
    for k, (i, name, conf) in enumerate(selected):
        block = channel_volume(volume, i, m=m, t=t, z_start=z_start, z_end=z_end)
        if "lut_lo" in conf and "lut_hi" in conf:
            lo, hi = float(conf["lut_lo"]), float(conf["lut_hi"])
        else:
            lo, hi = auto_contrast(block)
        gamma = float(conf.get("lut_gamma", 1.0))
        u8 = to_uint8(block, lo, hi, gamma)
        default_color = name if name in CHANNEL_COLORS else "gray"
        color = color_rgb_float(conf.get("color", default_color))
        out.append(ChannelVolume(name=name, data=u8, color=color,
                                 lut=(lo, hi, gamma)))
        if progress_cb is not None:
            progress_cb(int((k + 1) / total * 100))
    return out


def build_render_volumes(volume: Any,
                         channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                         m: int = 0, t: int = 0,
                         z_start: Optional[int] = None,
                         z_end: Optional[int] = None,
                         xy_max: int = 512, z_max: int = 128,
                         max_voxels: int = 48_000_000,
                         only_enabled: bool = True,
                         progress_cb: Optional[Callable[[int], None]] = None,
                         cancel_cb: Optional[Callable[[], bool]] = None,
                         ) -> Tuple[List[ChannelVolume], "Spacing"]:
    """Build **downsampled**, bounded uint8 ``(Zd, Hd, Wd)`` volumes for GPU
    volume rendering, plus the matching physical :class:`Spacing`.

    A full-resolution stack (e.g. ``100 × 4096 × 4096`` per channel) is far too
    large for VTK's volume mapper — it exhausts GPU/host memory and crashes. So
    the render volume is bounded: Z is subsampled to ≤ ``z_max`` planes, XY is
    strided so the larger spatial axis is ≤ ``xy_max``, and the stride is raised
    further if the result would exceed ``max_voxels``. Planes are read **one at a
    time** (only the selected Z indices) and downsampled immediately, so
    transient RAM stays ~one full plane regardless of the source size — no giant
    float32 intermediate, no full-volume allocation.

    The returned :class:`Spacing` accounts for the strides so the rendered volume
    keeps correct physical proportions (anisotropic Z included).
    """
    names = list(getattr(volume, "channel_names", []) or [])
    cd = channel_display or {}
    n_z = int(getattr(volume, "n_zslices", 1) or 1)
    height = int(getattr(volume, "height", 0) or 0)
    width = int(getattr(volume, "width", 0) or 0)
    zs, ze = clamp_z_range(n_z, z_start, z_end)
    z_span = max(1, ze - zs)

    xy_stride = max(1, int(np.ceil(max(height, width, 1) / max(1, xy_max))))
    z_count = min(z_span, max(1, int(z_max)))
    z_indices = np.unique(np.linspace(zs, ze - 1, z_count).astype(int))

    def _down_hw(stride: int) -> Tuple[int, int]:
        return ((height + stride - 1) // stride, (width + stride - 1) // stride)

    hd, wd = _down_hw(xy_stride)
    while len(z_indices) * hd * wd > int(max_voxels) and xy_stride < max(height, width, 1):
        xy_stride += 1
        hd, wd = _down_hw(xy_stride)

    selected: List[Tuple[int, str, Dict[str, Any]]] = []
    for i, name in enumerate(names):
        conf = dict(cd.get(name, {}))
        if only_enabled and cd and not conf.get("enabled", True):
            continue
        selected.append((i, name, conf))

    out: List[ChannelVolume] = []
    total = max(1, len(selected) * max(1, len(z_indices)))
    done = 0
    for i, name, conf in selected:
        planes: List[np.ndarray] = []
        for z in z_indices:
            if cancel_cb is not None and cancel_cb():
                return [], Spacing(dz=1.0, dy=1.0, dx=1.0)
            blk = np.asarray(volume.get_volume(int(i), m=int(m), t=int(t),
                                               z_start=int(z), z_end=int(z) + 1))
            plane = blk[0] if blk.ndim == 3 else blk
            if xy_stride > 1:
                plane = plane[::xy_stride, ::xy_stride]
            planes.append(np.ascontiguousarray(plane))
            done += 1
            if progress_cb is not None:
                progress_cb(int(done / total * 100))
        vol_small = np.stack(planes, axis=0)      # (Zd, Hd, Wd) — small

        if "lut_lo" in conf and "lut_hi" in conf:
            lo, hi = float(conf["lut_lo"]), float(conf["lut_hi"])
        else:
            lo, hi = auto_contrast(vol_small)
        gamma = float(conf.get("lut_gamma", 1.0))
        u8 = to_uint8(vol_small, lo, hi, gamma)
        default_color = name if name in CHANNEL_COLORS else "gray"
        color = color_rgb_float(conf.get("color", default_color))
        out.append(ChannelVolume(name=name, data=u8, color=color,
                                 lut=(lo, hi, gamma)))

    dz = float(getattr(volume, "z_step_um", 1.0) or 1.0)
    dxy = float(getattr(volume, "pixel_size_um", 1.0) or 1.0)
    z_downsample = z_span / max(1, len(z_indices))
    spacing = Spacing(dz=dz * max(1.0, z_downsample),
                      dy=dxy * xy_stride, dx=dxy * xy_stride)
    return out, spacing
