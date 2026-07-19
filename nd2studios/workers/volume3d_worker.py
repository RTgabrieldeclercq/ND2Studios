"""``VolumeBuildWorker`` — build LUT-mapped uint8 ``(Z,H,W)`` render blocks off
the GUI thread for the PyVista 3-D viewer.

The worker only produces **numpy** (via the Qt-free ``backend.viz3d.prep``): all
PyVista/VTK actor creation happens on the GUI thread in the viewer's ``finished``
slot, because VTK/OpenGL objects may only be touched on the main thread.

``finished`` payload: ``VolumeBuildResult(channels, spacing, m, t)`` — the
``(m, t)`` it was built for lets the viewer drop a stale result if the user has
already scrubbed on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal

from nd2studios.backend.viz3d import prep
from nd2studios.workers.base_worker import BaseWorker


def render_bounds() -> tuple:
    """(xy_max, z_max, max_voxels) render bounds from Settings (with defaults)."""
    try:
        from nd2studios.core.settings import Settings
        return (int(getattr(Settings, "VIEW3D_XY_MAX", 512) or 512),
                int(getattr(Settings, "VIEW3D_Z_MAX", 128) or 128),
                int(getattr(Settings, "VIEW3D_MAX_VOXELS", 48_000_000) or 48_000_000))
    except Exception:  # noqa: BLE001
        return (512, 128, 48_000_000)


@dataclass
class VolumeBuildResult:
    channels: List[prep.ChannelVolume]
    spacing: prep.Spacing
    m: int
    t: int


class VolumeBuildWorker(BaseWorker):
    """Build ``ChannelVolume`` blocks for one ``(m, t)`` position."""

    def __init__(self, volume: Any,
                 channel_display: Optional[Dict[str, Dict[str, Any]]],
                 m: int, t: int,
                 z_start: Optional[int] = None,
                 z_end: Optional[int] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._volume = volume
        self._channel_display = channel_display or {}
        self._m = int(m)
        self._t = int(t)
        self._z_start = z_start
        self._z_end = z_end

    def run_task(self) -> Optional[VolumeBuildResult]:
        if self._volume is None:
            return None
        self.set_status("Building 3-D volume…")

        def _progress(pct: int) -> None:
            if not self.cancelled:
                self.set_progress(int(pct))

        # Downsampled, memory-bounded build (reads one plane at a time and
        # strides Z/XY to a GPU-safe voxel budget). A full-resolution deep stack
        # would exhaust GPU/host memory and crash VTK's volume mapper. Bounds are
        # tunable via Settings for weak GPUs.
        try:
            from nd2studios.core.settings import Settings
            xy_max = int(getattr(Settings, "VIEW3D_XY_MAX", 512) or 512)
            z_max = int(getattr(Settings, "VIEW3D_Z_MAX", 128) or 128)
            max_voxels = int(getattr(Settings, "VIEW3D_MAX_VOXELS", 48_000_000)
                             or 48_000_000)
        except Exception:  # noqa: BLE001
            xy_max, z_max, max_voxels = 512, 128, 48_000_000

        channels, spacing = prep.build_render_volumes(
            self._volume,
            channel_display=self._channel_display,
            m=self._m, t=self._t,
            z_start=self._z_start, z_end=self._z_end,
            xy_max=xy_max, z_max=z_max, max_voxels=max_voxels,
            only_enabled=True,
            progress_cb=_progress,
            cancel_cb=lambda: bool(self.cancelled),
        )
        if self.cancelled or not channels:
            return None
        self.set_status("")
        return VolumeBuildResult(channels=channels, spacing=spacing,
                                 m=self._m, t=self._t)


class TimeSeriesBuildWorker(BaseWorker):
    """Prebuild downsampled render volumes for a list of timepoints so 3-D time
    playback is smooth — played from RAM with no disk reads per frame.

    Emits ``frame_ready(t, VolumeBuildResult)`` as each timepoint completes so the
    viewer can cache it and begin/continue playback. ``max_voxels_per_frame``
    lets the caller shrink per-frame volumes so the whole series fits a RAM
    budget when there are many timepoints.
    """

    frame_ready = Signal(int, object)   # (t, VolumeBuildResult)

    def __init__(self, volume: Any,
                 channel_display: Optional[Dict[str, Dict[str, Any]]],
                 m: int, t_list: List[int],
                 z_start: Optional[int] = None,
                 z_end: Optional[int] = None,
                 max_voxels_per_frame: Optional[int] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._volume = volume
        self._channel_display = channel_display or {}
        self._m = int(m)
        self._t_list = [int(t) for t in t_list]
        self._z_start = z_start
        self._z_end = z_end
        self._max_voxels_per_frame = max_voxels_per_frame

    def run_task(self) -> None:
        if self._volume is None:
            return None
        xy_max, z_max, max_voxels = render_bounds()
        if self._max_voxels_per_frame:
            max_voxels = min(max_voxels, int(self._max_voxels_per_frame))
        total = max(1, len(self._t_list))
        for i, t in enumerate(self._t_list):
            if self.cancelled:
                return None
            self.set_status(f"Preparing smooth playback… frame {i + 1}/{total}")
            channels, spacing = prep.build_render_volumes(
                self._volume, channel_display=self._channel_display,
                m=self._m, t=t, z_start=self._z_start, z_end=self._z_end,
                xy_max=xy_max, z_max=z_max, max_voxels=max_voxels,
                only_enabled=True, cancel_cb=lambda: bool(self.cancelled))
            if self.cancelled or not channels:
                return None
            self.frame_ready.emit(t, VolumeBuildResult(
                channels=channels, spacing=spacing, m=self._m, t=t))
            self.set_progress(int((i + 1) / total * 100))
        return None
