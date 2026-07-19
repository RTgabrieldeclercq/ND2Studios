"""viz3d — pure-numpy preparation of image volumes and result overlays for the
V1.65 PyVista 3-D viewer.

This subpackage is **Qt-free and PyVista/VTK-free** (per the ``CLAUDE.md``
backend-purity rule): it turns the in-RAM volume + measurement results the app
already holds into contiguous numpy arrays and small dataclasses. The widget
layer (``nd2studios/widgets/viewer3d``) converts those into ``pyvista`` objects
on the GUI thread. Keeping the graphics dependency out of here lets every
adapter be unit-tested without a display.

Public API:

- :mod:`prep` — ``build_channel_volumes`` / ``channel_volume`` / ``to_uint8`` /
  ``Spacing`` / ``spacing_from_volume`` / ``auto_contrast`` / ``color_rgb_float``.
- :mod:`overlays` — ``dvc_field`` (→ :class:`~overlays.DVCField`),
  ``ptv_polylines`` (→ :class:`~overlays.PtvTracks`), ``label_volume``,
  ``dvc_object_field`` (→ :class:`~overlays.MaskedField`), and the V1.68
  surface adapters ``dvc_object_surface`` (→ :class:`~overlays.SurfaceField`) /
  ``unwrap_surface`` (→ :class:`~overlays.UnwrapMap`).
- :mod:`surface` — ``build_object_surface`` (→ :class:`~surface.ObjectSurface`) +
  on-surface displacement sampling / decomposition.
- :mod:`mdm` — Mean Deformation Metrics (Stout et al. 2016): ``MDMResult``,
  ``mean_displacement_gradient``, ``deformation_metrics``, ``cumulative_rotation``.
"""
from __future__ import annotations

from nd2studios.backend.viz3d.prep import (
    CHANNEL_COLORS,
    ChannelVolume,
    Spacing,
    auto_contrast,
    build_channel_volumes,
    channel_volume,
    clamp_z_range,
    color_rgb_float,
    spacing_from_volume,
    to_uint8,
)
from nd2studios.backend.viz3d.overlays import (
    DVCField,
    MaskedField,
    PtvTracks,
    SurfaceField,
    UnwrapMap,
    dvc_field,
    dvc_object_field,
    dvc_object_surface,
    label_volume,
    ptv_polylines,
    unwrap_surface,
)
from nd2studios.backend.viz3d.surface import (
    ObjectSurface,
    build_object_surface,
    decompose_surface_displacement,
    sample_displacement_on_surface,
    sample_scalars_on_surface,
)
from nd2studios.backend.viz3d.mdm import (
    MDMResult,
    cumulative_rotation,
    deformation_metrics,
    mean_displacement_gradient,
)

__all__ = [
    "CHANNEL_COLORS",
    "ChannelVolume",
    "Spacing",
    "auto_contrast",
    "build_channel_volumes",
    "channel_volume",
    "clamp_z_range",
    "color_rgb_float",
    "spacing_from_volume",
    "to_uint8",
    "DVCField",
    "MaskedField",
    "PtvTracks",
    "SurfaceField",
    "UnwrapMap",
    "dvc_field",
    "dvc_object_field",
    "dvc_object_surface",
    "label_volume",
    "ptv_polylines",
    "unwrap_surface",
    "ObjectSurface",
    "build_object_surface",
    "decompose_surface_displacement",
    "sample_displacement_on_surface",
    "sample_scalars_on_surface",
    "MDMResult",
    "cumulative_rotation",
    "deformation_metrics",
    "mean_displacement_gradient",
]
