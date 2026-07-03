"""V1.45 — SerialTrack tracking option on the "Track Objects" node.

Exercises the vendored headless SerialTrack linker (METHOD_SERIALTRACK) through
``backend.object_tracker.link_objects`` / ``link_objects_with_params``: synthetic
measurement rows for objects translating across frames must get one stable
``track_id`` per object and the expected ``track_length``, in both Incremental and
Cumulative modes. (The first call JIT-compiles the numba kernels — a one-time,
cached pause.)
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.object_tracker import (
    METHOD_CENTROID, METHOD_SERIALTRACK, TRACKING_METHODS,
    link_objects, link_objects_with_params,
)


def _make_rows(n_obj=6, n_frames=4, shift=(3.0, 2.0), seed=0, channel="DAPI"):
    """Rows for *n_obj* objects translating by *shift* each frame.

    Returns ``(rows, base_positions)``; rows are ordered frame-major then by
    object index, so ``rows[t * n_obj + i]`` is object *i* at frame *t*.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(20.0, 180.0, size=(n_obj, 2))
    shift = np.asarray(shift, dtype=float)
    rows = []
    for t in range(n_frames):
        pts = base + shift * t
        for i in range(n_obj):
            rows.append(dict(
                segmentation_channel=channel, m_position=0, frame=t,
                centroid_y_px=float(pts[i, 0]), centroid_x_px=float(pts[i, 1]),
                area_px=100.0,
            ))
    return rows, base


def _ids_per_object(rows, n_obj):
    """Map object index -> set of track ids it received across frames."""
    by_obj = {}
    for k, r in enumerate(rows):
        by_obj.setdefault(k % n_obj, set()).add(r["track_id"])
    return by_obj


def test_serialtrack_registered():
    assert METHOD_SERIALTRACK in TRACKING_METHODS
    assert METHOD_CENTROID in TRACKING_METHODS


@pytest.mark.parametrize("mode", ["Incremental", "Cumulative"])
def test_serialtrack_stable_ids(mode):
    n_obj, n_frames = 6, 4
    rows, _ = _make_rows(n_obj=n_obj, n_frames=n_frames)
    out = link_objects(
        rows, max_displacement_px=80.0, min_track_length=2,
        method=METHOD_SERIALTRACK, st_mode=mode,
    )
    by_obj = _ids_per_object(out, n_obj)

    # Each object keeps exactly one track id across all frames.
    assert all(len(s) == 1 for s in by_obj.values()), by_obj
    # Distinct objects get distinct tracks.
    all_ids = {next(iter(s)) for s in by_obj.values()}
    assert len(all_ids) == n_obj
    # Every detection is assigned and spans all frames.
    assert all(r["track_id"] is not None for r in out)
    assert all(r["track_length"] == n_frames for r in out)
    assert all(r["track_validation"] == "unvalidated" for r in out)


def test_serialtrack_min_track_length_filter():
    # Two frames only; min_track_length=3 must discard every track.
    rows, _ = _make_rows(n_obj=5, n_frames=2)
    out = link_objects(
        rows, max_displacement_px=80.0, min_track_length=3,
        method=METHOD_SERIALTRACK, st_mode="Incremental",
    )
    assert all(r["track_id"] is None for r in out)


def test_serialtrack_via_params_and_um_conversion():
    # link_objects_with_params reads the node param dict and forwards st_*; a µm
    # max_distance is converted to px via pixel_size_um before becoming f_o_s.
    rows, _ = _make_rows(n_obj=6, n_frames=4)
    params = dict(
        method=METHOD_SERIALTRACK, max_distance=40.0, distance_unit="µm",
        min_track_length=2, st_mode="Incremental", st_n_neighbors=25,
    )
    out = link_objects_with_params(rows, params, pixel_size_um=0.5)  # 40µm -> 80px
    n_tracks = len({r["track_id"] for r in out if r["track_id"] is not None})
    assert n_tracks == 6


def test_serialtrack_matches_centroid_on_simple_translation():
    # On clean rigid translation both linkers should recover the same track count.
    rows_a, _ = _make_rows(seed=7)
    rows_b = [dict(r) for r in rows_a]
    out_c = link_objects(rows_a, max_displacement_px=80.0, method=METHOD_CENTROID)
    out_s = link_objects(rows_b, max_displacement_px=80.0,
                         method=METHOD_SERIALTRACK, st_mode="Incremental")
    n_c = len({r["track_id"] for r in out_c if r["track_id"] is not None})
    n_s = len({r["track_id"] for r in out_s if r["track_id"] is not None})
    assert n_c == n_s == 6
