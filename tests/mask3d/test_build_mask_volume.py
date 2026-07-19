"""3D Mask Drawing — ``build_mask_volume`` propagation (Qt-free backend).

Regression for the single-drawn-plane bug: "Propagate now" (default mode
"Interpolate between planes") did nothing when the user had drawn on only one Z
plane, because the interpolate loop needs a *pair* of consecutive drawn planes.
A single drawn plane must fall back to copy-extrude through Z.
"""
from __future__ import annotations

import numpy as np

from nd2studios.backend.analysis import mask3d

Z, H, W = 8, 20, 20
_RECT = {"type": "rect", "vertices": [[5, 5], [12, 12]], "op": "add"}


def _filled(vol):
    return [z for z in range(vol.shape[0]) if vol[z].any()]


def test_single_plane_interpolate_extrudes_through_z():
    vol = mask3d.build_mask_volume({"3": [_RECT]}, Z, H, W,
                                   mask3d.PROPAGATE_INTERPOLATE)
    assert _filled(vol) == list(range(Z))          # every plane, not just z=3


def test_single_plane_interpolate_equals_copy():
    by_z = {"3": [_RECT]}
    vi = mask3d.build_mask_volume(by_z, Z, H, W, mask3d.PROPAGATE_INTERPOLATE)
    vc = mask3d.build_mask_volume(by_z, Z, H, W, mask3d.PROPAGATE_COPY)
    assert np.array_equal(vi, vc)


def test_single_plane_none_fills_only_drawn_plane():
    vol = mask3d.build_mask_volume({"3": [_RECT]}, Z, H, W, mask3d.PROPAGATE_NONE)
    assert _filled(vol) == [3]


def test_two_plane_interpolate_fills_throughout():
    # "Propagate across Z" fills the WHOLE stack from the drawn planes: morph
    # between them, copy the nearest drawn plane beyond the ends.
    a = {"type": "rect", "vertices": [[3, 3], [8, 8]], "op": "add"}
    b = {"type": "rect", "vertices": [[10, 10], [16, 16]], "op": "add"}
    vol = mask3d.build_mask_volume({"2": [a], "5": [b]}, Z, H, W,
                                   mask3d.PROPAGATE_INTERPOLATE)
    assert _filled(vol) == list(range(Z))                 # every plane, not just 2..5
    # drawn planes keep their exact shape
    assert np.array_equal(vol[2], mask3d.rasterize_plane([a], H, W))
    assert np.array_equal(vol[5], mask3d.rasterize_plane([b], H, W))
    # ends copy the nearest drawn plane
    assert np.array_equal(vol[0], vol[2]) and np.array_equal(vol[1], vol[2])
    assert np.array_equal(vol[6], vol[5]) and np.array_equal(vol[7], vol[5])


def test_two_plane_interpolate_morphs_between():
    # Planes strictly between two (overlapping) drawn planes are the signed-distance
    # blend — non-empty, and a shift of the drawn shapes. Uses overlapping rects
    # (a far-apart disjoint pair can legitimately vanish mid-morph — an SDF property).
    a = {"type": "rect", "vertices": [[4, 4], [14, 14]], "op": "add"}
    b = {"type": "rect", "vertices": [[7, 7], [17, 17]], "op": "add"}
    vol = mask3d.build_mask_volume({"1": [a], "6": [b]}, Z, H, W,
                                   mask3d.PROPAGATE_INTERPOLATE)
    assert _filled(vol) == list(range(Z))          # still fills throughout
    for z in (2, 3, 4, 5):
        assert vol[z].any()                        # morphed middle planes non-empty


def test_copy_extrudes_through_z():
    vol = mask3d.build_mask_volume({"5": [_RECT]}, Z, H, W, mask3d.PROPAGATE_COPY)
    assert _filled(vol) == list(range(Z))


def _poly(cy, cx, r):
    return {"type": "polygon",
            "vertices": [[cy - r, cx - r], [cy - r, cx + r],
                         [cy + r, cx + r], [cy + r, cx - r]], "op": "add"}


def test_volume_to_shapes_traces_all_components_by_default():
    # A plane with three DISJOINT shapes must trace back to three polygons — the
    # old default (max_regions=1) kept only the largest, so 2 of 3 were dropped.
    Zc, Hc, Wc = 4, 40, 40
    by_z = {"1": [_poly(6, 6, 3), _poly(6, 30, 3), _poly(32, 18, 4)]}
    vol = mask3d.build_mask_volume(by_z, Zc, Hc, Wc, mask3d.PROPAGATE_COPY)
    all_shapes = mask3d.volume_to_shapes(vol)               # default = all
    assert len(all_shapes[0]) == 3
    assert len(mask3d.volume_to_shapes(vol, max_regions=1)[0]) == 1


def test_multiple_disjoint_shapes_all_propagate():
    # Three freeform shapes drawn on one plane all propagate through the stack.
    Zc, Hc, Wc = 6, 40, 40
    by_z = {"2": [_poly(6, 6, 3), _poly(6, 30, 3), _poly(32, 18, 4)]}
    vol = mask3d.build_mask_volume(by_z, Zc, Hc, Wc, mask3d.PROPAGATE_INTERPOLATE)
    from skimage.measure import label as sk_label
    for z in range(Zc):
        assert int(sk_label(vol[z]).max()) == 3            # all 3 on every plane
