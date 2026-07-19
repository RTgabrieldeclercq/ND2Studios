"""Headless tests for ``nd2studios.backend.analysis.granule_boundary`` (V1.70 · P5).

Covers the outward boundary band: the isolated-cube shell, neighbour inclusion /
background-only restriction, dilation↔EDT agreement at N=1, EDT anisotropy
correctness, the combined-label overlap tie-break, and empty / one-voxel granules.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from nd2studios.backend.analysis.granule_boundary import extract_boundary_bands


# ── helpers ───────────────────────────────────────────────────────────────────
def _cityblock_dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Independent reference: grow a 3-D bool mask by taxicab distance ≤ n."""
    out = mask.copy()
    for _ in range(n):
        nb = out.copy()
        nb[1:, :, :] |= out[:-1, :, :]
        nb[:-1, :, :] |= out[1:, :, :]
        nb[:, 1:, :] |= out[:, :-1, :]
        nb[:, :-1, :] |= out[:, 1:, :]
        nb[:, :, 1:] |= out[:, :, :-1]
        nb[:, :, :-1] |= out[:, :, 1:]
        out = nb
    return out


def _labels_from_masks(shape, masks):
    lbl = np.zeros(shape, dtype=np.int32)
    for gid, m in masks.items():
        lbl[m] = gid
    return lbl


# ── single isolated cube ────────────────────────────────────────────────────────
def test_single_cube_dilation_two_voxel_shell():
    shape = (16, 16, 16)
    cube = np.zeros(shape, bool)
    cube[4:8, 4:8, 4:8] = True
    masks = {1: cube}
    labels = _labels_from_masks(shape, masks)

    bands, combined = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 2, "band_method": "dilation"},
    )
    band = bands[1]

    # Exactly the 2-voxel (taxicab) outward shell, and none of the granule itself.
    expected = _cityblock_dilate(cube, 2) & ~cube
    assert np.array_equal(band, expected)
    assert int((band & cube).sum()) == 0

    # Representative voxels off the -Z face centre: 1 & 2 out are in, 3 out is not.
    assert band[3, 5, 5] and band[2, 5, 5]
    assert not band[1, 5, 5]
    # A pure corner-diagonal voxel (taxicab dist 3) must be excluded (face structure).
    assert not band[3, 3, 3]

    # Combined label volume paints the band with the source id, nothing inside/self.
    assert np.array_equal(combined > 0, band)
    assert set(np.unique(combined)).issubset({0, 1})
    assert int((combined != 0)[cube].sum()) == 0


# ── two adjacent granules: neighbour inclusion toggle ───────────────────────────
def _two_adjacent():
    shape = (12, 12, 16)
    a = np.zeros(shape, bool)
    b = np.zeros(shape, bool)
    a[4:8, 4:8, 4:8] = True     # granule 1
    b[4:8, 4:8, 8:12] = True    # granule 2, face-adjacent along x at x=8
    masks = {1: a, 2: b}
    labels = _labels_from_masks(shape, masks)
    return shape, masks, labels


def test_neighbor_included_by_default():
    shape, masks, labels = _two_adjacent()
    bands, _ = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 1, "band_method": "dilation", "include_neighbors": True},
    )
    a_band = bands[1]
    # A's outward band reaches into granule B (label 2) within N=1.
    assert int((a_band & (labels == 2)).sum()) > 0
    # It never contains A's own voxels.
    assert int((a_band & masks[1]).sum()) == 0


def test_neighbor_excluded_when_background_only():
    shape, masks, labels = _two_adjacent()
    bands, _ = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 1, "band_method": "dilation", "include_neighbors": False},
    )
    a_band = bands[1]
    # Background-only: nothing that belongs to granule B is in A's band.
    assert int((a_band & (labels == 2)).sum()) == 0
    # But the band is non-empty (it still reaches background voxels).
    assert int(a_band.sum()) > 0
    assert np.all(labels[a_band] == 0)


# ── dilation vs EDT agree at N=1 on isotropic voxels ────────────────────────────
def test_dilation_and_edt_agree_at_n1_isotropic():
    shape = (9, 9, 9)
    cube = np.zeros(shape, bool)
    cube[3:6, 3:6, 3:6] = True
    masks = {1: cube}
    labels = _labels_from_masks(shape, masks)
    vox = (1.0, 1.0, 1.0)

    band_dil, _ = extract_boundary_bands(
        masks, labels, vox, {"band_voxels": 1, "band_method": "dilation"},
    )
    band_edt, _ = extract_boundary_bands(
        masks, labels, vox, {"band_voxels": 1, "band_method": "edt"},
    )
    assert np.array_equal(band_dil[1], band_edt[1])
    # And it is the 6-face shell (no diagonal neighbours).
    assert band_dil[1][2, 4, 4] and not band_dil[1][2, 2, 2]


# ── EDT anisotropy correctness ──────────────────────────────────────────────────
def test_edt_band_is_anisotropy_correct():
    shape = (5, 5, 5)
    seed = np.zeros(shape, bool)
    seed[2, 2, 2] = True
    masks = {1: seed}
    labels = _labels_from_masks(shape, masks)

    # dz=2 µm (coarse Z), dy=dx=1 µm; N=1 → threshold = 1*min = 1.0 µm.
    bands, _ = extract_boundary_bands(
        masks, labels, (2.0, 1.0, 1.0),
        {"band_voxels": 1, "band_method": "edt"},
    )
    band = bands[1]
    # In-plane (1 µm away) neighbours are inside the band...
    assert band[2, 1, 2] and band[2, 3, 2] and band[2, 2, 1] and band[2, 2, 3]
    # ...but the Z neighbours (2 µm away) are excluded by the metric threshold.
    assert not band[1, 2, 2] and not band[3, 2, 2]


# ── combined-label overlap tie-break ────────────────────────────────────────────
def test_overlap_prefers_nearest_surface_then_lowest_id():
    # 1-D line: granule 1 at x=1, granule 2 at x=6.
    shape = (1, 1, 8)
    a = np.zeros(shape, bool); a[0, 0, 1] = True
    b = np.zeros(shape, bool); b[0, 0, 6] = True
    masks = {1: a, 2: b}
    labels = _labels_from_masks(shape, masks)

    _, combined = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 3, "band_method": "dilation"},
    )
    # x=3: dist to 1 is 2, dist to 2 is 3 → nearest surface = granule 1.
    assert combined[0, 0, 3] == 1
    # x=4: dist to 1 is 3, dist to 2 is 2 → nearest surface = granule 2.
    assert combined[0, 0, 4] == 2


def test_equidistant_overlap_breaks_to_lowest_id():
    shape = (1, 1, 7)
    a = np.zeros(shape, bool); a[0, 0, 1] = True
    b = np.zeros(shape, bool); b[0, 0, 5] = True
    masks = {1: a, 2: b}
    labels = _labels_from_masks(shape, masks)

    _, combined = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 2, "band_method": "dilation"},
    )
    # x=3 is 2 voxels from each surface → tie → lowest id wins.
    assert combined[0, 0, 3] == 1


# ── degenerate granules ─────────────────────────────────────────────────────────
def test_empty_and_one_voxel_granules_handled():
    shape = (5, 6, 7)
    empty = np.zeros(shape, bool)
    single = np.zeros(shape, bool)
    single[2, 3, 3] = True
    masks = {1: empty, 2: single}
    labels = _labels_from_masks(shape, masks)

    bands, combined = extract_boundary_bands(
        masks, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 1, "band_method": "dilation"},
    )
    # Empty granule → an all-False band, no error.
    assert 1 in bands and not bands[1].any()
    # One-voxel granule → its 6-face neighbours (all in-bounds here).
    assert int(bands[2].sum()) == 6
    assert int((bands[2] & single).sum()) == 0
    assert set(np.unique(combined)).issubset({0, 2})


def test_no_masks_returns_empty():
    shape = (4, 4, 4)
    labels = np.zeros(shape, np.int32)
    bands, combined = extract_boundary_bands(
        {}, labels, (1.0, 1.0, 1.0), {"band_voxels": 2},
    )
    assert bands == {}
    assert combined.shape == shape and int(combined.sum()) == 0


def test_combined_labels_key_in_input_is_ignored():
    shape = (9, 9, 9)
    cube = np.zeros(shape, bool)
    cube[3:6, 3:6, 3:6] = True
    labels = _labels_from_masks(shape, {1: cube})
    # Caller hands us the whole record dict, including the reserved "_labels" entry.
    masks_with_reserved = {1: cube, "_labels": labels}
    bands, _ = extract_boundary_bands(
        masks_with_reserved, labels, (1.0, 1.0, 1.0),
        {"band_voxels": 1, "band_method": "dilation"},
    )
    assert set(bands.keys()) == {1}
