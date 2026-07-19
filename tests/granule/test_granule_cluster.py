"""Headless tests for ``backend.analysis.granule_cluster`` (V1.70 · P2).

Covers the three behaviours the P2 spec pins down:

1. Three well-separated planted blobs (seed n=5, relax 60%) → BIC picks k≈3 and
   the labels reproduce the planted partition (Rand index ≈ 1).
2. A single elongated (anisotropic-covariance) blob stays **one** cluster under
   the full-covariance GMM — the reason we use ``covariance_type="full"``.
3. Determinism: two calls with identical inputs give identical labels + BICs.

``pytest.importorskip`` skips the whole module cleanly when scikit-learn is
absent (it is an optional dependency, per CLAUDE.md).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from nd2studios.backend.analysis.granule_cluster import cluster_granules

# Strongly anisotropic voxel size (thick Z) to exercise the µm-scaling path.
VOXEL_SIZE_UM = (4.0, 0.5, 0.5)  # (dz, dy, dx)


def _rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Permutation-invariant Rand index (1.0 = identical partitions)."""
    a = np.asarray(a)
    b = np.asarray(b)
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    iu = np.triu_indices(len(a), k=1)
    return float((same_a[iu] == same_b[iu]).mean())


def _plant_three_blobs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three compact blobs, well separated in *voxel* space (z,y,x)."""
    # Centres are far apart after µm scaling; note z-separation of 20 voxels
    # * dz=4 µm = 80 µm, so anisotropy does not smear the blobs together.
    centres = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 80.0, 80.0],
        [40.0, 5.0, 80.0],
    ])
    pts, planted = [], []
    for gid, c in enumerate(centres):
        blob = c + rng.normal(scale=1.0, size=(30, 3))
        pts.append(blob)
        planted.append(np.full(30, gid, dtype=int))
    return np.vstack(pts), np.concatenate(planted)


def test_three_wellseparated_blobs_bic_picks_three():
    rng = np.random.default_rng(1234)
    pts, planted = _plant_three_blobs(rng)

    params = {"n_granules": 5, "relax_pct": 60.0, "method": "gmm", "n_init": 3}
    labels, info = cluster_granules(pts, VOXEL_SIZE_UM, params)

    # Sweep bounds: k in [max(1,round(5*0.4)), round(5*1.6)] = [2, 8].
    assert set(info["bic_by_k"]).issubset(set(range(2, 9)))
    assert info["k"] == 3
    assert labels.shape == (pts.shape[0],)
    assert info["means"].shape == (3, 3)
    # Perfect recovery of the planted partition.
    assert _rand_index(labels, planted) == pytest.approx(1.0)


def test_elongated_blob_stays_single_cluster_full_covariance():
    rng = np.random.default_rng(7)
    # One Gaussian blob, hugely elongated along x (in µm after scaling).
    n = 120
    zyx = np.zeros((n, 3))
    zyx[:, 0] = rng.normal(scale=0.5, size=n)   # z: tight
    zyx[:, 1] = rng.normal(scale=0.5, size=n)   # y: tight
    zyx[:, 2] = rng.normal(scale=40.0, size=n)  # x: very stretched
    zyx += np.array([20.0, 20.0, 40.0])         # shift off the origin

    # Offer k in [1, 3]: k_lo=max(1,round(2*0.5))=1, k_hi=round(2*1.5)=3.
    params = {"n_granules": 2, "relax_pct": 50.0, "method": "gmm", "n_init": 3}
    labels, info = cluster_granules(zyx, VOXEL_SIZE_UM, params)

    assert 1 in info["bic_by_k"] and 3 in info["bic_by_k"]
    # Full covariance models the elongation with a single Gaussian → k == 1.
    assert info["k"] == 1
    assert len(np.unique(labels)) == 1


def test_determinism_two_calls_identical():
    rng = np.random.default_rng(99)
    pts, _ = _plant_three_blobs(rng)
    params = {"n_granules": 4, "relax_pct": 50.0, "method": "gmm", "n_init": 2}

    labels_a, info_a = cluster_granules(pts, VOXEL_SIZE_UM, params)
    labels_b, info_b = cluster_granules(pts, VOXEL_SIZE_UM, params)

    assert np.array_equal(labels_a, labels_b)
    assert info_a["k"] == info_b["k"]
    assert info_a["bic_by_k"] == info_b["bic_by_k"]


def _clusters_are_pure(pred: np.ndarray, planted: np.ndarray) -> bool:
    """True if every predicted cluster lies within a single planted group."""
    return all(
        len(np.unique(planted[pred == lab])) == 1 for lab in np.unique(pred)
    )


def test_kmeans_fallback_never_merges_separated_blobs():
    # The KMeans path is the fallback for sparse/unstable clouds; its X-means
    # BIC may over-split ultra-tight blobs, but it must never *merge* two
    # well-separated granules. We assert that (purity), plus valid plumbing:
    # correct label shape, k inside the swept range, and the µm-scaling path.
    rng = np.random.default_rng(2024)
    pts, planted = _plant_three_blobs(rng)
    params = {"n_granules": 5, "relax_pct": 60.0, "method": "kmeans", "n_init": 5}
    labels, info = cluster_granules(pts, VOXEL_SIZE_UM, params)

    assert labels.shape == (pts.shape[0],)
    assert info["k"] in range(2, 9)          # within [k_lo, k_hi] = [2, 8]
    assert info["k"] >= 3                     # at least the three blobs resolved
    assert info["means"].shape == (info["k"], 3)
    assert _clusters_are_pure(labels, planted)


def test_empty_cloud_returns_empty():
    labels, info = cluster_granules(
        np.zeros((0, 3)), VOXEL_SIZE_UM,
        {"n_granules": 3, "relax_pct": 50.0},
    )
    assert labels.shape == (0,)
    assert info["k"] == 0
    assert info["means"].shape == (0, 3)


def test_tiny_n_clamps_k_to_points():
    # Two points but a seed asking for many clusters → k clamped to <= N.
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 40.0, 40.0]])
    labels, info = cluster_granules(
        pts, VOXEL_SIZE_UM,
        {"n_granules": 10, "relax_pct": 50.0, "method": "gmm", "n_init": 1},
    )
    assert labels.shape == (2,)
    assert info["k"] <= 2
    assert max(info["bic_by_k"]) <= 2
