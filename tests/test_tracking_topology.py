"""V1.46 — Cell-Tracker topology/fingerprint tracking: correctness + progress.

Guards the V1.46 refactor of ``backend.celltracker.tracking``:

* the **vectorized** ``compute_topology_features`` must be bit-for-bit identical
  to the original per-cell loop (so the topology cost — and therefore the track
  assignment — is unchanged); and
* ``track_timeseries`` / ``track_fingerprint`` must emit **monotonic** progress
  fractions ending at 1.0 (the fix for "no progress"), while still assigning a
  sensible track_id partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nd2studios.backend.celltracker.tracking import (
    compute_topology_features, track_fingerprint, track_timeseries,
)


def _topology_features_reference(centroids: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """The ORIGINAL per-cell loop, kept here as the equivalence oracle."""
    from scipy.spatial import cKDTree

    N = len(centroids)
    if N < 2:
        return np.zeros((N, 2 * n_neighbors))
    K = min(n_neighbors, N - 1)
    tree = cKDTree(centroids)
    dists, indices = tree.query(centroids, k=K + 1)
    dists = dists[:, 1:]
    indices = indices[:, 1:]
    features = np.zeros((N, 2 * n_neighbors))
    for i in range(N):
        k = min(K, dists.shape[1])
        features[i, :k] = dists[i, :k]
        neighbors = centroids[indices[i, :k]] - centroids[i]
        angles = np.arctan2(neighbors[:, 0], neighbors[:, 1])
        angles_sorted = np.sort(angles)
        if k > 1:
            gaps = np.diff(angles_sorted)
            gaps = np.append(gaps, 2 * np.pi + angles_sorted[0] - angles_sorted[-1])
            gaps_sorted = np.sort(gaps)
            features[i, n_neighbors:n_neighbors + len(gaps_sorted)] = gaps_sorted
    return features


@pytest.mark.parametrize("N", [1, 2, 3, 6, 25, 200])
@pytest.mark.parametrize("n_neighbors", [1, 3, 5, 8])
def test_topology_features_match_reference(N: int, n_neighbors: int) -> None:
    rng = np.random.default_rng(N * 100 + n_neighbors)
    centroids = rng.uniform(0, 512, size=(N, 2))
    got = compute_topology_features(centroids, n_neighbors)
    ref = _topology_features_reference(centroids, n_neighbors)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def _synthetic_tracks_df(T: int = 12, n: int = 40, seed: int = 0) -> pd.DataFrame:
    """n objects drifting slowly over T frames — a clean, linkable field."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(20, 480, size=(n, 2))
    vel = rng.uniform(-1.5, 1.5, size=(n, 2))
    recs = []
    for t in range(T):
        pos = start + vel * t + rng.normal(0, 0.3, size=(n, 2))
        for lbl in range(n):
            recs.append({
                "frame": t, "label": lbl + 1,
                "centroid_y": pos[lbl, 0], "centroid_x": pos[lbl, 1],
                "area": 100.0 + rng.normal(0, 3),
            })
    return pd.DataFrame(recs)


def _assert_monotonic_progress(fractions):
    assert fractions, "no progress reported"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions), "progress went backwards"
    assert fractions[-1] == pytest.approx(1.0), "progress did not reach 1.0"


def test_track_timeseries_progress_and_ids() -> None:
    df = _synthetic_tracks_df()
    seen = []
    out = track_timeseries(
        df, max_dist=30.0, n_neighbors=5, use_topology=True, topo_weight=0.3,
        progress_cb=lambda frac, msg: seen.append(frac),
    )
    _assert_monotonic_progress(seen)
    assert "track_id" in out.columns
    # Clean slow-drift field → most objects should form full-length tracks.
    counts = out[out["track_id"] >= 0]["track_id"].value_counts()
    assert (counts >= df["frame"].nunique() - 2).sum() >= 30


def test_track_fingerprint_progress_and_ids() -> None:
    df = _synthetic_tracks_df(seed=1)
    seen = []
    out = track_fingerprint(
        df, max_dist=30.0, area_weight=0.3, max_gap=2,
        progress_cb=lambda frac, msg: seen.append(frac),
    )
    _assert_monotonic_progress(seen)
    assert "track_id" in out.columns
    assert (out["track_id"] >= 0).sum() > 0
