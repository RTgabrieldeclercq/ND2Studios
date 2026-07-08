"""V1.58 — birth/death LAP + mask-overlap (IoU) tracking.

Guards the two fixes for the post-StarDist tracking failure modes:

* the per-frame assignment is now a Jaqaman LAP with birth/death "no-match"
  nodes (:func:`solve_lap`), so a detection whose true partner is out of range
  starts/ends a track instead of being force-linked onto a neighbor — the
  "tracking currents" (low ``max_dist``) and spurious long links (high
  ``max_dist``) both disappear; and
* :func:`track_overlap` links segmented objects by mask IoU, the robust default
  for dense, slowly-moving nuclei.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from nd2studios.backend.celltracker.tracking import (
    _solve_lap_block, link_frames, solve_lap, track_overlap, track_timeseries,
)


# ─────────────────────────────── solve_lap ───────────────────────────────────
@pytest.mark.parametrize("seed", range(25))
def test_solve_lap_decomposition_matches_full(seed: int) -> None:
    """The connected-component split must give the same optimum as one big
    augmented solve (same total cost + match count)."""
    rng = np.random.default_rng(seed)
    N, M = int(rng.integers(1, 9)), int(rng.integers(1, 9))
    cost = rng.uniform(0, 10, size=(N, M))
    cost[rng.uniform(size=(N, M)) < 0.5] = np.inf   # gate ~half the pairs
    d = float(rng.uniform(2, 8))
    matches, up, uc = solve_lap(cost, d)
    ref = _solve_lap_block(cost, d)
    assert len(matches) == len(ref)
    assert abs(sum(cost[r, c] for r, c in matches)
               - sum(cost[r, c] for r, c in ref)) < 1e-9
    # bookkeeping is self-consistent
    assert len(matches) + len(up) == N
    assert len(matches) + len(uc) == M
    for r, c in matches:
        assert np.isfinite(cost[r, c])


def test_solve_lap_all_forbidden_is_all_births_deaths() -> None:
    cost = np.full((3, 4), np.inf)
    matches, up, uc = solve_lap(cost, 5.0)
    assert matches == []
    assert up == [0, 1, 2] and uc == [0, 1, 2, 3]


def _old_bare_link(prev: np.ndarray, curr: np.ndarray, max_dist: float) -> set:
    """The pre-V1.58 linker: bare Hungarian + post-gate (no no-match node)."""
    dist = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=2)
    cost = dist.copy()
    cost[dist > max_dist] = 1e6
    ri, ci = linear_sum_assignment(cost)
    return {(int(r), int(c)) for r, c in zip(ri, ci) if dist[r, c] < max_dist}


def test_birth_death_breaks_the_chaining_current() -> None:
    """Row of cells where the leftmost leaves and a new one enters on the right.

    The bare linker force-matches every cell onto its neighbor (the propagating
    "current"); the birth/death LAP instead links each cell to its own position
    and records one death + one birth.
    """
    prev = np.array([[0, 0], [0, 18], [0, 36], [0, 54], [0, 72]], float)
    curr = np.array([[0, 18], [0, 36], [0, 54], [0, 72], [0, 90]], float)

    old = _old_bare_link(prev, curr, 20.0)
    assert (0, 0) in old and len(old) == 5           # chains: P0→C@18, all shift

    matches, up, uc = link_frames(prev, curr, max_dist=20.0, topo_weight=0.0)
    assert set(matches) == {(1, 0), (2, 1), (3, 2), (4, 3)}
    assert up == [0] and uc == [4]                   # P0 dies, C@90 is born


def test_high_max_dist_does_not_force_a_long_link() -> None:
    """One cell moves a little; a far detection exists within a wide gate. The
    LAP must NOT bridge the far one — it becomes a birth, not a long link."""
    prev = np.array([[0.0, 0.0]])
    curr = np.array([[0.0, 3.0], [0.0, 80.0]])       # true partner + a distractor
    matches, up, uc = link_frames(prev, curr, max_dist=100.0, topo_weight=0.0)
    assert set(matches) == {(0, 0)}                  # links the 3px move only
    assert uc == [1]                                 # far detection is a birth


# ─────────────────────────────── track_overlap ───────────────────────────────
def _square(mask, label, y, x, s=10):
    mask[y:y + s, x:x + s] = label


def _df_from_masks(masks: np.ndarray) -> pd.DataFrame:
    recs = []
    for t in range(masks.shape[0]):
        for lab in np.unique(masks[t]):
            if lab == 0:
                continue
            ys, xs = np.nonzero(masks[t] == lab)
            recs.append({"frame": t, "label": int(lab),
                         "centroid_y": ys.mean(), "centroid_x": xs.mean(),
                         "area": float(ys.size)})
    return pd.DataFrame(recs)


def test_overlap_keeps_identity_through_close_approach() -> None:
    """Two nuclei approach each other; overlap keeps each identity stable where
    a nearest-centroid linker could swap them."""
    T, H, W = 6, 60, 120
    masks = np.zeros((T, H, W), np.int32)
    for t in range(T):
        _square(masks[t], 1, 25, 10 + 3 * t)   # →
        _square(masks[t], 2, 25, 80 - 3 * t)   # ←
    res = track_overlap(_df_from_masks(masks), masks, min_iou=0.1, max_gap=1)
    t1 = res[res.label == 1].sort_values("frame")["track_id"].tolist()
    t2 = res[res.label == 2].sort_values("frame")["track_id"].tolist()
    assert len(set(t1)) == 1 and t1[0] != -1
    assert len(set(t2)) == 1 and t2[0] != t1[0]


def test_overlap_new_object_is_a_birth() -> None:
    T, H, W = 6, 60, 120
    masks = np.zeros((T, H, W), np.int32)
    for t in range(T):
        _square(masks[t], 1, 25, 10 + 3 * t)
        if t >= 3:
            _square(masks[t], 3, 5, 5 + 2 * t)
    res = track_overlap(_df_from_masks(masks), masks, min_iou=0.1, max_gap=1)
    n3 = res[res.label == 3]
    assert n3["frame"].min() == 3
    assert n3["track_id"].nunique() == 1 and -1 not in set(n3["track_id"])
    # its id never appears before it was born
    assert not (set(res[res.frame < 3]["track_id"]) & set(n3["track_id"]))


def test_overlap_relinks_across_a_gap() -> None:
    T, H, W = 6, 60, 120
    masks = np.zeros((T, H, W), np.int32)
    for t in range(T):
        _square(masks[t], 1, 25, 10 + 3 * t)
        if t != 3:                              # nucleus 2 stationary, absent at t=3
            _square(masks[t], 2, 25, 80)
    res = track_overlap(_df_from_masks(masks), masks, min_iou=0.1, max_gap=1)
    n2 = res[res.label == 2].sort_values("frame")
    before = n2[n2.frame <= 2]["track_id"].tolist()
    after = n2[n2.frame >= 4]["track_id"].tolist()
    assert before and after and len(set(before + after)) == 1


def test_track_timeseries_clean_field_still_links() -> None:
    """Regression: the LAP change must not fragment an easy slow-drift field."""
    rng = np.random.default_rng(0)
    start = rng.uniform(20, 480, size=(40, 2)); vel = rng.uniform(-1.5, 1.5, size=(40, 2))
    recs = []
    for t in range(12):
        pos = start + vel * t + rng.normal(0, 0.3, size=(40, 2))
        for lbl in range(40):
            recs.append({"frame": t, "label": lbl + 1,
                         "centroid_y": pos[lbl, 0], "centroid_x": pos[lbl, 1], "area": 100.0})
    out = track_timeseries(pd.DataFrame(recs), max_dist=30.0, n_neighbors=5,
                           use_topology=True, topo_weight=0.3)
    counts = out[out["track_id"] >= 0]["track_id"].value_counts()
    assert (counts >= 10).sum() >= 30           # most objects → full-length tracks
