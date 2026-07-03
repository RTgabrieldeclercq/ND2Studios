"""Per-cell spatial metrics and self-fold-change.

Vendored from CellTracker ``backend/measurement.py`` (``compute_spatial_metrics``)
and ``backend/fields.py`` (``compute_self_fold_change``). The basic per-cell
measurement (``measure_timeseries`` etc.) is **not** copied — ND2Studios already
produces per-object measurements via :mod:`nd2studios.backend.results_engine`.
Only the post-tracking metrics CellTracker uniquely adds are kept here, and the
upstream skimage import is dropped (these functions need only numpy/scipy/pandas).

Both operate on a tracked pandas DataFrame (``frame``, ``track_id``,
``centroid_y``, ``centroid_x``, and an intensity column for self-fold-change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  Post-tracking spatial metrics
# ═══════════════════════════════════════════════════════════════

def compute_spatial_metrics(
    df: pd.DataFrame,
    n_neighbors: int = 6,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Compute per-cell spatial metrics from tracked measurements.
    Should be called AFTER tracking (needs track_id for velocity).

    Adds columns:
    - neighbor_dist_mean : average distance to K nearest neighbors
    - neighbor_dist_std  : std of neighbor distances
    - local_divergence   : divergence of the local displacement field
                           (positive = cells spreading apart, negative = converging)
    - local_curl         : curl magnitude of the local displacement field
                           (rotation of neighbors around the cell)

    Parameters
    ----------
    df : DataFrame with frame, track_id, centroid_y, centroid_x
    n_neighbors : int, number of nearest neighbors to consider
    progress_cb : callable(int)

    Returns
    -------
    df with new columns added
    """
    from scipy.spatial import cKDTree

    df = df.copy()
    df["neighbor_dist_mean"] = np.nan
    df["neighbor_dist_std"] = np.nan
    df["local_divergence"] = np.nan
    df["local_curl"] = np.nan

    frames = sorted(df["frame"].unique())

    # Pre-compute per-frame velocity from tracking
    # velocity = centroid(t) - centroid(t-1) for the same track_id
    df = df.sort_values(["track_id", "frame"])
    df["vy"] = np.nan
    df["vx"] = np.nan
    for tid, grp in df.groupby("track_id"):
        if len(grp) < 2:
            continue
        g = grp.sort_values("frame")
        dy = g["centroid_y"].diff()
        dx = g["centroid_x"].diff()
        df.loc[g.index, "vy"] = dy
        df.loc[g.index, "vx"] = dx

    for fi, t in enumerate(frames):
        fdf = df[df["frame"] == t]
        if len(fdf) < 3:
            if progress_cb and fi % 10 == 0:
                progress_cb(int((fi + 1) / len(frames) * 100))
            continue

        centroids = fdf[["centroid_y", "centroid_x"]].values
        K = min(n_neighbors, len(centroids) - 1)
        tree = cKDTree(centroids)
        dists, indices = tree.query(centroids, k=K + 1)

        # Neighbor distances (skip self at index 0)
        nb_dists = dists[:, 1:]
        df.loc[fdf.index, "neighbor_dist_mean"] = nb_dists.mean(axis=1)
        df.loc[fdf.index, "neighbor_dist_std"] = nb_dists.std(axis=1)

        # Local divergence and curl from velocity field
        vy = fdf["vy"].values
        vx = fdf["vx"].values

        if np.isnan(vy).all():
            if progress_cb and fi % 10 == 0:
                progress_cb(int((fi + 1) / len(frames) * 100))
            continue

        for i in range(len(centroids)):
            nb_idx = indices[i, 1:K+1]
            nb_valid = ~(np.isnan(vy[nb_idx]) | np.isnan(vx[nb_idx]))
            if nb_valid.sum() < 2:
                continue

            # Relative positions of neighbors
            dy_pos = centroids[nb_idx[nb_valid], 0] - centroids[i, 0]
            dx_pos = centroids[nb_idx[nb_valid], 1] - centroids[i, 1]
            r = np.sqrt(dy_pos**2 + dx_pos**2)
            r = np.clip(r, 1.0, None)

            # Relative velocities
            dvy = vy[nb_idx[nb_valid]] - (vy[i] if not np.isnan(vy[i]) else 0)
            dvx = vx[nb_idx[nb_valid]] - (vx[i] if not np.isnan(vx[i]) else 0)

            # Divergence: radial component of relative velocity / distance
            # div ~ <(dr . dv) / |dr|^2>
            radial = (dy_pos * dvy + dx_pos * dvx) / (r**2)
            div_val = float(np.mean(radial))

            # Curl: tangential component of relative velocity / distance
            # curl ~ <(dr x dv) / |dr|^2>  (2D cross product = dy*dvx - dx*dvy)
            tangential = (dy_pos * dvx - dx_pos * dvy) / (r**2)
            curl_val = float(np.mean(tangential))

            df.loc[fdf.index[i], "local_divergence"] = div_val
            df.loc[fdf.index[i], "local_curl"] = curl_val

        if progress_cb and fi % 10 == 0:
            progress_cb(int((fi + 1) / len(frames) * 100))

    # Clean up temp columns
    df = df.drop(columns=["vy", "vx"], errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════
#  Self fold change (per-cell intensity vs. its own time-average)
# ═══════════════════════════════════════════════════════════════

def compute_self_fold_change(
    df: pd.DataFrame,
    intensity_col: str,
    out_col: str = "_self_fold",
) -> pd.DataFrame:
    """
    Compute per-cell fold change against each cell's own time-average.

    For each tracked cell, computes:
        self_fold = intensity(t) / mean(intensity over all t for this track)

    This shows whether a cell is brighter or dimmer than its own baseline,
    independent of other cells.

    Parameters
    ----------
    df : tracked DataFrame with track_id and an intensity column
    intensity_col : e.g. "ERK-mRuby2_mean"
    out_col : name of the column to write (default "_self_fold")

    Returns
    -------
    df with the fold-change column added
    """
    if out_col in df.columns:
        return df

    df = df.copy()
    # Each cell's time-average
    track_means = df.groupby("track_id")[intensity_col].transform("mean")
    df[out_col] = df[intensity_col] / track_means.clip(lower=1e-10)
    return df
