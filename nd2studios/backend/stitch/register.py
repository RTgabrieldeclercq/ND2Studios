"""
Built-in phase-correlation registration (spec §3) — the always-available
overlap engine (no ashlar/m2stitch, no Java).

Pipeline per adjacent pair:
  1. Extract the expected overlap patches from the two tiles under the
     coordinate seed placement.
  2. High-pass (band-pass) + Hann-window, then sub-pixel phase correlation
     (`skimage.registration.phase_cross_correlation`) → residual shift.
  3. Score with normalized cross-correlation of the aligned overlap; reject
     matches below `ncc_threshold` or beyond `max_shift`.
  4. Globally optimize absolute positions: weighted least squares over accepted
     edges, Tikhonov-anchored to the coordinate seeds. The anchor fixes the
     gauge, keeps disconnected tiles at their seed, and bounds drift. Each axis
     (row/col) decouples and is solved independently.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.stitch.config import StitchConfig
from nd2studios.backend.stitch.dataset import Dataset


# ── low-level image ops ──

def _highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    img = img.astype(np.float32)
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        img = img - gaussian_filter(img, sigma=sigma)
    return img


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float((a * b).sum() / (da * db))


def _overlap_boxes(dr: int, dc: int, h: int, w: int):
    """Overlap of tile A[0:h,0:w] with tile B placed at (dr,dc). Returns
    ((ar0,ar1,ac0,ac1), (br0,br1,bc0,bc1)) or None if there is no overlap."""
    ar0, ar1 = max(0, dr), min(h, dr + h)
    ac0, ac1 = max(0, dc), min(w, dc + w)
    if ar1 <= ar0 or ac1 <= ac0:
        return None
    br0, br1 = ar0 - dr, ar1 - dr
    bc0, bc1 = ac0 - dc, ac1 - dc
    return (ar0, ar1, ac0, ac1), (br0, br1, bc0, bc1)


def _pair_shift(patch_a: np.ndarray, patch_b: np.ndarray, config: StitchConfig
                ) -> Tuple[Optional[np.ndarray], float]:
    """Return (shift (row,col) applied to align B→A assumption, ncc) or (None, 0)."""
    if patch_a.shape != patch_b.shape:
        return None, 0.0
    if min(patch_a.shape) < 8:
        return None, 0.0
    fa = _highpass(patch_a, config.filter_sigma)
    fb = _highpass(patch_b, config.filter_sigma)
    win = _hann2d(fa.shape)
    fa_w, fb_w = fa * win, fb * win
    if fa_w.std() < 1e-6 or fb_w.std() < 1e-6:
        return None, 0.0
    try:
        from skimage.registration import phase_cross_correlation
        shift, _err, _phase = phase_cross_correlation(
            fa_w, fb_w, upsample_factor=max(1, int(config.upsample_factor)),
            normalization=None,
        )
    except Exception:
        return None, 0.0
    shift = np.asarray(shift, dtype=np.float64)
    # Confidence: NCC of the overlap after aligning B by `shift`.
    from scipy.ndimage import shift as nd_shift
    b_aligned = nd_shift(fb, shift=shift, order=1, mode="constant", cval=0.0)
    ncc = _ncc(fa, b_aligned)
    return shift, ncc


# ── adjacency ──

def _candidate_pairs(dataset: Dataset,
                     seed: Dict[int, Tuple[float, float]]) -> List[Tuple[int, int]]:
    """Adjacent tile pairs to register.

    Regular grids use 4-neighborhood (row±1 / col±1). Otherwise any two tiles
    whose seed rectangles overlap by ≥ 8 px on both axes are candidates.
    """
    tiles = dataset.tiles
    h, w = dataset.tile_h, dataset.tile_w
    pairs: List[Tuple[int, int]] = []
    if dataset.is_regular_grid and all(t.row is not None for t in tiles):
        by_cell = {(t.row, t.col): t.index for t in tiles}
        for t in tiles:
            for dr, dc in ((0, 1), (1, 0)):
                nb = by_cell.get((t.row + dr, t.col + dc))
                if nb is not None:
                    pairs.append((t.index, nb))
        return pairs
    idx = [t.index for t in tiles]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ia, ib = idx[a], idx[b]
            (ya, xa), (yb, xb) = seed[ia], seed[ib]
            oy = min(ya + h, yb + h) - max(ya, yb)
            ox = min(xa + w, xb + w) - max(xa, xb)
            if oy >= 8 and ox >= 8:
                pairs.append((ia, ib))
    return pairs


# ── global optimization ──

def _solve_axis(n: int, node_of: Dict[int, int],
                edges: List[Tuple[int, int, float, float]],
                seed_vals: np.ndarray, lam: float) -> np.ndarray:
    """Weighted least squares: min Σ w((p_j-p_i)-d)² + λΣ(p_i-s_i)².

    ``edges`` are (i_node, j_node, d, w); returns the solved positions vector.
    """
    L = np.zeros((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    for i, j, d, w in edges:
        L[i, i] += w; L[j, j] += w
        L[i, j] -= w; L[j, i] -= w
        c[i] -= w * d
        c[j] += w * d
    A = L + lam * np.eye(n)
    rhs = lam * seed_vals + c
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return seed_vals.copy()


def refine_positions(dataset: Dataset,
                     ref_frames: Dict[int, np.ndarray],
                     seed: Dict[int, Tuple[float, float]],
                     config: StitchConfig
                     ) -> Tuple[Dict[int, Tuple[float, float]], Dict, Dict]:
    """Refine coordinate ``seed`` positions by registering the align channel.

    Returns ``(positions, confidences, info)`` where ``confidences`` maps each
    accepted ``(i, j)`` pair to its NCC and ``info`` carries diagnostics.
    """
    h, w = dataset.tile_h, dataset.tile_w
    px = dataset.pixel_size_um
    if config.max_shift_um is not None:
        max_shift_px = config.max_shift_um / max(px, 1e-6)
    else:
        # Default: half the overlap width, min 8 px, capped at half a tile.
        ov = max(dataset.overlap_frac, 0.0)
        max_shift_px = float(np.clip(0.5 * ov * min(h, w), 8.0, 0.5 * min(h, w)))

    pairs = _candidate_pairs(dataset, seed)
    edges_row: List[Tuple[int, int, float, float]] = []
    edges_col: List[Tuple[int, int, float, float]] = []
    confidences: Dict[Tuple[int, int], float] = {}

    node_of = {t.index: k for k, t in enumerate(dataset.tiles)}
    n = len(dataset.tiles)

    n_accept = 0
    for ia, ib in pairs:
        if ia not in ref_frames or ib not in ref_frames:
            continue
        (ya, xa), (yb, xb) = seed[ia], seed[ib]
        dr, dc = int(round(yb - ya)), int(round(xb - xa))
        boxes = _overlap_boxes(dr, dc, h, w)
        if boxes is None:
            continue
        (ar0, ar1, ac0, ac1), (br0, br1, bc0, bc1) = boxes
        A = ref_frames[ia][ar0:ar1, ac0:ac1]
        B = ref_frames[ib][br0:br1, bc0:bc1]
        shift, ncc = _pair_shift(A, B, config)
        if shift is None or ncc < config.ncc_threshold:
            continue
        if abs(shift[0]) > max_shift_px or abs(shift[1]) > max_shift_px:
            continue
        # Measured relative offset j−i = integer extraction offset + residual.
        d_row = dr + float(shift[0])
        d_col = dc + float(shift[1])
        w_e = float(ncc)
        edges_row.append((node_of[ia], node_of[ib], d_row, w_e))
        edges_col.append((node_of[ia], node_of[ib], d_col, w_e))
        confidences[(ia, ib)] = float(ncc)
        n_accept += 1

    seed_row = np.array([seed[t.index][0] for t in dataset.tiles], dtype=np.float64)
    seed_col = np.array([seed[t.index][1] for t in dataset.tiles], dtype=np.float64)

    if n_accept == 0:
        # No trustworthy edges — fall back to coordinates (spec §5 fallback).
        return dict(seed), confidences, {
            "engine": "phase_correlation", "n_pairs": len(pairs),
            "n_accepted": 0, "fell_back_to_coordinates": True,
            "max_shift_px": max_shift_px,
        }

    lam = 0.05
    p_row = _solve_axis(n, node_of, edges_row, seed_row, lam)
    p_col = _solve_axis(n, node_of, edges_col, seed_col, lam)

    # Drift is bounded by (a) rejecting any *per-pair* shift beyond max_shift and
    # (b) the Tikhonov seed anchor (λ) in the global solve, which softly pulls the
    # solution toward the coordinate seeds. We deliberately do NOT hard-clip the
    # absolute solved position to seed ± max_shift: across a large connected
    # mosaic, legitimate corrections accumulate along the spanning path and can
    # exceed a single pair's bound — clipping there would wrongly pin far tiles.

    positions = {t.index: (float(p_row[k]), float(p_col[k]))
                 for k, t in enumerate(dataset.tiles)}
    info = {
        "engine": "phase_correlation", "n_pairs": len(pairs),
        "n_accepted": n_accept, "fell_back_to_coordinates": False,
        "max_shift_px": max_shift_px,
        "mean_confidence": float(np.mean(list(confidences.values()))),
    }
    return positions, confidences, info
