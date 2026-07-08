"""
QC report (spec §1 / §9): what regime and engine ran, where each tile landed,
pairwise registration confidence, and which tiles fell back to coordinate
placement. Written as a JSON sidecar plus a rendered PNG.

Backend-pure: uses matplotlib's Agg backend, never PySide6.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from nd2studios.backend.stitch.dataset import Dataset


def _qc_paths(out_path: str) -> Tuple[str, str]:
    base = out_path
    for ext in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base + ".stitch_qc.json", base + ".stitch_qc.png"


def write_qc(out_path: str,
             dataset: Dataset,
             positions: Dict[int, Tuple[float, float]],
             confidences: Dict,
             regime: str,
             info: Dict) -> Dict[str, str]:
    """Write JSON + PNG QC sidecars. Returns {'json':…, 'png':…}."""
    json_path, png_path = _qc_paths(out_path)

    conf_vals = list(confidences.values()) if confidences else []
    report = {
        "regime": regime,
        "engine": info.get("engine", "?"),
        "n_tiles": dataset.n_tiles,
        "grid_shape": list(dataset.grid_shape) if dataset.grid_shape else None,
        "overlap_frac": dataset.overlap_frac,
        "overlap_frac_x": dataset.overlap_frac_x,
        "overlap_frac_y": dataset.overlap_frac_y,
        "pixel_size_um": dataset.pixel_size_um,
        "stage_present": dataset.stage_present,
        "is_regular_grid": dataset.is_regular_grid,
        "axis_flip_x": dataset.axis_flip_x,
        "axis_flip_y": dataset.axis_flip_y,
        "swap_xy": dataset.swap_xy,
        "fell_back_to_coordinates": info.get("fell_back_to_coordinates", regime == "zero_overlap"),
        "n_pairs": info.get("n_pairs"),
        "n_accepted_pairs": info.get("n_accepted"),
        "mean_confidence": (sum(conf_vals) / len(conf_vals)) if conf_vals else None,
        "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "positions": {str(m): [round(y, 2), round(x, 2)] for m, (y, x) in positions.items()},
        "pair_confidences": {f"{a}-{b}": round(v, 3) for (a, b), v in confidences.items()},
    }
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except Exception:
        pass

    try:
        _render_png(png_path, dataset, positions, confidences, regime, info)
    except Exception:
        png_path = ""
    return {"json": json_path, "png": png_path}


def _render_png(png_path: str,
                dataset: Dataset,
                positions: Dict[int, Tuple[float, float]],
                confidences: Dict,
                regime: str,
                info: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    h, w = dataset.tile_h, dataset.tile_w
    fig, ax = plt.subplots(figsize=(8, 8))
    for m, (y, x) in positions.items():
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="#7c5cff", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, str(m), ha="center", va="center",
                fontsize=7, color="#7c5cff")
    if positions:
        xs = [x for (_y, x) in positions.values()]
        ys = [y for (y, _x) in positions.values()]
        ax.set_xlim(min(xs) - w * 0.1, max(xs) + w * 1.1)
        ax.set_ylim(max(ys) + h * 1.1, min(ys) - h * 0.1)  # invert Y (image order)
    ax.set_aspect("equal")
    eng = info.get("engine", "?")
    ov = dataset.overlap_frac
    ax.set_title(f"Stitch QC — regime={regime}, engine={eng}, "
                 f"overlap≈{ov:.1%}, tiles={dataset.n_tiles}")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
