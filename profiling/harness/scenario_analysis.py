"""Measure runtime for every registered :class:`AnalysisPipeline`.

Generates a small synthetic ``(T, H, W)`` uint16 channel (a few
simulated bright spots on a low-intensity background) and runs each
pipeline against it. Synthetic data keeps the scenario reproducible
across machines — no real ND2 file required.

Pipelines whose required parameters can't be satisfied by the defaults
record an ``error`` field rather than a timing so the harness keeps
going. Cellpose-backed ``Nuclei Segmentation`` is allowed to fail
gracefully when ``cellpose`` is not installed.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from nd2studios.utils.profiling import measure

from profiling.harness.fixtures import SYNTH_H, SYNTH_T, SYNTH_W


def _build_synth_channels() -> Dict[str, np.ndarray]:
    """Construct a small (T, H, W) uint16 channel with simulated foci.

    Deterministic (seeded RNG). Two channels under deliberately-typical
    names — DAPI-style nuclear-ish background plus a bright-spot
    channel. Names chosen so a pipeline that asks for ``channel_name``
    has something sensible to pick.
    """
    rng = np.random.default_rng(seed=42)
    background = rng.integers(80, 130, size=(SYNTH_T, SYNTH_H, SYNTH_W),
                              dtype=np.uint16)
    spot_field = np.zeros_like(background)
    # 12 bright spots per frame at fixed coordinates.
    coords = [(50, 60), (180, 200), (120, 80), (40, 220),
              (200, 90), (220, 220), (160, 160), (80, 180),
              (110, 30), (30, 130), (240, 130), (140, 240)]
    for t in range(SYNTH_T):
        for (y, x) in coords:
            spot_field[t, max(0, y - 2):y + 3, max(0, x - 2):x + 3] += 800
    foci = np.clip(background.astype(np.int32) + spot_field, 0, 65535).astype(np.uint16)

    return {
        "DAPI": background,
        "Foci": foci,
    }


def _build_metadata() -> Dict[str, Any]:
    """Minimal metadata dict that satisfies the common pipeline reads."""
    return {
        "filepath": "<synthetic>",
        "n_timepoints": SYNTH_T,
        "n_zslices": 1,
        "n_channels": 2,
        "n_multipoints": 1,
        "height": SYNTH_H,
        "width": SYNTH_W,
        "dtype": "uint16",
        "pixel_size_um": 0.325,
        "z_step_um": 1.0,
        "voxel_size_um": (1.0, 0.325, 0.325),
        "channel_names": ["DAPI", "Foci"],
        "frame_timestamps_s": [float(t) for t in range(SYNTH_T)],
        "stage_xy_um": [(0.0, 0.0)],
        "stage_z_um": [],
        "loops": [],
    }


def _populate_choices(specs, channel_names: List[str]) -> Dict[str, Any]:
    """Build a params dict from ParamSpec defaults, injecting channel names."""
    params: Dict[str, Any] = {}
    for spec in specs:
        default = spec.default
        # Many pipelines populate `channel_name` "choice" params live in
        # the GUI. The registry returns an empty `choices=[]` until then;
        # inject the synthetic channel names so calls succeed headless.
        if spec.name == "channel_name":
            default = channel_names[-1] if channel_names else default
        elif (spec.param_type == "choice" and (default is None or default == "")
              and spec.choices):
            default = spec.choices[0]
        params[spec.name] = default
    return params


def _ensure_registry_loaded() -> None:
    """Import every pipeline module so AnalysisPipeline._registry is populated."""
    for mod in (
        "nd2studios.backend.analysis.tear_detection",
        "nd2studios.backend.analysis.histogram_threshold_pipeline",
        "nd2studios.backend.analysis.spots_pipeline",
        "nd2studios.backend.analysis.manual_mask",
    ):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 — optional / heavy deps may be missing
            pass
    # Nuclei Segmentation requires Cellpose; we *try* but accept failure.
    try:
        __import__("nd2studios.backend.analysis.nuclei_segmentation")
    except Exception:
        pass


def run() -> List[Dict[str, Any]]:
    """Time every registered AnalysisPipeline on the synthetic channels."""
    _ensure_registry_loaded()

    from nd2studios.core.analysis_registry import AnalysisPipeline

    results: List[Dict[str, Any]] = []
    channels = _build_synth_channels()
    metadata = _build_metadata()
    channel_names = list(channels.keys())

    for pipeline_cls in AnalysisPipeline.get_pipelines():
        name = pipeline_cls.name
        slug = name.lower().replace(" ", "_").replace("/", "_")
        try:
            pipeline = pipeline_cls()
            specs = pipeline.get_params()
            params = _populate_choices(specs, channel_names)

            with measure(f"analysis_{slug}") as m:
                pipeline.run(
                    channels=channels,
                    metadata=metadata,
                    params=params,
                    progress_cb=None,
                    cancelled_cb=lambda: False,
                )
            m.extra["pipeline"] = name
            m.extra["input_shape"] = list(next(iter(channels.values())).shape)
            results.append(m.to_dict())
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": f"analysis_{slug}",
                "pipeline": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
