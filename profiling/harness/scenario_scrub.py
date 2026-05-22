"""Measure per-frame latency when scrubbing T (and separately Z and M).

We do not actually animate a Qt slider; we call the same
``volume.get_frame(...)`` code path the viewer triggers in response to
slider events, one position at a time, and record the wall time for
each call. The resulting distribution determines how smooth slider
scrubbing feels at the GUI layer.

For each axis we report mean / p50 / p99 frame durations and the
corresponding FPS figures via
:func:`nd2studios.utils.profiling.fps_from_durations`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from nd2studios.utils.profiling import fps_from_durations, measure

from profiling.harness.fixtures import (
    SCRUB_FRAMES, SCRUB_INTERVAL_MS, TestFile, available_files,
)


def _open_volume(tf: TestFile):
    ext = tf.path.suffix.lower()
    if ext == ".nd2":
        from nd2studios.backend.nd2_volume import LazyND2Volume
        return LazyND2Volume(str(tf.path))
    if ext in (".tif", ".tiff"):
        from nd2studios.backend.tiff_loader import LazyMultiFileTIFFVolume
        return LazyMultiFileTIFFVolume([str(tf.path)], chain_axis="Z")
    raise ValueError(f"Unsupported extension: {ext!r}")


def _scrub_axis(volume, axis: str, n: int) -> List[float]:
    """Visit positions ``0..n-1`` along ``axis`` and return per-frame ms."""
    durations: List[float] = []
    for i in range(n):
        kwargs: Dict[str, Any] = {"c": 0, "m": 0, "t": 0, "z": 0, "z_mode": "none"}
        kwargs[axis] = i
        t0 = time.perf_counter()
        volume.get_frame(**kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        durations.append(dt_ms)

        # Pace the inner loop so we don't sample faster than the slider
        # would ever fire. Skip the sleep if reading already exceeded
        # the slider interval (the disk is slower than the slider).
        slack = (SCRUB_INTERVAL_MS - dt_ms) / 1000.0
        if slack > 0:
            time.sleep(slack)
    return durations


def run() -> List[Dict[str, Any]]:
    """Scrub T, Z, and M (if present) for every available test file."""
    results: List[Dict[str, Any]] = []
    files = available_files()
    if not files:
        return results

    for tf in files:
        try:
            volume = _open_volume(tf)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": f"scrub_open_{tf.label}",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        try:
            # Axes the volume actually has at least 2 positions on.
            axis_sizes = {
                "t": int(getattr(volume, "n_timepoints", 1) or 1),
                "z": int(getattr(volume, "n_zslices", 1) or 1),
                "m": int(getattr(volume, "n_multipoints", 1) or 1),
            }
            for axis, n_available in axis_sizes.items():
                n = min(SCRUB_FRAMES, n_available)
                if n <= 1:
                    continue
                with measure(f"scrub_{axis}_{tf.label}", track_pyalloc=False) as m:
                    durations = _scrub_axis(volume, axis, n)
                m.extra.update(fps_from_durations(durations))
                m.extra["axis"] = axis
                m.extra["filepath"] = str(tf.path)
                results.append(m.to_dict())
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": f"scrub_{tf.label}",
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            close = getattr(volume, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
