"""Measure cold-load latency: open a file and fetch its first frame.

Drives ND2Studios's real volume readers
(:class:`nd2studios.backend.nd2_volume.LazyND2Volume`,
:class:`nd2studios.backend.tiff_loader.LazyMultiFileTIFFVolume`) — no
mocks. For each :class:`~profiling.harness.fixtures.TestFile`, we time
two things:

1. constructing the lazy volume (metadata probe + handle setup), and
2. ``get_frame(c=0, m=0, t=0, z=0)`` — the first decoded plane.

Files that don't exist on disk are silently skipped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from nd2studios.utils.profiling import measure

from profiling.harness.fixtures import TestFile, available_files


def _open_volume(tf: TestFile):
    """Open ``tf`` with the right backend for its extension.

    Returns the lazy volume object so the caller can close it after
    measuring the first-frame fetch.
    """
    ext = tf.path.suffix.lower()
    if ext == ".nd2":
        from nd2studios.backend.nd2_volume import LazyND2Volume
        return LazyND2Volume(str(tf.path))
    if ext in (".tif", ".tiff"):
        from nd2studios.backend.tiff_loader import LazyMultiFileTIFFVolume
        return LazyMultiFileTIFFVolume([str(tf.path)], chain_axis="Z")
    raise ValueError(f"Unsupported extension for {tf.path!s}: {ext!r}")


def run() -> List[Dict[str, Any]]:
    """Open each available test file, record open + first-frame timings."""
    results: List[Dict[str, Any]] = []

    files = available_files()
    if not files:
        return results

    for tf in files:
        try:
            with measure(f"open_{tf.label}") as m_open:
                volume = _open_volume(tf)
            m_open.extra.update({
                "filepath": str(tf.path),
                "n_timepoints": getattr(volume, "n_timepoints", None),
                "n_zslices": getattr(volume, "n_zslices", None),
                "n_channels": getattr(volume, "n_channels", None),
                "n_multipoints": getattr(volume, "n_multipoints", None),
            })
            results.append(m_open.to_dict())

            with measure(f"first_frame_{tf.label}") as m_first:
                frame = volume.get_frame(c=0, m=0, t=0, z=0, z_mode="none")
            m_first.extra.update({
                "frame_shape": list(getattr(frame, "shape", ())),
                "dtype": str(getattr(frame, "dtype", "")),
            })
            results.append(m_first.to_dict())
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            results.append({
                "name": f"open_{tf.label}",
                "error": f"{type(exc).__name__}: {exc}",
                "filepath": str(tf.path),
            })
        finally:
            close = getattr(locals().get("volume"), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
