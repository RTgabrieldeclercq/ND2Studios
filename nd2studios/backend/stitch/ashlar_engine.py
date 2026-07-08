"""
ASHLAR overlap-registration engine (in-memory, no BioFormats/JVM).

ASHLAR's *registration* (``EdgeAligner``) is pure Python/NumPy/SciPy/networkx —
only its *readers* use Java/BioFormats. So we feed ASHLAR an in-memory
``Metadata`` + ``Reader`` built from our own tiles and stage-coordinate seed, run
``EdgeAligner``, and read back ``aligner.positions`` (absolute top-left pixel
positions). The JVM never starts.

The one catch: ``ashlar.reg`` does an unconditional ``import jnius`` at module
load, and jnius needs a JDK (``JAVA_HOME``). We satisfy that by pointing
``JAVA_HOME`` at the pip-installed ``jdk4py`` (packaged OpenJDK) when the user
hasn't set one — no system install or PATH change required.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Dict, Tuple

import numpy as np

from nd2studios.backend.stitch.config import StitchConfig
from nd2studios.backend.stitch.dataset import Dataset

Positions = Dict[int, Tuple[float, float]]

_JDK_HINT = (
    "The 'ashlar' engine needs a Java JDK: ashlar.reg imports jnius (BioFormats) "
    "at load time. Install the packaged JDK with `pip install jdk4py` (auto-"
    "detected) or set JAVA_HOME to a JDK. The 'm2stitch' and 'phase_correlation' "
    "engines work without Java."
)


def _ensure_java_home() -> None:
    """Point JAVA_HOME at jdk4py's bundled OpenJDK if the user set none."""
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isdir(jh):
        return
    try:
        import jdk4py
        os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    except Exception:
        pass


def ashlar_available() -> bool:
    """True only if ashlar.reg can actually be imported (JDK resolvable)."""
    if importlib.util.find_spec("ashlar") is None:
        return False
    _ensure_java_home()
    try:
        import ashlar.reg  # noqa: F401
        return True
    except Exception:
        return False


def ashlar_positions(dataset: Dataset,
                     ref_frames: Dict[int, np.ndarray],
                     seed: Positions,
                     config: StitchConfig) -> Tuple[Positions, Dict, Dict]:
    """Register the reference tiles with ASHLAR's EdgeAligner.

    Returns ``(positions, confidences, info)``; positions are (row, col) px.
    """
    if importlib.util.find_spec("ashlar") is None:
        raise ImportError("ashlar is not installed (pip install ashlar).")
    _ensure_java_home()
    try:
        import ashlar.reg as R
    except Exception as exc:  # pragma: no cover - depends on JDK presence
        raise ImportError(f"{_JDK_HINT} ({exc})")

    order = [t.index for t in dataset.tiles if t.index in ref_frames]
    if len(order) < 2:
        raise RuntimeError("ashlar needs ≥2 tiles with reference frames")

    px = float(dataset.pixel_size_um)
    pos_dtype = np.asarray(ref_frames[order[0]]).dtype
    # Seed positions in pixels, (y, x) — ashlar uses pixel units internally
    # (overlap = tile_size − |Δposition|, both in pixels).
    seed_arr = np.array([[seed[m][0], seed[m][1]] for m in order], dtype=np.float64)
    tile_hw = np.array([dataset.tile_h, dataset.tile_w], dtype=np.float64)

    class _Meta(R.Metadata):
        @property
        def _num_images(self):
            return len(order)

        @property
        def num_channels(self):
            return 1

        @property
        def pixel_size(self):
            return px

        @property
        def pixel_dtype(self):
            return np.dtype(pos_dtype)

        def tile_position(self, i):
            return seed_arr[i]

        def tile_size(self, i):
            return tile_hw

    class _Reader(R.Reader):
        def __init__(self, meta):
            self.metadata = meta

        def read(self, series, c):
            return np.asarray(ref_frames[order[series]])

    reader = _Reader(_Meta())

    # ashlar's max_shift is in µm.
    if config.max_shift_um is not None:
        max_shift_um = float(config.max_shift_um)
    else:
        ov = max(dataset.overlap_frac, 0.0)
        max_shift_px = max(8.0, 0.5 * ov * min(dataset.tile_h, dataset.tile_w))
        max_shift_um = max_shift_px * px

    aligner = R.EdgeAligner(
        reader, channel=0, max_shift=max_shift_um,
        filter_sigma=float(config.filter_sigma),
        do_make_thumbnail=False, verbose=False,
    )
    aligner.run()

    pos = np.asarray(aligner.positions, dtype=np.float64)  # (N, 2) y, x px
    positions: Positions = {order[i]: (float(pos[i, 0]), float(pos[i, 1]))
                            for i in range(len(order))}
    for t in dataset.tiles:  # tiles without a ref frame keep their seed
        positions.setdefault(t.index, seed[t.index])
    info = {"engine": "ashlar", "n_tiles": len(order),
            "max_shift_um": round(max_shift_um, 3)}
    return positions, {}, info
