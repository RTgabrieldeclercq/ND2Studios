"""Profiling test fixtures.

The harness looks at ``$PROFILING_TEST_DATA_DIR`` (falling back to
``~/profiling_data``) for representative ND2 / TIFF files. Each
:class:`TestFile` records expected dimensions so scenarios know how far
to scrub without hard-coding magic numbers.

Edit the file *paths* below to match the user's data, but keep the
labels stable — they end up as keys in the baseline JSON and as
columns when diffing snapshots across phases.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class TestFile:
    """One on-disk file to profile against."""

    path: Path
    label: str
    n_t: int = 1
    n_m: int = 1
    n_z: int = 1
    n_c: int = 1


# Where the harness looks for files. Override per-machine with
# `PROFILING_TEST_DATA_DIR=/some/folder python -m profiling.harness.run_all`.
DATA_DIR: Path = Path(
    os.environ.get("PROFILING_TEST_DATA_DIR", str(Path.home() / "profiling_data"))
)

# Files the harness will try to profile. Missing files are skipped
# cleanly so the harness still runs end-to-end on a clean checkout.
# Customize the dimensions to match real lab files when you wire them
# up — scenarios cap their inner loops by these numbers.
SMALL_ND2 = TestFile(
    path=DATA_DIR / "small.nd2",
    label="small_nd2",
    n_t=50, n_m=1, n_z=10, n_c=2,
)

LARGE_ND2 = TestFile(
    path=DATA_DIR / "large.nd2",
    label="large_nd2",
    n_t=500, n_m=4, n_z=30, n_c=3,
)

SMALL_TIFF = TestFile(
    path=DATA_DIR / "small.tif",
    label="small_tiff",
    n_t=20, n_m=1, n_z=1, n_c=1,
)

LARGE_TIFF = TestFile(
    path=DATA_DIR / "large.tif",
    label="large_tiff",
    n_t=200, n_m=1, n_z=1, n_c=1,
)


# Per-axis scrub parameters.
SCRUB_FRAMES: int = 100         # how many slider positions to visit
SCRUB_INTERVAL_MS: int = 16     # ~60 Hz simulated slider event spacing

# Tab-switch parameters.
TAB_SWITCH_COUNT: int = 12      # number of consecutive page transitions

# Analysis-scenario synthetic dimensions. Kept tiny so every registered
# pipeline can run inside a few seconds without lab data on disk.
SYNTH_T: int = 32
SYNTH_H: int = 256
SYNTH_W: int = 256


def all_files() -> List[TestFile]:
    """Every TestFile the harness knows about, in iteration order."""
    return [SMALL_ND2, LARGE_ND2, SMALL_TIFF, LARGE_TIFF]


def available_files() -> List[TestFile]:
    """Subset of :func:`all_files` whose paths exist on disk."""
    return [tf for tf in all_files() if tf.path.exists()]
