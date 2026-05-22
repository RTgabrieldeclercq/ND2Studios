"""
Storage helpers for V1.38 Phase 6 stage artifacts.

The only on-disk format choice the rest of the package makes is
"label stack" (chunked ``(T, H, W)`` int32). Everything else is JSON.
This module wraps the label-stack write/read behind a tiny API so the
Zarr-or-NPZ fallback decision lives in one place.

Zarr is treated as *optional*. When the import fails the fallback is
NPZ (``np.savez``), which is already used by
:func:`ND2StudiosManager.save_session` so the dependency surface does
not grow when zarr is absent. NPZ reads are eager (no partial-load
support); Zarr reads expose a tiny lazy proxy via
:func:`open_label_stack` so the Results page can stream T-slices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ── Zarr availability check ──────────────────────────────────────────
try:
    import zarr  # type: ignore

    HAS_ZARR = True
except Exception:  # noqa: BLE001 — any zarr import failure → NPZ fallback
    zarr = None  # type: ignore
    HAS_ZARR = False


# Backing-format suffixes. The relative path returned by
# :func:`write_label_stack` always ends in one of these so the manifest
# is self-describing.
_ZARR_SUFFIX = ".zarr"
_NPZ_SUFFIX = ".npz"


def _zarr_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Pick chunk dims that make single-T reads cheap.

    The viewer / Results page reads label masks one T at a time; a
    chunk of ``(1, H, W)`` makes that a single block read regardless of
    T length.
    """
    if len(shape) == 3:
        return (1, shape[1], shape[2])
    if len(shape) == 2:
        return shape  # already a single plane
    return shape


def write_label_stack(base_path: Path, labels: np.ndarray) -> str:
    """Write a label stack and return its actual relative path.

    Parameters
    ----------
    base_path :
        Path *without* an extension. The function appends ``.zarr`` or
        ``.npz`` depending on backend availability.
    labels :
        ``(T, H, W)`` int32 (or any int dtype) array. Cast to int32 on
        write to match :class:`AnalysisResult.label_masks` semantics.

    Returns
    -------
    str
        The basename + actual suffix that was written. The caller is
        expected to store this in :class:`StageRecord.artifacts` so the
        rehydrate path knows which backend to use.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(labels, dtype=np.int32)

    if HAS_ZARR:
        target = base_path.with_suffix(_ZARR_SUFFIX)
        # Re-create if exists — overwrite is the intended commit semantic.
        store = zarr.open(  # type: ignore[union-attr]
            str(target),
            mode="w",
            shape=arr.shape,
            chunks=_zarr_chunks(arr.shape),
            dtype=arr.dtype,
        )
        store[...] = arr
        return target.name

    target = base_path.with_suffix(_NPZ_SUFFIX)
    np.savez(target, labels=arr)
    return target.name


def read_label_stack(path: Path) -> np.ndarray:
    """Read a label stack written by :func:`write_label_stack`.

    Returns a regular numpy array regardless of backend. NPZ files are
    materialized; Zarr files are read into RAM here — callers that need
    lazy access should call :func:`open_label_stack` instead.
    """
    suffix = path.suffix.lower()
    if suffix == _ZARR_SUFFIX:
        if not HAS_ZARR:
            raise RuntimeError(
                f"Cannot read {path}: it was written as Zarr but the "
                "`zarr` package is not installed in this environment."
            )
        return np.asarray(zarr.open(str(path), mode="r")[...])  # type: ignore[union-attr]
    if suffix == _NPZ_SUFFIX:
        with np.load(path) as data:
            return np.asarray(data["labels"])
    raise ValueError(f"Unknown label-stack format: {path}")


def open_label_stack(path: Path):
    """Open a label stack lazily when possible.

    Returns a zarr ``Array`` for Zarr-backed stores (supports ``[t]``
    indexing without loading the whole stack) and a materialized numpy
    array for NPZ.
    """
    suffix = path.suffix.lower()
    if suffix == _ZARR_SUFFIX and HAS_ZARR:
        return zarr.open(str(path), mode="r")  # type: ignore[union-attr]
    return read_label_stack(path)
