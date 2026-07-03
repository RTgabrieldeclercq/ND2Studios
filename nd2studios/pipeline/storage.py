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
# V1.46 — incremental (per-frame) streaming sink uses a memmapped .npy when
# zarr is unavailable, since NPZ (np.savez) can only be written all-at-once.
_NPY_SUFFIX = ".npy"


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

    Returns a regular numpy array regardless of backend. NPZ / NPY files are
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
    if suffix == _NPY_SUFFIX:
        return np.asarray(np.load(path))
    raise ValueError(f"Unknown label-stack format: {path}")


def open_label_stack(path: Path):
    """Open a label stack lazily when possible.

    Returns a zarr ``Array`` for Zarr-backed stores (supports ``[t]``
    indexing without loading the whole stack), a memory-mapped numpy array
    for ``.npy`` (also supports lazy ``[t]`` reads), and a materialized numpy
    array for NPZ.
    """
    suffix = path.suffix.lower()
    if suffix == _ZARR_SUFFIX and HAS_ZARR:
        return zarr.open(str(path), mode="r")  # type: ignore[union-attr]
    if suffix == _NPY_SUFFIX:
        return np.load(path, mmap_mode="r")
    return read_label_stack(path)


class LabelStackWriter:
    """Incremental, per-frame label-stack writer (V1.46 streaming sink).

    Unlike :func:`write_label_stack` (which needs the whole ``(T, H, W)`` array
    in RAM), this opens a fixed-shape store up front and accepts one ``(H, W)``
    frame at a time via :meth:`write_frame`, so an analysis pipeline can spill
    masks to disk as it processes without ever holding the full stack. This is
    the Fiji/NIS-Elements "streaming sink" used on memory-constrained machines.

    Backend: a Zarr array with ``(1, H, W)`` chunks when available (per-frame
    writes touch a single chunk), else a memory-mapped ``.npy`` (``np.memmap``
    supports random-access ``mm[t] = frame`` writes without buffering the rest).
    Writes may arrive out of T-order — both backends are random-access.

    Usage::

        with LabelStackWriter(base, (T, H, W)) as w:
            for t in ...:
                w.write_frame(t, label_frame)
        reader = w.reader          # lazy [t]/.shape/.ndim reader after close
        rel = w.relative_name      # basename+suffix for the manifest
    """

    def __init__(self, base_path: Path, shape: Tuple[int, ...],
                 dtype=np.int32) -> None:
        self._base = Path(base_path)
        self._base.parent.mkdir(parents=True, exist_ok=True)
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)
        self._closed = False
        self.reader = None  # populated on close()

        if HAS_ZARR:
            self._target = self._base.with_suffix(_ZARR_SUFFIX)
            self._store = zarr.open(  # type: ignore[union-attr]
                str(self._target),
                mode="w",
                shape=self._shape,
                chunks=_zarr_chunks(self._shape),
                dtype=self._dtype,
            )
            self._memmap = None
        else:
            self._target = self._base.with_suffix(_NPY_SUFFIX)
            self._memmap = np.lib.format.open_memmap(
                str(self._target), mode="w+",
                dtype=self._dtype, shape=self._shape,
            )
            self._store = None

    @property
    def relative_name(self) -> str:
        return self._target.name

    @property
    def path(self) -> Path:
        return self._target

    def write_frame(self, t: int, frame: np.ndarray) -> None:
        """Write one ``(H, W)`` plane at index ``t`` (random-access)."""
        f = np.ascontiguousarray(frame, dtype=self._dtype)
        if self._store is not None:
            self._store[int(t)] = f
        else:
            self._memmap[int(t)] = f

    def close(self):
        """Flush and return a lazy reader (zarr Array or mmap numpy)."""
        if self._closed:
            return self.reader
        self._closed = True
        if self._memmap is not None:
            self._memmap.flush()
            del self._memmap
            self._memmap = None
        self.reader = open_label_stack(self._target)
        return self.reader

    def __enter__(self) -> "LabelStackWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
