"""Vendored headless subset of CellTracker's analysis backend.

CellTracker (McGhee Lab) is a sibling fluorescence-microscopy analysis app that
shares ND2Studios' application DNA. ND2Studios already overlaps it on
segmentation, basic per-cell measurement, and SerialTrack tracking, so only the
parts CellTracker *uniquely* contributes to cell-tracking pipelines are copied
here:

* :mod:`~nd2studios.backend.celltracker.tracking` — the topology-Hungarian and
  spatial-fingerprint linkers (``track_timeseries`` / ``track_fingerprint`` and
  their helpers). CellTracker's ``track_serialtrack`` is **not** vendored: it
  carries a hardcoded developer path and is already covered by
  :mod:`nd2studios.backend.serialtrack`.
* :mod:`~nd2studios.backend.celltracker.metrics` — per-cell spatial metrics
  (neighbor distance, local divergence/curl) and self-fold-change.
* :mod:`~nd2studios.backend.celltracker.fields` — Eulerian gridded field maps
  (density / velocity / divergence / curl / intensity / fold-change).

These functions are pure numpy / scipy / pandas (Qt-free), so the backend-purity
rule (no Qt imports under ``nd2studios/backend/``) holds. They operate on pandas
DataFrames with CellTracker's column convention (``frame``, ``label``,
``centroid_y``, ``centroid_x``, ``area``, ``track_id``); the row-dict bridge in
:mod:`nd2studios.backend.celltracker_bridge` adapts ND2Studios' measurement rows
to and from that shape.

Source: c:/Users/gabri/Documents/GitHub/Cell-Tracker/CellTracker/backend
(see that repo's CELLTRACKER_REFERENCE.md for the full API guide).
"""
from __future__ import annotations
