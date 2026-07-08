"""
``nd2studios.backend.dvc`` — pure (Qt-free) Digital Volume/Image
Correlation engine.

This is a clean-room Python port of FranckLab's Augmented Lagrangian
Digital Volume Correlation (ALDVC, Yang/Hazlett/Landauer/Franck 2020).
See ``Research/aldvc_literature_review.md`` and
``CodeLog/ClaudesPlan/V1.45_DVC.md`` for the algorithm decomposition and
the phased implementation plan.

Backend-purity rule (CLAUDE.md): nothing under this package may import
PySide6. Functions take numpy arrays in and return numpy arrays /
``DVCResult`` out, with an optional ``progress_cb`` / ``cancelled_cb`` so
they are equally callable from the :class:`DVCWorker` QThread and from
headless scripts.

Module map (built out phase by phase — see the plan):

- ``engine``        — top-level ``run_aldvc`` orchestration (Stage 0 + glue).
- ``integer_search``— Stage 1: FFT cross-correlation integer seed.   (Phase 2)
- ``mesh``          — Stage 2: regular hex/quad mesh + DOF pack/unpack.(Phase 1)
- ``outliers``      — Stage 2: median test + cc-threshold + inpaint.  (Phase 1)
- ``icgn``          — Stages 3 & 5: inverse-compositional Gauss-Newton.(Phase 3)
- ``global_step``   — Stage 4: FD operator + L-curve beta selection.   (Phase 4)
- ``admm``          — Stage 6: ADMM outer loop + dual updates.         (Phase 4)
- ``strain``        — Stage 7: F-field -> strain tensor + units.       (Phase 5)
- ``parallel``      — shared-memory process-pool dispatch helper.      (Phase 2)
"""
from __future__ import annotations

from nd2studios.backend.dvc.engine import run_aldvc

__all__ = ["run_aldvc"]
