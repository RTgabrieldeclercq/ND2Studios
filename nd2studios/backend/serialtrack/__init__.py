"""Vendored headless subset of SerialTrack (Yang et al., SoftwareX 19 (2022) 101204).

SerialTrack — *ScalE and Rotation Invariant Augmented Lagrangian Particle
Tracking* — is a scale/rotation-invariant topology PTV linker. ND2Studios uses
it as an alternative linker behind the "Track Objects" pipeline node (the
``METHOD_SERIALTRACK`` path in :mod:`nd2studios.backend.object_tracker`).

Only the **headless** modules needed for the ``track_coordinates`` linking path
are vendored here — the upstream ``io.py`` / ``results.py`` / ``run.py`` (which
pull in HDF5/TIFF I/O and the Qt GUI bridge) are deliberately omitted, so the
backend-purity rule (no Qt imports under ``nd2studios/backend/``) holds.

Upstream is a PEP-420 namespace package with no ``__init__.py``; this vendored
copy adds one so it imports as a regular package
(``nd2studios.backend.serialtrack.tracking``).

Source: c:/Users/gabri/Documents/GitHub/SerialTrack_Python/serialtrack
(see that repo's SERIALTRACK_REFERENCE.md for the full API guide). Only numpy,
scipy, and numba are required to import the linking path; scikit-image and
scikit-learn are imported lazily inside code paths the linker never hits.
"""
from __future__ import annotations
