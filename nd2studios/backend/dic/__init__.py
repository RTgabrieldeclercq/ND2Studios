"""2D Augmented-Lagrangian Digital Image Correlation (AL-DIC) backend.

Wraps the optional third-party package `al-dic <https://pypi.org/project/al-dic/>`_
(a.k.a. pyALDIC, by Zach Tong / Jin Yang — the same AL-DIC lineage as the app's
3D ALDVC engine) as a :class:`~nd2studios.core.dvc_registry.DVCMethod` so it plugs
into the existing DVC node infrastructure (``DVCResult`` container, ``DVCPanel``
viewer, off-thread job runner).

The 2D DIC node returns a **2D** :class:`DVCResult` (``dim == 2``), which the DVC
panel already renders. ``al-dic`` is an *optional, lazily-imported* dependency
(``pip install al-dic``): it is never in ``requirements.txt`` and is gated behind
``importlib.util.find_spec`` with a friendly :class:`ImportError` — mirroring the
stardist / cellpose convention in CLAUDE.md.

Pure backend package — no PySide6 imports (callable from workers + headless).
"""
from __future__ import annotations
