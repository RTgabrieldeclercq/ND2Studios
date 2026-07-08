"""
Registration Method Registry for ND2Studios (V1.56).

A **registration method** estimates a spatial transform aligning a *moving*
image/series onto a *reference*, then resamples the moving data through it. This
is distinct from the three existing paradigms:

- ``EnhancementPlugin`` — per-channel ``(T,H,W) -> (T,H,W)`` recipe step (the
  ``"Registration (Drift Correction)"`` plugin lives there for the single-channel
  quick win).
- ``AnalysisPipeline`` — label masks + per-object measurements.
- ``DVCMethod`` — a dense displacement/strain **field**.

Registration returns a **parametric transform** (per-frame shift or affine) plus
the aligned data, and is driven by **one reference channel** whose transform is
applied to every channel — the "register once, apply to all" rule that preserves
colocalization. That cross-channel contract does not fit the per-channel
enhancement plugin, so registration gets its own registry, modeled byte-for-byte
on :class:`~nd2studios.core.dvc_registry.DVCMethod` and reusing
:class:`~nd2studios.core.plugin_registry.ParamSpec`.

Pure module — no PySide6 (backend-callable / headless).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from nd2studios.core.plugin_registry import ParamSpec  # reuse, do not duplicate


@dataclass
class RegistrationResult:
    """Structured output from a :meth:`RegistrationMethod.run` call.

    - ``model``: the transform model actually used (``translation`` / ``euclidean``
      / ``affine``).
    - ``shifts_px``: ``(T, 2)`` recovered per-frame translation in **pixels**,
      axis order ``(row, col)`` == ``(y, x)`` (the effective shift onto the anchor;
      for affine models this is the translation part of the warp).
    - ``transforms``: ``(T, 2, 3)`` effective affine warp matrices, or ``None`` for
      the pure-translation model. Each maps the anchor's coordinates → that frame's
      sample locations (ECC convention), so it is applied with ``WARP_INVERSE_MAP``.
    - ``aligned``: the resampled **reference-channel** series (same shape/dtype as
      the input), for the before/after preview.
    - ``confidence``: ``(T,)`` per-frame NCC / ECC correlation for QC.
    - ``pixel_size_um``: ``(y, x)`` so shifts can be reported in micrometers.
    - ``diagnostics``: carries the raw transform bundle (``"transforms"``) so the
      caller can apply it to the *other* channels via ``estimate.apply_series``.
    """
    model: str = "translation"
    shifts_px: Optional[np.ndarray] = None
    transforms: Optional[np.ndarray] = None
    aligned: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    pixel_size_um: Tuple[float, ...] = ()
    reference_mode: str = ""
    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def shifts_um(self) -> Optional[np.ndarray]:
        """Per-frame shift in micrometers, or ``shifts_px`` if no pixel size."""
        if self.shifts_px is None or not self.pixel_size_um:
            return self.shifts_px
        scale = np.asarray(self.pixel_size_um, dtype=np.float64)
        # pixel_size_um is (y, x); shifts_px columns are (row=y, col=x).
        return np.asarray(self.shifts_px, dtype=np.float64) * scale[:2]

    @property
    def magnitude_px(self) -> Optional[np.ndarray]:
        """``(T,)`` per-frame shift magnitude in pixels."""
        if self.shifts_px is None:
            return None
        return np.sqrt(np.sum(np.square(np.asarray(self.shifts_px)), axis=-1))


class RegistrationMethod(ABC):
    """Base class for all registration methods.

    Subclasses declare ``name`` and ``description`` and decorate with
    ``@RegistrationMethod.register`` so the registration node (and any headless
    caller) can discover them and read their :meth:`get_params`.
    """
    name: str = "Unnamed"
    description: str = ""

    _registry: Dict[str, Type["RegistrationMethod"]] = {}

    @classmethod
    def register(cls, method_cls: Type["RegistrationMethod"]) -> Type["RegistrationMethod"]:
        RegistrationMethod._registry[method_cls.name] = method_cls
        return method_cls

    @classmethod
    def get_methods(cls) -> List[Type["RegistrationMethod"]]:
        return list(RegistrationMethod._registry.values())

    @classmethod
    def get_method(cls, name: str) -> Optional[Type["RegistrationMethod"]]:
        return RegistrationMethod._registry.get(name)

    @abstractmethod
    def get_params(self) -> List[ParamSpec]:
        """Return a fresh list of ParamSpec objects (called on each activation so
        the node popup can rebuild the form)."""
        ...

    @abstractmethod
    def run(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
        pixel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> RegistrationResult:
        """Estimate the transform aligning ``moving`` onto ``reference`` and
        resample.

        For temporal drift correction ``reference`` and ``moving`` are the **same**
        ``(T,H,W)`` reference-channel series; the ``reference`` *param* (first /
        previous / mean) selects the anchor within it. ``pixel_size_um`` is
        ``(y, x)``. ``progress_cb`` takes an int 0-100; ``cancelled_cb`` returns
        True to abort.
        """
        ...
