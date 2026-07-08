"""
DVC Method Registry for ND2Studios.

Digital Volume/Image Correlation (DVC/DIC) methods measure a dense
displacement field — and the strain tensor derived from it — between a
*reference* volume and a *deformed* volume. This is distinct from both
enhancement plugins (``(T,H,W) -> (T,H,W)`` recipe steps) and analysis
pipelines (``AnalysisResult`` = label masks + per-object measurements):
DVC consumes full ``(Z,H,W)`` volumes (or ``(H,W)`` images in the 2D
DIC case) and produces a dense vector/tensor field that neither of the
other two result types can represent.

So DVC gets its own registry, modeled byte-for-byte on
:class:`~nd2studios.core.analysis_registry.AnalysisPipeline`'s idiom and
reusing :class:`~nd2studios.core.plugin_registry.ParamSpec`, with its own
:class:`DVCResult` container.

Usage::

    @DVCMethod.register
    class ALDVCMethod(DVCMethod):
        name = "ALDVC"
        description = "Augmented Lagrangian Digital Volume Correlation"
        def get_params(self): ...
        def run(self, ref_vol, def_vol, voxel_size_um, params,
                progress_cb, cancelled_cb): ...

This module imports only the standard library + numpy + ``ParamSpec``; it
deliberately has **no PySide6 dependency** so the backend engine can call
it (and run headless) without pulling in Qt.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from nd2studios.core.plugin_registry import ParamSpec  # reuse, do not duplicate


@dataclass
class DVCResult:
    """Structured output from a :meth:`DVCMethod.run` call.

    Shapes (``d`` = 2 for 2D DIC, 3 for 3D DVC):

    - ``grid_coords``: ``(*grid, d)`` subset-center coordinates **in voxels**,
      where ``grid`` is ``(Gy, Gx)`` (2D) or ``(Gz, Gy, Gx)`` (3D).
    - ``displacement_field``: ``(*grid, d)`` displacement **in voxels**, axis
      order matching ``grid_coords`` (i.e. ``[..., 0]`` is the slowest spatial
      axis: y in 2D, z in 3D).
    - ``strain_field``: ``(*grid, n_components)`` or ``None`` until computed.

    ``voxel_size_um`` is ``(y, x)`` (2D) or ``(z, y, x)`` (3D); use
    :meth:`displacement_um` to convert the displacement field to micrometers.
    """
    dim: int
    grid_coords: np.ndarray
    displacement_field: np.ndarray
    voxel_size_um: Tuple[float, ...] = ()

    strain_field: Optional[np.ndarray] = None
    strain_type: str = ""

    qfactor: Optional[np.ndarray] = None        # (*grid,) correlation confidence
    converged: bool = False
    iterations: int = 0
    mu: float = 0.0
    beta: float = 0.0

    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── convenience accessors ──
    @property
    def magnitude(self) -> np.ndarray:
        """``(*grid,)`` displacement magnitude in voxels."""
        return np.sqrt(np.sum(np.square(self.displacement_field), axis=-1))

    def displacement_um(self) -> np.ndarray:
        """Displacement field converted to micrometers (per-axis scaling)."""
        if not self.voxel_size_um or len(self.voxel_size_um) != self.dim:
            return self.displacement_field
        scale = np.asarray(self.voxel_size_um, dtype=np.float64)
        return self.displacement_field * scale

    def magnitude_um(self) -> np.ndarray:
        return np.sqrt(np.sum(np.square(self.displacement_um()), axis=-1))


@dataclass
class DVCParams:
    """Convenience typed view over the param dict a DVCMethod receives.

    Kept thin (and optional) on purpose — methods may continue to read the
    raw ``params`` dict produced by ``ParamEditor.get_values()``. This exists
    so callers building params programmatically (headless scripts, batch) get
    sane defaults and IDE help.
    """
    subset_size: int = 16
    subset_spacing: int = 10
    correlation: str = "zncc"          # "zncc" | "phase"
    tracking_mode: str = "cumulative"  # "cumulative" | "incremental"
    strain_type: str = "infinitesimal"  # "infinitesimal" | "green-lagrange"
    search_radius: int = 0             # 0 → auto (= subset_size)
    seed_levels: int = 3               # multigrid FFT-seed pyramid levels (1 = single)
    admm_iterations: int = 4
    mu: float = 1e-3
    strain_smooth: float = 0.0         # Gaussian σ on û before ∂u/∂x (0 = off)
    n_workers: int = 0                 # IC-GN process-pool workers (0 = auto)
    newFFTSearch: bool = False         # re-seed every frame vs cross-frame warm-start
    use_gpu: bool = False              # route the FFT seed through CuPy if present

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DVCParams":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class DVCMethod(ABC):
    """Base class for all DVC/DIC methods.

    Subclasses declare ``name`` and ``description`` and decorate with
    ``@DVCMethod.register`` so the DVC pipeline node (and any headless caller)
    can discover them and read their :meth:`get_params`.
    """
    name: str = "Unnamed"
    description: str = ""

    _registry: Dict[str, Type["DVCMethod"]] = {}

    @classmethod
    def register(cls, method_cls: Type["DVCMethod"]) -> Type["DVCMethod"]:
        DVCMethod._registry[method_cls.name] = method_cls
        return method_cls

    @classmethod
    def get_methods(cls) -> List[Type["DVCMethod"]]:
        return list(DVCMethod._registry.values())

    @classmethod
    def get_method(cls, name: str) -> Optional[Type["DVCMethod"]]:
        return DVCMethod._registry.get(name)

    @abstractmethod
    def get_params(self) -> List[ParamSpec]:
        """Return a fresh list of ParamSpec objects describing this method's
        tunable parameters (called on each activation so the page can rebuild
        the form)."""
        ...

    @abstractmethod
    def run(
        self,
        ref_vol: np.ndarray,
        def_vol: np.ndarray,
        voxel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> DVCResult:
        """Correlate ``def_vol`` against ``ref_vol`` and return a DVCResult.

        Args:
            ref_vol:  reference image/volume — ``(H,W)`` (2D) or ``(Z,H,W)`` (3D).
            def_vol:  deformed image/volume, same shape as ``ref_vol``.
            voxel_size_um: physical voxel size, ``(y,x)`` or ``(z,y,x)``.
            params:   ``{param_name: value}`` from ParamEditor.
            progress_cb:  optional callable accepting an int 0-100.
            cancelled_cb: optional zero-arg callable returning True to abort.
        """
        ...
