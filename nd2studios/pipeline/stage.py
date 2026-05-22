"""
``PipelineStage`` ABC (V1.38 Phase 6).

A *stage* is the persistent representation of one node in the
ND2Studios workflow (recipe, analysis). Stages are pure Python — no
Qt imports — so they can be exercised from worker threads and from
headless test harnesses without an event loop.

Stages neither own threads nor schedule work. ``commit()`` runs on
the GUI thread after the appropriate worker has already produced the
data, and the I/O is synchronous (a JSON manifest plus, for
:class:`~nd2studios.pipeline.stages.analysis_stage.AnalysisStage`,
one label-stack write per M).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from nd2studios.pipeline.session import Session, StageRecord


class PipelineStage(ABC):
    """Abstract base for one stage of an ND2Studios session.

    Subclasses fill in:

    - :attr:`name` — manifest key. Subclasses with multiple flavours
      (``AnalysisStage`` per pipeline) override ``__init__`` to compose
      a unique name like ``"analysis:Histogram Threshold"``.
    - :meth:`commit` — write artifacts and stamp the manifest. Returns
      the :class:`StageRecord` that was just persisted so callers can
      log / display it.
    - :meth:`is_committed` — usually delegates to
      :meth:`Session.is_committed`.

    Release / rehydrate semantics belong to the *page* that owns the
    in-memory state (Recipe / Analysis), not the stage. The stage is
    just the persistent surface.
    """

    name: str = ""

    def __init__(self, session: Session):
        self._session = session

    @property
    def session(self) -> Session:
        return self._session

    @abstractmethod
    def commit(self) -> StageRecord:
        """Persist this stage's outputs and update the manifest."""

    def is_committed(self) -> bool:
        return self._session.is_committed(self.name)

    def record(self) -> StageRecord | None:
        return self._session.get_stage(self.name)
