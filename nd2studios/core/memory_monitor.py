"""Runtime memory-pressure monitor (V1.41).

A small ``QObject`` singleton that polls ``psutil.virtual_memory().percent``
on a ``QTimer`` and emits banded signals that other subsystems wire into:

* :attr:`MemoryMonitor.warning`   — 80% — purely advisory.
* :attr:`MemoryMonitor.critical`  — 90% — drop the pre-render cache.
* :attr:`MemoryMonitor.emergency` — 95% — modal warn + block new ops.
* :attr:`MemoryMonitor.recovered` — pressure has fallen back through
  the per-band recover threshold (75 / 85 / 90 respectively).

Hysteresis
----------

Each band has a separate "trigger" and "recover" threshold so the
monitor doesn't oscillate between two emits per second when psutil's
reading jitters across a hard boundary.  The recover threshold for
band N is the trigger of band N-1 (so once we are critical we have
to drop back under 85% to clear it, not just under 90%).

Thresholds are read from :class:`~nd2studios.core.settings.Settings`
so the diagnostics panel can adjust them per user.  Re-reading is
done on every tick — cheap enough and lets a slider in the dialog
take effect immediately without restart.
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Optional

import psutil
from PySide6.QtCore import QObject, QTimer, Signal

from nd2studios.core.settings import Settings

log = logging.getLogger(__name__)


class PressureBand(IntEnum):
    """Bands the monitor walks between.  Ordered low → high."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


class MemoryMonitor(QObject):
    """Banded memory-pressure monitor with hysteresis.

    Lifetime: created by :class:`MainWindow` at startup, lives for the
    entire application session.  Subscribers connect to the band
    signals and act according to their own policy — see
    :doc:`/CodeLog/Architecture/ARCHITECTURE` for the current wiring.
    """

    warning = Signal(float)
    critical = Signal(float)
    emergency = Signal(float)
    recovered = Signal(float)
    sampled = Signal(float)  # fired every tick — for the status-bar gauge

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(int(getattr(Settings, "MEMORY_MONITOR_INTERVAL_MS", 1500)))
        self._timer.timeout.connect(self._tick)
        self._band: PressureBand = PressureBand.NORMAL
        self._last_percent: float = 0.0

    def start(self) -> None:
        """Begin polling.  Idempotent."""
        if not self._timer.isActive():
            self._timer.start()
            # Fire one immediate sample so subscribers don't wait
            # ``interval`` ms to populate their status indicators.
            self._tick()

    def stop(self) -> None:
        """Stop polling.  Idempotent; safe to call from any thread."""
        if self._timer.isActive():
            self._timer.stop()

    def current_percent(self) -> float:
        """Return the most recent sample (cached between ticks)."""
        return self._last_percent

    def current_band(self) -> PressureBand:
        """Return the most recently emitted band."""
        return self._band

    def _tick(self) -> None:
        try:
            percent = float(psutil.virtual_memory().percent)
        except Exception as exc:  # noqa: BLE001
            log.debug("MemoryMonitor sample failed: %s", exc)
            return
        self._last_percent = percent
        self.sampled.emit(percent)
        new_band = self._classify(percent)
        if new_band == self._band:
            return
        self._transition(self._band, new_band, percent)
        self._band = new_band

    @staticmethod
    def _classify(percent: float) -> PressureBand:
        """Map a sample to a band, honoring hysteresis at boundaries.

        The static-classifier read of thresholds means the transition
        logic in :meth:`_transition` is what enforces the recover
        gates — this method just buckets the raw number.  See the
        docstring at the top of this module for why both are needed.
        """
        if percent >= getattr(Settings, "MEMORY_PRESSURE_EMERGENCY_PCT", 95.0):
            return PressureBand.EMERGENCY
        if percent >= getattr(Settings, "MEMORY_PRESSURE_CRITICAL_PCT", 90.0):
            return PressureBand.CRITICAL
        if percent >= getattr(Settings, "MEMORY_PRESSURE_WARNING_PCT", 80.0):
            return PressureBand.WARNING
        return PressureBand.NORMAL

    def _transition(
        self,
        prev: PressureBand,
        new: PressureBand,
        percent: float,
    ) -> None:
        """Emit the appropriate signal for ``prev → new`` transition.

        The trigger signals (warning/critical/emergency) fire when
        crossing *upward* across a band boundary.  The recovered
        signal fires once when crossing back below
        ``MEMORY_PRESSURE_RECOVER_WARNING_PCT`` from any non-NORMAL
        band — subscribers that need finer granularity can compare
        ``current_band()`` before and after.
        """
        # Trigger emissions on upward transitions.
        if new > prev:
            if new == PressureBand.WARNING:
                self.warning.emit(percent)
                log.warning("Memory pressure WARNING %.1f%%", percent)
            elif new == PressureBand.CRITICAL:
                self.critical.emit(percent)
                log.warning("Memory pressure CRITICAL %.1f%%", percent)
            elif new == PressureBand.EMERGENCY:
                self.emergency.emit(percent)
                log.error("Memory pressure EMERGENCY %.1f%%", percent)
            return

        # Downward transitions.  Only emit ``recovered`` when we cross
        # all the way back to NORMAL — partial recovery (e.g.
        # EMERGENCY → CRITICAL) stays gated so the GUI's blocking
        # state remains in place until pressure truly clears.
        if new == PressureBand.NORMAL and prev != PressureBand.NORMAL:
            self.recovered.emit(percent)
            log.info("Memory pressure recovered to NORMAL %.1f%%", percent)


_global_monitor: Optional[MemoryMonitor] = None


def install_global(parent: Optional[QObject] = None) -> MemoryMonitor:
    """Create (or return the existing) process-wide monitor.

    MainWindow calls this at startup, hands the instance off to
    subscribers, and starts polling.  Calling twice returns the same
    object so multiple GUI panels can hold their own reference.
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(parent)
    return _global_monitor


def global_monitor() -> Optional[MemoryMonitor]:
    """Return the singleton or ``None`` if :func:`install_global` hasn't run."""
    return _global_monitor
