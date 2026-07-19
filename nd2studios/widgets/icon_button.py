"""DPI-aware icon button factory (V1.44).

Centralizes two things the GUI overhaul needs everywhere:

* **`ui_scale()` / `scaled()`** — a single DPI scale factor derived from the
  primary screen's logical DPI, so button / icon / tile sizes track the monitor
  instead of being hard-coded in physical pixels.
* **`icon_button()` / `make_icon()` / `bind_toggle_icon()`** — build
  `qtawesome` (Font-Awesome) vector icons recolored to the Dracula palette.
  Vector icons stay crisp at any DPI, giving the "modern, polished graphics"
  the revamp asks for (e.g. a real play/pause control).

`qtawesome` is a hard dependency, but the import is guarded so a missing install
degrades to plain text buttons rather than crashing the app.
"""
from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QPushButton, QToolButton, QWidget

from nd2studios.core.settings import Settings

try:  # pragma: no cover - exercised only when the optional dep is absent
    import qtawesome as qta
    _HAVE_QTA = True
except Exception:  # noqa: BLE001
    qta = None  # type: ignore[assignment]
    _HAVE_QTA = False


# Screen-size text/UI growth (V1.64). On monitors larger than the 1080p
# baseline the whole UI — fonts *and* the controls that hold them — scales up
# together so text is legible on big lab displays without overflowing buttons.
# At/below the baseline the factor is 1.0, leaving the UI byte-identical to
# earlier versions.
SCREEN_BASELINE_HEIGHT = 1080.0  # px; a "standard" 1080p display gets 1.0
SCREEN_SCALE_SLOPE = 0.5         # fraction of the excess height turned into growth
SCREEN_SCALE_MAX = 1.5           # never grow the UI by more than 50% from screen size


def screen_scale() -> float:
    """Screen-size growth factor (``1.0`` at/below 1080p, capped at ``1.5``).

    Larger monitors get proportionally larger text and controls. Uses the
    primary screen's available **logical height** so it tracks usable desktop
    space and is not inflated by an unusually wide (ultrawide) aspect ratio.
    """
    app = QGuiApplication.instance()
    if app is not None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            height = screen.availableGeometry().height()
            if height > 0:
                excess = max(0.0, height / SCREEN_BASELINE_HEIGHT - 1.0)
                return min(SCREEN_SCALE_MAX, 1.0 + SCREEN_SCALE_SLOPE * excess)
    return 1.0


def ui_scale() -> float:
    """Combined DPI × screen-size scale factor (``1.0`` at 96 DPI / 1080p).

    The DPI term keeps physical sizes consistent across monitors of differing
    pixel density (unchanged from earlier versions); the :func:`screen_scale`
    term additionally enlarges the whole UI on physically larger displays.
    Multiplying them means every ``scaled()`` control grows in lockstep with
    the text, so enlarged type can never overflow its container.
    """
    dpi_factor = 1.0
    app = QGuiApplication.instance()
    if app is not None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            dpi = screen.logicalDotsPerInch()
            if dpi > 0:
                dpi_factor = max(1.0, dpi / 96.0)
    return dpi_factor * screen_scale()


def scaled(px: float) -> int:
    """Scale a base (96-DPI, 1080p) pixel value to the current display."""
    return int(round(px * ui_scale()))


def scaled_pt(base_pt: float) -> float:
    """Scale a base (1080p) point size for the current screen (half-pt steps)."""
    return round(base_pt * screen_scale() * 2) / 2


_PT_RE = re.compile(r"(\d+(?:\.\d+)?)pt")
_PX_RE = re.compile(r"(\d+(?:\.\d+)?)px")


def scale_qss(style: str, factor: Optional[float] = None) -> str:
    """Scale every ``pt`` font size and ``px`` dimension in a QSS string.

    V1.64 — fonts and the controls that hold them grow by the same screen-size
    *factor* (defaults to :func:`screen_scale`), so enlarged text stays
    proportional to its button and never overflows. Point sizes keep half-point
    precision; pixels round to a whole number ≥ 1. A *factor* of ``1.0``
    (baseline 1080p display) returns the string unchanged, so on existing
    setups the rendered UI is byte-identical to earlier versions. Safe to run on
    inline ``setStyleSheet`` strings — it only touches ``…pt`` / ``…px`` tokens,
    leaving colors, URLs, and percentages alone.
    """
    if factor is None:
        factor = screen_scale()
    if abs(factor - 1.0) < 1e-3:
        return style

    def _pt(match: "re.Match[str]") -> str:
        return f"{round(float(match.group(1)) * factor * 2) / 2:g}pt"

    def _px(match: "re.Match[str]") -> str:
        return f"{max(1, int(round(float(match.group(1)) * factor)))}px"

    return _PX_RE.sub(_px, _PT_RE.sub(_pt, style))


def make_icon(
    name: str,
    color: Optional[str] = None,
    *,
    color_active: Optional[str] = None,
    color_disabled: Optional[str] = None,
) -> QIcon:
    """A recolored qtawesome icon (e.g. ``"fa5s.play"``).

    Falls back to an empty :class:`QIcon` when qtawesome is unavailable so
    callers can still set button text.
    """
    if not _HAVE_QTA:
        return QIcon()
    return qta.icon(
        name,
        color=color or Settings.FG_PRIMARY,
        color_active=color_active or Settings.ACCENT_PURPLE,
        color_disabled=color_disabled or Settings.BORDER_COLOR,
    )


def icon_button(
    name: str,
    tooltip: str = "",
    *,
    text: str = "",
    checkable: bool = False,
    object_name: Optional[str] = None,
    color: Optional[str] = None,
    button_px: Optional[int] = None,
    icon_px: int = 16,
    parent: Optional[QWidget] = None,
) -> QPushButton:
    """Build a DPI-scaled `QPushButton` carrying a qtawesome icon.

    ``button_px`` (if given) sets a fixed square size; otherwise the button
    sizes to its content. ``icon_px`` is the base (96-DPI) icon edge.
    """
    btn = QPushButton(text, parent)
    if name:
        btn.setIcon(make_icon(name, color))
        btn.setIconSize(QSize(scaled(icon_px), scaled(icon_px)))
    if not _HAVE_QTA and not text and tooltip:
        # No icon available — show a short text fallback so the button isn't blank.
        btn.setText(tooltip.split()[0])
    if tooltip:
        btn.setToolTip(tooltip)
    if checkable:
        btn.setCheckable(True)
    if object_name:
        btn.setObjectName(object_name)
    if button_px is not None:
        s = scaled(button_px)
        btn.setFixedSize(s, s)
    return btn


def tool_button(
    name: str,
    tooltip: str = "",
    *,
    text: str = "",
    checkable: bool = False,
    object_name: Optional[str] = None,
    color: Optional[str] = None,
    icon_px: int = 18,
    parent: Optional[QWidget] = None,
) -> QToolButton:
    """A `QToolButton` variant (used for the top tab bar — icon over text)."""
    btn = QToolButton(parent)
    if name:
        btn.setIcon(make_icon(name, color))
        btn.setIconSize(QSize(scaled(icon_px), scaled(icon_px)))
    if text:
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    if tooltip:
        btn.setToolTip(tooltip)
    if checkable:
        btn.setCheckable(True)
    if object_name:
        btn.setObjectName(object_name)
    return btn


def arrow_png(name: str, color: str, size_px: int = 12) -> str:
    """Render a qtawesome icon to a cached PNG and return a QSS-friendly path.

    Qt's QSS ``::down-arrow`` / ``::up-arrow`` only render reliably from an
    ``image: url(...)`` — the CSS border-triangle trick often shows nothing.
    We render the icon once to ``%TEMP%/nd2studios_icons`` and hand back a
    forward-slashed absolute path for ``url()``.
    """
    import os
    import tempfile

    safe = name.replace(".", "_")
    key = f"{safe}_{color.lstrip('#')}_{int(size_px)}.png"
    cache_dir = os.path.join(tempfile.gettempdir(), "nd2studios_icons")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key)
    if not os.path.exists(path):
        if not _HAVE_QTA:
            return ""
        s = scaled(size_px)
        try:
            qta.icon(name, color=color).pixmap(QSize(s, s)).save(path, "PNG")
        except Exception:  # noqa: BLE001
            return ""
    return path.replace(os.sep, "/")


def bind_toggle_icon(
    btn: QPushButton,
    icon_unchecked: str,
    icon_checked: str,
    *,
    color: Optional[str] = None,
    color_checked: Optional[str] = None,
) -> None:
    """Swap a checkable button's icon on toggle (e.g. play ⇄ pause).

    Sets the initial icon and connects ``toggled`` so the icon tracks state.
    """
    def _apply(checked: bool) -> None:
        if checked:
            btn.setIcon(make_icon(icon_checked, color_checked or color))
        else:
            btn.setIcon(make_icon(icon_unchecked, color))

    _apply(btn.isChecked())
    btn.toggled.connect(_apply)
