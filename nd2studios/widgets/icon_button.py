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


def ui_scale() -> float:
    """Logical-DPI scale factor (1.0 at 96 DPI). Never below 1.0."""
    app = QGuiApplication.instance()
    if app is not None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            dpi = screen.logicalDotsPerInch()
            if dpi > 0:
                return max(1.0, dpi / 96.0)
    return 1.0


def scaled(px: float) -> int:
    """Scale a base (96-DPI) pixel value to the current display."""
    return int(round(px * ui_scale()))


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
