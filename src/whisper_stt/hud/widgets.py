"""
Custom PyQt6 widgets for the HUD overlay — status indicator, volume EQ,
status label, and mode indicator.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import (
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from whisper_stt.hud import theme
from whisper_stt.hud.animations import (
    PulseAnimation,
    lerp_levels,
)


# ---------------------------------------------------------------------------
# StatusIndicator — circular LED with glow
# ---------------------------------------------------------------------------

class StatusIndicator(QWidget):
    """Circular indicator that changes color & animation based on state."""

    clicked = pyqtSignal()

    _STATE_COLORS: dict[str, str] = {
        "idle": "#64748b",         # Slate Grey
        "speaking": "#33d5ee",     # Neon Cyan
        "transcoding": "#a56cdb",  # Deep Purple Arc
        "loading_model": "#a56cdb",  # Same as transcoding — text distinguishes them
        "input": "#34d399",        # Emerald Green
    }

    _DIAMETER = 18


    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._DIAMETER + 12, self._DIAMETER + 12)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to toggle recording (Ctrl+Shift+Space)")


        self._state: str = "idle"
        self._color = QColor(self._STATE_COLORS["idle"])

        # Animations
        self._pulse = PulseAnimation(self, duration=1200)

        self._flash_opacity: float = 0.0

    # -- public API --

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        # Stop previous animations
        self._pulse.stop()

        self._state = state
        self._color = QColor(self._STATE_COLORS.get(state, theme.TEXT_MUTED))

        if state == "speaking":
            self._pulse.start()
        elif state == "input":
            self._flash_opacity = 1.0
            QTimer.singleShot(400, self._end_flash)

        self.update()

    # -- internals --

    def _end_flash(self) -> None:
        self._flash_opacity = 0.0
        self.update()

    # -- painting --

    def mousePressEvent(self, event: object) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()

    def paintEvent(self, _event: object) -> None:

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        r = self._DIAMETER / 2

        # Outer glow
        glow = QRadialGradient(cx, cy, r + 6)
        glow.setColorAt(0.0, QColor(self._color.red(), self._color.green(), self._color.blue(), 80))
        glow.setColorAt(1.0, QColor(self._color.red(), self._color.green(), self._color.blue(), 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(int(cx - r - 6), int(cy - r - 6), int((r + 6) * 2), int((r + 6) * 2))

        # Microphone glyph — explicit affordance that this is the record
        # toggle, not just a status LED (was a plain filled circle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)

        cap_w = r * 0.85
        cap_h = r * 1.35
        cap_x = cx - cap_w / 2
        cap_y = cy - r * 1.05
        cap_path = QPainterPath()
        cap_path.addRoundedRect(cap_x, cap_y, cap_w, cap_h, cap_w / 2, cap_w / 2)
        painter.drawPath(cap_path)

        # Stand (U-shaped cradle) + stem, below the capsule
        pen = QPen(self._color, max(1.4, r * 0.16))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        stand_r = r * 1.0
        stand_rect_x = cx - stand_r
        stand_rect_y = cy - r * 0.25 - stand_r
        painter.drawArc(
            int(stand_rect_x), int(stand_rect_y), int(stand_r * 2), int(stand_r * 2),
            200 * 16, 140 * 16,
        )
        stem_top = cy + r * 0.55
        stem_bottom = cy + r * 0.95
        painter.drawLine(int(cx), int(stem_top), int(cx), int(stem_bottom))

        # Flash overlay
        if self._flash_opacity > 0:
            flash_color = QColor(theme.SUCCESS)
            flash_color.setAlphaF(self._flash_opacity * 0.6)
            painter.setBrush(flash_color)
            painter.drawEllipse(int(cx - r - 4), int(cy - r - 4), int((r + 4) * 2), int((r + 4) * 2))

        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(self._DIAMETER + 12, self._DIAMETER + 12)


# ---------------------------------------------------------------------------
# VolumeEqualizer — animated bar EQ
# ---------------------------------------------------------------------------

class VolumeEqualizer(QWidget):
    """EQ_BAR_COUNT vertical bars that visualize audio levels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bar_count = theme.EQ_BAR_COUNT
        self._levels: list[float] = [0.0] * self._bar_count
        self._display: list[float] = [0.0] * self._bar_count

        total_w = (
            self._bar_count * theme.EQ_BAR_WIDTH
            + (self._bar_count - 1) * theme.EQ_BAR_GAP
        )
        self._max_bar_h = 30
        self.setFixedSize(total_w + 4, self._max_bar_h + 4)

        # Smooth interpolation timer (~60 fps) — only runs while there's
        # something to animate; see _interpolate_and_repaint/update_levels.
        # Left running 24/7 (as it was before), this repaints 60x/sec
        # forever even with the HUD hidden in the tray — the app's normal
        # idle state — for no visible effect.
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._interpolate_and_repaint)
        self._timer.start()

    # -- public API --

    def update_levels(self, levels: Sequence[float]) -> None:
        """Set target bar levels (each 0.0–1.0).  List can be shorter/longer."""
        for i in range(self._bar_count):
            self._levels[i] = levels[i] if i < len(levels) else 0.0
        if not self._timer.isActive():
            self._timer.start()

    # -- internals --

    def _interpolate_and_repaint(self) -> None:
        self._display = lerp_levels(self._display, self._levels, factor=0.30)
        self.update()
        # Nothing left to animate (bars at rest, no new levels coming in) —
        # stop ticking until update_levels() has real data again.
        if all(v < 0.001 for v in self._display) and all(v < 0.001 for v in self._levels):
            self._timer.stop()

    # -- painting --

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bw = theme.EQ_BAR_WIDTH
        gap = theme.EQ_BAR_GAP
        max_h = self._max_bar_h
        base_y = self.height()

        primary = QColor(theme.PRIMARY)
        accent = QColor(theme.ACCENT)

        for i, level in enumerate(self._display):
            x = 2 + i * (bw + gap)
            bar_h = max(2, int(level * max_h))

            # Gradient per bar — bottom = primary, top = accent
            grad = QLinearGradient(x, base_y, x, base_y - bar_h)
            grad.setColorAt(0.0, primary)
            grad.setColorAt(1.0, accent)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)

            path = QPainterPath()
            radius = bw / 2
            path.addRoundedRect(float(x), float(base_y - bar_h), float(bw), float(bar_h), radius, radius)
            painter.drawPath(path)

        painter.end()

    def sizeHint(self) -> QSize:
        total_w = (
            self._bar_count * theme.EQ_BAR_WIDTH
            + (self._bar_count - 1) * theme.EQ_BAR_GAP
        )
        return QSize(total_w + 4, self._max_bar_h + 4)


# ---------------------------------------------------------------------------
# StatusLabel — state text with fade transition
# ---------------------------------------------------------------------------

_STATE_TEXTS: dict[str, str] = {
    "idle": "Ready",
    "speaking": "Listening…",
    "transcoding": "Processing…",
    "loading_model": "Loading model (first run)…",
    "input": "Typed",
}


class StatusLabel(QLabel):
    """Label that shows the current state text with a fade transition on change."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Ready", parent)
        self.setObjectName("statusLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._state = "idle"

        # Opacity effect for fade transition
        self._opacity_fx = QGraphicsOpacityEffect(self)
        self._opacity_fx.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_fx)

        self._fade_out = QPropertyAnimation(self._opacity_fx, b"opacity")
        self._fade_out.setDuration(120)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)

        self._fade_in = QPropertyAnimation(self._opacity_fx, b"opacity")
        self._fade_in.setDuration(180)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)

        self._pending_text: str | None = None
        self._fade_out.finished.connect(self._on_fade_out_done)

    # -- public API --

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        new_text = _STATE_TEXTS.get(state, state.capitalize())
        self._pending_text = new_text
        self._fade_out.stop()
        self._fade_in.stop()
        self._fade_out.start()

    # -- internals --

    def _on_fade_out_done(self) -> None:
        if self._pending_text is not None:
            self.setText(self._pending_text)
            self._pending_text = None
        self._fade_in.start()


# ---------------------------------------------------------------------------
# ModeIndicator — "LIVE" / "ONCE" pill badge
# ---------------------------------------------------------------------------

class ModeIndicator(QLabel):
    """Small clickable pill badge showing/toggling the operating mode."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("LIVE", parent)
        self.setObjectName("modeIndicator")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to toggle mode (Live / One-time)")
        self.set_mode("live")

    def set_mode(self, mode: str) -> None:
        """Set mode to ``'live'`` or ``'once'``."""
        mode = mode.lower()
        self.setText(mode.upper())
        self.setProperty("mode", mode)
        # Force QSS re-evaluation
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
