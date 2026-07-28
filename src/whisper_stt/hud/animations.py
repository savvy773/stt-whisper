"""
Animation helpers for the HUD overlay — pulse, spin, flash, and EQ interpolation.
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QVariant,
)
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------

def create_transition(
    target: QObject,
    prop: bytes,
    start: QVariant,
    end: QVariant,
    duration: int = 300,
    easing: QEasingCurve.Type = QEasingCurve.Type.InOutCubic,
    loop: bool = False,
) -> QPropertyAnimation:
    """Create and return a configured QPropertyAnimation."""
    anim = QPropertyAnimation(target, prop)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setDuration(duration)
    anim.setEasingCurve(easing)
    if loop:
        anim.setLoopCount(-1)  # infinite
    return anim


# ---------------------------------------------------------------------------
# Pulse (opacity 0.5 → 1.0 → 0.5 loop)
# ---------------------------------------------------------------------------

class PulseAnimation:
    """Pulsing opacity animation for a widget."""

    def __init__(self, widget: QWidget, duration: int = 1200) -> None:
        self._effect = QGraphicsOpacityEffect(widget)
        self._effect.setOpacity(1.0)
        widget.setGraphicsEffect(self._effect)

        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setStartValue(0.45)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setLoopCount(-1)

    def start(self) -> None:
        if self._anim.state() != QPropertyAnimation.State.Running:
            self._anim.start()

    def stop(self) -> None:
        self._anim.stop()
        self._effect.setOpacity(1.0)


# ---------------------------------------------------------------------------
# Flash (quick green flash then fade)
# ---------------------------------------------------------------------------

class FlashAnimation:
    """One-shot opacity flash: 1.0 → 0.0 over *duration* ms."""

    def __init__(self, widget: QWidget, duration: int = 600) -> None:
        self._effect = QGraphicsOpacityEffect(widget)
        self._effect.setOpacity(0.0)
        widget.setGraphicsEffect(self._effect)

        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setLoopCount(1)

    def trigger(self) -> None:
        """Play the flash once."""
        self._anim.stop()
        self._anim.start()


# ---------------------------------------------------------------------------
# EQ bar smooth interpolation
# ---------------------------------------------------------------------------

def lerp_levels(
    current: list[float],
    target: list[float],
    factor: float = 0.35,
) -> list[float]:
    """Linearly interpolate *current* toward *target* per-element.

    *factor* controls how quickly bars chase their target (0 = frozen, 1 = instant).
    Returns a new list with the interpolated values clamped to [0.0, 1.0].
    """
    result: list[float] = []
    for c, t in zip(current, target):
        v = c + (t - c) * factor
        result.append(max(0.0, min(1.0, v)))
    return result
