"""
Main HUD overlay window — frameless, translucent, always-on-top pill.

This is the primary visible component of the Whisper STT application.
"""

from __future__ import annotations

import logging
from typing import Sequence

from PyQt6.QtCore import (
    QPoint,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from whisper_stt.config import DEFAULT_HOTKEY
from whisper_stt.hud import sound, theme
from whisper_stt.hud.widgets import (
    ModeIndicator,
    StatusIndicator,
    StatusLabel,
    VolumeEqualizer,
)

log = logging.getLogger(__name__)


class OverlayWindow(QWidget):
    """Frameless, translucent, always-on-top HUD pill overlay.

    Signals
    -------
    state_changed(str)
        Emitted whenever the overlay state changes.
    mode_changed(str)
        ``'live'`` or ``'once'``.
    timer_changed(int)
        Mic-off timer in minutes, or ``0`` for disabled.
    hotkey_changed(str)
        pynput combo string for the global dictation hotkey.
    quit_requested()
        User clicked the HUD's own close (✕) button.
    """

    # -- Signals --
    state_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    device_changed = pyqtSignal(str)
    beam_changed = pyqtSignal(int)
    vad_changed = pyqtSignal(bool)
    low_vram_changed = pyqtSignal(bool)
    timer_changed = pyqtSignal(int)
    hotkey_changed = pyqtSignal(str)
    silence_threshold_changed = pyqtSignal(float)
    silence_threshold_once_changed = pyqtSignal(float)
    mic_clicked = pyqtSignal()
    quit_requested = pyqtSignal()
    settings_requested = pyqtSignal()


    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)


        # Window flags — frameless, transparent, tool window, always on top, no focus activation
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(theme.HUD_WIDTH, theme.HUD_HEIGHT_COLLAPSED)
        self.setStyleSheet(theme.get_stylesheet())

        # Enable Windows WS_EX_NOACTIVATE style so window clicks never steal focus
        try:
            import ctypes
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOPMOST = 0x00000008
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOPMOST)
        except Exception as e:
            log.warning("Could not set WS_EX_NOACTIVATE window style: %s", e)


        # Internal state
        self._state: str = "idle"
        self._mode: str = "live"
        self._device: str = "cuda"
        self._beam_size: int = 5

        self._vad_filter: bool = True
        self._low_vram: bool = False
        self._mic_timer_minutes: int = 30
        self._hotkey: str = DEFAULT_HOTKEY
        self._silence_threshold_sec: float = 1.8
        self._silence_threshold_sec_once: float = 5.0


        # Drag tracking
        self._drag_pos: QPoint | None = None

        # Build UI
        self._build_layout()


        # Text preview popup (tooltip-like)
        self._preview = QLabel(self)
        self._preview.setObjectName("textPreview")
        self._preview.setWordWrap(True)
        self._preview.hide()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._preview.hide)

        log.info("OverlayWindow initialized")

    def set_mode(self, value: str) -> None:
        """Set mode ('live' or 'once'); emits mode_changed only on change."""
        value = str(value).lower()
        if value != self._mode:
            self._mode = value
            self._mode_ind.set_mode(value)
            self.mode_changed.emit(value)

    def _on_mode_indicator_clicked(self) -> None:
        """Toggle mode from the HUD badge (quick access; Mode is also a row
        in the tray icon's Settings dialog)."""
        self.set_mode("once" if self._mode == "live" else "live")

    def set_device(self, value: str) -> None:
        """Set compute device ('cuda' or 'cpu'); emits device_changed only on change."""
        value = str(value).lower()
        if value != self._device:
            self._device = value
            self.device_changed.emit(value)
            log.info("Device changed → %s", value)

    def set_low_vram(self, value: bool) -> None:
        """Toggle low-VRAM (int8+fp16) mode; emits low_vram_changed only on change."""
        if value != self._low_vram:
            self._low_vram = value
            self.low_vram_changed.emit(value)
            log.info("Low VRAM mode → %s", value)

    def set_beam_size(self, value: int) -> None:
        """Set beam size (3 or 5); emits beam_changed only on change."""
        if value != self._beam_size:
            self._beam_size = value
            self.beam_changed.emit(value)
            log.info("Beam size changed → %d", value)

    def set_vad_filter(self, value: bool) -> None:
        """Toggle VAD noise filtering; emits vad_changed only on change."""
        if value != self._vad_filter:
            self._vad_filter = value
            self.vad_changed.emit(value)
            log.info("VAD filter changed → %s", value)

    def set_mic_timer(self, minutes: int) -> None:
        """Set the mic-off timer in minutes (0 = disabled); emits timer_changed only on change."""
        if minutes != self._mic_timer_minutes:
            self._mic_timer_minutes = minutes
            self.timer_changed.emit(minutes)
            log.info("Mic-off timer changed → %s min", minutes or "disabled")

    def set_hotkey(self, combo: str) -> None:
        """Set the global dictation hotkey (pynput combo string); emits
        hotkey_changed only on change."""
        if combo != self._hotkey:
            self._hotkey = combo
            self.hotkey_changed.emit(combo)
            log.info("Hotkey changed → %s", combo)

    def set_silence_threshold_sec(self, seconds: float) -> None:
        """Set Live mode's trailing-silence auto-cutoff, in seconds; emits
        silence_threshold_changed only on change."""
        if seconds != self._silence_threshold_sec:
            self._silence_threshold_sec = seconds
            self.silence_threshold_changed.emit(seconds)
            log.info("Live silence timeout changed → %.1fs", seconds)

    def set_silence_threshold_sec_once(self, seconds: float) -> None:
        """Set One-time mode's (longer) trailing-silence auto-cutoff, in
        seconds; emits silence_threshold_once_changed only on change. Both
        modes now auto-cut and keep recording — this is just how patient
        One-time is before committing a chunk."""
        if seconds != self._silence_threshold_sec_once:
            self._silence_threshold_sec_once = seconds
            self.silence_threshold_once_changed.emit(seconds)
            log.info("One-time silence timeout changed → %.1fs", seconds)


    # --------------------------------------------------------------------- #
    # Layout
    # --------------------------------------------------------------------- #

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 8)
        root.setSpacing(2)

        self._status_ind = StatusIndicator(self)
        self._status_ind.clicked.connect(self.mic_clicked.emit)
        self._eq = VolumeEqualizer(self)

        self._status_lbl = StatusLabel(self)
        self._mode_ind = ModeIndicator(self)
        self._mode_ind.clicked.connect(self._on_mode_indicator_clicked)

        from PyQt6.QtWidgets import QPushButton
        self._close_btn = QPushButton("✕", self)
        self._close_btn.setObjectName("closeButton")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setToolTip("Close App")
        self._close_btn.clicked.connect(self.quit_requested.emit)

        self._settings_btn = QPushButton("⚙", self)
        self._settings_btn.setObjectName("settingsButton")
        self._settings_btn.setFixedSize(20, 20)
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setToolTip("Settings")
        self._settings_btn.clicked.connect(self.settings_requested.emit)

        # Top row: mode badge + status text read together as one phrase
        # ("LIVE — Listening…"), close button tucked in the corner
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(self._mode_ind)
        top_row.addWidget(self._status_lbl, stretch=1)
        top_row.addWidget(self._close_btn)

        # Bottom row: every control lives in one tight cluster (no internal
        # gap) so nothing reads as wasted space; the cluster as a whole is
        # centered via equal stretch on both sides.
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        bottom_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self._status_ind)
        bottom_row.addWidget(self._eq)
        bottom_row.addWidget(self._settings_btn)
        bottom_row.addStretch(1)

        root.addLayout(top_row)
        root.addLayout(bottom_row)
        self.setLayout(root)


    # --------------------------------------------------------------------- #
    # State management
    # --------------------------------------------------------------------- #

    def set_state(self, state: str) -> None:
        """Transition the overlay to *state* ('idle', 'speaking', 'transcoding', 'input')."""
        if state == self._state:
            return
        self._state = state
        self._status_ind.set_state(state)
        self._status_lbl.set_state(state)
        self.state_changed.emit(state)
        log.debug("Overlay state → %s", state)

    @property
    def state(self) -> str:
        return self._state

    @property
    def mode(self) -> str:
        return self._mode

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #

    def update_volume(self, levels: Sequence[float]) -> None:
        """Forward audio levels to the equalizer widget."""
        self._eq.update_levels(levels)

    def show_text_preview(self, text: str, duration_ms: int = 3000) -> None:
        """Show a brief text preview near the HUD."""
        self._preview.setText(text)
        self._preview.adjustSize()
        # Position below the pill
        self._preview.move(10, self.height() + 4)
        self._preview.setFixedWidth(self.width() - 20)
        self._preview.show()
        self._preview_timer.start(duration_ms)

    def restore_position(self, x: int, y: int) -> None:
        """Move the overlay to a saved position."""
        self.move(x, y)

    def get_position(self) -> tuple[int, int]:
        """Return current (x, y) position on screen."""
        pos = self.pos()
        return pos.x(), pos.y()

    # --------------------------------------------------------------------- #
    # Drag support
    # --------------------------------------------------------------------- #

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event and self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        # Position is only persisted at quit (see main.py) — no need to save
        # on every drag, this just ends the drag gesture.
        self._drag_pos = None

    # --------------------------------------------------------------------- #
    # Show/hide feedback — a soft chime, not a visual change, so it has to
    # live here rather than in a paintEvent
    # --------------------------------------------------------------------- #

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        sound.play_show()

    def hideEvent(self, event: object) -> None:
        super().hideEvent(event)
        sound.play_hide()

    # Settings live in a single SettingsDialog (hud/settings_dialog.py),
    # opened from the tray icon's "Settings..." action — no HUD right-click
    # menu, no nested per-option submenus.

    # --------------------------------------------------------------------- #
    # Custom painting — rounded rect with glassmorphism background
    # --------------------------------------------------------------------- #

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        r = theme.BORDER_RADIUS

        # Background path
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(w), float(h), r, r)

        # Fill — dark glass
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 10, 15, 217))  # ~0.85 alpha
        painter.drawPath(path)

        # Inner highlight (top edge shine)
        highlight = QPainterPath()
        highlight.addRoundedRect(0.5, 0.5, float(w - 1), float(h - 1), r, r)
        pen = QPen(QColor(255, 255, 255, 18))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(highlight)

        # Subtle glow along the bottom when active
        if self._state in ("speaking", "transcoding", "loading_model"):
            glow_color = QColor(theme.PRIMARY)
            glow_color.setAlpha(30)
            painter.setPen(QPen(glow_color, 1.5))
            glow_path = QPainterPath()
            glow_path.addRoundedRect(0.0, 0.0, float(w), float(h), r, r)
            painter.drawPath(glow_path)

        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(theme.HUD_WIDTH, theme.HUD_HEIGHT_COLLAPSED)
