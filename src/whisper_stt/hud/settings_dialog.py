"""
Settings dialog — every configuration option in one window.

One radio-button group per setting (current value pre-selected), changes
apply and persist immediately (no Save button) by calling straight through
to the same OverlayWindow setter methods the rest of the app already uses.
A "Set Default" button restores the recommended baseline in one click.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from whisper_stt.config import BLACKLIST_PATH, DEFAULT_HOTKEY
from whisper_stt.hotkey import format_hotkey
from whisper_stt.hud import theme

if TYPE_CHECKING:
    from whisper_stt.hud.overlay import OverlayWindow
    from whisper_stt.settings import SettingsManager


_STYLESHEET = f"""
    QDialog {{
        background-color: #16161d;
        color: {theme.TEXT};
    }}
    QLabel {{
        color: {theme.TEXT_MUTED};
        font-size: {theme.FONT_SIZE_NORMAL}px;
    }}
    QLabel#sectionHeader {{
        color: {theme.PRIMARY_LIGHT};
        font-size: {theme.FONT_SIZE_SMALL}px;
        font-weight: 600;
        margin-top: 4px;
    }}
    QFrame#divider {{
        background: rgba(255, 255, 255, 0.08);
        max-height: 1px;
        min-height: 1px;
        border: none;
        margin: 8px 0 6px 0;
    }}
    QRadioButton {{
        color: {theme.TEXT};
        font-size: {theme.FONT_SIZE_NORMAL}px;
        spacing: 6px;
    }}
    QRadioButton::indicator {{
        width: 12px;
        height: 12px;
        border-radius: 6px;
        border: 1px solid {theme.TEXT_MUTED};
        background: transparent;
    }}
    QRadioButton::indicator:checked {{
        background: {theme.PRIMARY};
        border: 1px solid {theme.PRIMARY};
    }}
    QRadioButton::indicator:hover {{
        border: 1px solid {theme.PRIMARY_LIGHT};
    }}
    QPushButton#defaultButton {{
        color: {theme.TEXT};
        background: {theme.SURFACE};
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: {theme.FONT_SIZE_NORMAL}px;
    }}
    QPushButton#defaultButton:hover {{
        border: 1px solid {theme.PRIMARY_LIGHT};
        background: rgba(255, 255, 255, 0.12);
    }}
"""


class SettingsDialog(QDialog):
    """One-window settings editor. Non-modal, single instance (see main.py,
    which raises the existing dialog instead of creating a second one)."""

    # Each option list is ordered so the recommended default sits first
    # (leftmost) — matches the "Set Default" button below.
    # Exactly one hotkey is ever active — no built-in second fallback (see
    # project memory: a prior always-on-second-combo design was rejected as
    # risking collisions with other apps'/system shortcuts). Both presets
    # are hand-picked known-valid combos, so no runtime capture/validation
    # is needed here.
    _HOTKEY_OPTIONS: list[tuple[str, Any]] = [
        (f"{format_hotkey(DEFAULT_HOTKEY)} (Default)", DEFAULT_HOTKEY),
        (format_hotkey("<cmd>+<space>"), "<cmd>+<space>"),
    ]
    _MODE_OPTIONS: list[tuple[str, Any]] = [("Live", "live"), ("One-time", "once")]
    _DEVICE_OPTIONS: list[tuple[str, Any]] = [("GPU (CUDA)", "cuda"), ("CPU", "cpu")]
    _VRAM_OPTIONS: list[tuple[str, Any]] = [
        ("Full Accuracy (fp16)", False),
        ("Half weight (int8+fp16)", True),
    ]
    _BEAM_OPTIONS: list[tuple[str, Any]] = [
        ("High Accuracy (Beam 5)", 5),
        ("Balanced (Beam 3)", 3),
    ]
    _VAD_OPTIONS: list[tuple[str, Any]] = [("Strict (On)", True), ("Sensitive (Off)", False)]
    _TIMER_OPTIONS: list[tuple[str, Any]] = [
        ("10 min", 10),
        ("5 min", 5),
        ("30 min", 30),
        ("None (disabled)", 0),
    ]
    # Both modes auto-cut on a trailing pause and keep recording straight
    # through — these just control how patient each is before committing
    # a chunk. Live: short/responsive. One-time: long, so mid-thought
    # pauses don't fragment the sentence.
    _SILENCE_OPTIONS: list[tuple[str, Any]] = [
        ("1.8s (Default)", 1.8),
        ("1.0s (Fast)", 1.0),
        ("2.5s (Relaxed)", 2.5),
    ]
    _SILENCE_OPTIONS_ONCE: list[tuple[str, Any]] = [
        ("5.0s (Default)", 5.0),
        ("3.0s (Fast)", 3.0),
        ("8.0s (Patient)", 8.0),
    ]

    # Recommended baseline applied by the "Set Default" button.
    _DEFAULTS: dict[str, Any] = {
        "hotkey": DEFAULT_HOTKEY,
        "mode": "live",
        "device": "cuda",
        "low_vram": False,
        "beam_size": 5,
        "vad_filter": True,
        "mic_timer": 30,
        "silence_threshold_sec": 1.8,
        "silence_threshold_sec_once": 5.0,
    }

    def __init__(self, overlay: "OverlayWindow", settings: "SettingsManager", parent=None) -> None:
        import logging
        self._log = logging.getLogger(__name__)
        super().__init__(parent)
        self._overlay = overlay
        self._settings = settings
        self._default_specs: list[tuple[QButtonGroup, Any, Any]] = []

        # Fusion respects the custom ::indicator border-radius; the native
        # Windows style rasterizes it jagged/broken at small sizes. Must be
        # applied per-widget — setting it on the dialog alone does not
        # cascade down to child radio buttons.
        self._fusion_style = QStyleFactory.create("Fusion")

        self.setWindowTitle("Whisper STT Settings")
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)
        # Real title bar + close button, but never steals the always-on-top
        # HUD's spot — just a normal utility window. Qt.WindowType.Tool
        # (same as the HUD itself) keeps it out of the Windows taskbar —
        # without it, a parentless QDialog gets its own persistent taskbar
        # button that lingers even after the dialog is closed/hidden.
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(4)

        root.addWidget(self._section_header("Hotkey"))
        hotkey_form = QFormLayout()
        hotkey_form.setSpacing(12)
        hotkey_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._add_row(hotkey_form, "Dictation", self._HOTKEY_OPTIONS, overlay._hotkey,
                      overlay.set_hotkey, self._DEFAULTS["hotkey"], overlay.hotkey_changed)
        root.addLayout(hotkey_form)

        root.addWidget(self._divider())
        root.addWidget(self._section_header("General"))
        general_form = QFormLayout()
        general_form.setSpacing(12)
        general_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._add_row(general_form, "Mode", self._MODE_OPTIONS, overlay._mode,
                      overlay.set_mode, self._DEFAULTS["mode"], overlay.mode_changed)
        self._add_row(general_form, "Mic Off Timer", self._TIMER_OPTIONS, overlay._mic_timer_minutes,
                      overlay.set_mic_timer, self._DEFAULTS["mic_timer"], overlay.timer_changed)
        root.addLayout(general_form)

        root.addWidget(self._divider())
        root.addWidget(self._section_header("Recognition Engine"))
        engine_form = QFormLayout()
        engine_form.setSpacing(12)
        engine_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._add_row(engine_form, "Device", self._DEVICE_OPTIONS, overlay._device,
                      overlay.set_device, self._DEFAULTS["device"], overlay.device_changed)
        self._add_row(engine_form, "GPU Memory", self._VRAM_OPTIONS, overlay._low_vram,
                      overlay.set_low_vram, self._DEFAULTS["low_vram"], overlay.low_vram_changed)
        self._add_row(engine_form, "Quality", self._BEAM_OPTIONS, overlay._beam_size,
                      overlay.set_beam_size, self._DEFAULTS["beam_size"], overlay.beam_changed)
        self._add_row(engine_form, "VAD Noise Filter", self._VAD_OPTIONS, overlay._vad_filter,
                      overlay.set_vad_filter, self._DEFAULTS["vad_filter"], overlay.vad_changed)
        root.addLayout(engine_form)

        root.addWidget(self._divider())
        root.addWidget(self._section_header("Silence Timeout"))
        silence_form = QFormLayout()
        silence_form.setSpacing(12)
        silence_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._add_row(silence_form, "Live", self._SILENCE_OPTIONS, overlay._silence_threshold_sec,
                      overlay.set_silence_threshold_sec, self._DEFAULTS["silence_threshold_sec"],
                      overlay.silence_threshold_changed)
        self._add_row(silence_form, "One-time", self._SILENCE_OPTIONS_ONCE, overlay._silence_threshold_sec_once,
                      overlay.set_silence_threshold_sec_once, self._DEFAULTS["silence_threshold_sec_once"],
                      overlay.silence_threshold_once_changed)
        root.addLayout(silence_form)

        root.addWidget(self._divider())
        default_btn = QPushButton("Set Default")
        default_btn.setObjectName("defaultButton")
        default_btn.clicked.connect(self._apply_defaults)
        blacklist_btn = QPushButton("Edit Blacklist...")
        blacklist_btn.setObjectName("defaultButton")
        blacklist_btn.setToolTip(str(BLACKLIST_PATH))
        blacklist_btn.clicked.connect(self._open_blacklist)
        # The native title-bar X is small and hard to hit reliably on the
        # thinner title bar Qt.WindowType.Tool windows get on Windows — an
        # explicit, larger in-body Close button is a much easier click
        # target for the same action (closeEvent() below still saves
        # geometry either way).
        close_btn = QPushButton("Close")
        close_btn.setObjectName("defaultButton")
        close_btn.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addWidget(default_btn)
        btn_row.addWidget(blacklist_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        geometry = settings.get("settings_dialog_geometry")
        if isinstance(geometry, dict) and geometry.get("w", 0) > 0 and geometry.get("h", 0) > 0:
            self.resize(int(geometry["w"]), int(geometry["h"]))
            if geometry.get("x", -1) >= 0 and geometry.get("y", -1) >= 0:
                self.move(int(geometry["x"]), int(geometry["y"]))

        self._log.info("SettingsDialog.__init__ done — size=%s", self.sizeHint())

    def closeEvent(self, event: QCloseEvent) -> None:
        pos = self.pos()
        self._settings.set("settings_dialog_geometry", {
            "x": pos.x(), "y": pos.y(), "w": self.width(), "h": self.height(),
        })
        super().closeEvent(event)

    def _section_header(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("sectionHeader")
        return label

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _add_row(
        self,
        form: QFormLayout,
        label: str,
        options: list[tuple[str, Any]],
        current_value: Any,
        apply_fn,
        default_value: Any,
        change_signal=None,
    ) -> QButtonGroup:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(14)

        group = QButtonGroup(row)
        for text, value in options:
            btn = QRadioButton(text)
            btn.setProperty("optionValue", value)
            if self._fusion_style is not None:
                btn.setStyle(self._fusion_style)
            if value == current_value:
                btn.setChecked(True)
            group.addButton(btn)
            row_layout.addWidget(btn)
        row_layout.addStretch(1)

        def _on_clicked(btn: QRadioButton) -> None:
            apply_fn(btn.property("optionValue"))

        group.buttonClicked.connect(_on_clicked)
        form.addRow(label, row)
        self._default_specs.append((group, apply_fn, default_value))

        # Keep this row's selection in sync with changes made elsewhere
        # (e.g. the HUD's clickable LIVE/ONCE and KO/EN badges) while this
        # non-modal dialog is open. setChecked() here is programmatic, so
        # it does not re-trigger buttonClicked/apply_fn — no feedback loop.
        if change_signal is not None:
            def _on_external_change(value: Any, group: QButtonGroup = group) -> None:
                for btn in group.buttons():
                    if btn.property("optionValue") == value and not btn.isChecked():
                        btn.setChecked(True)
                        break

            change_signal.connect(_on_external_change)

        return group

    def _apply_defaults(self) -> None:
        for group, apply_fn, default_value in self._default_specs:
            apply_fn(default_value)
            for btn in group.buttons():
                if btn.property("optionValue") == default_value:
                    btn.setChecked(True)
                    break

    def _open_blacklist(self) -> None:
        """Open blacklist.json in Notepad, creating it first if this is a
        fresh install that has never transcribed (transcriber.py otherwise
        only creates the file lazily on first hallucination-filter check)."""
        try:
            if not BLACKLIST_PATH.exists():
                BLACKLIST_PATH.write_text("[]\n", encoding="utf-8")
            subprocess.Popen(["notepad.exe", str(BLACKLIST_PATH)])
        except OSError:
            self._log.exception("Failed to open blacklist.json in Notepad")
