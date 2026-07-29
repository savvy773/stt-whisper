"""
Entry point for the Whisper STT desktop application.

Wires together the HUD overlay, audio capture, transcription engine,
input controller, and global hotkey into a cohesive real-time pipeline.

Pipeline:
    Ctrl+Alt+Space → SPEAKING (mic on, EQ active)
    → trailing silence (Live: short/1.8s default, One-time: long/5s
      default — see Settings) → cut the buffer WITHOUT stopping the mic
      → TRANSCODING (faster-whisper in QThread) while recording continues
    → done          → INPUT (SendInput text injection)
    → session still active (either mode) → loop back to SPEAKING
    → hotkey again → stop the mic for real, transcribe the tail, IDLE

Both modes now auto-cut and keep recording straight through
transcription — the only difference is pause tolerance. The session
itself always starts/ends via the hotkey, in both modes.
"""

from __future__ import annotations

import ctypes
import logging
import logging.handlers
import math
import os
import subprocess
import sys
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication

from whisper_stt.audio import AudioCapture
from whisper_stt.config import APP_DIR, BLACKLIST_PATH, MIC_TIMER_MIN_CHARS, MIN_SPEECH_DURATION_SEC
from whisper_stt.hotkey import DEFAULT_HOTKEY, HotkeyManager, format_hotkey
from whisper_stt.hud import theme
from whisper_stt.hud.overlay import OverlayWindow
from whisper_stt.hud.settings_dialog import SettingsDialog
from whisper_stt.input_controller import InputController
from whisper_stt.settings import SettingsManager
from whisper_stt.transcriber import TranscriberEngine, TranscriptionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

from pathlib import Path

# Setup logs directory and file handler
_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_APP_LOG_PATH = _LOGS_DIR / "app.log"

_file_handler = logging.handlers.RotatingFileHandler(
    _APP_LOG_PATH, maxBytes=2_000_000, backupCount=2, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), _file_handler]
)
log = logging.getLogger("whisper_stt")



# ---------------------------------------------------------------------------
# Thread-safe bridges  (background threads → Qt main thread via signals)
# ---------------------------------------------------------------------------


class _HotkeyBridge(QObject):
    """Relay pynput hotkey callback (background thread) into a Qt signal."""

    triggered = pyqtSignal()

    def fire(self) -> None:
        """Called from the pynput listener thread."""
        self.triggered.emit()


class _AudioBridge(QObject):
    """Relay sounddevice callbacks (PortAudio thread) into Qt signals."""

    volume_updated = pyqtSignal(float)
    silence_detected = pyqtSignal()

    def __init__(self, capture: AudioCapture) -> None:
        super().__init__()
        self._capture = capture
        self._silence_fired = False
        capture.on_volume_level = self._on_volume
        capture.on_audio_data = self._on_audio

    def reset(self) -> None:
        """Reset silence guard for a new recording session."""
        self._silence_fired = False

    def _on_volume(self, rms: float) -> None:
        self.volume_updated.emit(rms)

    def _on_audio(self, chunk: np.ndarray) -> None:
        # Always feed the detector (not just until the first fire) — Live
        # mode's continuous recording relies on is_speaking/current_silence
        # staying accurate for the whole session, including while a
        # previous utterance is still transcribing. _silence_fired only
        # gates the Qt signal emission (once per silence episode); it
        # self-clears the moment speech resumes so the next pause fires again.
        is_silent = self._capture.silence_detector.feed(chunk)
        if is_silent:
            if not self._silence_fired:
                self._silence_fired = True
                self.silence_detected.emit()
        else:
            self._silence_fired = False


# ---------------------------------------------------------------------------
# Transcription worker  (runs in a dedicated QThread)
# ---------------------------------------------------------------------------


class _TranscribeWorker(QThread):
    """Run a single transcription off the GUI thread."""

    # NOTE: deliberately not named "finished" — that would shadow QThread.finished
    result_ready = pyqtSignal(object)  # emits TranscriptionResult

    def __init__(
        self,
        engine: TranscriberEngine,
        audio: AudioCapture,
        chunks: list[np.ndarray],
        language: str | None,
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._audio = audio
        self._chunks = chunks
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._initial_prompt = initial_prompt

    def run(self) -> None:
        # DSP (high-pass/resample/normalize) deliberately runs here, off the
        # GUI thread — it's real CPU work (tens of ms, more for long
        # one-time recordings) that used to run synchronously in
        # _stop_and_transcribe() on the main thread before this worker even
        # started.
        audio_data = self._audio.process_chunks(self._chunks)
        result = self._engine.transcribe(
            audio_data,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            initial_prompt=self._initial_prompt,
        )
        self.result_ready.emit(result)




# ---------------------------------------------------------------------------
# Mic-off countdown timer
# ---------------------------------------------------------------------------


class _MicOffTimer:
    """QTimer wrapper — fires once after *minutes* and calls *callback*."""

    def __init__(self) -> None:
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._cb: Any = None

    def start(self, minutes: int, callback: Any) -> None:
        self.cancel()
        if minutes <= 0:
            return
        self._cb = callback
        self._timer.timeout.connect(self._fire)
        self._timer.start(minutes * 60_000)
        log.info("Mic-off timer set: %d min", minutes)

    def cancel(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._timer.timeout.disconnect(self._fire)
        except TypeError:
            pass

    def _fire(self) -> None:
        log.info("Mic-off timer expired")
        if self._cb:
            self._cb()


# ---------------------------------------------------------------------------
# Mutable application state
# ---------------------------------------------------------------------------


class _AppState:
    """Holds the pipeline's mutable flags in one place (avoids nonlocal)."""

    session_active: bool = False
    worker: _TranscribeWorker | None = None
    eq_phase: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Default label for the single-instance-already-running message below,
    # which can fire before settings are loaded. Recomputed from the saved
    # hotkey right after SettingsManager() is constructed.
    hotkey_label = format_hotkey(DEFAULT_HOTKEY)

    # Single Instance Guard via Windows Named Mutex
    if sys.platform == "win32":
        import ctypes
        mutex_name = "Global\\WhisperSTT_SingleInstance_Mutex"
        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, True, mutex_name)
        last_error = kernel32.GetLastError()
        ERROR_ALREADY_EXISTS = 183
        if last_error == ERROR_ALREADY_EXISTS:
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Whisper STT is already running in your System Tray (near clock).\nPress {hotkey_label} anytime to talk!",
                "Whisper STT Already Running",
                0x00000040  # MB_ICONINFORMATION
            )
            sys.exit(0)


    app = QApplication(sys.argv)
    app.setApplicationName("Whisper STT")
    # Qt defaults to quitting the whole app when the last visible top-level
    # window closes. This is a tray-resident app — closing the Settings
    # dialog (or the HUD) must never quit it; only "Exit App" (_on_quit)
    # should.
    app.setQuitOnLastWindowClosed(False)

    # -- Core services --
    settings = SettingsManager()
    hotkey_label = format_hotkey(str(settings.get("hotkey", DEFAULT_HOTKEY)))
    saved_device = settings.get("device") or "cuda"
    saved_low_vram = bool(settings.get("low_vram", False))
    transcriber = TranscriberEngine(device=saved_device, low_vram=saved_low_vram)

    # Warm the model in the background so tray startup stays instant but the
    # model is likely already resident by the user's first utterance. A
    # *first-ever* load on a given GPU can be far slower than the normal
    # 2-4s (measured ~46s on this machine's RTX 5070 Ti — CUDA/cuDNN JIT-
    # compiling kernels for a new GPU architecture, then caching them to
    # disk; every load after that is fast). This thread used to lower its
    # own OS priority (THREAD_PRIORITY_BELOW_NORMAL) on the theory that it
    # would protect the GUI/hotkey-hook thread from CPU contention — that
    # was never actually confirmed to matter, while the latency cost was
    # real and measured: a short first utterance regularly finished before
    # a deprioritized warm-up did, forcing the first transcription to wait
    # on it. Removed — warm_up() now runs at normal thread priority so it
    # finishes sooner, shrinking that wait on every launch.
    threading.Thread(target=transcriber.warm_up, daemon=True).start()

    audio = AudioCapture(
        silence_rms=float(settings.get("silence_rms_threshold", 0.028)),
        silence_sec=float(settings.get("silence_threshold_sec", 1.8)),
    )
    input_ctrl = InputController()
    hotkey_mgr = HotkeyManager()
    mic_timer = _MicOffTimer()

    # -- Thread bridges --
    hotkey_bridge = _HotkeyBridge()
    audio_bridge = _AudioBridge(audio)

    # -- HUD overlay --
    overlay = OverlayWindow()
    pos = settings.get("window_position")
    
    # Ignore (0,0) or invalid positions
    if isinstance(pos, dict) and pos.get("x", 0) > 0 and pos.get("y", 0) > 0:
        overlay.restore_position(int(pos["x"]), int(pos["y"]))
        log.info("Restored saved HUD position: x=%d, y=%d", pos["x"], pos["y"])
    else:
        # First launch or reset: center near bottom of primary screen
        screen = app.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            cx = (geo.width() - overlay.width()) // 2
            cy = geo.height() - overlay.height() - 80
            overlay.restore_position(cx, cy)
            settings.set("window_position", {"x": cx, "y": cy})
            log.info("Calculated center-bottom position: x=%d, y=%d", cx, cy)

    # Start tray-only — the HUD appears when the hotkey is first pressed
    state = _AppState()

    # ------------------------------------------------------------------ #
    # Volume RMS → EQ bar levels
    # ------------------------------------------------------------------ #

    def _rms_to_levels(rms: float) -> list[float]:
        # Scale relative to the configured speech threshold so the gauge tracks
        # the same sensitivity the silence detector uses
        thr = audio.silence_detector.threshold_rms
        if rms < thr * 0.5:
            return [0.0] * theme.EQ_BAR_COUNT

        state.eq_phase += 0.15
        levels: list[float] = []
        # rms == threshold → ~35% bar height, saturates around 3x threshold
        base = min((rms - thr * 0.5) / (thr * 2.5) + 0.15, 1.0)
        for i in range(theme.EQ_BAR_COUNT):
            v = base * (0.3 + 0.7 * abs(math.sin(state.eq_phase + i * 0.5)))
            levels.append(max(0.05, min(v, 1.0)))
        return levels


    # ------------------------------------------------------------------ #
    # Pipeline actions
    # ------------------------------------------------------------------ #

    def _start_listening() -> None:
        """Begin microphone capture and transition HUD to SPEAKING."""
        if not state.session_active:
            overlay.set_state("idle")
            return
        input_ctrl.capture_target_window()
        audio_bridge.reset()
        audio.start()
        overlay.set_state("speaking")
        log.info("Listening started")

    def _resume_speaking() -> None:
        """The mic stream never stopped between utterances while the
        session is active (see _stop_and_transcribe, both modes) — resuming
        just means re-checking the focus target and flipping the HUD back.
        No stream restart, no delay, and nothing said during the previous
        utterance's transcription was ever missed."""
        input_ctrl.capture_target_window()
        overlay.set_state("speaking")


    def _stop_and_transcribe() -> None:
        """Grab recorded audio and launch a transcription thread.

        Both modes now auto-cut on a trailing pause and keep recording
        straight through the transcription — cut_recorded_audio() atomically
        grabs+clears the buffer and re-arms the silence detector, so nothing
        the user says in the meantime is lost. They differ only in pause
        tolerance (see _apply_silence_threshold_for_mode). The stream is
        only ever fully stopped via get_recorded_audio()/audio.stop() when
        the session itself is ending (state.session_active already False —
        set by _on_hotkey's deactivate branch before this runs).
        """
        continuing = state.session_active
        try:
            # Captured before cutting — cut_recorded_audio() resets the
            # detector (including speech_frames) for the next utterance.
            speech_frames = audio.silence_detector.speech_frames
            if continuing:
                chunks = audio.cut_recorded_audio()
            else:
                audio.stop()
                chunks = audio.get_recorded_audio()

            # Skip when too short (<0.4s) or when too little of the buffer was
            # actually above the RMS gate — transcribing mostly-silent/noisy
            # audio makes Whisper hallucinate filler phrases (e.g. the classic
            # Korean YouTube-outro line) instead of returning nothing. Gating
            # on speech_frames (cumulative above-gate duration) rather than
            # the old speech_detected boolean matters because that boolean is
            # a one-shot latch — a single short blip (click/cough/creak) sets
            # it True for the whole utterance even if everything after it is
            # near-silent background, which used to let mostly-noise buffers
            # through. Checked BEFORE switching the HUD to "transcoding" so a
            # stray noise blip or the pre-speech grace-period timeout never
            # flashes the "Processing" state for something that was never
            # actually processed. Counting raw samples here (not the
            # DSP-processed length) keeps this check cheap on the main thread
            # — the DSP pipeline itself now runs inside the worker thread, off
            # the GUI thread. 6400 samples was the old threshold at the 16kHz
            # post-resample rate (0.4s); scaled to the mic's native rate so
            # the same 0.4s cutoff still applies to the raw, not-yet-resampled
            # sample count — same scaling applies to the speech-duration gate.
            min_samples = int(0.4 * audio.device_sample_rate)
            min_speech_samples = int(MIN_SPEECH_DURATION_SEC * audio.device_sample_rate)
            total_samples = sum(len(c) for c in chunks)
            if total_samples < min_samples or speech_frames < min_speech_samples:
                # DEBUG, not INFO: in Live mode this fires every ~8s (the
                # pre-speech grace period, MAX_WAIT_FOR_SPEECH_SEC) for as
                # long as the session sits idle with the mic still open —
                # at INFO it was the single largest contributor to app.log's
                # growth, dwarfing every other line combined.
                log.debug("No usable speech captured — skipping transcription")
                if continuing:
                    _resume_speaking()
                else:
                    state.session_active = False
                    overlay.set_state("idle")
                return

            # Distinguish "model still warming up" (can take tens of seconds
            # on a genuinely cold GPU kernel cache, e.g. right after launch —
            # see transcriber.warm_up()) from ordinary per-utterance
            # transcoding (usually well under a second once the model is
            # resident) — otherwise a slow cold warm-up reads as a stuck/
            # broken "Processing" state instead of a one-time loading delay.
            overlay.set_state("transcoding" if transcriber.model_loaded else "loading_model")

            language = str(settings.get("input_language", "ko")).lower()
            beam_size = int(settings.get("beam_size", 5) or 5)
            vad_filter = bool(settings.get("vad_filter", True))
            initial_prompt = settings.get("initial_prompt")

            worker = _TranscribeWorker(
                transcriber,
                audio,
                chunks,
                language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )
            worker.result_ready.connect(_on_transcription_done)
            # Keep the reference until the QThread fully finishes, then let Qt delete it
            worker.finished.connect(worker.deleteLater)
            worker.finished.connect(_release_worker)
            state.worker = worker
            worker.start()
            log.info("Transcription thread started (lang=%s, beam=%d, vad=%s)", language, beam_size, vad_filter)


        except Exception as e:
            log.error("Failed to start transcription thread: %s", e, exc_info=True)
            overlay.set_state("idle")
            if continuing:
                _resume_speaking()
            else:
                state.session_active = False



    def _release_worker() -> None:
        state.worker = None
        # Fires after _on_transcription_done (QThread.finished follows
        # result_ready), so overlay.state is already back to "speaking" for
        # a continuing session by this point (either mode). If the NEXT
        # utterance's silence cutoff already elapsed while we were still
        # transcribing the previous one (recording never stopped), dispatch
        # it now instead of waiting for another silence event that may
        # never come if the user has gone quiet since.
        if (
            state.session_active
            and overlay.state == "speaking"
            and audio.silence_detector.is_silent()
        ):
            _stop_and_transcribe()

    def _on_transcription_done(result: TranscriptionResult) -> None:
        """Handle completed transcription — inject text and decide next step."""
        try:
            if result and result.text and result.text.strip():
                text = result.text.strip()
                prob = (result.language_probability or 0.0) * 100.0
                lang = result.language or "unknown"
                overlay.set_state("input")
                overlay.show_text_preview(result.text)
                # Push the mic-off deadline out to `mic_timer_minutes` from
                # now — restarting the countdown on real speech, rather than
                # firing on a fixed wall-clock schedule since launch/last
                # settings change, means it actually tracks trailing silence
                # since the user last said something. Gated on length so an
                # isolated short interjection (or a hallucination the filter
                # missed) can't keep re-arming an otherwise-idle session.
                if len(text) >= MIC_TIMER_MIN_CHARS:
                    mic_timer.start(overlay._mic_timer_minutes, _force_idle)
                # type_text() blocks on a per-character time.sleep() pacing
                # loop (SendInput/clipboard injection) — on the main thread
                # that freezes the whole Qt event loop (HUD, tray, hotkey
                # queue) for the duration. It doesn't touch any Qt widgets,
                # just Win32 ctypes calls, so it's safe to run off-thread.
                threading.Thread(target=input_ctrl.type_text, args=(result.text,), daemon=True).start()
                log.info(
                    "Transcribed [%s, %.0f%%] in %.2fs: %s",
                    lang,
                    prob,
                    result.processing_time or 0.0,
                    result.text,
                )
            else:
                log.warning("No speech detected (empty text)")
                overlay.show_text_preview("No text detected")
        except Exception as e:
            log.error("Error in transcription completion handler: %s", e, exc_info=True)
            overlay.set_state("idle")

        # Session still active (either mode) → resume immediately (mic
        # never stopped); session ended while we were transcribing → go idle
        if state.session_active:
            _resume_speaking()
        else:
            audio.stop()
            state.session_active = False
            overlay.set_state("idle")



    def _force_idle() -> None:
        """Unconditionally stop everything and go idle."""
        state.session_active = False
        audio.stop()
        overlay.set_state("idle")
        log.info("Forced idle")

    # ------------------------------------------------------------------ #
    # Hotkey toggle
    # ------------------------------------------------------------------ #

    def _on_hotkey(hide_on_deactivate: bool = True) -> None:
        """Toggle the dictation session. `hide_on_deactivate` is False for
        the HUD's own mic-icon click (see overlay.mic_clicked below) — the
        HUD has a dedicated ✕ button for closing itself, so clicking the
        mic to just stop recording shouldn't also hide the window. The
        global hotkey (which has no such separate close affordance) keeps
        the original hide-on-deactivate behavior."""
        log.info("Hotkey action triggered inside main thread. Active session: %s", state.session_active)
        if state.session_active:
            # Deactivate session — HUD goes back to tray-only (unless this
            # came from the mic-icon click)
            state.session_active = False
            if overlay.state == "speaking":
                _stop_and_transcribe()
            elif overlay.state in ("transcoding", "loading_model"):
                # session_active just went False, so Live mode's
                # cut_recorded_audio() path in the in-flight
                # _stop_and_transcribe() call already stopped taking new
                # audio into account for a "continuing" session — but the
                # stream itself is still open (that path deliberately never
                # calls audio.stop()). Stop it now so the mic doesn't keep
                # capturing after the user explicitly ended the session;
                # the in-flight transcription still finishes normally.
                audio.stop()
            else:
                overlay.set_state("idle")
            if hide_on_deactivate:
                overlay.hide()
        else:
            # Activate session — HUD appears
            state.session_active = True
            overlay.show()
            _start_listening()

    # Explicitly use QueuedConnection for cross-thread hotkey signals
    hotkey_bridge.triggered.connect(_on_hotkey, Qt.ConnectionType.QueuedConnection)
    hotkey_mgr.register(hotkey_bridge.fire)
    hotkey_mgr.start(str(settings.get("hotkey", DEFAULT_HOTKEY)))



    # ------------------------------------------------------------------ #
    # Audio signals
    # ------------------------------------------------------------------ #

    audio_bridge.volume_updated.connect(
        lambda rms: overlay.update_volume(_rms_to_levels(rms))
    )

    def _on_silence() -> None:
        # Both modes auto-cut on their own trailing-silence threshold now
        # (Live: short/responsive, One-time: long — see
        # _apply_silence_threshold_for_mode). The session itself still only
        # starts/ends via the hotkey. state.worker is None gate: if a
        # transcription is already running, don't dispatch a second one —
        # recording continues uninterrupted (nothing lost) and
        # _release_worker() picks up this pending silence as soon as the
        # current one finishes.
        if overlay.state == "speaking" and state.worker is None:
            _stop_and_transcribe()

    audio_bridge.silence_detected.connect(_on_silence)

    # ------------------------------------------------------------------ #
    # Overlay UI signals → settings persistence
    # ------------------------------------------------------------------ #

    def _apply_silence_threshold_for_mode() -> None:
        """Both modes auto-cut on a trailing pause now — this just picks
        which threshold currently applies (Live: short, One-time: long)."""
        seconds = (
            overlay._silence_threshold_sec
            if overlay.mode == "live"
            else overlay._silence_threshold_sec_once
        )
        audio.silence_detector.set_threshold_sec(seconds)

    def _on_mode_changed(mode: str) -> None:
        mapped = "one-time" if mode == "once" else mode
        settings.set("mode", mapped)
        _apply_silence_threshold_for_mode()
        log.info("Mode → %s", mapped)

    def _on_device_changed(dev: str) -> None:
        settings.set("device", dev)
        transcriber.set_device(dev)
        log.info("Device mode → %s", dev)

    def _on_low_vram_changed(low_vram: bool) -> None:
        settings.set("low_vram", low_vram)
        transcriber.set_low_vram(low_vram)
        log.info("Low VRAM mode → %s", low_vram)

    def _on_beam_changed(b: int) -> None:
        settings.set("beam_size", b)
        log.info("Saved beam_size → %d", b)

    def _on_vad_changed(v: bool) -> None:
        settings.set("vad_filter", v)
        log.info("Saved vad_filter → %s", v)

    def _on_timer_changed(minutes: int) -> None:
        # Store the raw value (0 = explicitly disabled) rather than None —
        # None is indistinguishable from "never set" on the next load, which
        # silently reset a deliberate "disable the timer" choice back to the
        # 10-minute default on every restart.
        settings.set("mic_off_timer", minutes)
        if minutes > 0:
            mic_timer.start(minutes, _force_idle)
        else:
            mic_timer.cancel()

    def _on_hotkey_combo_changed(combo: str) -> None:
        nonlocal hotkey_label
        if not hotkey_mgr.set_hotkey(combo):
            # Reverted internally to the previous combo — don't persist or
            # relabel a change that didn't actually take effect.
            log.error("Hotkey swap to '%s' failed; kept previous hotkey.", combo)
            return
        settings.set("hotkey", combo)
        hotkey_label = format_hotkey(combo)
        tray_icon.setToolTip(f"Whisper STT (Press {hotkey_label} to Record)")
        log.info("Hotkey → %s", hotkey_label)

    def _on_silence_threshold_changed(seconds: float) -> None:
        settings.set("silence_threshold_sec", seconds)
        _apply_silence_threshold_for_mode()
        log.info("Live silence timeout → %.1fs", seconds)

    def _on_silence_threshold_once_changed(seconds: float) -> None:
        settings.set("silence_threshold_sec_once", seconds)
        _apply_silence_threshold_for_mode()
        log.info("One-time silence timeout → %.1fs", seconds)

    def _on_quit() -> None:
        x, y = overlay.get_position()
        settings.set("window_position", {"x": x, "y": y})
        hotkey_stopped_cleanly = hotkey_mgr.stop()
        audio.stop()
        mic_timer.cancel()

        # Give an in-flight transcription a chance to finish cleanly instead
        # of having Qt force-kill the QThread mid-CTranslate2-call on exit
        if state.worker is not None and state.worker.isRunning():
            log.info("Waiting for in-progress transcription to finish before exit...")
            state.worker.wait(2000)

        # Either a wedged transcription (e.g. a native CT2 call stuck after
        # a device switch) or a lingering pynput listener thread (not a
        # daemon thread — see HotkeyManager.stop()) can keep the process
        # alive as a Task Manager zombie even after app.quit(). Exit App
        # must always actually exit.
        if (state.worker is not None and state.worker.isRunning()) or not hotkey_stopped_cleanly:
            log.warning(
                "Worker or hotkey listener still running after grace period — forcing process exit."
            )
            log.info("Shutting down — settings saved")
            os._exit(0)

        log.info("Shutting down — settings saved")
        app.quit()

    # Restore and sync saved settings into OverlayWindow
    saved_mode = str(settings.get("mode", "live")).lower()
    target_mode = "once" if saved_mode in ("one-time", "once") else "live"
    overlay.set_mode(target_mode)


    overlay._hotkey = str(settings.get("hotkey", DEFAULT_HOTKEY))

    saved_dev = str(settings.get("device", "cuda")).lower()
    overlay._device = saved_dev
    transcriber.set_device(saved_dev)
    log.info("Initialized computation device -> %s", saved_dev)

    overlay._low_vram = saved_low_vram
    transcriber.set_low_vram(saved_low_vram)
    log.info("Initialized low VRAM mode -> %s", saved_low_vram)


    saved_beam = settings.get("beam_size") or 5
    overlay._beam_size = int(saved_beam)


    saved_vad = settings.get("vad_filter")
    overlay._vad_filter = bool(saved_vad if saved_vad is not None else True)

    # `is not None` (not plain truthiness) so a saved 0 — explicitly disabled
    # by the user — is honored instead of being reset back to the 30-min default
    saved_timer = settings.get("mic_off_timer")
    timer_mins = int(saved_timer) if saved_timer is not None else 30
    overlay._mic_timer_minutes = timer_mins
    if timer_mins > 0:
        mic_timer.start(timer_mins, _force_idle)
        log.info("Initialized Mic Off Timer -> %d min", timer_mins)

    # AudioCapture(silence_sec=...) at construction only knew about the Live
    # value — sync both dialog display values, then apply whichever one
    # actually matches the restored mode.
    overlay._silence_threshold_sec = float(settings.get("silence_threshold_sec", 1.8))
    overlay._silence_threshold_sec_once = float(settings.get("silence_threshold_sec_once", 5.0))
    _apply_silence_threshold_for_mode()



    # ------------------------------------------------------------------ #
    # System Tray Icon (Resident App)
    # ------------------------------------------------------------------ #
    from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
    from PyQt6.QtGui import QIcon, QPixmap, QColor, QPainter

    # Prefer the project's own .ico (lives in scripts/, relative to APP_DIR);
    # fall back to a drawn circle if it's ever missing (e.g. a fresh clone
    # without the asset). Used for both the window/taskbar icon and the tray.
    def _create_app_icon() -> QIcon:
        ico_path = APP_DIR / "scripts" / "whisper-stt.ico"
        if ico_path.exists():
            return QIcon(str(ico_path))

        pix = QPixmap(32, 32)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(99, 102, 241))  # Primary Blue
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()
        return QIcon(pix)

    app_icon = _create_app_icon()
    app.setWindowIcon(app_icon)
    tray_icon = QSystemTrayIcon(app_icon, app)
    tray_icon.setToolTip(f"Whisper STT (Press {hotkey_label} to Record)")

    tray_menu = QMenu()
    tray_menu.setStyleSheet(theme.get_stylesheet())
    toggle_hud_act = tray_menu.addAction("Show / Hide HUD")
    settings_act = tray_menu.addAction("Settings...")
    blacklist_act = tray_menu.addAction("Edit Blacklist...")
    tray_menu.addSeparator()
    exit_act = tray_menu.addAction("Exit App")

    # One settings window, all options in one place, instead of navigating
    # nested per-option submenus. Keep a single instance and raise it if
    # already open rather than creating a new one every click.
    settings_dialog: SettingsDialog | None = None

    def _open_settings() -> None:
        nonlocal settings_dialog
        try:
            if settings_dialog is not None and settings_dialog.isVisible():
                # Toggle: pressing the Settings button/menu item again closes
                # it instead of just re-raising an already-open window.
                settings_dialog.close()
                return

            if settings_dialog is None:
                settings_dialog = SettingsDialog(overlay, settings, parent=None)
            settings_dialog.show()
            settings_dialog.raise_()
            settings_dialog.activateWindow()
            # Qt's show()/activateWindow() sometimes leaves the native HWND's
            # WS_VISIBLE flag unset on this app (isVisible() reports True,
            # the window never actually appears — confirmed via direct
            # Win32 ShowWindow/IsWindowVisible probing). Force it natively
            # as a fallback so the dialog is guaranteed to actually surface.
            hwnd = int(settings_dialog.winId())
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            log.exception("Failed to open SettingsDialog")

    settings_act.triggered.connect(_open_settings)

    def _open_blacklist() -> None:
        try:
            if not BLACKLIST_PATH.exists():
                BLACKLIST_PATH.write_text("", encoding="utf-8")
            subprocess.Popen(["notepad.exe", str(BLACKLIST_PATH)])
        except OSError:
            log.exception("Failed to open blacklist.txt in Notepad")

    blacklist_act.triggered.connect(_open_blacklist)

    def _toggle_hud_visibility() -> None:
        if overlay.isVisible():
            overlay.hide()
            _force_idle()
            log.info("HUD hidden to system tray — mic stopped.")
        else:
            overlay.show()
            overlay.raise_()
            overlay.activateWindow()
            log.info("HUD shown from system tray.")

    toggle_hud_act.triggered.connect(_toggle_hud_visibility)
    exit_act.triggered.connect(_on_quit)

    def _on_tray_activated(reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            _toggle_hud_visibility()

    tray_icon.activated.connect(_on_tray_activated)
    # Windows requires a SetForegroundWindow-style handoff immediately before
    # showing a tray icon's context menu, or clicks inside it don't register
    # (the menu appears but nothing happens when you click an item).
    # QSystemTrayIcon.setContextMenu() handles that correctly internally;
    # manually positioning via menu.popup() does not, which is what broke
    # "Settings..." — the menu is only 3 items now anyway, so the taskbar-
    # clipping risk that motivated manual positioning is no longer a concern.
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()

    # ✕ Close button on HUD now hides overlay to system tray instead of quitting
    def _on_close_hud() -> None:
        overlay.hide()
        _force_idle()
        tray_icon.showMessage(
            "Whisper STT Running in Background",
            f"Press {hotkey_label} anytime to talk, or click tray icon to restore HUD.",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    overlay.quit_requested.connect(_on_close_hud)

    overlay.mode_changed.connect(_on_mode_changed)
    overlay.device_changed.connect(_on_device_changed)
    overlay.low_vram_changed.connect(_on_low_vram_changed)
    overlay.beam_changed.connect(_on_beam_changed)
    overlay.vad_changed.connect(_on_vad_changed)
    overlay.timer_changed.connect(_on_timer_changed)
    overlay.hotkey_changed.connect(_on_hotkey_combo_changed)
    overlay.silence_threshold_changed.connect(_on_silence_threshold_changed)
    overlay.silence_threshold_once_changed.connect(_on_silence_threshold_once_changed)
    # HUD mic-icon click: toggle recording without hiding the HUD — the ✕
    # button is the only thing that should close it (see _on_hotkey above).
    overlay.mic_clicked.connect(lambda: _on_hotkey(hide_on_deactivate=False))
    overlay.settings_requested.connect(_open_settings)



    log.info("Whisper STT ready — press %s to toggle recording", hotkey_label)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
