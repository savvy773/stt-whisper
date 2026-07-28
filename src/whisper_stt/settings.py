"""Settings manager for Whisper STT with fully exposed, user-editable parameters."""

import json
import logging
import os
from threading import Lock
from typing import Any

from whisper_stt.config import DEFAULT_HOTKEY, SETTINGS_PATH

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings with persistent JSON storage."""

    DEFAULT_SETTINGS: dict[str, Any] = {
        # ================================================================
        # HIDDEN / EXPERT TUNING — no UI control in the Settings dialog.
        # These are the "golden values" tuned over many earlier debugging
        # rounds (see docs/ARCHITECTURE.md). To change one: either hand-edit
        # the value below (takes effect for a fresh settings.json, i.e. a
        # clean install) or edit the key directly in settings.json on disk
        # (takes effect on next app launch — no code change needed either
        # way, since everything in this dict is just a JSON-backed default).
        # ================================================================
        "input_language": "ko",  # Spoken language passed to Whisper (avoids auto-detect misfires)
        "initial_prompt": "한국어 대화 및 영어 단어가 포함된 문장 (VS Code, Python, Whisper, API). 매끄러운 띄어쓰기와 한영 혼용 작성.",  # Korean context prompt for high accuracy
        "silence_rms_threshold": 0.028,  # Volume threshold to trigger speech detection

        # ================================================================
        # Exposed in the Settings dialog (tray icon → Settings...) —
        # editing these here only changes the default for a fresh install;
        # normal changes should go through the dialog, not this file.
        # ================================================================
        "window_position": None,  # dict {"x": int, "y": int} or None
        "settings_dialog_geometry": None,  # dict {"x","y","w","h"} or None
        "hotkey": DEFAULT_HOTKEY,  # pynput combo string — see hotkey.py's SettingsDialog options
        "mode": "live",  # 'live' or 'one-time'
        "mic_off_timer": 30,  # minutes of trailing silence before auto-off; 0 = disabled
        "device": "cuda",  # 'cuda' or 'cpu'
        "low_vram": False,  # True = int8_float16 compute (~half VRAM), for 8GB-class GPUs
        "beam_size": 5,  # 3 (balanced), 5 (optimal accuracy - DEFAULT)
        "vad_filter": True,  # Silero VAD noise filtering
        # Both modes now auto-cut on a trailing pause and keep recording
        # straight through — the only difference is how patient each is
        # before committing a chunk. Live: short/responsive. One-time:
        # long, so a mid-thought pause doesn't fragment the sentence.
        "silence_threshold_sec": 1.8,  # Live mode pause tolerance
        "silence_threshold_sec_once": 5.0,  # One-time mode pause tolerance
    }

    def __init__(self) -> None:
        """Initialize the settings manager with default values."""
        self._lock = Lock()
        self._settings: dict[str, Any] = dict(self.DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        """Load settings from the JSON file."""
        with self._lock:
            if not SETTINGS_PATH.exists():
                logger.info(f"Settings file not found. Creating with defaults at {SETTINGS_PATH}")
                self._save_unlocked()
                return

            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)

                # Update settings with loaded values
                for key, value in loaded_settings.items():
                    self._settings[key] = value
                logger.info(f"Loaded settings from {SETTINGS_PATH}")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")

    def save(self) -> None:
        """Save current settings to the JSON file."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        # Write to a temp file and atomically replace the real one — every
        # settings.set() call re-saves the whole file, and main.py's
        # _on_quit() sometimes force-exits via os._exit(0) (a wedged worker
        # or hotkey thread). A write-in-place truncates the file first, so a
        # kill landing mid-write leaves settings.json invalid, silently
        # resetting every saved preference (hotkey, position, mode, ...) back
        # to defaults on next launch. os.replace() is atomic on Windows/NTFS
        # — the old file is never observed in a half-written state.
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, SETTINGS_PATH)
            logger.info(f"Saved settings to {SETTINGS_PATH}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional fallback default."""
        with self._lock:
            if default is not None:
                return self._settings.get(key, default)
            return self._settings.get(key, self.DEFAULT_SETTINGS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Set a setting value and save immediately."""
        with self._lock:
            self._settings[key] = value
        self.save()
