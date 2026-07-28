"""Short synthesized UI feedback tones for HUD show/hide.

Pure sine tones with a fast attack/release envelope, generated on the fly —
no audio asset files, no new dependency (sounddevice is already used for
mic capture elsewhere in the app).
"""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
_AMPLITUDE = 0.15  # quiet — this is a UI accent, not an alert
_ATTACK_RELEASE_SEC = 0.01


def _tone(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)

    ramp = max(1, int(_SAMPLE_RATE * _ATTACK_RELEASE_SEC))
    envelope = np.ones_like(wave)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)

    return (wave * envelope * _AMPLITUDE).astype(np.float32)


def _play(samples: np.ndarray) -> None:
    try:
        sd.play(samples, _SAMPLE_RATE)
    except Exception:
        # Never let a missing/busy output device break the HUD show/hide itself.
        logger.debug("UI sound playback failed (non-fatal)", exc_info=True)


def play_show() -> None:
    """Soft rising two-note chime — HUD appearing."""
    gap = np.zeros(int(_SAMPLE_RATE * 0.015), dtype=np.float32)
    _play(np.concatenate([_tone(740.0, 0.07), gap, _tone(988.0, 0.10)]))


def play_hide() -> None:
    """Soft falling two-note chime — HUD disappearing."""
    gap = np.zeros(int(_SAMPLE_RATE * 0.015), dtype=np.float32)
    _play(np.concatenate([_tone(988.0, 0.07), gap, _tone(659.0, 0.10)]))
