"""High-fidelity audio capture with smart mic selection and precise 16kHz resampling."""

import logging
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from whisper_stt.config import (
    HIGHPASS_HZ,
    MAX_WAIT_FOR_SPEECH_SEC,
    SAMPLE_RATE,
    VAD_EXIT_RATIO,
)

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Detects end-of-utterance with hysteresis (lower gate while speaking).

    Two different cutoffs apply depending on whether speech has started yet:
    before the first word, a long grace period tolerates hesitation after the
    hotkey press; once speech has begun, a short trailing-silence window ends
    the utterance promptly — natural pauses between sentences are what
    segment a long live-mode session, no separate max-duration cap needed.
    """

    def __init__(
        self,
        threshold_rms: float,
        threshold_sec: float,
        sample_rate: int,
        max_wait_sec: float = MAX_WAIT_FOR_SPEECH_SEC,
    ) -> None:
        self.threshold_rms = threshold_rms
        self.threshold_sec = threshold_sec
        self.sample_rate = sample_rate
        self.max_wait_sec = max_wait_sec
        self.silence_frames_threshold = int(threshold_sec * sample_rate)
        self.max_wait_frames = int(max_wait_sec * sample_rate)
        self.current_silence_frames = 0
        self.is_speaking = False
        self.speech_detected = False
        # Cumulative frames actually above the RMS gate this utterance —
        # unlike speech_detected (a one-shot latch that never un-sets once
        # any single blip crosses the gate), this tracks how much of the
        # buffer was really speech-level. A single click/cough/creak can
        # set speech_detected=True and then be followed by seconds of
        # near-silent background noise; sending that whole buffer to
        # Whisper makes it hallucinate plausible-sounding filler (e.g. the
        # classic Korean YouTube-outro phrase) on the low-information audio.
        # main.py gates on this instead of speech_detected before deciding
        # whether to transcribe at all.
        self.speech_frames = 0

    def feed(self, chunk: np.ndarray) -> bool:
        if len(chunk) == 0:
            return False

        rms = np.sqrt(np.mean(chunk**2))

        # Hysteresis: while already speaking, use lower gate (0.70x) so mid-word dips don't split speech
        effective_gate = self.threshold_rms * VAD_EXIT_RATIO if self.is_speaking else self.threshold_rms

        if rms >= effective_gate:
            self.is_speaking = True
            self.speech_detected = True
            self.speech_frames += len(chunk)
            self.current_silence_frames = 0
        else:
            self.current_silence_frames += len(chunk)

        limit = self.silence_frames_threshold if self.speech_detected else self.max_wait_frames
        if self.current_silence_frames >= limit:
            self.is_speaking = False
            return True

        return False

    def reset(self) -> None:
        self.current_silence_frames = 0
        self.is_speaking = False
        self.speech_detected = False
        self.speech_frames = 0

    def set_threshold_sec(self, seconds: float) -> None:
        """Update the trailing-silence cutoff at runtime (Settings dialog)."""
        self.threshold_sec = seconds
        self.silence_frames_threshold = int(seconds * self.sample_rate)

    def set_sample_rate(self, sample_rate: int) -> None:
        """Recompute frame thresholds after the capture device's sample
        rate changes (e.g. AudioCapture.restart() re-discovers a
        different device) — the frame counts above are only valid for the
        rate they were computed against."""
        self.sample_rate = sample_rate
        self.silence_frames_threshold = int(self.threshold_sec * sample_rate)
        self.max_wait_frames = int(self.max_wait_sec * sample_rate)

    def is_silent(self) -> bool:
        """Whether the current silence run has already crossed the active
        cutoff (mirrors the condition inside feed(), without consuming a
        new chunk) — used to check for a pending cutoff that elapsed while
        a transcription was in flight and recording never stopped."""
        limit = self.silence_frames_threshold if self.speech_detected else self.max_wait_frames
        return self.current_silence_frames >= limit


class AudioCapture:
    """Manages audio recording with smart microphone auto-discovery and high-quality resampling."""

    def __init__(
        self,
        # Real values always come from main.py via SettingsManager
        # (settings.py's DEFAULT_SETTINGS is the actual source of truth for
        # these two); these literals only matter if AudioCapture() is ever
        # constructed without explicit args.
        silence_rms: float = 0.028,
        silence_sec: float = 1.8,
    ) -> None:
        self.stream: sd.InputStream | None = None
        self.is_recording = False
        self._audio_buffer: list[np.ndarray] = []
        self._callback_error_count = 0
        self._last_callback_ts = 0.0

        self.on_audio_data: Callable[[np.ndarray], None] | None = None
        self.on_volume_level: Callable[[float], None] | None = None

        self.device_index: int | None = None
        self.device_name: str = "Default Microphone"
        self.device_sample_rate: int = 44100
        self.channels: int = 1

        self._discover_best_microphone()

        self.silence_detector = SilenceDetector(
            threshold_rms=silence_rms,
            threshold_sec=silence_sec,
            sample_rate=self.device_sample_rate,
        )

    def _discover_best_microphone(self) -> None:
        """Discover active USB/External microphone (e.g. MATA STUDIO C10) or default input device."""
        try:
            devices = sd.query_devices()
            best_idx = None

            # Priority search for USB/dedicated microphones
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    name = d.get("name", "").lower()
                    if "mata" in name or "c10" in name or "usb" in name:
                        best_idx = idx
                        break

            # Fallback to system default input device
            if best_idx is None:
                default_input = sd.default.device[0]
                if default_input is not None and default_input >= 0:
                    best_idx = default_input

            if best_idx is not None:
                info = sd.query_devices(best_idx)
                self.device_index = best_idx
                self.device_name = info.get("name", "Microphone")
                self.device_sample_rate = int(info.get("default_samplerate", 44100))
                self.channels = min(int(info.get("max_input_channels", 1)), 2)
                logger.info("Selected Microphone [%d]: '%s' at %d Hz (%d channels)",
                            self.device_index, self.device_name, self.device_sample_rate, self.channels)
            else:
                logger.warning("No input devices found!")
        except Exception as e:
            logger.error("Error discovering microphone: %s", e)
            self.device_sample_rate = 44100

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        # Recorded regardless of what follows — this alone proves the
        # stream is still alive, which is what the health-check watchdog
        # in main.py relies on to tell "dead stream" apart from "no one's
        # talking right now".
        self._last_callback_ts = time.monotonic()

        if status:
            logger.warning("Audio stream status: %s", status)

        try:
            # Convert multi-channel (stereo) to mono if needed
            if indata.ndim > 1 and indata.shape[1] > 1:
                chunk = np.mean(indata, axis=1)
            else:
                chunk = indata[:, 0]

            self._audio_buffer.append(chunk.copy())

            rms = float(np.sqrt(np.mean(chunk**2)))
            if self.on_volume_level:
                self.on_volume_level(rms)

            if self.on_audio_data:
                self.on_audio_data(chunk)
        except Exception:
            # sounddevice registers this callback with error=paAbort (see
            # its cffi trampoline): any uncaught exception here — not just
            # ones we raise on purpose — permanently aborts the PortAudio
            # stream with no exception reaching sys.excepthook (this runs
            # on PortAudio's own C thread) and no further callbacks ever
            # again, i.e. a silently dead mic. Log once, not every call —
            # this fires ~100x/sec, so a recurring error would otherwise
            # turn into a disk-I/O storm.
            self._callback_error_count += 1
            if self._callback_error_count == 1:
                logger.exception("Audio callback error (suppressing repeats)")

    def start(self) -> bool:
        """Open and start the input stream. Returns whether it's actually
        recording afterward — callers must check this instead of assuming
        success, since a transient device-open failure (e.g. WASAPI not
        ready yet right after a fresh Windows boot) used to be silently
        swallowed here while the caller went on to show a normal
        "Listening…" HUD with a dead mic behind it."""
        if self.is_recording:
            return True

        self._audio_buffer.clear()
        self.silence_detector.reset()
        self._callback_error_count = 0
        self._last_callback_ts = time.monotonic()

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.device_sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_recording = True
            logger.info("Started mic capture on '%s' (%d Hz)", self.device_name, self.device_sample_rate)
            return True
        except Exception as e:
            logger.error("Failed to start audio stream on device %s: %s", self.device_index, e)
            self.stream = None
            self.is_recording = False
            return False

    def restart(self) -> bool:
        """Recover a stalled/dead stream: close it, re-discover the input
        device (in case it was replugged, or wasn't ready yet — the same
        fresh-boot timing this whole file is trying to be resilient to),
        resync the silence detector to any sample-rate change, and reopen.
        """
        logger.warning("Restarting audio stream (recovery)")
        self.stop()
        self._discover_best_microphone()
        self.silence_detector.set_sample_rate(self.device_sample_rate)
        return self.start()

    def seconds_since_last_callback(self) -> float:
        """How long since the PortAudio callback last fired. Only
        meaningful while is_recording — 0.0 otherwise so callers don't
        mistake "never started" for "just stalled"."""
        if not self.is_recording:
            return 0.0
        return time.monotonic() - self._last_callback_ts

    def stop(self) -> None:
        if not self.is_recording or self.stream is None:
            return

        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            logger.error("Failed to stop audio stream: %s", e)
        finally:
            # Always clear state even if stop()/close() raised (e.g. the
            # stream was already aborted by PortAudio) — otherwise
            # is_recording stays stuck True and every future start() call
            # silently no-ops forever.
            self.stream = None
            self.is_recording = False
        logger.info("Stopped mic capture.")

    def get_recorded_audio(self) -> list[np.ndarray]:
        """Return the raw recorded chunks (call after stop()).

        Does NOT run the DSP pipeline — that's real CPU work (tens of ms,
        more for long recordings) that must not run on the calling thread
        when the caller is the main Qt thread. Call process_chunks() on the
        result, off the GUI thread (see _TranscribeWorker in main.py).
        """
        return self._audio_buffer

    def cut_recorded_audio(self) -> list[np.ndarray]:
        """Atomically grab+clear the buffer WITHOUT stopping the stream.

        Used by Live mode between utterances: the mic keeps recording
        straight through transcription, so nothing the user says while the
        previous utterance is being transcribed is lost. Also re-arms the
        silence detector for the next utterance. Returns raw chunks — same
        DSP-deferral reasoning as get_recorded_audio() above; the swap +
        detector reset must stay cheap since this runs on the main thread.
        """
        chunks, self._audio_buffer = self._audio_buffer, []
        self.silence_detector.reset()
        return chunks

    def process_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Shared DSP pipeline: high-pass → anti-alias low-pass → 16kHz
        resample → peak normalization.

        Pure function of `chunks` (only reads self.device_sample_rate, never
        mutates instance state) — safe to call from any thread, including
        the transcription worker thread it's meant to run on.
        """
        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float64)
        if len(audio) == 0:
            return audio.astype(np.float32)

        sr = self.device_sample_rate

        # High-pass via moving-average subtraction (removes DC offset and
        # <~80Hz rumble). Box-filter moving average computed via cumsum in
        # O(N) instead of np.convolve's O(N*window) direct convolution —
        # verified bit-for-bit equivalent to convolve(..., mode="same") for
        # both odd and even window sizes (see scratchpad verification), just
        # far cheaper for the ~500-tap window this works out to at 44.1kHz.
        hp_window = int(sr / HIGHPASS_HZ)
        if hp_window > 1 and len(audio) > hp_window:
            start = (hp_window - 1) // 2
            pad_left = hp_window - 1 - start
            pad_right = hp_window - 1 - pad_left
            padded = np.concatenate([np.zeros(pad_left), audio, np.zeros(pad_right)])
            cs = np.concatenate([[0.0], np.cumsum(padded)])
            moving_avg = (cs[hp_window:] - cs[:-hp_window]) / hp_window
            audio = audio - moving_avg

        if sr != SAMPLE_RATE:
            # Anti-alias low-pass (windowed-sinc FIR) before decimation —
            # bare linear interpolation aliases HF noise into the speech band
            cutoff = 0.45 * SAMPLE_RATE
            taps = 101
            t = np.arange(taps) - taps // 2
            sinc = np.sinc(2.0 * cutoff / sr * t) * np.hamming(taps)
            sinc /= sinc.sum()
            audio = np.convolve(audio, sinc, mode="same")

            num_target = int(round(len(audio) * (SAMPLE_RATE / sr)))
            audio = np.interp(
                np.linspace(0, len(audio), num_target, endpoint=False),
                np.arange(len(audio)),
                audio,
            )
            logger.debug("Resampled %d Hz -> 16000 Hz (%d samples)", sr, num_target)

        # Peak normalization (gain capped so near-silence isn't blown up into noise)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            audio = audio * min(0.95 / peak, 20.0)

        return audio.astype(np.float32)
