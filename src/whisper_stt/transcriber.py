"""Faster-whisper engine wrapper for transcription with safe device switching."""

from dataclasses import dataclass, field
import gc
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, List

import numpy as np

from whisper_stt.config import (
    BLACKLIST_PATH,
    FILLER_REPEAT_SYLLABLES,
    HALLUCINATION_PHRASES,
    HF_HUB_ETAG_TIMEOUT_SEC,
    MODEL_DIR,
    MODEL_ID,
    SAMPLE_RATE,
)

# Bound how long huggingface_hub waits on its "is this the latest revision?"
# check before falling back to the local cache — avoids a flaky connection
# stalling model load on every launch
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_SEC))

# Add nvidia pip package DLL directories (cublas, cudnn) for Windows.
# Scan sys.path rather than site.getsitepackages() — the latter reflects
# the RUNNING interpreter's own configured site-packages, which is wrong
# whenever this app ends up executed by a different pythonw.exe than the
# project's own .venv (e.g. Windows' Python launcher/app-execution-alias
# shim resolving to the global install instead). whisper-stt.pyw always
# prepends the venv's site-packages to sys.path itself, so scanning
# sys.path finds the real nvidia/cublas folder regardless of which
# interpreter binary is actually running.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for site_dir in dict.fromkeys(sys.path):  # de-dup, preserve order
        nvidia_path = Path(site_dir) / "nvidia"
        if nvidia_path.exists():
            for sub in nvidia_path.iterdir():
                for bin_folder in (sub / "bin", sub / "lib"):
                    if bin_folder.exists():
                        try:
                            os.add_dll_directory(str(bin_folder))
                            os.environ["PATH"] = str(bin_folder) + os.path.pathsep + os.environ.get("PATH", "")
                        except Exception:
                            pass

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[\s.,!?~…·]+")


def _normalize_for_hallucination_check(text: str) -> str:
    return _PUNCT_RE.sub("", text).lower()


_HALLUCINATION_SET = frozenset(_normalize_for_hallucination_check(p) for p in HALLUCINATION_PHRASES)

# mtime-cached loader for BLACKLIST_PATH (see config.py) — a user-hand-edited
# JSON array of extra phrases to drop, same exact-match semantics as
# _HALLUCINATION_SET. Re-parsed only when the file's mtime changes so calling
# this once per transcribe() stays a single cheap stat() in the common case.
_user_blacklist_cache: dict[str, Any] = {"mtime": None, "set": frozenset()}


def _load_user_blacklist() -> frozenset[str]:
    try:
        mtime = BLACKLIST_PATH.stat().st_mtime
    except OSError:
        try:
            BLACKLIST_PATH.write_text("[]\n", encoding="utf-8")
        except OSError:
            pass
        return frozenset()

    if mtime == _user_blacklist_cache["mtime"]:
        return _user_blacklist_cache["set"]

    try:
        entries = json.loads(BLACKLIST_PATH.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("blacklist.json must be a JSON array of strings")
        normalized = frozenset(
            _normalize_for_hallucination_check(p) for p in entries if isinstance(p, str) and p.strip()
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.error("Failed to load blacklist.json (%s) — treating as empty until fixed", e)
        normalized = frozenset()

    _user_blacklist_cache["mtime"] = mtime
    _user_blacklist_cache["set"] = normalized
    return normalized

# Matches a segment made ENTIRELY of one filler syllable repeated 2+ times
# (e.g. "음음음", "어어") with nothing else — see config.FILLER_REPEAT_SYLLABLES.
_FILLER_REPEAT_RE = re.compile(
    "^(?:" + "|".join(FILLER_REPEAT_SYLLABLES) + "){2,}$"
)

# CPU device cap: use half the logical cores so a CPU transcription doesn't
# saturate every core on the machine (irrelevant on CUDA — CTranslate2 only
# consults this for CPU execution).
_CPU_THREADS = max(1, (os.cpu_count() or 4) // 2)


@dataclass
class TranscriptionResult:
    """Data class holding transcription results."""
    text: str
    segments: List[dict[str, Any]] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    processing_time: float = 0.0


class TranscriberEngine:
    """Wrapper for the faster-whisper model supporting safe device switching.

    On CUDA, ``low_vram`` swaps ``float16`` weights for ``int8_float16``
    (int8-quantized weights, fp16 compute) — roughly half the VRAM footprint
    of plain float16 for a small, usually imperceptible accuracy cost. Useful
    on 8GB cards; large-v3 in plain float16 is already fairly tight there
    once CUDA context + decode-time activation memory are added on top of
    the ~3GB of model weights.
    """

    def __init__(self, device: str = "cuda", low_vram: bool = False) -> None:
        self._model: WhisperModel | None = None
        self._device = device.lower()
        self._low_vram = low_vram
        self._compute_type = self._resolve_compute_type()
        self._lock = threading.Lock()
        # Separate, short-lived lock just for the (device, compute_type,
        # model) triple — distinct from self._lock, which transcribe() holds
        # for the whole transcription (seconds). Without this, set_device()/
        # set_low_vram() (called on the main thread, deliberately NOT taking
        # self._lock — see below) could interleave with _get_model()'s read
        # of self._device/self._compute_type mid-update, e.g. reading a new
        # device with a stale compute_type. This lock is only ever held for
        # plain attribute reads/writes, never across the slow model load.
        self._state_lock = threading.Lock()
        self._verify_device()

    def _resolve_compute_type(self) -> str:
        if self._device != "cuda":
            return "int8"
        return "int8_float16" if self._low_vram else "float16"

    def _verify_device(self) -> None:
        if self._device == "cuda":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() == 0:
                    logger.warning("No CUDA devices found. Falling back to CPU.")
                    self._device = "cpu"
            except ImportError:
                self._device = "cpu"

        self._compute_type = self._resolve_compute_type()
        logger.info(f"Transcriber engine configured for {self._device} with {self._compute_type}")

    def set_device(self, device: str) -> None:
        """Safely switch computation device between 'cuda' and 'cpu'.

        Deliberately does NOT take ``self._lock`` — that lock is held by
        ``transcribe()`` for the whole duration of a transcription (including
        a several-second model reload), and this is called synchronously
        from the main/UI thread when the user flips the Settings dialog
        radio. Blocking on it there froze the entire app until any in-flight
        transcription finished. ``self._state_lock`` below is a separate,
        short-lived lock guarding only the metadata swap, never the slow
        model load — an in-flight transcribe() call keeps using the model
        it already fetched, and the next call picks up the new device via
        the None check in _get_model().
        """
        target_device = device.lower()
        if target_device not in ("cuda", "cpu"):
            logger.warning(f"Unsupported device '{device}' requested.")
            return

        with self._state_lock:
            if self._device == target_device:
                return

            logger.info(f"Safely switching computation device to {target_device}...")
            self._device = target_device
            self._verify_device()

            # Explicitly cleanup C++ CTranslate2 model memory and force GC to prevent Access Violation
            if self._model is not None:
                self._model = None
                gc.collect()
                logger.info("Previous model memory safely garbage collected.")

    def set_low_vram(self, low_vram: bool) -> None:
        """Toggle int8_float16 (low VRAM) vs plain float16 on CUDA.

        Same reasoning as set_device() above — uses only the short-lived
        ``self._state_lock``, never ``self._lock``, to avoid freezing the
        UI thread against an in-flight transcribe().
        """
        with self._state_lock:
            if self._low_vram == low_vram:
                return

            self._low_vram = low_vram
            old_compute_type = self._compute_type
            self._compute_type = self._resolve_compute_type()
            logger.info(f"Low VRAM mode → {low_vram} (compute_type {old_compute_type} -> {self._compute_type})")

            if self._compute_type != old_compute_type and self._model is not None:
                self._model = None
                gc.collect()
                logger.info("Previous model memory safely garbage collected (compute_type changed).")

    @property
    def model_loaded(self) -> bool:
        """Non-blocking check for HUD feedback only (see main.py's
        "loading_model" vs "transcoding" state split) — doesn't take
        self._lock, so it can't stall behind an in-flight warm_up()/
        transcribe() call. A plain bool read/write is safe without a lock
        under the GIL; the small window where this could be stale by one
        _get_model() call is harmless for a UI hint.
        """
        return self._model is not None

    def warm_up(self) -> None:
        """Pre-load the model AND run one dummy inference, off the calling
        thread.

        Call this once from a background thread right after startup so both
        are likely done by the time the user finishes their first utterance,
        without delaying tray-icon startup itself.

        Loading the weights (_get_model()) alone isn't enough to absorb the
        real first-run cost: CTranslate2 only touches the cuBLAS/cuDNN DLLs
        (~2GB combined, under .venv/Lib/site-packages/nvidia) on the first
        actual GEMM/conv call, not at model construction. Right after a
        Windows boot those DLLs are cold on disk (and re-scanned by Windows
        Defender, since .venv isn't a trusted path) — measured ~17s for that
        first real inference on this machine, even though model load itself
        stayed ~3s. A plain zeros buffer runs the exact same kernels a real
        utterance would, so this pays that cost here instead of on the
        user's first sentence. Every launch after the first-post-boot one
        finds the DLLs already in the OS page cache and this dummy call
        finishes in well under a second.
        """
        with self._lock:
            model = self._get_model()
            if model is None:
                return
            try:
                dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence
                segments, _ = model.transcribe(
                    dummy_audio,
                    language="ko",
                    task="transcribe",
                    beam_size=1,
                    vad_filter=False,
                    without_timestamps=True,
                )
                list(segments)  # force the lazy generator to actually run
            except Exception:
                logger.exception("Warm-up dummy inference failed (non-fatal)")

    def _get_model(self) -> WhisperModel | None:
        """Lazy load the model, with CUDA→CPU fallback.

        Snapshots (device, compute_type) under self._state_lock — coherent
        with whatever set_device()/set_low_vram() last wrote — then releases
        it before the several-second WhisperModel(...) construction, so a
        concurrent device switch is never blocked on the load itself.
        """
        with self._state_lock:
            if self._model is not None:
                return self._model
            device, compute_type = self._device, self._compute_type

        logger.info(f"Loading model {MODEL_ID} on {device} ({compute_type})...")
        start_time = time.time()
        try:
            model = WhisperModel(
                MODEL_ID, device=device, compute_type=compute_type,
                cpu_threads=_CPU_THREADS if device == "cpu" else 0,
                download_root=str(MODEL_DIR),
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model on {device}: {e}")
            if device != "cuda":
                return None
            logger.warning("Falling back to CPU...")
            try:
                model = WhisperModel(
                    MODEL_ID, device="cpu", compute_type="int8",
                    cpu_threads=_CPU_THREADS, download_root=str(MODEL_DIR),
                )
                device, compute_type = "cpu", "int8"
            except Exception as ex:
                logger.error(f"CPU fallback also failed: {ex}")
                return None

        with self._state_lock:
            if self._device == device and self._compute_type == compute_type:
                self._model = model
            else:
                # set_device()/set_low_vram() changed the target while this
                # load was in flight — don't cache a model for a config
                # that's no longer current; the next _get_model() call will
                # load correctly for whatever is current now. This call's
                # in-flight transcription still uses the model it just built.
                logger.info(
                    "Config changed during model load (%s/%s -> %s/%s) — not caching.",
                    device, compute_type, self._device, self._compute_type,
                )
        return model

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: str | None = "ko",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio with thread lock safety during device switches."""
        start_time = time.time()

        with self._lock:
            try:
                model = self._get_model()
                if model is None:
                    raise RuntimeError("Model is not loaded.")

                vad_params = {
                    "threshold": 0.35,
                    "min_speech_duration_ms": 200,
                    "min_silence_duration_ms": 450,
                    "speech_pad_ms": 400,
                }

                segments_iter, info = model.transcribe(
                    audio_data,
                    language=language,
                    task="transcribe",
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    without_timestamps=True,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    vad_parameters=vad_params if vad_filter else None,
                )

                # Loaded once per transcribe() call (not per segment) — a
                # single stat() in the common case where blacklist.json
                # hasn't changed since the last call.
                user_blacklist = _load_user_blacklist()

                segments = []
                full_text = []
                for segment in segments_iter:
                    # Drop segments that are an exact match (modulo
                    # whitespace/punctuation) for a known hallucination
                    # filler — see config.HALLUCINATION_PHRASES, or a
                    # user-added entry in config.BLACKLIST_PATH. Per-segment
                    # rather than whole-text so a real utterance with a
                    # hallucinated trailing segment keeps everything else.
                    normalized = _normalize_for_hallucination_check(segment.text)
                    if normalized in _HALLUCINATION_SET:
                        logger.info("Dropped known hallucination segment: %r", segment.text)
                        continue
                    if normalized in user_blacklist:
                        logger.info("Dropped custom blacklist segment: %r", segment.text)
                        continue
                    if _FILLER_REPEAT_RE.match(normalized):
                        logger.info("Dropped repeated-filler hallucination segment: %r", segment.text)
                        continue
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                    full_text.append(segment.text)

                text = " ".join(full_text).strip()
                processing_time = time.time() - start_time

                result = TranscriptionResult(
                    text=text,
                    segments=segments,
                    language=info.language if info else "",
                    language_probability=info.language_probability if info else 0.0,
                    duration=info.duration if info else 0.0,
                    processing_time=processing_time
                )

                logger.debug(f"Transcription complete in {processing_time:.2f}s: {text}")
                return result

            except Exception as e:
                logger.error(f"Transcription failed: {e}", exc_info=True)
                return TranscriptionResult(
                    text="",
                    processing_time=time.time() - start_time
                )
