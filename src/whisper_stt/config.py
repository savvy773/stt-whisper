"""Configuration constants for Whisper STT."""

from pathlib import Path

# ============================================================================
# HAND-TUNABLE CONSTANTS — no UI control, no settings.json entry. Edit the
# value below and restart the app to change. These are Python-level (not
# per-user JSON) because they're either startup-only (MODEL_ID, paths) or
# rarely-touched engine-tuning knobs (VAD_EXIT_RATIO, HIGHPASS_HZ) rather
# than something a user would flip per-session. For the handful of "golden
# value" tuning knobs that ARE JSON-backed but still hidden from the
# Settings dialog (input_language, initial_prompt, silence_rms_threshold),
# see settings.py's DEFAULT_SETTINGS — those can be edited there OR directly
# in settings.json without any code change.
# ============================================================================
SAMPLE_RATE = 16000
MAX_WAIT_FOR_SPEECH_SEC = 8.0   # Grace period to START speaking before giving up (avoids HUD flapping on hotkey-press hesitation)
VAD_EXIT_RATIO = 0.70           # Lower RMS gate while speaking so mid-word dips don't split
HIGHPASS_HZ = 80.0              # Drop 80Hz AC hum/desk rumble before decoding
MIN_SPEECH_DURATION_SEC = 0.2   # Minimum cumulative above-gate audio required before transcribing at
                                 # all (matches transcriber.py's own vad_params min_speech_duration_ms).
                                 # Blocks the case where a single short noise blip (click/cough/creak)
                                 # crosses the RMS gate once and the rest of the buffer is near-silent
                                 # background — sending that to Whisper makes it hallucinate filler
                                 # phrases instead of returning nothing.
MIC_TIMER_MIN_CHARS = 3         # Minimum transcribed length (~syllables) that counts as "the user is
                                 # still talking" for the mic-off timer (main.py's _MicOffTimer). The
                                 # timer restarts from now on every utterance that clears this bar, so
                                 # it measures trailing silence since the last real thing said, not wall
                                 # -clock time since launch/settings change. A single short interjection
                                 # ("어", "네") or a hallucination-filtered empty result stays below it
                                 # and doesn't push the deadline out.

# Known Whisper hallucination fillers — what the model falls back to when
# forced to decode audio that cleared the MIN_SPEECH_DURATION_SEC gate above
# (so it wasn't pure silence/a blip) but still isn't real intelligible
# speech, e.g. sustained background noise/chatter. faster-whisper's own
# confidence heuristics (no_speech_threshold, and the log_prob/
# compression_ratio thresholds it already applies at their library defaults)
# don't reliably catch this — these specific phrases are common Korean
# video/ad filler the model over-fits to. transcriber.py drops any segment
# that's an exact match after stripping whitespace/punctuation, so only
# segments essentially identical to these get removed — real dictated
# content (even content that happens to mention these topics in a longer
# sentence) is never touched.
HALLUCINATION_PHRASES = (
    "안녕하세요 반갑습니다",
    "시청해주셔서 감사합니다",
    "배달의민족",
    "thank you for watching",
    "한글자막 by 한효정",
)

# Same hallucination class as above but with a variable repeat count the
# model produces (e.g. "음음음", "음음", "어어어") — a fixed-string blocklist
# entry can't cover every count, so transcriber.py matches this as a regex
# instead: one or more of these single filler syllables, repeated 2+ times,
# with nothing else in the segment. A single occurrence ("음") is left alone
# since that's a real, meaningful short utterance (backchannel "mm"/"uh"),
# not a hallucination — only the repeated-with-nothing-else form is safe to drop.
FILLER_REPEAT_SYLLABLES = ("음", "어", "으", "흠", "엄")

# Built-in default hotkey combo, applied on fresh install and by the
# Settings dialog's "Set Default" button. The dialog's Hotkey row
# (hud/settings_dialog.py's _HOTKEY_OPTIONS) offers this plus one
# alternate — only one combo is ever active at a time. pynput combo
# syntax, e.g.: "<ctrl>+<f9>", "<ctrl>+<alt>+<space>", "<alt>+<shift>+r"
DEFAULT_HOTKEY = "<ctrl>+<alt>+<space>"


# Network resilience: cap how long huggingface_hub waits on its startup
# "is this the latest revision?" check before falling back to the local cache
# (prevents a flaky connection from stalling model load / app startup)
HF_HUB_ETAG_TIMEOUT_SEC = 3

# ============================================================================
# Structural — not really "tuning," don't touch without a reason.
# ============================================================================
MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"

APP_DIR = Path(__file__).resolve().parents[2]
SETTINGS_PATH = APP_DIR / "settings.json"

# User-editable hallucination blocklist — JSON array of strings, hand-edited
# directly (no restart needed, transcriber.py hot-reloads on mtime change).
# Separate from HALLUCINATION_PHRASES above because that list is a code
# constant tuned for hallucinations common across installs; this one is for
# phrases specific to this mic/room/model that only show up on this machine
# (e.g. "아멘") — adding one here doesn't need a code change or PR.
BLACKLIST_PATH = APP_DIR / "blacklist.json"

# Cache downloaded model weights under the project folder instead of the
# global ~/.cache/huggingface — keeps the app self-contained and survives a
# HF cache wipe elsewhere on the machine.
MODEL_DIR = APP_DIR / "models"
