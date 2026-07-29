"""Robust and crash-free global hotkey listener for Whisper STT using pynput GlobalHotKeys."""

import ctypes
import logging
from typing import Callable
from pynput import keyboard

from whisper_stt.config import DEFAULT_HOTKEY

logger = logging.getLogger(__name__)

_DISPLAY_OVERRIDES = {"page_up": "Page Up", "page_down": "Page Down", "cmd": "Win"}

# --------------------------------------------------------------------------
# Fire-time "no extra modifiers held" guard — see _on_trigger() below for why
# this is needed on top of pynput's own matching.
# --------------------------------------------------------------------------

_VK_CONTROL = 0x11
_VK_MENU = 0x12  # Alt
_VK_SHIFT = 0x10
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C

_user32 = ctypes.windll.user32
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]


def _is_down(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def _currently_held_modifiers() -> set[str]:
    """Query Windows directly (not pynput's own internal state) for which
    modifier keys are physically down right now."""
    held = set()
    if _is_down(_VK_CONTROL):
        held.add("ctrl")
    if _is_down(_VK_MENU):
        held.add("alt")
    if _is_down(_VK_SHIFT):
        held.add("shift")
    if _is_down(_VK_LWIN) or _is_down(_VK_RWIN):
        held.add("cmd")
    return held


def format_hotkey(combo: str) -> str:
    """pynput combo string ("<ctrl>+<alt>+<space>") -> human label
    ("Ctrl+Alt+Space") for tray tooltips, messages, and the Settings
    dialog's option labels."""
    labels = []
    for part in combo.split("+"):
        if part.startswith("<") and part.endswith(">"):
            name = part[1:-1]
            labels.append(_DISPLAY_OVERRIDES.get(name, name.capitalize()))
        else:
            labels.append(part.upper())
    return "+".join(labels)


class HotkeyManager:
    """Manages the single fixed global hotkey (DEFAULT_HOTKEY) reliably
    without ctypes thread exceptions."""

    def __init__(self) -> None:
        self._callback: Callable[[], None] | None = None
        self._listener: keyboard.GlobalHotKeys | None = None
        self._current: str = DEFAULT_HOTKEY

    def register(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def start(self, combo: str = DEFAULT_HOTKEY) -> bool:
        if self._listener is not None:
            return True

        try:
            parsed_keys = keyboard.HotKey.parse(combo)
        except ValueError as e:
            logger.error("Invalid hotkey combo '%s': %s", combo, e)
            return False

        required_mods = {
            name for name, key in (
                ("ctrl", keyboard.Key.ctrl),
                ("alt", keyboard.Key.alt),
                ("shift", keyboard.Key.shift),
                ("cmd", keyboard.Key.cmd),
            )
            if key in parsed_keys
        }
        # The combo's non-modifier key(s) (e.g. Space), by virtual-key code —
        # see the GetAsyncKeyState re-check below for why these are verified
        # too, not just the modifiers.
        required_main_vks = {
            key.vk for key in parsed_keys
            if isinstance(key, keyboard.KeyCode) and key.vk is not None
        }

        def _on_trigger():
            # pynput's own HotKey.press() fires as soon as this combo's
            # required keys are all down, regardless of any OTHER key also
            # currently held — its internal _state can only ever be a
            # subset of _keys, never a strict superset, since press()
            # ignores any key not in _keys entirely. That means a shorter
            # combo (e.g. Ctrl+Space) also fires while physically pressing
            # a longer one that contains it (e.g. Ctrl+Alt+Space), because
            # performing the longer combo necessarily also performs the
            # shorter one along the way. Guard against this by checking, at
            # the moment of activation, the ACTUAL modifier state queried
            # directly from Windows (not from pynput's own tracking) —
            # only fire if it's exactly this combo's required set, no more.
            held = _currently_held_modifiers()
            if held != required_mods:
                logger.debug(
                    "Hotkey chord matched but extra/missing modifiers held (%s != %s) — ignoring.",
                    held, required_mods,
                )
                return
            # Windows swallows the key-up for Win when it's used in a
            # system-reserved shortcut (e.g. Win+Plus/Minus for Magnifier),
            # so pynput's internal press-state can be left thinking Win (and
            # whatever combo used it last) is still held, and a later
            # unrelated key then spuriously completes this combo. The
            # modifier re-check above can't catch this on its own since Win
            # may still be genuinely down; also confirm the combo's actual
            # non-modifier key (e.g. Space) is physically down right now.
            if required_main_vks and not all(_is_down(vk) for vk in required_main_vks):
                logger.debug(
                    "Hotkey chord matched but main key not physically down — ignoring spurious trigger.",
                )
                return
            logger.info("Global hotkey triggered!")
            if self._callback:
                try:
                    self._callback()
                except Exception as e:
                    logger.error("Error in hotkey callback: %s", e)

        try:
            self._listener = keyboard.GlobalHotKeys({combo: _on_trigger})
            self._listener.start()
            self._current = combo
            logger.info("Global hotkey listener started (%s).", combo)
            return True
        except Exception as e:
            logger.error("Failed to start GlobalHotKeys listener: %s", e)
            return False

    def set_hotkey(self, combo: str) -> bool:
        """Live-swap the active combo: stop the current listener and start
        a new one on `combo`. Reverts to the previous combo (rather than
        leaving the app with zero working hotkeys) if the swap can't be
        confirmed safe — either the old listener won't confirm stopped (see
        stop()'s docstring on why starting a second listener alongside a
        possibly-still-alive one is unsafe) or the new combo fails to start.
        """
        previous = self._current
        if not self.stop():
            logger.error(
                "Could not confirm old hotkey listener stopped — refusing to swap to '%s'.", combo,
            )
            return False

        if self.start(combo):
            return True

        logger.error("Failed to start hotkey '%s' — reverting to '%s'.", combo, previous)
        self.start(previous)
        return False

    def stop(self) -> bool:
        """Stop the listener. Returns False if the underlying pynput thread
        is still alive after a short grace join — in that case `self._listener`
        is deliberately left as-is (still tracked as active) rather than
        cleared. main.py's _on_quit() force-exits the process on a False
        return.

        pynput never marks its Listener thread as a daemon thread (it
        inherits daemon=False from whatever thread calls start(), which
        here is the main thread) — so if it's ever slow to notice the stop
        signal (e.g. a backlog of queued keyboard events it hasn't drained
        yet), a plain app.quit() leaves pythonw.exe running as a zombie
        process in Task Manager even after the tray icon/HUD have already
        vanished.
        """
        if self._listener is None:
            return True

        try:
            self._listener.stop()
            self._listener.join(timeout=1.5)
        except Exception as e:
            logger.error("Error stopping GlobalHotKeys listener: %s", e)
            return False

        if self._listener.is_alive():
            logger.warning("Global hotkey listener thread still alive after grace period.")
            return False

        self._listener = None
        logger.info("Global hotkey listener stopped.")
        return True
