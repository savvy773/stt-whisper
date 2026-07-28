"""Universal Windows Text Injector with Native Win32 Clipboard & SendInput."""

import ctypes
from ctypes import wintypes
import logging
import threading
import time

logger = logging.getLogger(__name__)

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", Input_I)]


INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56
CF_UNICODETEXT = 13


class InputController:
    """Universal text injection controller supporting Win32 native clipboard and SendInput."""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._last_target_hwnd: int = 0
        # Registered once; used to mark clipboard fallback content as
        # excluded from Win+V Clipboard History and Cloud Clipboard sync,
        # since dictated text is transient input, not something the user
        # meant to save.
        self._cf_exclude_monitor = self._user32.RegisterClipboardFormatW(
            "ExcludeClipboardContentFromMonitorProcessing"
        )
        self._cf_can_include_history = self._user32.RegisterClipboardFormatW(
            "CanIncludeInClipboardHistory"
        )
        self._cf_can_upload_cloud = self._user32.RegisterClipboardFormatW(
            "CanUploadToCloudClipboard"
        )
        # main.py fires type_text() on its own daemon thread per transcription
        # result so injection never blocks the Qt thread. With continuous
        # auto-cut recording, two results can land in quick succession —
        # without this lock, two SendInput loops would interleave characters
        # into the target window.
        self._type_lock = threading.Lock()

    def capture_target_window(self) -> None:
        """Call before recording to remember user's active window."""
        hwnd = self._user32.GetForegroundWindow()
        if hwnd:
            self._last_target_hwnd = hwnd
            logger.debug("Captured target window HWND: %s", hwnd)

    def type_text(self, text: str) -> None:
        """Inject text into whatever window currently has focus.

        The user may switch windows while speaking or during transcription,
        so the *current* foreground window always wins; the window captured
        at recording-start is only a fallback for when there's momentarily
        no valid foreground window (e.g. an activation race with the HUD).
        """
        if not text:
            return

        with self._type_lock:
            target_hwnd = self._user32.GetForegroundWindow()
            if not target_hwnd or not self._user32.IsWindow(target_hwnd):
                target_hwnd = self._last_target_hwnd

            if target_hwnd and self._user32.IsWindow(target_hwnd):
                try:
                    self._user32.SetForegroundWindow(target_hwnd)
                    time.sleep(0.08)
                except Exception:
                    pass

            logger.info("Injecting text (%d chars): '%s'", len(text), text)

            # 2. Try Direct SendInput (Unicode) — Instant typing
            if self._type_sendinput_unicode(text):
                return

            # 3. Fallback to Win32 Native Clipboard + Ctrl+V
            self._type_win32_clipboard(text)

    def _set_clipboard_format_dword(self, fmt: int, value: int) -> None:
        """Attach a DWORD-valued clipboard format to the currently open clipboard."""
        if not fmt:
            return
        h_mem = self._kernel32.GlobalAlloc(0x0042, ctypes.sizeof(wintypes.DWORD))
        if not h_mem:
            return
        p_mem = self._kernel32.GlobalLock(h_mem)
        ctypes.memmove(p_mem, ctypes.byref(wintypes.DWORD(value)), ctypes.sizeof(wintypes.DWORD))
        self._kernel32.GlobalUnlock(h_mem)
        self._user32.SetClipboardData(fmt, h_mem)

    def _type_sendinput_unicode(self, text: str) -> bool:
        """Inject characters one at a time as Unicode keyboard events.

        A small delay between characters gives the target window's message
        loop time to process each keystroke — sending many events in one
        large batch can silently drop keystrokes in slower-processing apps.
        Returns False when Windows blocks the injection (SendInput reports
        fewer events than sent, e.g. UIPI against an elevated window) so the
        caller can fall back to clipboard paste.
        """
        try:
            # UTF-16 code units, so surrogate pairs (emoji etc.) inject correctly
            code_units: list[int] = []
            for ch in text:
                encoded = ch.encode("utf-16-le")
                for i in range(0, len(encoded), 2):
                    code_units.append(int.from_bytes(encoded[i:i + 2], "little"))

            extra = ctypes.c_ulong(0)
            for cu in code_units:
                ii_down = Input_I(ki=KeyBdInput(0, cu, KEYEVENTF_UNICODE, 0, ctypes.pointer(extra)))
                ii_up = Input_I(ki=KeyBdInput(0, cu, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra)))

                inputs = (Input * 2)(Input(INPUT_KEYBOARD, ii_down), Input(INPUT_KEYBOARD, ii_up))
                sent = self._user32.SendInput(2, ctypes.pointer(inputs[0]), ctypes.sizeof(Input))
                if sent != 2:
                    logger.warning("SendInput injected %d/2 events — blocked (UIPI?)", sent)
                    return False
                time.sleep(0.003)

            logger.info("Injected text directly via SendInput Unicode.")
            return True
        except Exception as e:
            logger.warning("SendInput Unicode failed: %s", e)
            return False

    def _type_win32_clipboard(self, text: str) -> None:
        """Fallback: Win32 native OpenClipboard + SendInput Ctrl+V."""
        try:
            # Set clipboard data natively via Win32 API
            if self._user32.OpenClipboard(None):
                self._user32.EmptyClipboard()
                
                # Allocate global memory
                encoded_text = text.encode("utf-16le") + b"\x00\x00"
                h_mem = self._kernel32.GlobalAlloc(0x0042, len(encoded_text))
                if h_mem:
                    p_mem = self._kernel32.GlobalLock(h_mem)
                    ctypes.memmove(p_mem, encoded_text, len(encoded_text))
                    self._kernel32.GlobalUnlock(h_mem)
                    self._user32.SetClipboardData(CF_UNICODETEXT, h_mem)

                # Mark this clipboard content as dictation scratch data, not
                # something to remember — keeps it out of Win+V Clipboard
                # History and Cloud Clipboard sync.
                self._set_clipboard_format_dword(self._cf_exclude_monitor, 1)
                self._set_clipboard_format_dword(self._cf_can_include_history, 0)
                self._set_clipboard_format_dword(self._cf_can_upload_cloud, 0)

                self._user32.CloseClipboard()

                time.sleep(0.05)

                # Send Ctrl + V
                extra = ctypes.c_ulong(0)
                ctrl_down = Input(INPUT_KEYBOARD, Input_I(ki=KeyBdInput(VK_CONTROL, 0, 0, 0, ctypes.pointer(extra))))
                v_down = Input(INPUT_KEYBOARD, Input_I(ki=KeyBdInput(VK_V, 0, 0, 0, ctypes.pointer(extra))))
                v_up = Input(INPUT_KEYBOARD, Input_I(ki=KeyBdInput(VK_V, 0, KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))))
                ctrl_up = Input(INPUT_KEYBOARD, Input_I(ki=KeyBdInput(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))))

                inputs = (Input * 4)(ctrl_down, v_down, v_up, ctrl_up)
                self._user32.SendInput(4, ctypes.pointer(inputs[0]), ctypes.sizeof(Input))
                logger.info("Injected text via Win32 Native Clipboard Paste.")

                # Give the target app time to actually read the clipboard on
                # paste before clearing it — SendInput only queues the
                # keystrokes, it doesn't wait for them to be processed.
                time.sleep(0.15)
                if self._user32.OpenClipboard(None):
                    self._user32.EmptyClipboard()
                    self._user32.CloseClipboard()
        except Exception as e:
            logger.error("Win32 Clipboard injection failed: %s", e)
