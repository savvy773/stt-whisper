"""
Direct pythonw launcher for Whisper STT with Exception Dialog & Crash Logging.
Smart path resolution: Works whether placed in project root or subdirectories.
"""

import ctypes
import os
import sys
import traceback
from pathlib import Path

# Smart project directory resolution
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "src" / "whisper_stt").exists():
    PROJECT_DIR = SCRIPT_DIR
else:
    PROJECT_DIR = SCRIPT_DIR.parent

VENV_DIR = PROJECT_DIR / ".venv"
SRC_DIR = PROJECT_DIR / "src"
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CRASH_LOG_PATH = LOGS_DIR / "crash.log"



def log_exception_and_show_dialog(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions to crash.log and display a Windows error dialog."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- CRASH AT {Path(__file__).name} ---\n{error_msg}\n")
    except Exception:
        pass

    # Show Windows Error Message Box so user sees exact error instead of silent exit
    try:
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Whisper STT encountered an error:\n\n{exc_value}\n\nCheck crash.log for details.",
            "Whisper STT Error",
            0x00000010  # MB_ICONERROR
        )
    except Exception:
        pass


# Register uncaught exception handler
sys.excepthook = log_exception_and_show_dialog


# Auto-elevate to Admin if not running as Admin (Required for Windows Admin Console hotkeys & UIPI)
def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


if not is_admin():
    pythonw_exe = VENV_DIR / "Scripts" / "pythonw.exe"
    if not pythonw_exe.exists():
        pythonw_exe = Path(sys.executable).parent / "pythonw.exe"

    try:
        res = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            str(pythonw_exe),
            f'"{__file__}"',
            str(PROJECT_DIR),
            0  # SW_HIDE
        )
        if res > 32:
            sys.exit(0)
    except Exception as e:
        print(f"Admin elevation failed: {e}")

# 1. Add virtualenv site-packages to sys.path
if sys.platform == "win32":
    site_packages = VENV_DIR / "Lib" / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))

# 2. Add src directory to sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 3. Add DLL search directory for CUDA / CTranslate2 on Windows
if hasattr(os, "add_dll_directory") and VENV_DIR.exists():
    bindir = VENV_DIR / "Scripts"
    if bindir.exists():
        try:
            os.add_dll_directory(str(bindir))
        except Exception:
            pass

from whisper_stt.main import main

if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        log_exception_and_show_dialog(exc_type, exc_value, exc_tb)
