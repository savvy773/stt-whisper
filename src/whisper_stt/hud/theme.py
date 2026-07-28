"""
Design tokens, styling constants, and QSS stylesheet generator for the HUD overlay.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG = "rgba(10, 10, 15, 0.85)"
SURFACE = "rgba(255, 255, 255, 0.08)"
PRIMARY = "#6366f1"
PRIMARY_LIGHT = "#818cf8"
ACCENT = "#22d3ee"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"
TEXT = "#f8fafc"
TEXT_MUTED = "#94a3b8"

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
HUD_WIDTH = 240
HUD_HEIGHT_COLLAPSED = 84  # two rows: status label on top, controls below (was 60/one row)
HUD_HEIGHT_EXPANDED = 180
BORDER_RADIUS = 20
EQ_BAR_COUNT = 12
EQ_BAR_WIDTH = 4
EQ_BAR_GAP = 3

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILY = "Segoe UI"
FONT_SIZE_SMALL = 10
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 16


def get_stylesheet() -> str:
    """Return the main QSS stylesheet string for the overlay and its children."""
    return f"""
        /* ---- Global ---- */
        QWidget {{
            font-family: "{FONT_FAMILY}";
            font-size: {FONT_SIZE_NORMAL}px;
            color: {TEXT};
        }}

        /* ---- Status label ---- */
        QLabel#statusLabel {{
            font-size: {FONT_SIZE_NORMAL}px;
            font-weight: 500;
            color: {TEXT};
            padding: 0 4px;
        }}

        /* ---- Mode indicator pill (clickable) ---- */
        QLabel#modeIndicator {{
            font-size: {FONT_SIZE_SMALL}px;
            font-weight: 700;
            letter-spacing: 1px;
            padding: 2px 8px;
            border-radius: 8px;
        }}
        QLabel#modeIndicator[mode="live"] {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {PRIMARY}, stop:1 {ACCENT}
            );
            color: {TEXT};
            border: 1px solid transparent;
        }}
        QLabel#modeIndicator[mode="once"] {{
            background: {SURFACE};
            color: {TEXT_MUTED};
            border: 1px solid {TEXT_MUTED};
        }}
        QLabel#modeIndicator:hover {{
            border: 1px solid {ACCENT};
        }}

        /* ---- Context menu ---- */
        #textPreview {{
            background-color: rgba(15, 15, 20, 0.95);
            color: #f8fafc;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 8px 12px;
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
        }}

        #closeButton {{
            background: transparent;
            color: #94a3b8;
            border: none;
            font-size: 13px;
            font-weight: bold;
            padding: 2px 4px;
            border-radius: 10px;
        }}
        #closeButton:hover {{
            color: #f87171;
            background: rgba(248, 113, 113, 0.15);
        }}

        #settingsButton {{
            background: transparent;
            color: #94a3b8;
            border: none;
            font-size: 13px;
            padding: 2px 4px;
            border-radius: 10px;
        }}
        #settingsButton:hover {{
            color: {TEXT};
            background: rgba(255, 255, 255, 0.1);
        }}

        QMenu {{
            background-color: rgba(18, 18, 24, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 8px 0;
            font-size: {FONT_SIZE_NORMAL}px;
            color: {TEXT};
        }}
        QMenu::item {{
            padding: 9px 28px 9px 14px;
        }}
        QMenu::item:selected {{
            background: {SURFACE};
        }}
        QMenu::separator {{
            height: 1px;
            background: rgba(255, 255, 255, 0.08);
            margin: 6px 10px;
        }}
        QMenu::indicator {{
            width: 14px;
            height: 14px;
            margin-left: 6px;
        }}
        QMenu::indicator:checked {{
            image: none;
            background: {PRIMARY};
            border-radius: 7px;
        }}
        QMenu::indicator:unchecked {{
            image: none;
            background: transparent;
            border: 1px solid {TEXT_MUTED};
            border-radius: 7px;
        }}

        /* ---- Tooltip / text preview ---- */
        QLabel#textPreview {{
            font-size: {FONT_SIZE_SMALL}px;
            color: {TEXT_MUTED};
            padding: 4px 10px;
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
        }}
    """
