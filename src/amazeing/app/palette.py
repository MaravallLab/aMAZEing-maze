"""One colour vocabulary for the interface and the figures.

A stimulus category keeps the same colour in the application as it has in the
session figures, so a blue bar in a results figure and a blue trace in the
stimulus preview mean the same thing. The grammar categories come straight
from ``amazeing.auditory.analysis`` so there is a single source of truth; the
categories the other experiment modes use are added here in the same spirit
(cool colours for smooth or consonant sounds, warm for rough or dissonant,
green for vocalisations, grey for silence).
"""

from __future__ import annotations

from typing import Dict

from amazeing.auditory.analysis import STIM_COLORS

# Extra categories for the non-grammar modes.
_EXTRA: Dict[str, str] = {
    # temporal envelope modulation
    "smooth": "#1565C0",
    "rough": "#EF6C00",
    "rough_complex": "#B71C1C",
    # consonant / dissonant intervals
    "consonant": "#2E7D32",
    "dissonant": "#C62828",
    "unison": "#5E35B1",
    # custom mode and plain tones
    "tone": "#1565C0",
    "am_tone": "#EF6C00",
    "wav": "#00838F",
    "interval": "#5E35B1",
    "pattern": "#6A1B9A",
    "control": "#546E7A",
    "silent": "#9E9E9E",
    "vocalisation": "#2E7D32",
    "unknown": "#78909C",
}

STIMULUS_COLORS: Dict[str, str] = {**_EXTRA, **STIM_COLORS}


def colour_for(category: str) -> str:
    """Colour for a stimulus category, falling back to a neutral grey."""
    if not category:
        return STIMULUS_COLORS["unknown"]
    if category in STIMULUS_COLORS:
        return STIMULUS_COLORS[category]
    return STIMULUS_COLORS["unknown"]


def is_dark_theme() -> bool:
    """True when the application palette is darker than mid grey."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return False
        return app.palette().window().color().lightness() < 128
    except Exception:
        return False


def figure_style() -> Dict[str, object]:
    """matplotlib rcParams that make a figure sit inside the window's theme."""
    if is_dark_theme():
        fg, bg, grid = "#E0E0E0", "#1E1E1E", "#4F4F4F"
    else:
        fg, bg, grid = "#202020", "#FFFFFF", "#C8C8C8"
    return {
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.edgecolor": grid,
        "xtick.color": fg,
        "ytick.color": fg,
        "grid.color": grid,
        "axes.titlecolor": fg,
        "font.size": 8,
    }
