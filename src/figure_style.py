"""Shared physical-size and typography contract for paper figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


TEXT_WIDTH_IN = 5.5

# Directory holding the Source Sans Pro OTF files (any TeX Live install has them under
# texmf-dist/fonts/opentype/adobe/sourcesanspro). Override with CICM_FONT_DIR; if the
# files are not found, matplotlib falls back to its default sans-serif font.
FONT_DIR = Path(os.environ.get("CICM_FONT_DIR", "texmf-dist/fonts/opentype/adobe/sourcesanspro"))
FONT_FILES = (
    "SourceSansPro-Regular.otf",
    "SourceSansPro-Semibold.otf",
    "SourceSansPro-Bold.otf",
    "SourceSansPro-RegularIt.otf",
)


def register_paper_font() -> str:
    """Register the exact embeddable font files used by the paper figures."""
    for filename in FONT_FILES:
        path = FONT_DIR / filename
        if path.is_file():
            font_manager.fontManager.addfont(path)
    regular = FONT_DIR / "SourceSansPro-Regular.otf"
    if not regular.is_file():
        raise FileNotFoundError(f"Paper font not found: {regular}")
    return font_manager.FontProperties(fname=regular).get_name()


FIGURE_FONT = register_paper_font()

COLORS = {
    "correct": "#384653",
    "stale": "#D87558",
    "cross": "#3D829B",
    "current": "#4D8B69",
    "other": "#A4A8AD",
    "ink": "#24292F",
    "muted": "#68717A",
    "grid": "#D9DEE3",
    "paper": "#FAFAF8",
    "null": "#E5E8EC",
}

# These palettes encode ordered context length or model identity. They are kept
# separate from the current/old/other semantic colors above.
LENGTH_COLORS = ("#9FB9CF", "#5F87A7", "#2E5D7B")
MODEL_COLORS = ("#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#79706E")


def apply_paper_style() -> None:
    """Set legible sizes at the ICLR template's final 5.5-inch width."""
    plt.rcParams.update(
        {
            "font.family": FIGURE_FONT,
            "font.size": 8.4,
            "font.weight": "regular",
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.7,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
