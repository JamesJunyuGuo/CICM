#!/usr/bin/env python3
"""Render the paper's reader-facing overview at its final ICLR width."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_figure_style import COLORS, TEXT_WIDTH_IN, apply_paper_style


def rounded_box(ax, x, y, width, height, *, face, edge="#D9DEE3", radius=0.012):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=0.65, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    return patch


def panel_frame(ax, x, width, accent):
    rounded_box(ax, x, 0.04, width, 0.92, face="#FBFCFC", edge="#E1E5E8")
    ax.plot([x + 0.008, x + width - 0.008], [0.945, 0.945], color=accent,
            linewidth=1.2, solid_capstyle="round")


def panel_heading(ax, x, letter, title, accent, tint):
    rounded_box(ax, x, 0.846, 0.032, 0.057, face=tint, edge=tint, radius=0.008)
    ax.text(x + 0.016, 0.875, letter, fontsize=8.3, fontweight="semibold",
            color=accent, ha="center", va="center")
    ax.text(x + 0.043, 0.875, title, fontsize=9.1, fontweight="semibold",
            color=COLORS["ink"], va="center")


def draw_example(ax):
    x, width = 0.012, 0.302
    panel_frame(ax, x, width, COLORS["stale"])
    panel_heading(ax, x + 0.016, "A", "An old value wins",
                  COLORS["stale"], "#FCF1ED")

    rows = [
        (0.660, "Earlier", "meal = keto", COLORS["stale"], "#FCF1ED"),
        (0.515, "Update", "meal = gluten-free", COLORS["current"], "#EEF6F1"),
    ]
    for y, role, text, accent, fill in rows:
        rounded_box(ax, x + 0.018, y, width - 0.036, 0.102, face=fill, edge="#DEE3E6")
        ax.plot([x + 0.025, x + 0.025], [y + 0.012, y + 0.090], color=accent,
                linewidth=1.15)
        ax.text(x + 0.040, y + 0.051, role, fontsize=7.2, color=COLORS["muted"],
                va="center")
        ax.text(x + 0.112, y + 0.051, text, fontsize=8.0, color=COLORS["ink"],
                va="center")

    rounded_box(ax, x + 0.018, 0.380, width - 0.036, 0.078,
                face="#F3F5F6", edge="#DDE2E5")
    ax.text(x + 0.035, 0.419, "Query: current meal?", fontsize=7.9,
            color=COLORS["ink"], va="center")
    ax.add_patch(FancyArrowPatch(
        (x + width / 2, 0.374), (x + width / 2, 0.326), arrowstyle="-|>",
        mutation_scale=7.5, linewidth=0.75, color=COLORS["muted"],
    ))
    ax.text(x + 0.018, 0.282, "Answer", fontsize=7.2,
            color=COLORS["muted"], va="center")
    rounded_box(ax, x + 0.090, 0.242, width - 0.108, 0.080,
                face="#FCF1ED", edge=COLORS["stale"])
    ax.text(x + 0.110, 0.282, "keto", fontsize=8.7, fontweight="semibold",
            color=COLORS["stale"], va="center")
    ax.text(x + 0.018, 0.105, "Right variable, old value", fontsize=7.8,
            color=COLORS["stale"], fontweight="semibold", va="center")


def draw_competition(ax):
    x, width = 0.349, 0.302
    panel_frame(ax, x, width, COLORS["cross"])
    panel_heading(ax, x + 0.016, "B", "Conflicting updates",
                  COLORS["cross"], "#EDF5F7")

    ax.text(x + 0.018, 0.765, "Matched context length", fontsize=7.5,
            color=COLORS["muted"], va="center")

    rows = [
        (0.605, "Extra text", "answers stay\nreliable", COLORS["current"]),
        (0.425, "Old values", "accuracy\nfalls", COLORS["stale"]),
    ]
    for y, left_text, right_text, accent in rows:
        rounded_box(ax, x + 0.018, y, 0.118, 0.105, face="#F4F6F7")
        ax.text(x + 0.077, y + 0.052, left_text, fontsize=7.7, ha="center",
                va="center", color=COLORS["ink"], wrap=True)
        ax.add_patch(FancyArrowPatch(
            (x + 0.145, y + 0.052), (x + 0.181, y + 0.052),
            arrowstyle="-|>", mutation_scale=7.5, linewidth=0.75, color=COLORS["muted"],
        ))
        rounded_box(ax, x + 0.188, y, 0.095, 0.105,
                    face="#EEF6F1" if accent == COLORS["current"] else "#FCF1ED",
                    edge=accent)
        ax.text(x + 0.236, y + 0.052, right_text, fontsize=7.5,
                ha="center", va="center", color=accent, fontweight="semibold", wrap=True)

    ax.plot([x + 0.018, x + width - 0.018], [0.337, 0.337],
            color="#DFE3E6", linewidth=0.65)
    ax.text(x + 0.018, 0.275, "Same length, different outcome",
            fontsize=7.7, color=COLORS["ink"], fontweight="semibold", va="center")
    ax.text(x + 0.018, 0.135, "Across model families and scales",
            fontsize=7.8, color=COLORS["ink"], fontweight="semibold", va="center")
    ax.text(x + 0.018, 0.092, "Scaling shifts the onset",
            fontsize=7.5, color=COLORS["muted"], va="center")


def draw_selection(ax):
    x, width = 0.686, 0.302
    panel_frame(ax, x, width, COLORS["current"])
    panel_heading(ax, x + 0.016, "C", "Available, not selected",
                  COLORS["current"], "#EEF6F1")

    ax.text(x + 0.018, 0.765, "Both values remain available", fontsize=7.5,
            color=COLORS["muted"], va="center")
    rounded_box(ax, x + 0.018, 0.625, 0.130, 0.105,
                face="#EEF6F1", edge=COLORS["current"])
    ax.text(x + 0.083, 0.700, "Current value", fontsize=7.0,
            color=COLORS["muted"], ha="center", va="center")
    ax.text(x + 0.083, 0.655, "gluten-free", fontsize=7.7,
            color=COLORS["current"], fontweight="semibold", ha="center", va="center")
    rounded_box(ax, x + 0.162, 0.625, 0.122, 0.105,
                face="#FCF1ED", edge=COLORS["stale"])
    ax.text(x + 0.223, 0.700, "Old value", fontsize=7.0,
            color=COLORS["muted"], ha="center", va="center")
    ax.text(x + 0.223, 0.655, "keto", fontsize=7.7,
            color=COLORS["stale"], fontweight="semibold", ha="center", va="center")

    ax.add_patch(FancyArrowPatch(
        (x + 0.223, 0.617), (x + 0.165, 0.525), arrowstyle="-|>",
        mutation_scale=8, linewidth=1.1, color=COLORS["stale"],
    ))
    ax.add_patch(FancyArrowPatch(
        (x + 0.083, 0.617), (x + 0.140, 0.525), arrowstyle="-|>",
        mutation_scale=7, linewidth=0.65, linestyle="--", color=COLORS["current"],
    ))
    rounded_box(ax, x + 0.078, 0.425, 0.150, 0.096, face="#F4F6F7")
    ax.text(x + 0.153, 0.473, "Answer selection", fontsize=7.9,
            color=COLORS["ink"], fontweight="semibold", ha="center", va="center")
    ax.add_patch(FancyArrowPatch(
        (x + 0.153, 0.421), (x + 0.153, 0.377), arrowstyle="-|>",
        mutation_scale=7.5, linewidth=0.75, color=COLORS["muted"],
    ))
    rounded_box(ax, x + 0.078, 0.292, 0.150, 0.082,
                face="#FCF1ED", edge=COLORS["stale"])
    ax.text(x + 0.153, 0.333, "selected: keto", fontsize=8.0,
            color=COLORS["stale"], fontweight="semibold", ha="center", va="center")

    ax.plot([x + 0.018, x + width - 0.018], [0.240, 0.240],
            color="#DFE3E6", linewidth=0.65)
    rounded_box(ax, x + 0.018, 0.085, width - 0.036, 0.105,
                face="#EEF6F1", edge="#D8E7DD")
    ax.text(x + width / 2, 0.155, "Late targeted change", fontsize=7.0,
            color=COLORS["muted"], ha="center", va="center")
    ax.text(x + width / 2, 0.115, "current answer restored", fontsize=7.7,
            color=COLORS["current"], fontweight="semibold", ha="center", va="center")


def render(out_dir: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.25))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.01)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_example(ax)
    draw_competition(ax)
    draw_selection(ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F1_memory_selection_overview.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_dir / "F1_memory_selection_overview.png", dpi=300,
                bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("paper/full_draft_assets/figures"))
    args = parser.parse_args()
    render(args.out)


if __name__ == "__main__":
    main()
