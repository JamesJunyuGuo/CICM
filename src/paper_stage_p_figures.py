#!/usr/bin/env python3
"""Paper-only Stage-P figures built from frozen analysis summaries.

The script emits three independent manuscript assets:

* FigP_timing: when replacing the successful run's key/query repairs errors;
* FigP_event_trace: how answer preference, attention, key matching, and
  current-value readability change as later context is processed;
* FigP_six_model: paired-prompt attention and key-matching shifts across models.

All values are read directly from saved Stage-P summaries. No inference or
statistical recomputation is performed here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paper_figure_style import COLORS, TEXT_WIDTH_IN, apply_paper_style


CHECKPOINTS = ("tau2", "tau3", "tau4", "tauQ")
CHECKPOINT_LABELS = ("after\nupdate", "later\ntext", "old\nmention", "final\nquery")
MODEL_SPECS = (
    ("Pythia-160M", "pythia160"),
    ("Pythia-1.4B", "pythia14"),
    ("Qwen2.5-1.5B", "qwen15"),
    ("Qwen2.5-7B", "qwen7"),
    ("Llama-3.1-8B", "llama8"),
    ("Gemma-2-2B", "gemma2"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(ax, label: str, x: float = -0.12) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="semibold",
        color=COLORS["ink"],
    )


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _trajectory(analysis: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    control, error = [], []
    for checkpoint in CHECKPOINTS:
        row = analysis["trajectories"][checkpoint][metric]["raw"]
        control.append(row["correct_mean"])
        error.append(row["stale_mean"])
    return np.asarray(control), np.asarray(error)


def _clean_comparison_axis(ax) -> None:
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def timing_figure(root: Path, out_dir: Path) -> None:
    """Figure 8: matched early-versus-final patching comparison."""
    patch = load_json(
        root / "results/stage_p/models/pythia160/patch_cpu_full/patch.summary.json"
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.75), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.27, wspace=0.42)

    specifications = (
        ("a", "Immediately after the update", ("early_key", "early_query"), COLORS["current"]),
        ("b", "At the final query", ("final_key", "final_query"), COLORS["stale"]),
    )
    labels = ("Replace current key", "Replace query")
    y = np.asarray([1, 0])

    for panel_index, (ax, (letter, title, keys, color)) in enumerate(
        zip(axes, specifications)
    ):
        rows = [patch["paths"][key] for key in keys]
        values = 100 * np.asarray([row["correction_rate"] for row in rows])
        intervals = 100 * np.asarray([row["correction_rate_bootstrap95"] for row in rows])
        errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))

        ax.errorbar(
            values,
            y,
            xerr=errors,
            fmt="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=color,
            ecolor=COLORS["ink"],
            elinewidth=0.85,
            capsize=2.0,
            zorder=3,
        )
        for yi, value, interval in zip(y, values, intervals):
            ax.text(
                min(interval[1] + 2.0, 104.0),
                yi,
                f"{value:.0f}%",
                color=color,
                fontsize=8.2,
                fontweight="semibold",
                va="center",
            )
        ax.set_xlim(0, 110)
        ax.set_ylim(-0.55, 1.55)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks(y, labels if panel_index == 0 else ("", ""))
        ax.set_title(title, loc="left")
        _clean_comparison_axis(ax)
        panel_label(ax, letter, x=-0.15)

    fig.supxlabel("Old-value answers corrected (%)", y=0.045, fontsize=8.4)
    save(fig, out_dir, "FigP_timing")


def _plot_trace_panel(ax, x, control, error, title, ylabel, show_xlabels) -> None:
    ax.plot(
        x,
        control,
        color=COLORS["current"],
        marker="o",
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="no conflicting update",
    )
    ax.plot(
        x,
        error,
        color=COLORS["stale"],
        marker="s",
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="old-value answer",
    )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.65, alpha=0.72)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, CHECKPOINT_LABELS if show_xlabels else ("", "", "", ""))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def event_trace_figure(root: Path, out_dir: Path) -> None:
    """Figure 9: aligned behavior, routing, matching, and retention traces."""
    analysis = load_json(
        root / "results/stage_p/models/pythia160/analysis/analysis.summary.json"
    )
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 3.18))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.91, bottom=0.20,
                        wspace=0.32, hspace=0.48)
    x = np.arange(len(CHECKPOINTS))

    panels = (
        ("behavioral_margin", "Answer preference", "Answer score\ncurrent minus old"),
        ("read_ratio", "Attention preference", "Attention log ratio\ncurrent over old"),
        ("qk_margin", "Key-matching preference", "Key-match score\ncurrent minus old"),
    )
    for index, (ax, (metric, title, ylabel)) in enumerate(zip(axes.flat[:3], panels)):
        control, error = _trajectory(analysis, metric)
        _plot_trace_panel(
            ax,
            x,
            control,
            error,
            title,
            ylabel,
            show_xlabels=index >= 2,
        )
        panel_label(ax, chr(ord("a") + index))

    ax = axes[1, 1]
    readable = np.asarray([
        analysis["trajectories"][checkpoint]["retention"]
        ["final_stale_mean_true_current_score"]
        for checkpoint in CHECKPOINTS
    ])
    null_ranges = np.asarray([
        analysis["trajectories"][checkpoint]["retention"]
        ["value_label_shuffle95"]
        for checkpoint in CHECKPOINTS
    ])
    ax.fill_between(
        x,
        null_ranges[:, 0],
        null_ranges[:, 1],
        color=COLORS["null"],
        linewidth=0,
        label="random-label range",
        zorder=1,
    )
    ax.plot(
        x,
        readable,
        color=COLORS["stale"],
        marker="s",
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="current-value score on errors",
        zorder=3,
    )
    ax.set_ylim(0, 0.9)
    ax.set_title("Current value remains readable", loc="left")
    ax.set_ylabel("Probe score for\ncurrent value")
    ax.set_xticks(x, CHECKPOINT_LABELS)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "d")

    condition_handles, condition_labels = axes[0, 0].get_legend_handles_labels()
    condition_handles.append(Patch(facecolor=COLORS["null"], edgecolor="none"))
    condition_labels.append("random-label range")
    fig.legend(
        condition_handles,
        condition_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.7,
        handletextpad=0.45,
    )
    save(fig, out_dir, "FigP_event_trace")


def six_model_figure(root: Path, out_dir: Path) -> None:
    """Figure 10: paired-prompt internal shifts across six models."""
    entries = []
    for label, slug in MODEL_SPECS:
        path = root / f"results/stage_p/models/{slug}/analysis/analysis.summary.json"
        entries.append((label, load_json(path)))

    metrics = (
        (
            "read_ratio",
            "Attention shifts toward old values",
            "Toward old (log-ratio change)",
        ),
        (
            "qk_margin",
            "Key matching shifts toward old values",
            "Toward old (score change)",
        ),
    )
    apply_paper_style()
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.36))
    gs = fig.add_gridspec(
        1,
        4,
        left=0.17,
        right=0.985,
        top=0.86,
        bottom=0.22,
        width_ratios=(1.0, 0.055, 1.0, 0.055),
        wspace=0.24,
    )

    for index, (metric, title, colorbar_label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, 2 * index])
        cax = fig.add_subplot(gs[0, 2 * index + 1])
        # Stored values are overwrite minus control and become negative when an
        # old assignment gains ground. Flip the sign for reader-facing plots.
        matrix = np.asarray([
            [
                -analysis["descriptive_full_pool"]["trajectories"][checkpoint][metric]
                ["length_distance_controlled"]["overwrite_minus_control"]
                for checkpoint in CHECKPOINTS
            ]
            for _, analysis in entries
        ])
        vmax = max(float(np.max(matrix)), 1e-8)
        image = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1) - 0.5,
            np.arange(matrix.shape[0] + 1) - 0.5,
            matrix,
            cmap="Oranges",
            vmin=0,
            vmax=vmax,
            shading="flat",
            rasterized=False,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text_color = "white" if value > 0.58 * vmax else COLORS["ink"]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.set_xticks(np.arange(len(CHECKPOINTS)), CHECKPOINT_LABELS)
        ax.set_yticks(
            np.arange(len(entries)),
            [label for label, _ in entries] if index == 0 else [],
        )
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color("#C9CDD2")
            spine.set_linewidth(0.6)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(colorbar_label, fontsize=7.5, labelpad=3.0)
        cbar.ax.tick_params(labelsize=7.2, length=2)
        panel_label(ax, chr(ord("a") + index), x=-0.16 if index == 0 else -0.12)

    save(fig, out_dir, "FigP_six_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("paper/full_draft_assets/figures"))
    args = parser.parse_args()
    timing_figure(args.root, args.out)
    event_trace_figure(args.root, args.out)
    six_model_figure(args.root, args.out)


if __name__ == "__main__":
    main()
