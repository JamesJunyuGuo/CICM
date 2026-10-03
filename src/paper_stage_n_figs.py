"""Stage N training-dynamics figure from frozen result summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_merged_figures import apply_style, grid, panel_label
from stage_l_paper_full_assets import COLORS


def load_curves(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return sorted(rows, key=lambda row: int(row["step"]))


def _plot_series(ax, x, rows, value_key, color, marker, label):
    values = np.asarray([row[value_key] for row in rows], dtype=float)
    ax.plot(
        x,
        values,
        color=color,
        marker=marker,
        markersize=3.2,
        markeredgecolor="white",
        markeredgewidth=0.45,
        linewidth=1.15,
        label=label,
        zorder=3,
    )


def _training_axis(ax, x):
    ax.set_xscale("log")
    ticks = [0.5, 10, 100, 1000, 10000, 100000]
    ax.set_xticks(ticks, ["0", "10", "100", "1k", "10k", "100k"])
    ax.set_xlim(0.42, max(x) * 1.25)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Training step (log)")
    grid(ax)


def render_dynamics(curves, transitions_path: str | Path, out_dir: str | Path):
    transitions = json.loads(Path(transitions_path).read_text(encoding="utf-8"))
    onset = int(transitions["transitions"]["S_ind"]["step"])
    x = np.asarray([0.5 if int(row["step"]) == 0 else int(row["step"]) for row in curves])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.86, bottom=0.29, wspace=0.30)

    _plot_series(
        axes[0], x, curves, "induction_global",
        COLORS["cross"], "s", "Repeated-token copying",
    )
    _plot_series(
        axes[0], x, curves, "k0_accuracy",
        COLORS["current"], "o", "No-overwrite binding",
    )
    axes[0].axvline(onset, color=COLORS["stale"], linestyle="--", linewidth=1.0)
    axes[0].text(
        onset * 1.18, 0.06, "onset\n(step 1k)", color=COLORS["stale"],
        fontsize=7.5, ha="left", va="bottom",
    )
    axes[0].set_ylabel("Score / accuracy")
    axes[0].set_title("Binding and copying emerge together")
    _training_axis(axes[0], x)
    panel_label(axes[0], "a")

    _plot_series(
        axes[1], x, curves, "error_share_within_stale",
        COLORS["stale"], "o", "Old-value error",
    )
    _plot_series(
        axes[1], x, curves, "error_share_other",
        COLORS["other"], "s", "Other",
    )
    _plot_series(
        axes[1], x, curves, "error_share_cross_variable",
        COLORS["cross"], "^", "Other-variable error",
    )
    axes[1].axvline(onset, color=COLORS["stale"], linestyle="--", linewidth=1.0)
    axes[1].axvspan(24000, 48000, color=COLORS["stale"], alpha=0.065, linewidth=0)
    axes[1].text(
        33000, 0.53, "old-value\ndominance", color=COLORS["stale"],
        fontsize=7.5, ha="center", va="center",
    )
    axes[1].set_ylabel("Share of overwrite errors")
    axes[1].set_title("Old-value errors dominate later")
    _training_axis(axes[1], x)
    panel_label(axes[1], "b")

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.6,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "Fig_dynamics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Fig_dynamics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves", default="results/stage_n/dynamics/curves.jsonl"
    )
    parser.add_argument(
        "--transitions", default="results/stage_n/dynamics/transitions.summary.json"
    )
    parser.add_argument(
        "--out-dir", default="paper/full_draft_assets/figures"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_dynamics(load_curves(args.curves), args.transitions, args.out_dir)
