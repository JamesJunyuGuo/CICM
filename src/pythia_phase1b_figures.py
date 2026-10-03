#!/usr/bin/env python3
"""Render the preregistered Stage N Phase 1b readings from saved summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "current": "#2878B5",
    "stale": "#D95F59",
    "query": "#2878B5",
    "key": "#E69F00",
    "both": "#3A9D70",
    "qk": "#7A5195",
    "final": "#D95F59",
    "random": "#777777",
}


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asymmetric_error(mean: float, interval: list[float]) -> np.ndarray:
    return np.asarray([[mean - interval[0]], [interval[1] - mean]])


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def panel_ablation(ax, summary: dict) -> None:
    arms = [summary["current_binding"], summary["stale_promoting"]]
    labels = ["Current-binding\nset", "Stale-promoting\nset"]
    colors = [COLORS["current"], COLORS["stale"]]
    x = np.arange(2)
    width = 0.28
    for index, (arm, color) in enumerate(zip(arms, colors)):
        ax.bar(
            x[index] - width / 2,
            arm["phase1_same_pool_effect"],
            width,
            color="white",
            edgecolor=color,
            linewidth=1.4,
            hatch="///",
            label="Phase 1 same-pool" if index == 0 else None,
            zorder=3,
        )
        heldout = arm["paired_effect"]
        ax.bar(
            x[index] + width / 2,
            heldout["mean"],
            width,
            color=color,
            alpha=0.9,
            label="Phase 1b held-out" if index == 0 else None,
            zorder=3,
        )
        ax.errorbar(
            x[index] + width / 2,
            heldout["mean"],
            yerr=asymmetric_error(heldout["mean"], heldout["ci"]),
            color="#222222",
            linewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
        random = arm["random_effect"]
        ax.errorbar(
            x[index] + 0.36,
            random["mean"],
            yerr=asymmetric_error(random["mean"], random["quantile95"]),
            fmt="o",
            markersize=3.5,
            color=COLORS["random"],
            linewidth=1.0,
            capsize=2.5,
            label="Matched random 95% range" if index == 0 else None,
            zorder=5,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.58)
    ax.set_ylabel("Directional answer-flip rate", fontsize=9)
    ax.set_title("A  Held-out ablation de-bias", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)


def panel_side(ax, summary: dict) -> None:
    targets = ["L8H10", "L8H2"]
    sides = ["query", "key", "both"]
    x = np.arange(len(targets))
    width = 0.22
    for side_index, side in enumerate(sides):
        means = []
        lows = []
        highs = []
        for target in targets:
            result = summary["b1"][target]["sides"][side]["qk_margin"]
            means.append(result["mean"])
            lows.append(result["mean"] - result["ci"][0])
            highs.append(result["ci"][1] - result["mean"])
        positions = x + (side_index - 1) * width
        ax.bar(
            positions,
            means,
            width,
            color=COLORS[side],
            alpha=0.9,
            label=side.capitalize(),
            zorder=3,
        )
        ax.errorbar(
            positions,
            means,
            yerr=np.asarray([lows, highs]),
            fmt="none",
            color="#222222",
            linewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, targets)
    ax.set_ylabel("Held-out QK restoration fraction", fontsize=9)
    ax.set_title("B  Query-key decomposition", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)


def panel_upstream(ax, summary: dict) -> None:
    paths = [(target, side) for target in ("L8H10", "L8H2") for side in ("query", "key")]
    x = np.arange(len(paths))
    width = 0.25
    for metric_index, (metric, label, color) in enumerate(
        [("qk", "QK margin", COLORS["qk"]), ("final_gap", "Final gap", COLORS["final"])]
    ):
        offset = (metric_index - 0.5) * width
        for path_index, (target, side) in enumerate(paths):
            result = summary["b2"][target][side]
            selected = result["selected_evaluation"][metric]
            random = result["random12_evaluation"][metric]
            position = x[path_index] + offset
            ax.bar(
                position,
                selected["mean"],
                width,
                color=color,
                alpha=0.9,
                label=label if path_index == 0 else None,
                zorder=3,
            )
            ax.errorbar(
                position,
                selected["mean"],
                yerr=asymmetric_error(selected["mean"], selected["ci"]),
                fmt="none",
                color="#222222",
                linewidth=0.9,
                capsize=2,
                zorder=4,
            )
            ax.vlines(
                position + width * 0.36,
                random["quantile95"][0],
                random["quantile95"][1],
                color=COLORS["random"],
                linewidth=2.0,
                zorder=5,
            )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [f"{target}\n{side}" for target, side in paths])
    ax.set_ylabel("Held-out restoration fraction", fontsize=9)
    ax.set_title("C  Discovery top-12 vs matched random", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ablation = load_json(args.ablation)
    upstream = load_json(args.upstream)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=True)
    panel_ablation(axes[0], ablation)
    panel_side(axes[1], upstream)
    panel_upstream(axes[2], upstream)

    handles = []
    labels = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(7, len(labels)),
        frameon=False,
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"N6_phase1b.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
