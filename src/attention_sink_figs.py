"""Figures for the Stage N-S attention-sink diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "gate": "#2B6F8A",
    "routing": "#B35C44",
    "value": "#6B7F3A",
    "full": "#454A52",
    "null": "#B9BEC5",
}


def make_figure(summary: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), constrained_layout=True)

    gate = summary["gate_observation_corrupt_minus_clean"]
    delta = gate["length_controlled_delta"]
    shuffle = gate["shuffle95"]
    controlled = gate.get("length_controlled", {})
    ci = controlled.get("ci95", [delta, delta])
    axes[0].axhspan(shuffle[0], shuffle[1], color=COLORS["null"], alpha=0.35, label="shuffle 95%")
    axes[0].errorbar(
        [0],
        [delta],
        yerr=[[delta - ci[0]], [ci[1] - delta]],
        fmt="o",
        color=COLORS["gate"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=3,
        linewidth=1.4,
    )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xlim(-0.65, 0.65)
    axes[0].set_xticks([0], ["stale - correct"])
    axes[0].set_ylabel("Content-gate difference")
    axes[0].set_title("a  Observed sink gate", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    arms = ["gate", "routing", "value", "full"]
    targeted = summary["targeted_arms"]
    means = np.asarray([targeted[arm]["mean"] for arm in arms])
    lower = means - np.asarray([targeted[arm]["ci95"][0] for arm in arms])
    upper = np.asarray([targeted[arm]["ci95"][1] for arm in arms]) - means
    colors = [COLORS[arm] for arm in arms]
    x = np.arange(len(arms))
    axes[1].bar(x, means, color=colors, width=0.62, edgecolor="white", linewidth=0.7)
    axes[1].errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#25282C", capsize=3, linewidth=1.0)
    random_gate = summary["controls"]["gate_random"]["quantile95"]
    if np.all(np.isfinite(random_gate)):
        axes[1].axhspan(random_gate[0], random_gate[1], color=COLORS["null"], alpha=0.28, label="matched random gate 95%")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, ["Gate", "Routing", "Value", "Full"])
    axes[1].set_ylabel("Current-minus-stale gap change")
    axes[1].set_title("b  Held-out causal decomposition", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    for axis in axes:
        axis.grid(axis="y", color="#D8DCE1", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Attention-sink diagnostic: {summary['verdict'].replace('_', ' ')}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(out_dir / "N8_attention_sink_diagnostic.png", dpi=240)
    fig.savefig(out_dir / "N8_attention_sink_diagnostic.pdf")
    plt.close(fig)
