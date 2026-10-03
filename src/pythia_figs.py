"""Publication-ready Stage N Phase-1 figures from saved summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "current": "#167D5A",
    "stale": "#D24D3E",
    "cross": "#3976AF",
    "other": "#777777",
    "accent": "#7656A5",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def behavior_figure(summary, out_dir):
    ks = sorted(int(k) for k in summary["by_k"])
    single = []
    for k in ks:
        cells = [summary["by_cell"][f"k{k}__single__{template}"] for template in ("arrow", "current", "latest")]
        counts = {label: sum(cell["counts"].get(label, 0) for cell in cells) for label in ("correct_current", "within_stale", "other")}
        total = sum(cell["n"] for cell in cells)
        single.append({label: count / total for label, count in counts.items()})
    accuracy = [row["correct_current"] for row in single]
    stale = [row["within_stale"] for row in single]
    other = [row["other"] for row in single]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].plot(ks, accuracy, marker="o", ms=4, lw=1.6, color=COLORS["current"], label="Current value")
    axes[0].plot(ks, stale, marker="s", ms=3.8, lw=1.4, color=COLORS["stale"], label="Stale value")
    axes[0].plot(ks, other, marker="^", ms=3.8, lw=1.2, color=COLORS["other"], label="Other token")
    axes[0].set(xlabel="Number of overwrites, k", ylabel="Fraction of examples", ylim=(-0.03, 1.03), title="a  Clean single-variable dose response")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))

    templates = ["arrow", "current", "latest"]
    variants = ["single", "multi"]
    chosen_k = summary["fixed_k_gate"]["chosen_k"]
    x = np.arange(len(templates))
    width = 0.35
    for offset, variant in enumerate(variants):
        values = [summary["by_cell"][f"k{chosen_k}__{variant}__{template}"]["accuracy"] for template in templates]
        axes[1].bar(x + (offset - 0.5) * width, values, width, label=variant.capitalize(), color=[COLORS["current"], COLORS["cross"]][offset])
    axes[1].set_xticks(x, ["Arrow", "Current", "Latest"])
    axes[1].set(ylabel="Current-value accuracy", ylim=(0, 1), title=f"b  Format sensitivity at fixed k={chosen_k}")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    save(fig, out_dir, "N1_behavior")


def mechanism_figure(mechanism, out_dir):
    layers = mechanism["probe"]["layers"]
    x = [row["layer"] for row in layers]
    correct = [row["correct_true_current_score"] for row in layers]
    stale = [row["within_stale_true_current_score"] for row in layers]
    lens_correct = mechanism["direct_logit_attribution"]["logit_lens_correct_mean"]
    lens_failure = mechanism["direct_logit_attribution"]["logit_lens_failure_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].plot(x, correct, color=COLORS["current"], marker="o", ms=3.5, lw=1.5, label="Correct trials")
    axes[0].plot(x, stale, color=COLORS["stale"], marker="s", ms=3.2, lw=1.4, label="Within-stale failures")
    null = mechanism["probe"]["value_label_shuffle95"]
    axes[0].axhspan(null[0], null[1], color="#B8B8B8", alpha=0.35, label="Shuffle 95%")
    axes[0].set(xlabel="Residual checkpoint", ylabel="OOS probability of current value", title="a  Current-value decodability")
    axes[0].legend(loc="best")
    axes[1].plot(range(len(lens_correct)), lens_correct, color=COLORS["current"], marker="o", ms=3.5, lw=1.5, label="Correct trials")
    axes[1].plot(range(len(lens_failure)), lens_failure, color=COLORS["stale"], marker="s", ms=3.2, lw=1.4, label="Within-stale failures")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Residual checkpoint", ylabel="Current - stale logit", title="b  Decision resolution")
    axes[1].legend(loc="best")
    save(fig, out_dir, "N2_retention_selection")


def circuit_heatmaps(mechanism, first_pass, out_dir):
    n_layers = 12
    n_heads = 12
    attention = np.full((n_layers, n_heads), np.nan)
    for row in mechanism["attention"]["all_heads"]:
        attention[row["layer"], row["head"]] = row["within_vs_correct"]["length_controlled_delta"]
    patch = np.full((n_layers, n_heads), np.nan)
    for row in first_pass["ranking"]:
        patch[row["layer"], row["head"]] = row["evaluation_recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    limits = [np.nanmax(np.abs(attention)), np.nanmax(np.abs(patch))]
    for axis, matrix, limit, title in zip(
        axes,
        [attention, patch],
        limits,
        ["a  Stale-attention shift (length controlled)", "b  Clean-to-corrupt head-patch recovery"],
    ):
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        axis.set(xlabel="Head", ylabel="Layer", xticks=range(n_heads), yticks=range(n_layers), title=title)
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    save(fig, out_dir, "N3_head_circuit")


def causal_figure(cumulative, ablation, out_dir):
    curve = cumulative["cumulative"]
    counts = np.asarray([row["count"] for row in curve])
    means = np.asarray([row["mean_recovery"] for row in curve])
    low = np.asarray([row["ci"][0] for row in curve])
    high = np.asarray([row["ci"][1] for row in curve])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].plot(counts, means, color=COLORS["accent"], lw=1.6)
    axes[0].fill_between(counts, low, high, color=COLORS["accent"], alpha=0.18, linewidth=0)
    axes[0].axhline(cumulative["target_recovery"], color=COLORS["stale"], ls="--", lw=1.1, label="80% of all-head joint")
    axes[0].axvline(cumulative["selected_count"], color="#333333", ls=":", lw=1.0)
    axes[0].set(xlabel="Discovery-ranked heads patched", ylabel="Held-out recovered gap fraction", title="a  Circuit concentration")
    axes[0].legend(loc="lower right")

    current = ablation["current_binding_ablation"]
    stale = ablation["stale_component_ablation"]
    labels = ["Current-head\nablation", "Stale-head\nablation"]
    targeted = [current["ablated_stale_rate"], stale["ablated_correct_rate"]]
    random = [current["random_same_size_stale_rate"]["mean"], stale["random_same_size_correct_rate"]["mean"]]
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, targeted, width, color=COLORS["stale"], label="Targeted")
    axes[1].bar(x + width / 2, random, width, color="#A6A6A6", label="Same-size random")
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel="Directional flip rate", ylim=(0, max(targeted + random + [0.05]) * 1.2), title="b  Causal ablation")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.76, -0.08), ncol=2)
    save(fig, out_dir, "N4_patch_ablation")


def qkov_induction_figure(qkov, induction, first_pass, out_dir):
    heads = qkov["heads"][:3]
    labels = [f"L{row['layer']}H{row['head']}" for row in heads]
    correct = [row["by_label"]["correct_current"]["qk_latest_margin_mean"] for row in heads]
    failure = [row["by_label"]["within_stale"]["qk_latest_margin_mean"] for row in heads]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    x = np.arange(len(heads))
    width = 0.35
    axes[0].bar(x - width / 2, correct, width, color=COLORS["current"], label="Correct trials")
    axes[0].bar(x + width / 2, failure, width, color=COLORS["stale"], label="Within-stale failures")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="QK current - strongest stale", title="a  Latest-write QK margin")
    axes[0].legend(loc="lower left")

    induction_map = {(row["layer"], row["head"]): row["induction_score"] for row in induction["all_heads"]}
    patch_rows = first_pass["ranking"]
    induction_values = [induction_map[(row["layer"], row["head"])] for row in patch_rows]
    recovery = [row["evaluation_recovery"] for row in patch_rows]
    axes[1].scatter(induction_values, recovery, s=15, alpha=0.45, color="#777777", edgecolors="none")
    for row in patch_rows[:12]:
        key = (row["layer"], row["head"])
        axes[1].scatter(induction_map[key], row["evaluation_recovery"], s=30, color=COLORS["accent"], edgecolors="white", linewidths=0.5)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Induction score", ylabel="Held-out individual patch recovery", title="b  Induction-head connection")
    save(fig, out_dir, "N5_qkov_induction")


def write_tables(mechanism, first_pass, qkov, out_dir):
    attention_fields = [
        "layer", "head", "correct_current_attention", "failure_current_attention",
        "correct_stale_attention", "failure_stale_attention", "correct_identity_attention",
        "failure_identity_attention", "failure_minus_correct_stale_ratio",
    ]
    with (out_dir / "attention_all_144_heads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=attention_fields + ["length_controlled_delta", "length_controlled_ci_low", "length_controlled_ci_high", "shuffle95_low", "shuffle95_high"])
        writer.writeheader()
        for row in mechanism["attention"]["all_heads"]:
            controlled = row["within_vs_correct"]
            writer.writerow({
                **{field: row[field] for field in attention_fields},
                "length_controlled_delta": controlled["length_controlled_delta"],
                "length_controlled_ci_low": controlled["length_controlled_ci"][0],
                "length_controlled_ci_high": controlled["length_controlled_ci"][1],
                "shuffle95_low": controlled["length_controlled_shuffle95"][0],
                "shuffle95_high": controlled["length_controlled_shuffle95"][1],
            })
    with (out_dir / "path_patch_all_144_heads.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank", "layer", "head", "discovery_recovery", "evaluation_recovery", "evaluation_ci_low", "evaluation_ci_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in first_pass["ranking"]:
            writer.writerow({**{field: row[field] for field in fields[:5]}, "evaluation_ci_low": row["evaluation_ci"][0], "evaluation_ci_high": row["evaluation_ci"][1]})
    with (out_dir / "qkov_key_heads.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["layer", "head", "label", "n", "qk_latest_margin_mean", "qk_position_slope_mean", "current_attention_mean", "stale_attention_sum_mean", "current_ov_copy_margin_mean", "stale_ov_copy_margin_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for head in qkov["heads"]:
            for label, values in head["by_label"].items():
                writer.writerow({"layer": head["layer"], "head": head["head"], "label": label, **values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-summary", required=True)
    parser.add_argument("--mechanism-summary", required=True)
    parser.add_argument("--first-pass-summary", required=True)
    parser.add_argument("--cumulative-summary", required=True)
    parser.add_argument("--ablation-summary", required=True)
    parser.add_argument("--qkov-summary", required=True)
    parser.add_argument("--induction-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_figure(load(args.behavior_summary), out_dir)
    mechanism = load(args.mechanism_summary)
    mechanism_figure(mechanism, out_dir)
    first_pass = load(args.first_pass_summary)
    circuit_heatmaps(mechanism, first_pass, out_dir)
    causal_figure(load(args.cumulative_summary), load(args.ablation_summary), out_dir)
    qkov = load(args.qkov_summary)
    qkov_induction_figure(qkov, load(args.induction_summary), first_pass, out_dir)
    write_tables(mechanism, first_pass, qkov, out_dir)


if __name__ == "__main__":
    main()
