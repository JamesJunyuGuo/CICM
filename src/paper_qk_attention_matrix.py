"""Cross-model attention figure from frozen Stage-P and Stage-L summaries.

The main figure separates matched overwrite/control contrasts from natural
dialogue failure/correct contrasts, retaining stored bootstrap intervals.
The appendix preserves the Qwen layer/head detail from its Stage-L harvest.
No inference or statistical refitting is performed.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from paper_merged_figures import apply_style, panel_label
from stage_l_paper_full_assets import COLORS, save
from paper_figure_style import apply_paper_style, TEXT_WIDTH_IN

QWEN_RESULTS = "results/stage_l/natural_factorial_otherdist_l1/full"


def load_model(root, rel):
    z = np.load(Path(root) / rel / "l1_harvest.npz", allow_pickle=True)
    rows = [json.loads(l) for l in (Path(root) / rel / "l1_index.jsonl").open()]
    labels = np.array([r["label"] for r in rows])
    ac = z["attn_current"].astype(np.float64)  # (N, L, H)
    as_ = z["attn_stale"].astype(np.float64)
    at = z["attn_total"].astype(np.float64)
    denom = ac + as_
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denom > 0, as_ / denom, np.nan)  # stale share of (cur+stale)
    return labels, rho, ac, as_, at


def build_head_detail(root, out_dir):
    labels, rho, ac, as_, at = load_model(root, QWEN_RESULTS)
    correct = labels == "correct_current"
    fail = labels == "within_stale"

    rho_correct = np.nanmean(rho[correct], axis=0)  # (L, H)
    rho_fail = np.nanmean(rho[fail], axis=0)
    delta = 100 * (rho_fail - rho_correct)  # points, per (layer, head)

    # Panel (b) reproduces the paper's headline quantity: rho_stale averaged over
    # heads at the FINAL layer, per example (stage_l_mech.py L552:
    # nanmean(attn_ratio, axis=2)[:, -1]), then compared across trial types.
    final_rho = np.nanmean(rho[:, -1, :], axis=1)  # (N,)
    stale_correct = float(np.nanmean(final_rho[correct]))
    stale_fail = float(np.nanmean(final_rho[fail]))
    print(f"[qwen] final-layer head-mean rho_stale: correct={100*stale_correct:.1f}%  "
          f"within-stale={100*stale_fail:.1f}%  raw diff={100*(stale_fail-stale_correct):.1f} pts "
          f"(paper length-controlled = 19.4)")
    fig = plt.figure(figsize=(5.5, 2.30))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        top=0.86,
        bottom=0.32,
        width_ratios=(2.05, 1.15),
        wspace=0.42,
    )

    # (a) layer x head diagonal-diverging heatmap
    ax = fig.add_subplot(gs[0, 0])
    vmax = np.nanmax(np.abs(delta))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    layer_edges = np.arange(delta.shape[0] + 1) - 0.5
    head_edges = np.arange(delta.shape[1] + 1) - 0.5
    im = ax.pcolormesh(head_edges, layer_edges, delta, cmap="RdBu_r", norm=norm,
                       shading="flat", rasterized=False)
    ax.set_xlim(-0.5, delta.shape[1] - 0.5)
    ax.set_ylim(-0.5, delta.shape[0] - 0.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Late heads shift toward old values", loc="left")
    cax = fig.add_axes([0.105, 0.075, 0.49, 0.032])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Error minus correct attention (points)", fontsize=7.5,
                   labelpad=2.0)
    cbar.ax.tick_params(labelsize=7.2, length=2)
    panel_label(ax, "a")

    # (b) Show the raw final-layer current/old allocation that underlies the
    # headline comparison. The controlled cross-model estimates remain in text
    # and the appendix table.
    ax = fig.add_subplot(gs[0, 1])
    stale_shares = 100 * np.asarray([stale_correct, stale_fail])
    current_shares = 100 - stale_shares
    y = np.asarray([1, 0])
    ax.barh(y, current_shares, color=COLORS["current"], height=0.48,
            edgecolor="none")
    ax.barh(y, stale_shares, left=current_shares, color=COLORS["stale"],
            height=0.48, edgecolor="none")
    for yi, current_share, stale_share in zip(y, current_shares, stale_shares):
        current_text = (
            f"{current_share:.0f}%\ncurrent" if current_share >= 30
            else f"{current_share:.0f}%"
        )
        ax.text(current_share / 2, yi, current_text,
                ha="center", va="center", color="white", fontsize=7.7,
                fontweight="semibold", linespacing=0.95)
        ax.text(current_share + stale_share / 2, yi, f"{stale_share:.0f}%\nold",
                ha="center", va="center", color="white", fontsize=7.7,
                fontweight="semibold", linespacing=0.95)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Final-layer attention share (%)")
    ax.set_yticks(y, ["Correct\nanswer", "Old-value\nanswer"])
    ax.set_title("Current vs old allocation", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")

    save(fig, out_dir, "Fig_qk_attention_heads")


def build(root, out_dir):
    """Contrast shared overwrite pressure with failure-conditioned evidence."""
    models = (
        ("Pythia-160M", "pythia160"),
        ("Pythia-1.4B", "pythia14"),
        ("Qwen2.5-1.5B", "qwen15"),
        ("Qwen2.5-7B", "qwen7"),
        ("Llama-3.1-8B", "llama8"),
        ("Gemma-2-2B*", "gemma2"),
    )
    entries = []
    for label, slug in models:
        path = root / f"results/stage_p/models/{slug}/analysis/analysis.summary.json"
        summary = json.loads(path.read_text())
        entries.append((label, summary["descriptive_full_pool"]["trajectories"]["tauQ"]))

    apply_paper_style()
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.12))
    axes = [fig.add_axes(bounds) for bounds in (
        [0.195, 0.255, 0.205, 0.50],
        [0.445, 0.255, 0.205, 0.50],
        [0.755, 0.255, 0.230, 0.50],
    )]
    fig.text(0.42, 0.965, "Matched overwrite / no-overwrite prompts",
             ha="center", va="top", fontsize=7.5, color=COLORS["muted"])
    fig.text(0.87, 0.965, "Natural dialogue", ha="center", va="top",
             fontsize=7.5, color=COLORS["muted"])
    for ax, title in zip(axes, ("a  Attention", "b  Key matching", "c  Failure association")):
        ax.set_title(title, loc="left", fontsize=8.3, pad=8)
        ax.axvline(0, color=COLORS["muted"], lw=0.65, ls="--", zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4, labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)
        ax.set_axisbelow(True)

    for ax, metric, color, xlabel in zip(
        axes[:2], ("read_ratio", "qk_margin"),
        (COLORS["stale"], COLORS["cross"]),
        ("Current / old attention\nlog-ratio change", "Current - strongest old\nQK score change"),
    ):
        for index, (label, row) in enumerate(entries):
            estimate = row[metric]["length_distance_controlled"]
            value = estimate["overwrite_minus_control"]
            low, high = estimate["paired_bootstrap95"]
            assert low <= value <= high
            ax.errorbar(value, 5 - index, xerr=[[value - low], [high - value]],
                        fmt="o", color=color, markersize=3.5, capsize=1.6,
                        elinewidth=0.8, markeredgewidth=0.7,
                        markerfacecolor="white" if label.endswith("*") else color,
                        zorder=3)
        ax.set_xlim(-0.70, 0.045)
        ax.set_ylim(-0.55, 5.55)
        ax.set_xticks([-0.6, -0.3, 0])
        ax.set_yticks(range(5, -1, -1),
                      [label for label, _ in entries] if ax is axes[0] else [""] * 6)
        ax.set_xlabel(xlabel, fontsize=7.5, labelpad=3)

    ax = axes[2]
    natural_models = (
        ("Qwen2.5-7B", QWEN_RESULTS, 1),
        ("Llama-3.1-8B", "results/stage_l/natural_factorial_otherdist_llama/l1/full", 0),
    )
    for label, rel, y in natural_models:
        summary = json.loads((root / rel / "l1_summary.json").read_text())
        estimate = summary["attention_stale_ratio_within_vs_correct"]
        value = 100 * estimate["length_controlled_delta"]
        low, high = 100 * np.asarray(estimate["length_controlled_delta_ci"])
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o",
                    color=COLORS["stale"], markersize=3.8, capsize=2,
                    elinewidth=0.85, zorder=3)
        ax.text(0.02, (y + 0.7) / 2.0, label, transform=ax.transAxes,
                fontsize=7.5, va="bottom", zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.set_xlim(-3, 24)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([])
    ax.set_xlabel("Error - correct old share\n(percentage points)", fontsize=7.5, labelpad=3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "Fig_qk_attention.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / "Fig_qk_attention.png", dpi=400,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="paper/full_draft_assets/figures")
    args = parser.parse_args()
    apply_style()
    build_head_detail(Path(args.root), Path(args.out))
    build(Path(args.root), Path(args.out))
    print("wrote Fig_qk_attention")


if __name__ == "__main__":
    main()
