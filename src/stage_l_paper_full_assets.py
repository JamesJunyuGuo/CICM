import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_figure_style import COLORS, apply_paper_style


SOURCES = {
    "p0": "results/p0_qwen7b.summary.json",
    "k1d": "results/stage_k/k1d_summary.json",
    "l0": "results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json",
    "qwen_l1": "results/stage_l/natural_factorial_otherdist_l1/full/l1_summary.json",
    "llama_l1": "results/stage_l/natural_factorial_otherdist_llama/l1/full/l1_summary.json",
    "validity_l2": "results/stage_l/validity_l2/summary.json",
    "geometry": "results/stage_l/geometry/summary.json",
    "stage_g_circuit": "results/stage_g/circuit/full_r1_ai/summary.json",
    "stage_g_ext_baseline": "results/stage_g/external/full_ai/v2/qwen_baseline/all/summary.json",
    "stage_g_ext_arm_l": "results/stage_g/external/full_ai/v2/arm_l/all/summary.json",
    "stage_g_ext_random": "results/stage_g/external/full_ai/v2/random_heads/all/summary.json",
    "stage_e_scale": "results/figures/F8_openrouter_scale_sweep.data.json",
    "stage_e_p0_errors": "results/stage_e/qwen_p0_errors.summary.json",
    "stage_f_hard": "results/stage_f/hard_tier/sweep.summary.json",
    "stage_k_mechanism": "results/stage_k/mech/k2k3b_summary.json",
}

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_inputs(root):
    return {name: load_json(root / rel) for name, rel in SOURCES.items()}


def apply_style():
    apply_paper_style()


def wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return center - half, center + half


def error_pair(value, ci):
    return max(0, value - ci[0]), max(0, ci[1] - value)


def panel_label(ax, label):
    ax.text(-0.1, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, size, title, body, face, edge):
    x, y = xy
    w, h = size
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + h - 0.05, title, fontsize=8.5, fontweight="bold", va="top")
    ax.text(x + 0.018, y + h - 0.105, body, fontsize=7.8, va="top", linespacing=1.25)


def figure_task_overview(out_dir):
    fig, ax = plt.subplots(figsize=(7.4, 2.85), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def card(x, y, w, h, eyebrow, value, detail, accent, fill="#FFFFFF"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=0.9, facecolor=fill, edgecolor="#BCC3C9",
        )
        ax.add_patch(patch)
        ax.add_patch(FancyBboxPatch(
            (x, y + h - 0.035), w, 0.035,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            linewidth=0, facecolor=accent,
        ))
        ax.text(x + 0.016, y + h - 0.068, eyebrow.upper(), fontsize=7.0,
                color=COLORS["muted"], fontweight="bold", va="top")
        ax.text(x + 0.016, y + h - 0.125, value, fontsize=9.1,
                color=COLORS["ink"], fontweight="bold", va="top")
        ax.text(x + 0.016, y + 0.032, detail, fontsize=7.2,
                color=COLORS["muted"], va="bottom")

    ax.text(0.02, 0.965, "CONTEXT TIMELINE", fontsize=7.2, color=COLORS["muted"],
            fontweight="bold", va="top")
    ax.text(0.02, 0.895, "The target slot is updated, then competes with a recent value from another slot.",
            fontsize=10.7, color=COLORS["ink"], fontweight="bold", va="top")

    y_top, h_top, w_top = 0.56, 0.245, 0.205
    top_cards = [
        (0.02, "same slot / old", "keto", "meal style", COLORS["stale"], "#FCF4F1"),
        (0.265, "same slot / current", "gluten free", "meal style", COLORS["current"], "#F1F7F3"),
        (0.51, "other slot / recent", "hiphop", "music genre", COLORS["cross"], "#F1F7F9"),
        (0.765, "anchored query", "Latest meal style?", "retrieve target slot", COLORS["correct"], "#F5F6F7"),
    ]
    for x, eyebrow, value, detail, accent, fill in top_cards:
        card(x, y_top, w_top, h_top, eyebrow, value, detail, accent, fill)
    for x0, x1 in ((0.225, 0.265), (0.47, 0.51), (0.715, 0.765)):
        ax.add_patch(FancyArrowPatch(
            (x0, y_top + 0.122), (x1, y_top + 0.122), arrowstyle="-|>",
            mutation_scale=9, linewidth=0.9, color="#8B949D",
        ))

    ax.text(0.02, 0.475, "PROGRAM-VERIFIABLE ANSWER TAXONOMY", fontsize=7.2,
            color=COLORS["muted"], fontweight="bold", va="top")
    branch_y = 0.435
    ax.plot([0.18, 0.82], [branch_y, branch_y], color="#B5BCC3", linewidth=0.9)
    ax.add_patch(FancyArrowPatch((0.867, y_top - 0.01), (0.82, branch_y),
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color="#8B949D"))

    outcomes = [
        (0.02, "CORRECT CURRENT", "gluten free", "same slot + latest", COLORS["current"], "#F1F7F3"),
        (0.355, "WITHIN-SLOT STALE", "keto", "same slot + superseded", COLORS["stale"], "#FCF4F1"),
        (0.69, "CROSS-SLOT CAPTURE", "hiphop", "wrong slot + recent", COLORS["cross"], "#F1F7F9"),
    ]
    for x, eyebrow, value, detail, accent, fill in outcomes:
        center = x + 0.145
        ax.add_patch(FancyArrowPatch((center, branch_y), (center, 0.35),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color="#A0A7AE"))
        card(x, 0.07, 0.29, 0.27, eyebrow, value, detail, accent, fill)
    save(fig, out_dir, "F1_task_overview")


def figure_case_studies(cases, out_dir):
    # The ICLR template sets \textwidth to 5.5 in.  Draw at that physical width
    # so the declared point sizes are the sizes readers see in the paper.
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.55))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.055, hspace=0.38)

    configurations = [
        {
            "title": "A  Variable identity retrieves an old value",
            "events": [
                ("Old value", "gluten free", "34 messages", "stale"),
                ("Current", "mediterranean", "30 messages", "current"),
                ("Other variable", "discussion", "24 messages", "cross"),
                ("Recent old value", "gluten free", "2 messages", "stale"),
            ],
        },
        {
            "title": "B  Recency retrieves a value of another variable",
            "events": [
                ("Old value", "keto", "32 messages", "stale"),
                ("Current", "gluten free", "28 messages", "current"),
                ("Other variable", "hiphop", "22 messages", "cross"),
                ("Context", "10 turns omitted", "", "other"),
            ],
        },
    ]

    colors = {
        "stale": COLORS["stale"],
        "current": COLORS["current"],
        "cross": COLORS["cross"],
        "other": "#929AA2",
    }

    for ax, case, config in zip(axes, cases, configurations):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.0, 0.97, config["title"], fontsize=10.0,
                fontweight="semibold", va="top", color=COLORS["ink"])

        xs = [0.075, 0.285, 0.495, 0.705, 0.925]
        y_line = 0.56
        ax.add_patch(FancyArrowPatch(
            (xs[0] - 0.025, y_line), (xs[-1] + 0.025, y_line),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
            color="#AAB1B7",
        ))

        for x, (role, value, distance, kind) in zip(xs[:4], config["events"]):
            ax.scatter([x], [y_line], s=34, facecolor="white",
                       edgecolor=colors[kind], linewidth=1.5, zorder=3)
            ax.text(x, 0.77, role, fontsize=7.8, fontweight="semibold",
                    color=colors[kind], ha="center", va="center")
            ax.text(x, 0.68, value, fontsize=8.5, color=COLORS["ink"],
                    ha="center", va="center")
            if distance:
                ax.text(x, 0.43, distance, fontsize=8.0, color=COLORS["muted"],
                        ha="center", va="center")

        ax.scatter([xs[-1]], [y_line], s=38, marker="D", facecolor=COLORS["ink"],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(xs[-1], 0.77, "Anchored query", fontsize=7.8,
                fontweight="semibold", color=COLORS["ink"], ha="center", va="center")
        ax.text(xs[-1], 0.68, "latest meal style?", fontsize=8.5,
                color=COLORS["ink"], ha="center", va="center")
        ax.text(xs[-1], 0.43, "query", fontsize=8.0,
                color=COLORS["muted"], ha="center", va="center")

        outcome_kind = "stale" if case["label"] == "within_stale" else "cross"
        ax.plot([0.0, 1.0], [0.27, 0.27], color="#D9DEE3", linewidth=0.8)
        ax.text(0.0, 0.13, "Model answer", fontsize=8.0,
                color=COLORS["muted"], va="center")
        ax.text(0.19, 0.13, case["model_output"], fontsize=10.0,
                fontweight="semibold", color=colors[outcome_kind], va="center")
        ax.text(0.63, 0.13, "Current value", fontsize=8.0,
                color=COLORS["muted"], va="center")
        ax.text(0.80, 0.13, case["current_value"], fontsize=9.2,
                fontweight="semibold", color=colors["current"], va="center")

    save(fig, out_dir, "F_case_studies")


def figure_behavioral_dose(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), constrained_layout=True)
    p0 = data["p0"]
    ax = axes[0]
    for n_lines, color, marker in zip((20, 40, 80), (COLORS["current"], COLORS["cross"], COLORS["stale"]), ("o", "s", "^")):
        rows = sorted(
            [row for row in p0["cells"] if row["condition"] == "interference" and row["n_lines"] == n_lines],
            key=lambda row: row["interference_load"],
        )
        xs = [row["interference_load"] for row in rows]
        ys = [100 * row["accuracy"] for row in rows]
        errs = [error_pair(row["accuracy"], wilson(round(row["accuracy"] * row["n"]), row["n"])) for row in rows]
        ax.errorbar(xs, ys, yerr=100 * np.asarray(errs).T, marker=marker, color=color, linewidth=2, capsize=2, label=f"{n_lines} lines")
    ax.scatter([0], [100], marker="D", s=32, color=COLORS["correct"], label="no overwrite")
    ax.set(xlabel="Number of stale competitors", ylabel="Accuracy (%)", ylim=(25, 104), xticks=[0, 2, 4, 8], title="Controlled overwrite task")
    grid(ax)
    ax.legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.29),
        columnspacing=1.0, handlelength=1.5, handletextpad=0.45,
    )
    panel_label(ax, "a")

    ax = axes[1]
    l0 = data["l0"]
    doses = sorted(l0["by_dose"], key=int)
    x = np.arange(len(doses))
    bottom = np.zeros(len(doses))
    series = (
        ("Correct", "correct_current", "correct"),
        ("Within-slot stale", "within_stale", "stale"),
        ("Cross-slot", "cross_slot", "cross"),
        ("Other", "other", "other"),
    )
    for label, key, color_key in series:
        vals = np.array([100 * l0["by_dose"][dose]["counts"].get(key, 0) / l0["by_dose"][dose]["n"] for dose in doses])
        ax.bar(x, vals, bottom=bottom, width=0.72, label=label, color=COLORS[color_key])
        bottom += vals
    ax.set(xlabel="Number of overwrites", ylabel="Response composition (%)", ylim=(0, 100), xticks=x, xticklabels=doses, title="Natural preference dialogues")
    ax.legend(
        frameon=False, fontsize=7.4, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.29), columnspacing=1.0,
        handlelength=1.5, handletextpad=0.45,
    )
    panel_label(ax, "b")
    save(fig, out_dir, "F2_behavioral_dose")


def figure_foundational_breakpoints(data, out_dir):
    fig, axes = plt.subplots(
        1, 3, figsize=(7.65, 3.45), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.45, 1.18)},
    )

    ax = axes[0]
    p0 = data["p0"]
    for n_lines, color, marker in zip(
        (20, 40, 80),
        (COLORS["current"], COLORS["cross"], COLORS["stale"]),
        ("o", "s", "^"),
    ):
        rows = sorted(
            [row for row in p0["cells"] if row["condition"] == "interference" and row["n_lines"] == n_lines],
            key=lambda row: row["interference_load"],
        )
        xs = [row["interference_load"] for row in rows]
        ys = [100 * row["accuracy"] for row in rows]
        errs = [error_pair(row["accuracy"], wilson(round(row["accuracy"] * row["n"]), row["n"])) for row in rows]
        ax.errorbar(
            xs,
            ys,
            yerr=100 * np.asarray(errs).T,
            marker=marker,
            color=color,
            linewidth=1.25,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.55,
            capsize=1.8,
            label=f"{n_lines} lines",
        )
    ax.scatter(
        [0],
        [100],
        marker="D",
        s=20,
        color=COLORS["correct"],
        edgecolors="white",
        linewidths=0.55,
        label="no overwrite",
    )
    ax.set(xlabel="Old competing values", ylabel="Accuracy (%)", ylim=(25, 104),
           xticks=[0, 2, 4, 8], title="Controlled dose response")
    grid(ax)
    ax.legend(frameon=False, fontsize=6.7, ncol=2, loc="lower left",
              columnspacing=0.7, handletextpad=0.35)
    panel_label(ax, "a")

    standard_rows = data["stage_e_scale"]["accuracy_by_i"]
    hard_rows = data["stage_f_hard"]["cells"]
    model_specs = [
        ("qwen/qwen-2.5-7b-instruct", "Qwen2.5-7B", COLORS["stale"], "o"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B", COLORS["cross"], "s"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B", COLORS["current"], "^"),
        ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B", COLORS["correct"], "D"),
        ("openai/gpt-4o", "GPT-4o", "#A56A2A", "P"),
    ]

    ax = axes[1]
    for model, label, color, marker in model_specs:
        rows = sorted([row for row in standard_rows if row["model"] == model],
                      key=lambda row: row["interference_load"])
        xs = [row["interference_load"] for row in rows]
        ys = [100 * row["accuracy"] for row in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linewidth=1.25,
            markersize=3.6,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=label,
        )
        hard = sorted(
            [row for row in hard_rows if row["model"] == model
             and row["condition"] == "interference" and row["n_lines"] == 320],
            key=lambda row: row["interference_load"],
        )
        if hard:
            hard_x = [xs[-1]] + [row["interference_load"] for row in hard]
            hard_y = [ys[-1]] + [100 * row["accuracy"] for row in hard]
            ax.plot(
                hard_x,
                hard_y,
                color=color,
                marker=marker,
                linewidth=1.1,
                markersize=3.6,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="--",
            )
    ax.axhspan(0, 90, color="#F7F4F1", zorder=0)
    ax.axvline(22.6, color="#AAB1B7", linewidth=0.8, linestyle=":")
    ax.text(24.0, 5, "hard tier", rotation=90, fontsize=6.4, color=COLORS["muted"], va="bottom")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Old competing values (log scale)", ylabel="Accuracy (%)",
           ylim=(0, 104), xticks=[2, 4, 8, 16, 32, 64],
           xticklabels=["2", "4", "8", "16", "32", "64"],
           title="Scale shifts the break point")
    grid(ax)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.25), columnspacing=0.7,
              handlelength=1.4, handletextpad=0.35)
    panel_label(ax, "b")

    hardest = []
    for model, label, color, marker in model_specs:
        if model in {"qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.1-8b-instruct"}:
            row = next(row for row in standard_rows if row["model"] == model and row["interference_load"] == 16)
            stale_share = row["within_stale_errors"] / row["wrong"]
        else:
            row = next(row for row in hard_rows if row["model"] == model and row["condition"] == "interference"
                       and row["n_lines"] == 320 and row["interference_load"] == 64)
            stale_share = row["within_stale_rate_among_errors"]
            row = {**row, "wrong": round((1 - row["accuracy"]) * row["n"])}
        hardest.append((label, 100 * row["accuracy"], 100 * stale_share, row["wrong"]))

    ax = axes[2]
    x = np.arange(len(hardest))
    stale = np.array([row[2] for row in hardest])
    other = 100 - stale
    ax.bar(x, stale, color=COLORS["stale"], width=0.68, label="stale value")
    ax.bar(x, other, bottom=stale, color=COLORS["other"], width=0.68, label="other error")
    for i, (_, accuracy, stale_share, wrong) in enumerate(hardest):
        ax.text(i, 104, f"{accuracy:.0f}%", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold")
        ax.text(i, max(4, stale_share / 2), f"{stale_share:.0f}%", ha="center",
                va="center", fontsize=6.6, color="white" if stale_share > 35 else COLORS["ink"],
                fontweight="bold")
        ax.text(i, -11, f"{wrong} err", ha="center", va="top", fontsize=6.1,
                color=COLORS["muted"])
    short_labels = ["Qwen\n7B", "Llama\n8B", "Llama\n70B", "Qwen\n72B", "GPT-4o"]
    ax.set(ylabel="Composition of errors (%)", ylim=(-17, 115), xticks=x,
           xticklabels=short_labels, title="Old values dominate at failure")
    grid(ax)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.34), columnspacing=0.8, handletextpad=0.35)
    panel_label(ax, "c")

    save(fig, out_dir, "F2_foundational_breakpoints")


def figure_format_effect(data, out_dir):
    k1d = data["k1d"]
    rows = [
        ("DP\nfree-form", k1d["dynamic_preference_freeform"]),
        ("DP\nmultiple-choice", k1d["dynamic_preference_mc_contrast"]),
        ("IF\nfree-form", k1d["instructional_forgetting_k1c"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65), constrained_layout=True)
    x = np.arange(len(rows))
    acc = [100 * row["forget_accuracy"] for _, row in rows]
    axes[0].bar(x, acc, color=[COLORS["current"], COLORS["cross"], COLORS["stale"]], width=0.62)
    axes[0].set(ylabel="Task accuracy (%)", ylim=(0, 100), xticks=x, xticklabels=[name for name, _ in rows], title="Accuracy alone hides the error type")
    for i, value in enumerate(acc):
        axes[0].text(i, value + 3, f"{value:.1f}", ha="center", fontsize=8)
    grid(axes[0])
    panel_label(axes[0], "a")

    stale = [100 * row["within_stale_fraction"] for _, row in rows]
    errs = [error_pair(row["within_stale_fraction"], row["within_stale_ci"]) for _, row in rows]
    axes[1].bar(x, stale, color=[COLORS["stale"], COLORS["cross"], COLORS["stale"]], width=0.62)
    axes[1].errorbar(x, stale, yerr=100 * np.asarray(errs).T, fmt="none", color=COLORS["ink"], capsize=2)
    axes[1].set(ylabel="Within-slot stale among failures (%)", ylim=(0, 108), xticks=x, xticklabels=[name for name, _ in rows], title="Free-form exposes stale-value reuse")
    for i, (_, row) in enumerate(rows):
        axes[1].text(i, stale[i] + 5, f"{row['within_stale_failures']}/{row['forget_failures']}", ha="center", fontsize=8)
    grid(axes[1])
    panel_label(axes[1], "b")
    save(fig, out_dir, "F3_format_effect")


def figure_factorial(data, out_dir):
    cells = data["l0"]["by_factorial_cell"]
    row_keys = ["same_near", "same_far"]
    col_keys = ["other_far", "other_mid", "other_near2"]
    maps = []
    for metric in ("accuracy", "within_stale_rate", "cross_slot_rate"):
        maps.append(np.array([[100 * cells[f"{r}__{c}"][metric] for c in col_keys] for r in row_keys]))
    # Draw at the ICLR template's physical text width.  Keeping typography local
    # avoids inheriting the larger defaults used by the paper's overview figures.
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.72), constrained_layout=True)
    titles = ["Current value", "Old-value error", "Other-variable error"]
    palettes = ["Greens", "Oranges", "Blues"]
    for idx, (ax, values, title, cmap) in enumerate(zip(axes, maps, titles, palettes)):
        ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(3):
                text_color = "white" if values[i, j] > 55 else COLORS["ink"]
                ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center",
                        color=text_color, fontweight="medium", fontsize=7.2)
        ax.set_title(f"{chr(ord('a') + idx)}  {title}", loc="left",
                     fontsize=8.2, fontweight="semibold", pad=4)
        ax.set_xticks(range(3), ["far", "middle", "near\n(2 turns)"], fontsize=7.2)
        ax.set_yticks(range(2), ["near", "far"] if idx == 0 else ["", ""],
                      fontsize=7.0)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=2.5, width=0.6)
        if idx == 0:
            ax.set_ylabel("Old-value\ndistance", fontsize=7.8)
    fig.supxlabel("Distance of the competing value from another variable", fontsize=7.8)
    save(fig, out_dir, "F4_factorial_heatmap")


def failure_score(summary, label):
    return summary["probe"]["failure_pool_summary"]["failure_modes"][label]["decodability"]["mean_true_current_score"]


def figure_selection(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(6.35, 2.25), constrained_layout=True)
    models = [("Qwen2.5-7B", data["qwen_l1"]), ("Llama-3.1-8B", data["llama_l1"])]
    y = np.arange(2)[::-1]
    within = [100 * failure_score(summary, "within_stale") for _, summary in models]
    cross = [100 * failure_score(summary, "cross_slot") for _, summary in models]
    null_low = min(100 * summary["probe"]["within_stale_true_current_score_shuffle_null"]["ci"][0]
                   for _, summary in models)
    null_high = max(100 * summary["probe"]["within_stale_true_current_score_shuffle_null"]["ci"][1]
                    for _, summary in models)
    axes[0].axvspan(null_low, null_high, color=COLORS["null"], zorder=0, label="shuffle 95%")
    axes[0].scatter(within, y + 0.11, s=31, color=COLORS["stale"], marker="o",
                    label="within-slot stale", zorder=3)
    axes[0].scatter(cross, y - 0.11, s=31, color=COLORS["cross"], marker="s",
                    label="cross-slot", zorder=3)
    axes[0].set(xlabel="Current-value probe score (%)", xlim=(0, 100), yticks=y,
                yticklabels=[name for name, _ in models], title="Current remains\ndecodable")
    grid(axes[0], axis="x")
    panel_label(axes[0], "a")

    deltas = [100 * summary["probe"]["within_vs_correct_true_current_score"]["length_controlled_delta"] for _, summary in models]
    cis = [summary["probe"]["within_vs_correct_true_current_score"]["length_controlled_delta_ci"] for _, summary in models]
    errs = [error_pair(delta / 100, ci) for delta, ci in zip(deltas, cis)]
    axes[1].errorbar(deltas, y, xerr=100 * np.asarray(errs).T, fmt="none",
                     color=COLORS["ink"], capsize=2, elinewidth=1.0, zorder=2)
    axes[1].scatter(deltas, y, s=34, color=[COLORS["stale"], COLORS["cross"]], zorder=3)
    axes[1].axvline(0, color=COLORS["ink"], linewidth=0.8)
    axes[1].set(xlabel="Failure minus correct (points)", xlim=(-12, 1), yticks=y,
                yticklabels=[name for name, _ in models], title="Modest retention\ndecrease")
    grid(axes[1], axis="x")
    panel_label(axes[1], "b")

    qwen_attn = data["qwen_l1"]["attention_stale_ratio_within_vs_correct"]
    llama_attn = data["llama_l1"].get("attention_stale_ratio_within_vs_correct")
    attn_rows = [("Qwen2.5-7B", qwen_attn)]
    if llama_attn:
        attn_rows.append(("Llama-3.1-8B", llama_attn))
    vals = [100 * row["length_controlled_delta"] for _, row in attn_rows]
    err = [error_pair(row["length_controlled_delta"], row["length_controlled_delta_ci"]) for _, row in attn_rows]
    attn_y = np.arange(len(vals))[::-1]
    axes[2].errorbar(vals, attn_y, xerr=100 * np.asarray(err).T, fmt="none",
                     color=COLORS["ink"], capsize=2, elinewidth=1.0, zorder=2)
    axes[2].scatter(vals, attn_y, s=34,
                    color=[COLORS["stale"], COLORS["cross"]][: len(vals)], zorder=3)
    axes[2].axvline(0, color=COLORS["ink"], linewidth=0.8)
    axes[2].set(xlabel="Stale-attention shift (pt)", xlim=(-4, 23),
                yticks=attn_y, yticklabels=[name for name, _ in attn_rows],
                title="Qwen-only stale\nattention shift")
    grid(axes[2], axis="x")
    panel_label(axes[2], "c")
    for ax in axes:
        ax.title.set_fontsize(9.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.45,
    )
    save(fig, out_dir, "F5_selection_mechanism")


def run_by(data, model, kind):
    return next(row for row in data["validity_l2"]["runs"] if row["model"] == model and row["layer_kind"] == kind)


def figure_intervention(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.75), constrained_layout=True)
    alphas = data["validity_l2"]["alpha_list"]
    for model, label, color, marker in (("qwen", "Qwen2.5-7B", COLORS["stale"], "o"), ("llama", "Llama-3.1-8B", COLORS["cross"], "s")):
        row = run_by(data, model, "final")
        means = [100 * row["by_alpha"][str(alpha)]["target_error_reduction"]["mean"] for alpha in alphas]
        cis = [row["by_alpha"][str(alpha)]["target_error_reduction"]["ci"] for alpha in alphas]
        errs = [error_pair(mean / 100, ci) for mean, ci in zip(means, cis)]
        axes[0].errorbar(
            alphas,
            means,
            yerr=100 * np.asarray(errs).T,
            marker=marker,
            linewidth=1.25,
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=1.0,
            capsize=1.8,
            color=color,
            label=label,
        )
    axes[0].axhline(0, color=COLORS["ink"], linewidth=0.8)
    axes[0].set(xlabel=r"Steering strength $\alpha$ (fraction of residual norm)", ylabel="Targeted minus random\nerror reduction (points)", title="Dose response at the final layer")
    axes[0].legend(
        frameon=False,
        fontsize=8,
        handlelength=1.7,
        handletextpad=0.45,
    )
    grid(axes[0])
    panel_label(axes[0], "a")

    labels = []
    vals = []
    cis = []
    colors = []
    for model, model_label, color in (("qwen", "Qwen", COLORS["stale"]), ("llama", "Llama", COLORS["cross"])):
        for kind in ("middle", "final"):
            row = run_by(data, model, kind)
            labels.append(f"{model_label}\n{kind}")
            vals.append(100 * row["best_target_error_reduction"]["mean"])
            cis.append(row["best_target_error_reduction"]["ci"])
            colors.append(color if kind == "final" else COLORS["other"])
    errs = [error_pair(value / 100, ci) for value, ci in zip(vals, cis)]
    axes[1].bar(range(4), vals, color=colors, width=0.62)
    axes[1].errorbar(range(4), vals, yerr=100 * np.asarray(errs).T, fmt="none", color=COLORS["ink"], capsize=2)
    axes[1].axhline(0, color=COLORS["ink"], linewidth=0.8)
    axes[1].set(ylabel="Best targeted minus random\nerror reduction (points)", xticks=range(4), xticklabels=labels, title="The lever is concentrated at the output end")
    grid(axes[1])
    panel_label(axes[1], "b")
    save(fig, out_dir, "F6_intervention_validity")


def figure_geometry(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65), constrained_layout=True)
    models = [("Qwen2.5-7B", data["geometry"]["models"]["qwen"]), ("Llama-3.1-8B", data["geometry"]["models"]["llama"])]
    x = np.arange(2)
    width = 0.24
    geom = [100 * row["g1_selection_argmax"]["geometric_accuracy"] for _, row in models]
    identity = [100 * row["g1_selection_argmax"]["identity_only_accuracy"] for _, row in models]
    recency = [100 * row["g1_selection_argmax"]["recency_only_accuracy"] for _, row in models]
    axes[0].bar(x - width, geom, width, color=COLORS["current"], label="joint geometry")
    axes[0].bar(x, identity, width, color=COLORS["stale"], label="identity-only")
    axes[0].bar(x + width, recency, width, color=COLORS["cross"], label="recency-only")
    axes[0].set(ylabel="Out-of-sample choice accuracy (%)", ylim=(0, 100), xticks=x, xticklabels=[name for name, _ in models], title="Joint geometry loses to identity-only")
    axes[0].legend(frameon=False, fontsize=7)
    grid(axes[0])
    panel_label(axes[0], "a")

    sign = [100 * row["g2_crossover_prediction_length_controlled"]["crossover_sign_accuracy"] for _, row in models]
    mae = [100 * row["g2_crossover_prediction_length_controlled"]["mean_absolute_cross_share_error"] for _, row in models]
    axes[1].bar(x - 0.18, sign, 0.36, color=COLORS["current"], label="crossover sign accuracy")
    axes[1].bar(x + 0.18, mae, 0.36, color=COLORS["other"], label="cross-share absolute error")
    axes[1].set(ylabel="Rate / error (%)", ylim=(0, 100), xticks=x, xticklabels=[name for name, _ in models], title="The crossover remains unexplained")
    axes[1].legend(frameon=False, fontsize=7)
    grid(axes[1])
    panel_label(axes[1], "b")
    save(fig, out_dir, "F7_geometry_boundary")


def figure_circuit_localization(data, out_dir):
    circuit = data["stage_g_circuit"]
    effects = {row["component"]: row for row in circuit["patching"]["component_effects"]}
    components = [
        ("residual_stream_late", "Late residual stream"),
        ("qk_key_stale_writes", "Stale-write keys"),
        ("selection_head_output_proxy", "Selection-head output"),
        ("late_mlp_output", "Late MLP output"),
        ("selection_head_attention_pattern_proxy", "Selection-head attention"),
        ("qk_query_final", "Final query"),
        ("qk_key_all_writes", "All write keys"),
        ("residual_stream_early_mid", "Early/mid residual"),
        ("qk_key_current_write", "Current-write key"),
    ]
    recovery = np.array([100 * effects[key]["delta_recovery_mean"] for key, _ in components])
    labels = [label for _, label in components]
    colors = [
        COLORS["current"], COLORS["stale"], COLORS["cross"], "#7A8793",
        "#7A8793", "#7A8793", "#7A8793", COLORS["other"], "#C85D4A",
    ]

    fig = plt.figure(figsize=(7.5, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.62, 0.72, 1.02))

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(labels))
    ax.barh(y, recovery, color=colors, height=0.62)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-28, 88)
    ax.set_xlabel("Recovered clean-corrupt gap (%)")
    ax.set_title("Where patching restores the answer")
    grid(ax, axis="x")
    for yi, value in zip(y, recovery):
        if value < -8:
            xpos, align, color = value + 1.6, "left", "white"
        elif value < 0:
            xpos, align, color = value - 2.0, "right", COLORS["ink"]
        else:
            xpos, align, color = value + 2.0, "left", COLORS["ink"]
        ax.text(xpos, yi, f"{value:.1f}", va="center", ha=align,
                fontsize=7.3, fontweight="bold", color=color)
    panel_label(ax, "a")

    baseline = circuit["geometry"]["baseline"]
    arm_l = circuit["geometry"]["arm_l"]
    arm_names = ["Baseline", "Adapter"]
    arm_colors = [COLORS["correct"], COLORS["stale"]]

    ax = fig.add_subplot(gs[0, 1])
    margins = [
        baseline["mean_current_minus_stale_qk"],
        arm_l["mean_current_minus_stale_qk"],
    ]
    bars = ax.bar(np.arange(2), margins, color=arm_colors, width=0.62)
    ax.set_xticks(np.arange(2), arm_names, rotation=18, ha="right")
    ax.set_ylabel("Mean QK margin")
    ax.set_ylim(0, 58)
    ax.set_title("Current vs. stale keys")
    grid(ax)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in margins], padding=2,
                 fontsize=7.3, fontweight="bold")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    width = 0.34
    identity = [baseline["mean_query_identity_alignment"], arm_l["mean_query_identity_alignment"]]
    recency = [baseline["mean_query_recency_alignment"], arm_l["mean_query_recency_alignment"]]
    bars_identity = ax.bar(x - width / 2, identity, width, color=COLORS["current"], label="Identity")
    bars_recency = ax.bar(x + width / 2, recency, width, color=COLORS["cross"], label="Recency")
    ax.set_xticks(x, arm_names)
    ax.set_ylabel("Query-axis cosine")
    ax.set_ylim(0, 0.30)
    ax.set_title("What the query encodes")
    grid(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=7.2, columnspacing=0.8, handletextpad=0.35)
    ax.bar_label(bars_identity, labels=[f"{v:.2f}" for v in identity], padding=2,
                 fontsize=6.8)
    ax.bar_label(bars_recency, labels=[f"{v:.2f}" for v in recency], padding=2,
                 fontsize=6.8)
    panel_label(ax, "c")

    save(fig, out_dir, "F8_circuit_localization")


def tex_escape(value):
    text = str(value)
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_table(path, caption, label, columns, rows, notes=None):
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(tex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    if notes:
        lines.append(r"\vspace{2pt}\parbox{0.96\linewidth}{\footnotesize " + notes + "}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    k1d = data["k1d"]
    l0 = data["l0"]["overall"]
    behavior_rows = [
        ("Synthetic, no overwrite", "Qwen2.5-7B", "exact", 300, "100.0", 0, "--"),
        ("ICF dynamic preference", "Qwen2.5-7B", "free-form", 783, "69.3", 240, "82.1 [77.1, 87.1]"),
        ("ICF dynamic preference", "Qwen2.5-7B", "multiple-choice", 783, "70.9", 228, "29.4 [23.2, 35.5]"),
        ("ICF instruction forgetting", "Qwen2.5-7B", "free-form", 933, "1.4", 920, "98.5 [97.6, 99.2]"),
        ("Controlled preference dialogues", "Qwen2.5-7B", "free-form", l0["n"], f"{100*l0['accuracy']:.1f}", l0["n"] - l0["counts"]["correct_current"], f"{100*l0['within_stale_failure_share']:.1f}"),
    ]
    write_table(
        out_dir / "behavioral_evidence.tex",
        "Behavioral evidence across controlled and external settings. The last column is conditional on an error and therefore is not a prevalence estimate.",
        "tab:behavior",
        ("Setting", "Model", "Answer", "$n$", "Acc. (\%)", "Errors", "Within-slot stale / errors (\%)"),
        behavior_rows,
        "CICM and synthetic labels are deterministic. ICF Dynamic Preference free-form uses deterministic parsing plus a GPT-4o judge for unresolved responses; human agreement is 0.90, and its 82.1\% stale share is treated as an approximate upper bound.",
    )

    hard_cells = data["stage_f_hard"]["cells"]
    hard_models = [
        ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B"),
        ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B"),
        ("openai/gpt-4o", "GPT-4o"),
    ]
    hard_rows = []
    for model, display in hard_models:
        simple = next(row for row in hard_cells if row["model"] == model and row["condition"] == "simple")
        i32 = next(row for row in hard_cells if row["model"] == model and row["condition"] == "interference"
                   and row["n_lines"] == 320 and row["interference_load"] == 32)
        i64 = next(row for row in hard_cells if row["model"] == model and row["condition"] == "interference"
                   and row["n_lines"] == 320 and row["interference_load"] == 64)
        errors = round((1 - i64["accuracy"]) * i64["n"])
        hard_rows.append((
            display,
            f"{100 * simple['accuracy']:.0f}",
            f"{100 * i32['accuracy']:.0f}",
            f"{100 * i64['accuracy']:.0f}",
            errors,
            f"{100 * i64['within_stale_rate_among_errors']:.0f}",
        ))
    write_table(
        out_dir / "hard_tier_breakpoints.tex",
        "Hard-tier endpoints for models that saturated the standard sweep. Each cell contains 50 examples and 320 context lines.",
        "tab:hard-tier",
        ("Model", "No overwrite", "$I=32$", "$I=64$", "Errors at $I=64$", "Stale / errors (\\%)"),
        hard_rows,
        "Accuracy columns are percentages. The no-overwrite control holds length fixed without old competing assignments.",
    )

    def aggregate_external(summary, benchmark):
        cells = [row for row in summary["cells"] if row["benchmark"] == benchmark]
        n = sum(row["n"] for row in cells)
        metric_key = "headline_metric" if benchmark == "ruler_vt" else "accuracy"
        headline = sum(row[metric_key] * row["n"] for row in cells) / n
        errors = sum(row["n"] * (1 - row["accuracy"]) for row in cells)
        superseded = sum(
            row["n"] * (1 - row["accuracy"]) * row["superseded_error_rate"]
            for row in cells
        )
        cross_chain = sum(
            row["n"] * (1 - row["accuracy"]) * row["cross_chain_inclusion_error_rate"]
            for row in cells
        )
        return {
            "n": n,
            "headline": headline,
            "superseded_share": superseded / errors if errors else 0,
            "cross_chain_share": cross_chain / errors if errors else 0,
        }

    external_rows = []
    external_specs = [
        ("babilong", "BABILong", "accuracy", "superseded state"),
        ("entity_tracking", "Entity Tracking", "set accuracy", "superseded state"),
        ("ruler_vt", "RULER-VT", "variable recall", "cross-chain inclusion"),
    ]
    for key, display, metric, error_name in external_specs:
        baseline = aggregate_external(data["stage_g_ext_baseline"], key)
        arm_l = aggregate_external(data["stage_g_ext_arm_l"], key)
        random = aggregate_external(data["stage_g_ext_random"], key)
        error_share = (
            baseline["cross_chain_share"] if key == "ruler_vt"
            else baseline["superseded_share"]
        )
        external_rows.append((
            display,
            baseline["n"],
            metric,
            f"{100 * baseline['headline']:.1f}",
            f"{100 * error_share:.1f} {error_name}",
            f"{100 * arm_l['headline']:.1f}",
            f"{100 * random['headline']:.1f}",
            f"{100 * (arm_l['headline'] - random['headline']):+.1f}",
        ))
    write_table(
        out_dir / "community_external_benchmarks.tex",
        "Audited community-benchmark results for Qwen2.5-7B. Scores aggregate all evaluated cells within each benchmark.",
        "tab:community-external",
        ("Benchmark", "$n$", "Headline metric", "Baseline", "Baseline error signature", "Arm-L", "Random", "Arm-L $-$ random"),
        external_rows,
        "All numeric entries except n are percentages or percentage-point differences. Earlier-state returns are defined only for BABILong and Entity Tracking; RULER-VT uses its distinct cross-chain-inclusion taxonomy. Arm-L is the diagnostic latest-binding adapter, and Random is a matched random-head adapter.",
    )

    order = [
        "same_near__other_far", "same_near__other_mid", "same_near__other_near2",
        "same_far__other_far", "same_far__other_mid", "same_far__other_near2",
    ]
    factorial_rows = []
    for key in order:
        row = data["l0"]["by_factorial_cell"][key]
        same, other = key.split("__")
        factorial_rows.append((same.replace("same_", ""), other.replace("other_", "").replace("near2", "2 turns"), row["n"], f"{100*row['accuracy']:.1f}", f"{100*row['within_stale_rate']:.1f}", f"{100*row['cross_slot_rate']:.1f}", f"{100*row['other_rate']:.1f}"))
    write_table(
        out_dir / "factorial_cells.tex",
        "Complete response composition for the identity-by-recency factorial. Each row contains 200 examples.",
        "tab:factorial",
        ("Old same-slot", "Other-slot", "$n$", "Correct", "Stale", "Cross-slot", "Other"),
        factorial_rows,
        "All rates are percentages of all examples in the cell. The two rightmost far-same cells show a tie, not a global reversal.",
    )

    mechanism_rows = []
    for model_name, summary in (("Qwen2.5-7B", data["qwen_l1"]), ("Llama-3.1-8B", data["llama_l1"])):
        probe = summary["probe"]
        null = probe["within_stale_true_current_score_shuffle_null"]["ci"]
        comparison = probe["within_vs_correct_true_current_score"]
        mechanism_rows.append((model_name, probe["n"], f"{100*probe['cv_accuracy']:.1f}", f"{100*failure_score(summary, 'within_stale'):.1f}", f"{100*failure_score(summary, 'cross_slot'):.1f}", f"[{100*null[0]:.1f}, {100*null[1]:.1f}]", f"{100*comparison['length_controlled_delta']:.1f} [{100*comparison['length_controlled_delta_ci'][0]:.1f}, {100*comparison['length_controlled_delta_ci'][1]:.1f}]"))
    write_table(
        out_dir / "mechanism_results.tex",
        "Current-value decodability with grouped cross-validation. Failure scores remain far above the shuffled-label range, while the negative controlled difference shows a smaller retention degradation.",
        "tab:mechanism",
        ("Model", "$n$", "CV acc.", "Within stale", "Cross-slot", "Shuffle 95\%", "Failure $-$ correct"),
        mechanism_rows,
    )

    intervention_rows = []
    for model, display in (("qwen", "Qwen2.5-7B"), ("llama", "Llama-3.1-8B")):
        for kind in ("middle", "final"):
            row = run_by(data, model, kind)
            effect = row["best_target_error_reduction"]
            intervention_rows.append((display, kind, row["layer"], row["best_alpha"], f"{100*effect['mean']:.1f} [{100*effect['ci'][0]:.1f}, {100*effect['ci'][1]:.1f}]", "yes" if row["l2_gate_pass"] else "no", row["identity_gate_mismatches"]))
    write_table(
        out_dir / "intervention_validity.tex",
        "Residual-normalized steering validity. Large effects occur only at the final layer; middle-layer effects are null or small.",
        "tab:intervention",
        ("Model", "Location", "Layer", r"$\alpha$", "Targeted $-$ random (points)", "Gate", "Identity mismatches"),
        intervention_rows,
    )


def normalize_case(row):
    competition = row["competition"]
    selected_indices = {
        competition["target_current"]["message_index"],
        competition["same_slot_stale"][0]["message_index"],
        competition["same_slot_stale_nearest"]["message_index"],
        competition["recent_other_slot_nearest"]["message_index"],
        len(row["messages"]) - 1,
    }
    snippets = [
        {"message_index": index, "role": row["messages"][index]["role"], "content": row["messages"][index]["content"]}
        for index in sorted(selected_indices)
    ]
    return {
        "id": row["id"],
        "label": row["label"],
        "target_slot": row["slot_label"],
        "current_value": row["current_value"],
        "stale_value": competition["same_slot_stale_nearest"]["value"],
        "other_slot": competition["recent_other_slot_nearest"]["slot"],
        "other_value": competition["recent_other_slot_nearest"]["value"],
        "query": row["query"],
        "model_output": row["response"],
        "factorial_cell": row["factorial_cell"],
        "snippets": snippets,
    }


def select_cases(root):
    rows = load_jsonl(root / "results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_rows_merged.jsonl")
    within = next(row for row in rows if row["label"] == "within_stale" and row["id"] == "stage_l_natfact_1_00000")
    cross = next(row for row in rows if row["label"] == "cross_slot" and row["id"] == "stage_l_natfact_1_00015")
    return [normalize_case(within), normalize_case(cross)]


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def latex_highlight(text, terms, color):
    rendered = latex_escape(text)
    for term in sorted(terms, key=len, reverse=True):
        escaped = latex_escape(term)
        rendered = rendered.replace(
            escaped,
            rf"\textcolor{{{color}}}{{\textbf{{{escaped}}}}}",
        )
    return rendered


def role_turn(environment, role, content):
    return (
        rf"\begin{{{environment}}}" "\n"
        rf"\textbf{{{role}.}} {content}" "\n"
        rf"\end{{{environment}}}" "\n"
    )


def omitted_turn(count):
    return (
        "\\begin{omittedturn}\n"
        rf"\centering [{count} turns omitted]" "\n"
        "\\end{omittedturn}\n"
    )


def find_jsonl_item(path, item_id):
    wanted = str(item_id)
    for row in load_jsonl(path):
        if str(row.get("id")) == wanted:
            return row
    raise ValueError(f"item {item_id} not found in {path}")


def write_verbatim_transcripts(root, out_path):
    if_setup_path = root / "external_data/icf_bench/instructional_forgetting/instructional_forgetting.json"
    if_output_path = root / "results/stage_k/mech/smoke/instructional_forgetting_index.jsonl"
    dp_path = root / "results/stage_k/freeform_dp_sample_claude.json"
    cicm_path = root / "results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_rows_merged.jsonl"

    if_setup = next(row for row in load_json(if_setup_path) if str(row["id"]) == "0")
    if_output = find_jsonl_item(if_output_path, "0")
    assert if_setup["forget_instruction"] == "Forget the numbers involved in our previous calculations."
    assert if_output["final_error_type"] == "within_stale"
    assert "- 5 and 8" in if_output["raw"] and "- 25 and 13" in if_output["raw"]

    dp_item = next(row for row in load_json(dp_path)["items"] if int(row["id"]) == 120)
    assert dp_item["claude_hand_label"] == "within_stale"
    assert "Ultra Music Festival" in dp_item["freeform_response"]
    assert dp_item["new_pref"] == "I avoid electronic music."

    cicm = find_jsonl_item(cicm_path, "stage_l_natfact_1_00101")
    assert cicm["label"] == "cross_slot"
    assert cicm["current_value"] == "classical" and cicm["response"] == "pescatarian"
    assert "music genre" in cicm["query"] and "not any other preference" in cicm["query"]

    if_lines = [line.strip() for line in if_output["raw"].splitlines() if line.strip()]
    if_rendered = []
    for line in if_lines:
        if line.startswith("- "):
            if_rendered.append(rf"\textcolor{{staleink}}{{\textbf{{{latex_escape(line)}}}}}")
        else:
            if_rendered.append(latex_escape(line))
    if_response = r"\par ".join(if_rendered)

    dp_response = latex_highlight(
        dp_item["freeform_response"],
        ["For electronic music", "Ultra Music Festival", "Electric Daisy Carnival (EDC)"],
        "staleink",
    )
    cicm_messages = cicm["messages"]

    chunks = [
        r"\begin{minipage}{0.98\linewidth}",
        r"\textbf{Case 1: instructional forgetting exposes forbidden numbers}\par",
        role_turn("systemturn", "System", latex_escape("You are a helpful assistant.")),
        role_turn("userturn", "User", latex_escape(if_setup["conversations"][0]["value"])),
        role_turn("assistantturn", "Assistant", latex_escape(if_setup["conversations"][1]["value"])),
        omitted_turn(10),
        role_turn("userturn", "User", latex_escape(if_setup["forget_instruction"])),
        role_turn("userturn", "User", latex_escape(if_setup["test_query"])),
        role_turn("assistantturn", "Assistant", if_response),
        r"{\footnotesize\textbf{Expected:} \textcolor{currentink}{\textbf{withhold the numbers}}"
        r"\quad\textbf{Model:} \textcolor{staleink}{\textbf{re-states the forgotten numbers}}}\par",
        r"{\scriptsize\textbf{Source.} Setup: \texttt{instructional\_forgetting.json}, item 0; "
        r"model output: \texttt{instructional\_forgetting\_index.jsonl}, item 0.}",
        r"\end{minipage}\par",
        r"\vspace{2mm}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\textbf{Case 2: a revised preference is ignored}\par",
        role_turn("systemturn", "System", latex_escape("You are a useful assistant. Make the most reasonable recommendation according to my current preferences.")),
        role_turn("userturn", "User", latex_escape(dp_item["old_pref"])),
        role_turn("assistantturn", "Assistant", latex_escape("I understand your preference.")),
        role_turn("userturn", "User", latex_highlight(dp_item["new_pref"], [dp_item["new_pref"]], "currentink")),
        role_turn("assistantturn", "Assistant", latex_escape("I understand your updated preference.")),
        role_turn("userturn", "User", latex_escape(dp_item["bare_question"])),
        role_turn("assistantturn", "Assistant", dp_response),
        r"{\footnotesize\textbf{Expected:} \textcolor{currentink}{\textbf{avoid electronic-music recommendations}}"
        r"\quad\textbf{Model:} \textcolor{staleink}{\textbf{recommends electronic festivals}}}\par",
        r"{\scriptsize\textbf{Source.} \texttt{freeform\_dp\_sample\_claude.json}, item 120; "
        r"prompt wrapper: \texttt{icf\_k1d.py}.}",
        r"\end{minipage}\par",
        r"\vspace{2mm}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\textbf{Case 3: an anchored query captures another slot}\par",
        role_turn("systemturn", "System", latex_escape(cicm_messages[0]["content"])),
        role_turn("userturn", "User", latex_highlight(cicm_messages[3]["content"], ["folk"], "staleink")),
        role_turn("assistantturn", "Assistant", latex_escape(cicm_messages[4]["content"])),
        role_turn("userturn", "User", latex_highlight(cicm_messages[7]["content"], ["classical"], "currentink")),
        role_turn("assistantturn", "Assistant", latex_escape(cicm_messages[8]["content"])),
        omitted_turn(22),
        role_turn("userturn", "User", latex_highlight(cicm_messages[31]["content"], ["pescatarian"], "crossink")),
        role_turn("assistantturn", "Assistant", latex_escape(cicm_messages[32]["content"])),
        omitted_turn(2),
        role_turn("userturn", "User", latex_escape(cicm["query"])),
        role_turn("assistantturn", "Assistant", latex_highlight(cicm["response"], [cicm["response"]], "staleink")),
        r"{\footnotesize\textbf{Expected:} \textcolor{currentink}{\textbf{classical}}"
        r"\quad\textbf{Model:} \textcolor{staleink}{\textbf{pescatarian}}}\par",
        r"{\scriptsize\textbf{Source.} \texttt{openrouter\_qwen25\_7b\_rows\_merged.jsonl}, "
        r"item \texttt{stage\_l\_natfact\_1\_00101}.}",
        r"\end{minipage}\par",
        r"\vspace{1mm}",
        r"{\footnotesize Gray, blue, and green boxes denote system, user, and assistant turns. Red text marks the produced stale or wrong value; green text marks the current target.}",
    ]
    out_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def generate(root, out_dir):
    root = Path(root)
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(root)
    apply_style()
    figure_task_overview(figures)
    cases = select_cases(root)
    figure_case_studies(cases, figures)
    figure_behavioral_dose(data, figures)
    figure_foundational_breakpoints(data, figures)
    figure_format_effect(data, figures)
    figure_factorial(data, figures)
    figure_selection(data, figures)
    figure_intervention(data, figures)
    figure_geometry(data, figures)
    figure_circuit_localization(data, figures)
    write_tables(data, tables)
    write_verbatim_transcripts(root, out_dir / "verbatim_transcripts.tex")
    (out_dir / "case_studies.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "generated_by": "src/stage_l_paper_full_assets.py",
        "inputs": SOURCES,
        "figures": sorted(path.name for path in figures.glob("*.pdf")),
        "tables": sorted(path.name for path in tables.glob("*.tex")),
        "case_ids": [case["id"] for case in cases],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="paper/full_draft_assets")
    args = parser.parse_args()
    generate(args.root, args.out)


if __name__ == "__main__":
    main()
