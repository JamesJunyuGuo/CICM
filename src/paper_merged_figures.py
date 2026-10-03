"""Publication figures and compact result table for the stale-binding paper.

Keeps trends in figures and moves discrete comparisons into one dense table:

  Fig_phenomenon  (1x2) = controlled dose + scale break point
  Fig_causality   (1x1) = steering dose response
  discrete_results_summary.tex = endpoints + mechanism + causality + geometry

All numbers are read through the SAME loaders/helpers as
`stage_l_paper_full_assets.py`, so no value changes; only layout and style do.
Run with the shared verl python; writes PDF+PNG into
`paper/full_draft_assets/figures/`.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import stage_l_paper_full_assets as base
from paper_figure_style import LENGTH_COLORS, MODEL_COLORS, apply_paper_style
from stage_l_paper_full_assets import (
    COLORS,
    error_pair,
    failure_score,  # noqa: F401  (kept for parity / future mechanism merge)
    load_inputs,
    run_by,
    wilson,
)


# --- shared style -----------------------------------------------------------
def apply_style():
    apply_paper_style()


def panel_label(ax, label):
    ax.text(-0.14, 1.09, label, transform=ax.transAxes,
            fontweight="semibold", fontsize=9.5)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


MK = dict(markersize=4.2, markeredgecolor="white", markeredgewidth=0.6)


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# --- phenomenon panels ------------------------------------------------------
def panel_dose(ax, data):
    p0 = data["p0"]
    for n_lines, color, marker in zip(
        (20, 40, 80),
        LENGTH_COLORS,
        ("o", "s", "^"),
    ):
        rows = sorted(
            [r for r in p0["cells"] if r["condition"] == "interference" and r["n_lines"] == n_lines],
            key=lambda r: r["interference_load"],
        )
        xs = [r["interference_load"] for r in rows]
        ys = [100 * r["accuracy"] for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.2,
                label=f"{n_lines} lines", **MK)
    ax.scatter([0], [100], marker="D", s=26, color=COLORS["correct"],
               edgecolors="white", linewidths=0.6, label="no overwrite")
    ax.set(xlabel="Number of old values", ylabel="Accuracy (%)", ylim=(25, 104),
           xticks=[0, 2, 4, 8], title="More old values reduce accuracy")
    grid(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower left",
              columnspacing=0.7, handletextpad=0.35)


def panel_scale(ax, data):
    standard_rows = data["stage_e_scale"]["accuracy_by_i"]
    hard_rows = data["stage_f_hard"]["cells"]
    model_specs = [
        ("qwen/qwen-2.5-7b-instruct", "Qwen2.5-7B", MODEL_COLORS[0], "o"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B", MODEL_COLORS[1], "s"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B", MODEL_COLORS[2], "^"),
        ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B", MODEL_COLORS[3], "D"),
        ("openai/gpt-4o", "GPT-4o", MODEL_COLORS[4], "P"),
    ]
    for model, label, color, marker in model_specs:
        rows = sorted([r for r in standard_rows if r["model"] == model],
                      key=lambda r: r["interference_load"])
        xs = [r["interference_load"] for r in rows]
        ys = [100 * r["accuracy"] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=1.2, label=label, **MK)
        hard = sorted(
            [r for r in hard_rows if r["model"] == model and r["condition"] == "interference"
             and r["n_lines"] == 320],
            key=lambda r: r["interference_load"],
        )
        if hard:
            hx = [xs[-1]] + [r["interference_load"] for r in hard]
            hy = [ys[-1]] + [100 * r["accuracy"] for r in hard]
            ax.plot(hx, hy, color=color, marker=marker, linewidth=1.2, linestyle="--", **MK)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Number of old values (log scale)", ylabel="Accuracy (%)", ylim=(0, 104),
           xticks=[2, 4, 8, 16, 32, 64], xticklabels=["2", "4", "8", "16", "32", "64"],
           title="Larger models fail at higher loads")
    grid(ax)
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), columnspacing=0.7, handlelength=1.4, handletextpad=0.35)


def panel_composition(ax, data):
    standard_rows = data["stage_e_scale"]["accuracy_by_i"]
    hard_rows = data["stage_f_hard"]["cells"]
    specs = [
        ("qwen/qwen-2.5-7b-instruct", "Qwen\n7B"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama\n8B"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama\n70B"),
        ("qwen/qwen-2.5-72b-instruct", "Qwen\n72B"),
        ("openai/gpt-4o", "GPT-4o"),
    ]
    hardest = []
    for model, label in specs:
        if model in {"qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.1-8b-instruct"}:
            row = next(r for r in standard_rows if r["model"] == model and r["interference_load"] == 16)
            stale_share = row["within_stale_errors"] / row["wrong"]
            wrong = row["wrong"]
            acc = row["accuracy"]
        else:
            row = next(r for r in hard_rows if r["model"] == model and r["condition"] == "interference"
                       and r["n_lines"] == 320 and r["interference_load"] == 64)
            stale_share = row["within_stale_rate_among_errors"]
            acc = row["accuracy"]
            wrong = round((1 - acc) * row["n"])
        hardest.append((label, 100 * acc, 100 * stale_share, wrong))

    y = np.arange(len(hardest))[::-1]
    acc = np.array([r[1] for r in hardest])
    stale = np.array([r[2] for r in hardest])
    for yi, a, s in zip(y, acc, stale):
        ax.plot([a, s], [yi, yi], color=COLORS["grid"], linewidth=1.3, zorder=1)
    ax.scatter(acc, y, s=28, color=COLORS["correct"], marker="o",
               label="task accuracy", zorder=3)
    ax.scatter(stale, y, s=31, color=COLORS["stale"], marker="s",
               label="stale share of failures", zorder=3)
    labels = [f"{r[0].replace(chr(10), ' ')}  ({r[3]} err)" for r in hardest]
    ax.set(xlabel="Percent (%)", xlim=(0, 104), yticks=y, yticklabels=labels,
           title="Failures are predominantly stale")
    grid(ax, axis="x")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left",
              handletextpad=0.4, labelspacing=0.35)


def panel_format(ax, data):
    k1d = data["k1d"]
    rows = [
        ("DP\nfree-form", k1d["dynamic_preference_freeform"], COLORS["stale"]),
        ("DP\nmult.-choice", k1d["dynamic_preference_mc_contrast"], COLORS["cross"]),
        ("IF\nfree-form", k1d["instructional_forgetting_k1c"], COLORS["stale"]),
    ]
    y = np.arange(len(rows))[::-1]
    acc = [100 * r["forget_accuracy"] for _, r, _ in rows]
    stale = [100 * r["within_stale_fraction"] for _, r, _ in rows]
    errs = [error_pair(r["within_stale_fraction"], r["within_stale_ci"]) for _, r, _ in rows]
    for yi, a, s in zip(y, acc, stale):
        ax.plot([a, s], [yi, yi], color=COLORS["grid"], linewidth=1.3, zorder=1)
    ax.scatter(acc, y, s=28, color=COLORS["correct"], marker="o",
               label="task accuracy", zorder=3)
    ax.errorbar(stale, y, xerr=100 * np.asarray(errs).T, fmt="s", markersize=4.8,
                color=COLORS["stale"], capsize=2, elinewidth=0.9,
                label="stale share of failures", zorder=3)
    ax.set(xlabel="Percent (%)", xlim=(0, 104), yticks=y,
           yticklabels=[r[0].replace("\n", " ") for r in rows],
           title="Multiple choice masks stale reuse")
    grid(ax, axis="x")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left",
              handletextpad=0.4, labelspacing=0.35)


# --- causality panels -------------------------------------------------------
def panel_steer_dose(ax, data):
    alphas = data["validity_l2"]["alpha_list"]
    for model, label, color, marker in (("qwen", "Qwen2.5-7B", COLORS["stale"], "o"),
                                         ("llama", "Llama-3.1-8B", COLORS["cross"], "s")):
        row = run_by(data, model, "final")
        means = [100 * row["by_alpha"][str(a)]["target_error_reduction"]["mean"] for a in alphas]
        cis = [row["by_alpha"][str(a)]["target_error_reduction"]["ci"] for a in alphas]
        errs = [error_pair(m / 100, ci) for m, ci in zip(means, cis)]
        ax.errorbar(alphas, means, yerr=100 * np.asarray(errs).T, marker=marker,
                    linewidth=1.4, elinewidth=1.0, capsize=1.8, color=color, label=label, **MK)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set(xlabel=r"Steering strength $\alpha$ (fraction of residual norm)",
           ylabel="Targeted minus random\nerror reduction (points)",
           title="Dose response at the final layer")
    ax.legend(frameon=False, fontsize=7.4, handlelength=1.7, handletextpad=0.45)
    grid(ax)


def panel_layers(ax, data):
    labels, vals, cis, colors = [], [], [], []
    for model, ml, color in (("qwen", "Qwen", COLORS["stale"]), ("llama", "Llama", COLORS["cross"])):
        for kind in ("middle", "final"):
            row = run_by(data, model, kind)
            labels.append(f"{ml}\n{kind}")
            vals.append(100 * row["best_target_error_reduction"]["mean"])
            cis.append(row["best_target_error_reduction"]["ci"])
            colors.append(color if kind == "final" else COLORS["other"])
    errs = [error_pair(v / 100, ci) for v, ci in zip(vals, cis)]
    y = np.arange(4)[::-1]
    ax.errorbar(vals, y, xerr=np.asarray(errs).T * 100, fmt="none",
                color=COLORS["ink"], capsize=2, elinewidth=1.0, zorder=2)
    for yi, value, color, label in zip(y, vals, colors, labels):
        filled = "final" in label
        ax.scatter(value, yi, s=36, facecolor=color if filled else "white",
                   edgecolor=color, linewidth=1.3, zorder=3)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set(xlabel="Targeted minus random error reduction (points)", xlim=(-5, 76),
           yticks=y, yticklabels=[label.replace("\n", " ") for label in labels],
           title="Control localizes to the final layer")
    grid(ax, axis="x")


def panel_geometry(ax, data):
    models = [("Qwen2.5-7B", data["geometry"]["models"]["qwen"]),
              ("Llama-3.1-8B", data["geometry"]["models"]["llama"])]
    geom = [100 * r["g1_selection_argmax"]["geometric_accuracy"] for _, r in models]
    identity = [100 * r["g1_selection_argmax"]["identity_only_accuracy"] for _, r in models]
    recency = [100 * r["g1_selection_argmax"]["recency_only_accuracy"] for _, r in models]
    y = np.arange(2)[::-1]
    null_low = min(100 * r["g1_selection_argmax"]["shuffle_null"]["ci"][0] for _, r in models)
    null_high = max(100 * r["g1_selection_argmax"]["shuffle_null"]["ci"][1] for _, r in models)
    ax.axvspan(null_low, null_high, color=COLORS["null"], zorder=0, label="shuffle 95%")
    offsets = (0.16, 0.0, -0.16)
    for values, color, marker, label, offset in (
        (identity, COLORS["stale"], "s", "identity-only", offsets[0]),
        (geom, COLORS["current"], "o", "joint geometry", offsets[1]),
        (recency, COLORS["cross"], "^", "recency-only", offsets[2]),
    ):
        ax.scatter(values, y + offset, s=32, color=color, marker=marker,
                   label=label, zorder=3)
    ax.set(xlabel="Out-of-sample choice accuracy (%)", xlim=(0, 85), yticks=y,
           yticklabels=[n for n, _ in models], title="Joint geometry trails identity-only")
    ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="lower right",
              handletextpad=0.35, columnspacing=0.7)
    grid(ax, axis="x")


def panel_crossover(ax, data):
    models = [("Qwen2.5-7B", data["geometry"]["models"]["qwen"]),
              ("Llama-3.1-8B", data["geometry"]["models"]["llama"])]
    cell_order = [
        "same_far__other_far", "same_far__other_mid", "same_far__other_near2",
        "same_near__other_far", "same_near__other_mid", "same_near__other_near2",
    ]
    cell_labels = [
        "old far · other far", "old far · other mid", "old far · other near",
        "old near · other far", "old near · other mid", "old near · other near",
    ]
    y = np.arange(len(cell_order))[::-1]
    for offset, (name, row), color, marker in zip(
        (0.13, -0.13), models, (COLORS["stale"], COLORS["cross"]), ("o", "s")
    ):
        heldout = row["g2_crossover_prediction_length_controlled"]["heldout_cell_predictions"]
        observed = [100 * heldout[cell]["observed_cross_share_among_failures"] for cell in cell_order]
        ax.scatter(observed, y + offset, s=27, color=color, marker=marker,
                   label=name, zorder=3)
    ax.axvline(0, color=COLORS["other"], linewidth=2.0, zorder=1,
               label="prediction: 0% in every cell")
    ax.set(xlabel="Observed cross-slot share among failures (%)", xlim=(-2, 64),
           yticks=y, yticklabels=cell_labels,
           title="Held-out cells expose missing cross-slot mass")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right",
              handletextpad=0.35, labelspacing=0.3)
    grid(ax, axis="x")


# --- compose ----------------------------------------------------------------
def figure_phenomenon(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.87, bottom=0.31,
                        wspace=0.30)
    panel_dose(axes[0], data)
    panel_scale(axes[1], data)
    for ax, lab in zip(axes.flat, "ab"):
        panel_label(ax, lab)
    save(fig, out_dir, "Fig_phenomenon")


def figure_causality(data, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True)
    panel_steer_dose(ax, data)
    save(fig, out_dir, "Fig_causality")


def _fmt_interval(ci):
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def write_discrete_summary(data, out_dir):
    """Write exact discrete comparisons that do not benefit from plot geometry."""
    out_dir.mkdir(parents=True, exist_ok=True)

    standard = data["stage_e_scale"]["accuracy_by_i"]
    hard = data["stage_f_hard"]["cells"]
    scale_specs = [
        ("qwen/qwen-2.5-7b-instruct", "Qwen2.5-7B", 16, 100),
        ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B", 16, 100),
        ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B", 64, 50),
        ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B", 64, 50),
        ("openai/gpt-4o", "GPT-4o", 64, 50),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Compact summary of discrete comparisons removed from the result figures. Values are percentages or percentage-point differences; intervals are 95\%.}",
        r"\label{tab:discrete-summary}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.16\textwidth}p{0.18\textwidth}rYY@{}}",
        r"\toprule",
        r"Evidence & Setting & $n$ & Estimate & Comparator / interval \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Behavioral endpoints and format}} \\",
    ]

    for model, display, load, expected_n in scale_specs:
        if load == 16:
            row = next(r for r in standard if r["model"] == model and r["interference_load"] == load)
            wrong = row["wrong"]
            stale_share = row["within_stale_errors"] / wrong if wrong else float("nan")
        else:
            row = next(
                r for r in hard
                if r["model"] == model and r["condition"] == "interference"
                and r["n_lines"] == 320 and r["interference_load"] == load
            )
            wrong = round((1 - row["accuracy"]) * row["n"])
            stale_share = row["within_stale_rate_among_errors"]
        assert row["n"] == expected_n and wrong > 0
        lines.append(
            f"Scale endpoint & {display}, $I={load}$ & {row['n']} & "
            f"accuracy {100 * row['accuracy']:.1f} & stale/error {100 * stale_share:.1f} \\\\"
        )

    k1d = data["k1d"]
    format_rows = [
        ("DP, free-form", k1d["dynamic_preference_freeform"]),
        ("DP, multiple-choice", k1d["dynamic_preference_mc_contrast"]),
        ("IF, free-form", k1d["instructional_forgetting_k1c"]),
    ]
    for label, row in format_rows:
        lines.append(
            f"Format & {label} & {row['n']} & accuracy {100 * row['forget_accuracy']:.1f} & "
            f"stale/error {100 * row['within_stale_fraction']:.1f} "
            f"{_fmt_interval(row['within_stale_ci'])} \\\\"
        )

    lines.extend([
        r"\addlinespace[1pt]",
        r"\multicolumn{5}{@{}l}{\textit{Retention and selection}} \\",
    ])
    for display, summary in (("Qwen2.5-7B", data["qwen_l1"]), ("Llama-3.1-8B", data["llama_l1"])):
        probe = summary["probe"]
        within = failure_score(summary, "within_stale")
        cross = failure_score(summary, "cross_slot")
        null = probe["within_stale_true_current_score_shuffle_null"]["ci"]
        delta = probe["within_vs_correct_true_current_score"]
        attn = summary["attention_stale_ratio_within_vs_correct"]
        lines.append(
            f"Probe + attention & {display} & {probe['n']} & "
            f"current score {100 * within:.1f}/{100 * cross:.1f}; null {100 * null[0]:.1f}--{100 * null[1]:.1f} & "
            f"failure--correct {100 * delta['length_controlled_delta']:.1f} "
            f"{_fmt_interval(delta['length_controlled_delta_ci'])}; stale-attn. "
            f"{100 * attn['length_controlled_delta']:+.1f} "
            f"{_fmt_interval(attn['length_controlled_delta_ci'])} \\\\"
        )

    lines.extend([
        r"\addlinespace[1pt]",
        r"\multicolumn{5}{@{}l}{\textit{Answer-boundary steering}} \\",
    ])
    for model, display in (("qwen", "Qwen2.5-7B"), ("llama", "Llama-3.1-8B")):
        middle = run_by(data, model, "middle")
        final = run_by(data, model, "final")
        middle_effect = middle["best_target_error_reduction"]
        final_effect = final["best_target_error_reduction"]
        lines.append(
            f"Targeted $-$ random & {display}, $\\alpha=0.4$ & {final_effect['n']} & "
            f"middle {100 * middle_effect['mean']:.1f} {_fmt_interval(middle_effect['ci'])} & "
            f"final {100 * final_effect['mean']:.1f} {_fmt_interval(final_effect['ci'])}; "
            f"identity mismatches {final['identity_gate_mismatches']} \\\\"
        )

    lines.extend([
        r"\addlinespace[1pt]",
        r"\multicolumn{5}{@{}l}{\textit{Out-of-sample geometric account}} \\",
    ])
    for key, display in (("qwen", "Qwen2.5-7B"), ("llama", "Llama-3.1-8B")):
        model = data["geometry"]["models"][key]
        g1 = model["g1_selection_argmax"]
        g2 = model["g2_crossover_prediction_length_controlled"]
        null = g1["shuffle_null"]["ci"]
        lines.append(
            f"Choice + crossover & {display} & {g1['n']} & "
            f"joint/identity/recency {100 * g1['geometric_accuracy']:.1f}/"
            f"{100 * g1['identity_only_accuracy']:.1f}/{100 * g1['recency_only_accuracy']:.1f} & "
            f"null {100 * null[0]:.1f}--{100 * null[1]:.1f}; predicted cross-share 0 in 6/6 cells; "
            f"sign {100 * g2['crossover_sign_accuracy']:.1f}, MAE {100 * g2['mean_absolute_cross_share_error']:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{2pt}\parbox{0.98\linewidth}{\scriptsize Scale endpoints use $I=16$ for 7--8B models and $I=64$ for 70--72B models and GPT-4o. Stale/error is conditional on an error. Probe scores are within-slot/cross-slot; all probe and attention differences are length controlled. Geometry lists joint/identity-only/recency-only accuracy. DP and IF denote Dynamic Preference and Instructional Forgetting.}",
        r"\end{table}",
    ])
    (out_dir / "discrete_results_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="paper/full_draft_assets/figures")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    apply_style()
    # Unify the retained legacy figures with the new style: the base figure
    # builders resolve panel_label/grid from their own module namespace at call
    # time, so overriding them there restyles those figures without touching any
    # data path (numbers stay verbatim).
    base.panel_label = panel_label
    base.grid = grid
    data = load_inputs(root)

    # Retained figures, redrawn in the unified style.
    base.figure_task_overview(out_dir)
    base.figure_case_studies(base.select_cases(root), out_dir)
    base.figure_factorial(data, out_dir)
    # Discrete probe and comparison results are represented in a compact table.

    # New merged figures.
    figure_phenomenon(data, out_dir)
    figure_causality(data, out_dir)
    write_discrete_summary(data, out_dir.parent / "tables")
    print("wrote retained figures + trend figures + discrete result table to", out_dir.parent)


if __name__ == "__main__":
    main()
