import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#2f5f9f",
    "orange": "#d9822b",
    "gold": "#c9a227",
    "olive": "#6f8f3a",
    "pink": "#b8577a",
    "ink": "#20242a",
    "muted": "#6b7280",
    "grid": "#d6dbe3",
    "light": "#f5f7fa",
    "null": "#b8bec8",
}


SOURCES = {
    "p0": "results/overwrite_task/p0_qwen7b.summary.json",
    "stage_l_l0": "results/cicm/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json",
    "stage_k_k1d": "results/icf_bench/k1d_summary.json",
    "qwen_l1": "results/cicm/natural_factorial_otherdist_l1/full/l1_summary.json",
    "qwen_l1_rows": "results/cicm/natural_factorial_otherdist_l1/full/l1_probe_rows.jsonl",
    "qwen_l1_npz": "results/cicm/natural_factorial_otherdist_l1/full/l1_harvest.npz",
    "qwen_l1_index": "results/cicm/natural_factorial_otherdist_l1/full/l1_index.jsonl",
    "qwen_l2": "results/cicm/natural_factorial_otherdist_l2/full_alpha_sweep/l2_summary.json",
    "llama_l0": "results/cicm/natural_factorial_otherdist_llama/l0/openrouter_llama31_8b_summary.json",
    "llama_l1": "results/cicm/natural_factorial_otherdist_llama/l1/full/l1_summary.json",
    "llama_l1_rows": "results/cicm/natural_factorial_otherdist_llama/l1/full/l1_probe_rows.jsonl",
    "llama_l2": "results/cicm/natural_factorial_otherdist_llama/l2/full_alpha_sweep/l2_summary.json",
    "validity_l2": "results/cicm/validity_l2/summary.json",
    "geometry": "results/cicm/geometry/summary.json",
}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values, seed: int = 1009, n_boot: int = 2000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def pct(x: float) -> float:
    return 100.0 * float(x)


def yerr(mean: float, ci: tuple[float, float] | list[float]) -> list[list[float]]:
    lo, hi = ci
    return [[max(0.0, mean - lo)], [max(0.0, hi - mean)]]


def err_pair(value: float, ci: tuple[float, float] | list[float]) -> tuple[float, float]:
    lo, hi = ci
    return max(0.0, value - lo), max(0.0, hi - value)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_grid(ax) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def fig1(out_dir: Path) -> str:
    p0 = load_json(SOURCES["p0"])
    l0 = load_json(SOURCES["stage_l_l0"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    ax = axes[0]
    simple_cells = [c for c in p0["cells"] if c["condition"] == "simple"]
    simple_acc = np.mean([c["accuracy"] for c in simple_cells])
    simple_n = sum(c["n"] for c in simple_cells)
    simple_ci = wilson_ci(round(simple_acc * simple_n), simple_n)
    simple_err = err_pair(simple_acc, simple_ci)
    ax.errorbar([0], [pct(simple_acc)], yerr=[[pct(simple_err[0])], [pct(simple_err[1])]],
                fmt="o", color=COLORS["ink"], label="no overwrite")
    for n_lines, color in zip([20, 40, 80], [COLORS["blue"], COLORS["orange"], COLORS["olive"]]):
        cells = sorted(
            [c for c in p0["cells"] if c["condition"] == "interference" and c["n_lines"] == n_lines],
            key=lambda c: c["interference_load"],
        )
        xs = [c["interference_load"] for c in cells]
        ys = [pct(c["accuracy"]) for c in cells]
        errs = []
        for c in cells:
            lo, hi = wilson_ci(round(c["accuracy"] * c["n"]), c["n"])
            errs.append(tuple(pct(x) for x in err_pair(c["accuracy"], (lo, hi))))
        ax.errorbar(xs, ys, yerr=np.asarray(errs).T, marker="o", linewidth=1.8, color=color, label=f"{n_lines} lines")
    ax.set_title("Controlled overwrite dose response")
    ax.set_xlabel("Old-value competitors")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 2, 4, 8])
    add_grid(ax)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    doses = sorted(l0["by_dose"], key=lambda x: int(x))
    x = np.arange(len(doses))
    acc = [l0["by_dose"][d]["accuracy"] for d in doses]
    stale_share = [l0["by_dose"][d]["within_stale_failure_share"] for d in doses]
    acc_err = []
    share_err = []
    for d in doses:
        row = l0["by_dose"][d]
        n = row["n"]
        counts = row["counts"]
        lo, hi = wilson_ci(counts.get("correct_current", 0), n)
        acc_err.append(tuple(pct(x) for x in err_pair(row["accuracy"], (lo, hi))))
        fail = n - counts.get("correct_current", 0)
        lo, hi = wilson_ci(counts.get("within_stale", 0), fail)
        share_err.append(tuple(pct(x) for x in err_pair(row["within_stale_failure_share"], (lo, hi))))
    width = 0.36
    ax.bar(x - width / 2, [pct(v) for v in acc], width, color=COLORS["blue"], label="accuracy")
    ax.errorbar(x - width / 2, [pct(v) for v in acc], yerr=np.asarray(acc_err).T, fmt="none", color=COLORS["ink"], capsize=2)
    ax.bar(x + width / 2, [pct(v) for v in stale_share], width, color=COLORS["orange"], label="within-stale / failures")
    ax.errorbar(x + width / 2, [pct(v) for v in stale_share], yerr=np.asarray(share_err).T, fmt="none", color=COLORS["ink"], capsize=2)
    ax.set_title("Natural CICM dose response")
    ax.set_xlabel("Overwrite count")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, doses)
    add_grid(ax)
    ax.legend(frameon=False, loc="upper right")

    save_figure(fig, out_dir, "F1_phenomenon_dose_response")
    return (
        "F1. Phenomenon and dose response. Left: controlled Paper-1 overwrite accuracy "
        "from results/overwrite_task/p0_qwen7b.summary.json, with Wilson intervals derived from summary n "
        "and accuracy. Right: Stage-L natural CICM accuracy and within-stale share among "
        "failures by overwrite count from results/cicm/natural_factorial_otherdist_l0/"
        "openrouter_qwen25_7b_summary_merged.json."
    )


def fig2(out_dir: Path) -> str:
    k1d = load_json(SOURCES["stage_k_k1d"])
    rows = [
        ("DP free-form", k1d["dynamic_preference_freeform"]),
        ("DP multiple-choice", k1d["dynamic_preference_mc_contrast"]),
        ("IF free-form", k1d["instructional_forgetting_k1c"]),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    xs = np.arange(len(rows))
    vals = [r["within_stale_fraction"] for _, r in rows]
    cis = [r["within_stale_ci"] for _, r in rows]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["olive"]]
    ax.bar(xs, [pct(v) for v in vals], color=colors, width=0.62)
    ax.errorbar(
        xs,
        [pct(v) for v in vals],
        yerr=np.asarray([tuple(pct(x) for x in err_pair(v, ci)) for v, ci in zip(vals, cis)]).T,
        fmt="none",
        color=COLORS["ink"],
        capsize=2,
    )
    for i, (_, r) in enumerate(rows):
        ax.text(i, pct(vals[i]) + 5, f"{r['within_stale_failures']}/{r['forget_failures']}", ha="center", fontsize=8)
    ax.set_title("Multiple-choice masks stale-binding")
    ax.set_ylabel("Within-stale among failures (%)")
    ax.set_xticks(xs, [name.replace(" ", "\n") for name, _ in rows])
    ax.set_ylim(0, 110)
    add_grid(ax)
    save_figure(fig, out_dir, "F2_mc_masks_stale_binding")
    return (
        "F2. Multiple-choice masks stale-binding. Bars show within-stale fraction among "
        "failures for Dynamic Preference free-form, Dynamic Preference multiple-choice, "
        "and Instructional Forgetting free-form. Values and CIs are read from "
        "results/icf_bench/k1d_summary.json."
    )


def fig3(out_dir: Path) -> str:
    l0 = load_json(SOURCES["stage_l_l0"])
    cells = [
        "same_near__other_far",
        "same_near__other_mid",
        "same_near__other_near2",
        "same_far__other_far",
        "same_far__other_mid",
        "same_far__other_near2",
    ]
    labels = ["same near\nother far", "same near\nother mid", "same near\nother near2",
              "same far\nother far", "same far\nother mid", "same far\nother near2"]
    x = np.arange(len(cells))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
    series = [("within-stale", "within_stale", COLORS["blue"]), ("cross-slot", "cross_slot", COLORS["orange"])]
    for offset, (_, key, color) in zip([-width / 2, width / 2], series):
        vals, errs = [], []
        for cell in cells:
            row = l0["by_factorial_cell"][cell]
            n = row["n"]
            k = row["counts"].get(key, 0)
            val = k / n
            lo, hi = wilson_ci(k, n)
            vals.append(pct(val))
            errs.append(tuple(pct(x) for x in err_pair(val, (lo, hi))))
        ax.bar(x + offset, vals, width, color=color, label=series[0][0] if key == "within_stale" else series[1][0])
        ax.errorbar(x + offset, vals, yerr=np.asarray(errs).T, fmt="none", color=COLORS["ink"], capsize=2)
    for i, cell in enumerate(cells):
        row = l0["by_factorial_cell"][cell]
        if row["cross_slot_rate"] >= row["within_stale_rate"]:
            top = pct(max(row["cross_slot_rate"], row["within_stale_rate"])) + 12
            ax.text(i, top + 2, "crossover", ha="center", va="bottom", fontsize=7, color=COLORS["orange"])
            ax.plot([i - 0.38, i + 0.38], [top, top], color=COLORS["orange"], linewidth=1.2)
    ax.axvline(2.5, color=COLORS["grid"], linewidth=1.2)
    ax.set_title("Identity-vs-recency competition by factorial cell")
    ax.set_ylabel("Response mode rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    add_grid(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, out_dir, "F3_identity_recency_crossover")
    return (
        "F3. Identity-vs-recency competition. Grouped bars show within-stale and cross-slot "
        "response rates per factorial cell, with Wilson intervals derived from per-cell "
        "counts in results/cicm/natural_factorial_otherdist_l0/"
        "openrouter_qwen25_7b_summary_merged.json. Crossover labels mark cells where "
        "cross-slot rate meets or exceeds within-stale rate."
    )


def _probe_group_stats(rows_path: str) -> dict:
    rows = load_jsonl(rows_path)
    groups = defaultdict(list)
    for row in rows:
        score = row.get("probe_true_current_score")
        if score is not None:
            groups[row.get("label", "other")].append(float(score))
    stats = {}
    for label, vals in groups.items():
        stats[label] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "ci": bootstrap_ci(vals, seed=1009 + len(label)),
        }
    return stats


def fig4(out_dir: Path) -> str:
    l1 = load_json(SOURCES["qwen_l1"])
    stats = _probe_group_stats(SOURCES["qwen_l1_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    ax = axes[0]
    labels = ["correct_current", "within_stale", "cross_slot"]
    names = ["correct", "within-stale", "cross-slot"]
    vals = [stats[label]["mean"] for label in labels]
    cis = [stats[label]["ci"] for label in labels]
    xs = np.arange(len(labels))
    ax.axhspan(
        pct(l1["probe"]["within_stale_true_current_score_shuffle_null"]["ci"][0]),
        pct(l1["probe"]["within_stale_true_current_score_shuffle_null"]["ci"][1]),
        color=COLORS["null"],
        alpha=0.35,
        label="value-label shuffle 95%",
    )
    ax.bar(xs, [pct(v) for v in vals], color=[COLORS["blue"], COLORS["orange"], COLORS["olive"]], width=0.62)
    ax.errorbar(xs, [pct(v) for v in vals], yerr=np.asarray([tuple(pct(x) for x in err_pair(v, ci)) for v, ci in zip(vals, cis)]).T,
                fmt="none", color=COLORS["ink"], capsize=2)
    ax.set_title("Current value remains decodable")
    ax.set_ylabel("True-current probe score (%)")
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 105)
    add_grid(ax)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    attn = l1["attention_stale_ratio_within_vs_correct"]
    mean = attn["length_controlled_delta"]
    ci = attn["length_controlled_delta_ci"]
    null = attn["length_controlled_shuffle_null"]["ci"]
    ax.axhspan(pct(null[0]), pct(null[1]), color=COLORS["null"], alpha=0.35, label="shuffle 95%")
    ax.bar([0], [pct(mean)], color=COLORS["orange"], width=0.48)
    ax.errorbar([0], [pct(mean)], yerr=yerr(pct(mean), [pct(ci[0]), pct(ci[1])]), fmt="none", color=COLORS["ink"], capsize=2)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_title("Stale-attention signature")
    ax.set_ylabel("Within-stale minus correct\nstale/(stale+current), pp")
    ax.set_xticks([0], ["length-controlled"])
    ax.set_ylim(min(-5, pct(ci[0]) - 5), pct(ci[1]) + 8)
    add_grid(ax)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, out_dir, "F4_selection_failure_mechanism")
    return (
        "F4. Selection failure mechanism. Decodability bars use probe_true_current_score "
        "from results/cicm/natural_factorial_otherdist_l1/full/l1_probe_rows.jsonl "
        "with bootstrap intervals; the null band and attention delta are read from "
        "results/cicm/natural_factorial_otherdist_l1/full/l1_summary.json. "
        "The projection-independent quantitative result is that current values remain "
        "well above the value-label shuffle null on failure trials while stale attention is elevated."
    )


def _load_hidden_labels():
    rows = load_jsonl(SOURCES["qwen_l1_index"])
    hidden = np.load(SOURCES["qwen_l1_npz"])["hidden"][:, -1, :].astype(np.float32)
    labels = np.asarray([row.get("label", "other") for row in rows], dtype=object)
    return hidden, labels


def _scatter_projection(ax, xy: np.ndarray, labels: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    order = [
        ("correct_current", "correct", COLORS["blue"]),
        ("within_stale", "within-stale", COLORS["orange"]),
        ("cross_slot", "cross-slot", COLORS["olive"]),
        ("other", "other", COLORS["muted"]),
    ]
    for key, name, color in order:
        mask = labels == key
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, alpha=0.42, color=color, label=name, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=COLORS["grid"], linewidth=0.8)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, markerscale=1.8, loc="best")


def fig5(out_dir: Path) -> tuple[str, str]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    hidden, labels = _load_hidden_labels()
    z = StandardScaler().fit_transform(hidden)

    # Supervised mean-difference axes are used only for illustration.
    axis1 = z[labels == "within_stale"].mean(axis=0) - z[labels == "correct_current"].mean(axis=0)
    axis2 = z[labels == "cross_slot"].mean(axis=0) - z[labels == "within_stale"].mean(axis=0)
    axis1 = axis1 / max(float(np.linalg.norm(axis1)), 1e-12)
    axis2 = axis2 / max(float(np.linalg.norm(axis2)), 1e-12)
    xy = np.column_stack([z @ axis1, z @ axis2])
    xy = (xy - xy.mean(axis=0, keepdims=True)) / xy.std(axis=0, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    _scatter_projection(
        ax,
        xy,
        labels,
        "Illustrative supervised projection",
        "within-stale vs correct axis",
        "cross-slot vs within-stale axis",
    )
    ax.text(
        0.01,
        0.01,
        "Illustration only: quantitative claims come from F4 probes.",
        transform=ax.transAxes,
        fontsize=7,
        color=COLORS["muted"],
        va="bottom",
    )
    save_figure(fig, out_dir, "F5_representation_projection_illustration")

    pca = PCA(n_components=2, svd_solver="randomized", random_state=1009)
    pxy = pca.fit_transform(z)
    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    _scatter_projection(
        ax,
        pxy,
        labels,
        "Appendix: naive PCA projection",
        f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)",
        f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)",
    )
    ax.text(0.01, 0.01, "Unsupervised PCA included for honesty; not a mechanism claim.",
            transform=ax.transAxes, fontsize=7, color=COLORS["muted"], va="bottom")
    save_figure(fig, out_dir, "F5_appendix_naive_pca")

    main_caption = (
        "F5. Illustrative projection of Qwen decision-position representations. The main "
        "panel projects saved final-layer activations onto two supervised discriminative "
        "axes fit from results/cicm/natural_factorial_otherdist_l1/full/l1_harvest.npz "
        "and l1_index.jsonl. Illustrative projection; all quantitative claims come from "
        "the cross-validated probe in F4, not from this plot."
    )
    appendix_caption = (
        "F5 appendix. Naive PCA projection of the same saved Qwen decision-position "
        "representations from results/cicm/natural_factorial_otherdist_l1/full/"
        "l1_harvest.npz and l1_index.jsonl. This is an unsupervised visualization only."
    )
    return main_caption, appendix_caption


def fig6(out_dir: Path) -> str:
    validity = load_json(SOURCES["validity_l2"])
    runs = validity["runs"]
    final_runs = [r for r in runs if r["layer_kind"] == "final"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8), constrained_layout=True)

    ax = axes[0]
    for run, color in zip(sorted(final_runs, key=lambda r: r["model"]), [COLORS["olive"], COLORS["blue"]]):
        alphas = sorted(float(a) for a in run["by_alpha"])
        means, lo, hi = [], [], []
        for a in alphas:
            row = run["by_alpha"][str(a)]["target_error_reduction"]
            means.append(pct(row["mean"]))
            lo.append(pct(row["ci"][0]))
            hi.append(pct(row["ci"][1]))
        means = np.asarray(means)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        ax.plot(alphas, means, marker="o", linewidth=1.8, color=color, label=f"{run['model']} final L{run['layer']}")
        ax.fill_between(alphas, lo, hi, color=color, alpha=0.18)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_title("Causal dose response, residual-normalized")
    ax.set_xlabel("Alpha as residual-norm ratio")
    ax.set_ylabel("Targeted - random\nerror reduction (pp)")
    add_grid(ax)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    labels, vals, lows, highs, colors = [], [], [], [], []
    for run in sorted(runs, key=lambda r: (r["model"], r["layer"])):
        row = run["best_target_error_reduction"]
        labels.append(f"{run['model']}\n{run['layer_kind']} L{run['layer']}")
        vals.append(pct(row["mean"]))
        lows.append(pct(row["ci"][0]))
        highs.append(pct(row["ci"][1]))
        colors.append(COLORS["orange"] if run["layer_kind"] == "final" else COLORS["null"])
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.62)
    ax.errorbar(x, vals, yerr=np.asarray([(max(0.0, v - l), max(0.0, h - v)) for v, l, h in zip(vals, lows, highs)]).T,
                fmt="none", color=COLORS["ink"], capsize=2)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_title("Layer localization qualifies causality")
    ax.set_ylabel("Best alpha effect (pp)")
    ax.set_xticks(x, labels)
    add_grid(ax)
    save_figure(fig, out_dir, "F6_causal_dose_and_layer_localization")
    return (
        "F6. Qualified causal intervention result. Left: residual-normalized final-layer "
        "targeted-minus-random target-error reduction by alpha. Right: best-alpha layer "
        "localization shows large final-layer effects and null/tiny middle-layer effects. "
        "All values and CIs are read from results/cicm/validity_l2/summary.json; "
        "legacy absolute-alpha final-layer sweeps are stored in results/cicm/"
        "natural_factorial_otherdist_l2/full_alpha_sweep/l2_summary.json and the Llama "
        "equivalent, but the plotted comparison uses the normalized validity rerun."
    )


def fig7(out_dir: Path) -> str:
    sources = [
        ("Qwen2.5-7B", load_json(SOURCES["qwen_l1"]), _probe_group_stats(SOURCES["qwen_l1_rows"]), COLORS["blue"]),
        ("Llama-3.1-8B", load_json(SOURCES["llama_l1"]), _probe_group_stats(SOURCES["llama_l1_rows"]), COLORS["olive"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True, sharey=True)
    for ax, (model_name, summary, stats, color) in zip(axes, sources):
        labels = ["within_stale", "cross_slot"]
        names = ["within-stale", "cross-slot"]
        vals = [stats[label]["mean"] for label in labels]
        cis = [stats[label]["ci"] for label in labels]
        xs = np.arange(len(labels))
        null_ci = summary["probe"]["within_stale_true_current_score_shuffle_null"]["ci"]
        ax.axhspan(pct(null_ci[0]), pct(null_ci[1]), color=COLORS["null"], alpha=0.35, label="shuffle 95%")
        ax.bar(xs, [pct(v) for v in vals], color=[color, COLORS["orange"]], width=0.6)
        ax.errorbar(xs, [pct(v) for v in vals],
                    yerr=np.asarray([tuple(pct(x) for x in err_pair(v, ci)) for v, ci in zip(vals, cis)]).T,
                    fmt="none", color=COLORS["ink"], capsize=2)
        ax.set_title(model_name)
        ax.set_xticks(xs, names)
        ax.set_ylim(0, 105)
        add_grid(ax)
    axes[0].set_ylabel("True-current probe score (%)")
    axes[1].legend(frameon=False, loc="lower right")
    save_figure(fig, out_dir, "F7_cross_model_l1_replication")
    return (
        "F7. Cross-model L-1 replication. Bars show true-current probe score on failure "
        "pools for Qwen2.5-7B and Llama-3.1-8B, with bootstrap intervals from each "
        "model's l1_probe_rows.jsonl and shuffle-null bands from the corresponding "
        "l1_summary.json files."
    )


def write_captions(out_dir: Path, captions: dict[str, str]) -> None:
    lines = ["# Stage L Figure Captions", ""]
    for key in sorted(captions):
        lines.extend([f"## {key}", "", captions[key], ""])
    lines.extend(["## Source Files", ""])
    for name, path in sorted(SOURCES.items()):
        if Path(path).exists():
            lines.append(f"- {name}: `{path}`")
    (out_dir / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/cicm/figures")
    args = parser.parse_args()
    apply_style()
    out_dir = Path(args.out_dir)
    captions = {
        "F1": fig1(out_dir),
        "F2": fig2(out_dir),
        "F3": fig3(out_dir),
        "F4": fig4(out_dir),
        "F6": fig6(out_dir),
        "F7": fig7(out_dir),
    }
    captions["F5"], captions["F5_appendix"] = fig5(out_dir)
    write_captions(out_dir, captions)
    expected = [
        "F1_phenomenon_dose_response",
        "F2_mc_masks_stale_binding",
        "F3_identity_recency_crossover",
        "F4_selection_failure_mechanism",
        "F5_representation_projection_illustration",
        "F5_appendix_naive_pca",
        "F6_causal_dose_and_layer_localization",
        "F7_cross_model_l1_replication",
    ]
    missing = [name for name in expected for suffix in [".pdf", ".png"] if not (out_dir / f"{name}{suffix}").exists()]
    if missing:
        raise SystemExit(f"missing figure outputs: {missing}")
    print(json.dumps({"out_dir": str(out_dir), "figures": expected, "captions": str(out_dir / "CAPTIONS.md")}, indent=2))


if __name__ == "__main__":
    main()
