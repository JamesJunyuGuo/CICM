"""Publication figure pipeline for Stage E.

Matplotlib-only, explicit rcParams, one PDF+PNG+data sidecar per figure.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from analyze_a1_attention import compute_p_last, load_jsonl


COLORS = {
    "qwen": "#1f77b4",
    "llama": "#ff7f0e",
    "baseline": "#4c78a8",
    "arm_l": "#59a14f",
    "generic": "#f28e2b",
    "random_heads": "#b07aa1",
    "within_stale": "#e15759",
    "cross": "#76b7b2",
    "other": "#bab0ac",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_bundle(out_dir, name, fig, data, description, sources):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    sidecar = out / f"{name}.data.json"
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    write_json(sidecar, data)
    source_text = ", ".join(str(s) for s in sources)
    return f"- `{name}`: {description} Sources: {source_text}."


def _p0_series(summary):
    if not summary:
        return []
    rows = [
        cell
        for cell in summary.get("cells", [])
        if cell.get("condition") == "interference"
    ]
    return sorted(rows, key=lambda r: (r["n_lines"], r["interference_load"]))


def figure_f1(out_dir, sources):
    qwen = read_json(sources["qwen_p0"])
    llama = read_json(sources["llama_p0"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    plotted = {"qwen": _p0_series(qwen), "llama": _p0_series(llama)}
    for ax, label, data in zip(axes, ["qwen", "llama"], [plotted["qwen"], plotted["llama"]]):
        by_len = {}
        for row in data:
            by_len.setdefault(row["n_lines"], []).append(row)
        for n_lines, rows in sorted(by_len.items()):
            ax.plot(
                [r["interference_load"] for r in rows],
                [r["accuracy"] for r in rows],
                marker="o",
                label=f"n={n_lines}",
            )
        ax.set_title(label.upper())
        ax.set_xlabel("interference load I")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("accuracy")
    return save_bundle(
        out_dir,
        "F1_behavioral_accuracy",
        fig,
        plotted,
        "P0 accuracy vs interference load for Qwen and Llama.",
        [sources["qwen_p0"], sources["llama_p0"]],
    )


def _error_bar_rows(summary, label):
    if not summary:
        return []
    overall = summary.get("overall")
    if overall:
        return [
            {
                "label": label,
                "within_stale": overall.get("within_stale_rate_among_wrong", 0.0),
                "cross": overall.get("cross_value_rate_among_wrong", 0.0),
                "other": overall.get("other_rate_among_wrong", 0.0),
            }
        ]
    rows = []
    for cell in summary.get("cells", []):
        if cell.get("condition") == "simple":
            continue
        err = 1.0 - cell.get("accuracy", 0.0)
        if err <= 0:
            continue
        rows.append(
            {
                "label": f"{label}:{cell.get('stage_d_cell')}",
                "within_stale": cell.get("within_stale_rate_among_errors", 0.0),
                "cross": 0.0,
                "other": max(0.0, 1.0 - cell.get("within_stale_rate_among_errors", 0.0)),
            }
        )
    return rows


def figure_f2(out_dir, sources):
    rows = []
    for key, label in [
        ("qwen_p0_errors", "Qwen P0"),
        ("llama_p0_errors", "Llama P0"),
        ("natural_qwen", "Nat Qwen"),
        ("natural_llama", "Nat Llama"),
    ]:
        rows.extend(_error_bar_rows(read_json(sources[key]), label))
    rows = rows[:12]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for kind in ["within_stale", "cross", "other"]:
        vals = np.array([row[kind] for row in rows])
        ax.bar(x, vals, bottom=bottom, label=kind, color=COLORS[kind])
        bottom += vals
    ax.set_ylabel("fraction of errors")
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return save_bundle(
        out_dir,
        "F2_failure_signature",
        fig,
        {"rows": rows},
        "Stacked error-type bars for stale vs other failures.",
        [sources[k] for k in ["qwen_p0_errors", "llama_p0_errors", "natural_qwen", "natural_llama"]],
    )


def _a1_plot_data(npz_path, index_path):
    if not Path(npz_path).exists() or not Path(index_path).exists():
        return None
    rows = load_jsonl(index_path)
    data = np.load(npz_path)
    p_last = compute_p_last(data["attn_writes"], np.array([r["write_count"] for r in rows]))
    correct = np.array([bool(r["correct"]) for r in rows])
    heatmap = p_last[correct].mean(axis=0)
    by_i = []
    for load in [2, 4, 8]:
        idx = [
            i
            for i, row in enumerate(rows)
            if row["condition"] == "interference"
            and row["n_lines"] == 40
            and row["interference_load"] == load
        ]
        if idx:
            by_i.append(
                {
                    "interference_load": load,
                    "mean_p_last": float(p_last[idx].mean()),
                    "uniform": 1.0 / (load + 1),
                }
            )
    return {"heatmap": heatmap.tolist(), "by_i": by_i}


def figure_f3(out_dir, sources):
    qwen = _a1_plot_data(sources["qwen_a1_npz"], sources["qwen_a1_index"])
    llama = _a1_plot_data(sources["llama_a1_npz"], sources["llama_a1_index"])
    data = {"qwen": qwen, "llama": llama}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    heatmap = np.array((llama or qwen or {"heatmap": [[0.0]]})["heatmap"])
    im = axes[0].imshow(heatmap, aspect="auto")
    axes[0].set_title("p_last heatmap")
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, label, rows in [
        (axes[1], "Qwen", (qwen or {}).get("by_i", [])),
        (axes[2], "Llama", (llama or {}).get("by_i", [])),
    ]:
        ax.plot([r["interference_load"] for r in rows], [r["mean_p_last"] for r in rows], marker="o", label="observed")
        ax.plot([r["interference_load"] for r in rows], [r["uniform"] for r in rows], marker="x", label="uniform")
        ax.set_title(label)
        ax.set_xlabel("I")
        ax.set_ylim(0, 0.55)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylabel("mean p_last")
    return save_bundle(
        out_dir,
        "F3_mechanism_p_last",
        fig,
        data,
        "p_last heatmap and mean p_last vs uniform reference.",
        [sources["qwen_a1_npz"], sources["qwen_a1_index"], sources["llama_a1_npz"], sources["llama_a1_index"]],
    )


def figure_f4(out_dir, sources):
    qwen = read_json(sources["qwen_a2"])
    llama = read_json(sources["llama_a2"])
    data = {"qwen": qwen.get("mean_gap_by_layer", {}) if qwen else {}, "llama": llama.get("mean_gap_by_layer", {}) if llama else {}}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, label in zip(axes, ["qwen", "llama"]):
        for key, vals in data[label].items():
            ax.plot(range(len(vals)), vals, label=key)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label.upper())
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("gold - best stale gap")
    return save_bundle(
        out_dir,
        "F4_late_resolution_logitlens",
        fig,
        data,
        "Logit-lens gold minus best-stale gap by layer.",
        [sources["qwen_a2"], sources["llama_a2"]],
    )


def figure_f5(out_dir, sources):
    summary = read_json(sources["stage_b"])
    rows = []
    if summary:
        rows = [
            row
            for row in summary.get("groups", [])
            if row.get("run") in {"dose_response", "positive_control_answered_stale"}
            and row.get("set") == "W"
        ]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for scope in sorted({row.get("scope") for row in rows if row.get("run") == "dose_response"}):
        sr = [row for row in rows if row.get("run") == "dose_response" and row.get("scope") == scope]
        ax.plot([r["alpha"] for r in sr], [r["flip_to_correct_rate"] for r in sr], marker="o", label=scope)
    pos = [row for row in rows if row.get("run") == "positive_control_answered_stale"]
    if pos:
        ax.scatter([pos[0]["alpha"]], [pos[0]["matches_boosted_value_rate"]], marker="x", color="black", label="stale control")
    ax.set_xlabel("alpha")
    ax.set_ylabel("flip / boosted-match rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    return save_bundle(
        out_dir,
        "F5_causal_dose_response",
        fig,
        {"rows": rows},
        "Stage-B dose-response and answered-stale positive control.",
        [sources["stage_b"]],
    )


def _group_summary_rows(summary, prefix):
    rows = []
    if not summary:
        return rows
    for group, arms in summary.get("groups", {}).items():
        for arm, vals in arms.items():
            rows.append({"group": f"{prefix}:{group}", "arm": arm, "accuracy": vals["accuracy"]})
    return rows


def figure_f6(out_dir, sources):
    rows = _group_summary_rows(read_json(sources["stage_c"]), "stage_c")
    d = read_json(sources["stage_d_transfer"])
    if d:
        for arm, vals in d.get("headline", {}).get("parallel_entities_accuracy", {}).items():
            rows.append({"group": "stage_d:parallel_entities", "arm": arm, "accuracy": vals})
    natural_transfer = read_json(sources["natural_transfer"])
    if natural_transfer:
        for family, arms in natural_transfer.get("high_i_summary", {}).items():
            for arm, vals in arms.items():
                rows.append(
                    {
                        "group": f"natural:{family}",
                        "arm": arm,
                        "accuracy": vals["accuracy"],
                    }
                )
    for key, label in [("natural_qwen", "natural:qwen"), ("natural_llama", "natural:llama")]:
        summary = read_json(sources[key])
        if not summary:
            continue
        family_vals = {}
        for cell in summary.get("cells", []):
            if cell.get("condition") == "interference" and cell.get("interference_load") in {4, 8}:
                family_vals.setdefault(cell["stage_d_cell"], []).append(cell["accuracy"])
        for family, vals in family_vals.items():
            rows.append({"group": f"{label}:{family}", "arm": "baseline", "accuracy": float(np.mean(vals))})
    groups = list(dict.fromkeys(row["group"] for row in rows))[:12]
    arms = ["baseline", "arm_l", "generic", "random_heads"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(groups))
    width = 0.18
    for i, arm in enumerate(arms):
        vals = []
        for group in groups:
            match = next((row for row in rows if row["group"] == group and row["arm"] == arm), None)
            vals.append(math.nan if match is None else match["accuracy"])
        ax.bar(x + (i - 1.5) * width, vals, width, label=arm, color=COLORS.get(arm))
    ax.set_ylabel("accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return save_bundle(
        out_dir,
        "F6_repair_transfer",
        fig,
        {"rows": rows},
        "Repair accuracy across Stage-C, Stage-D, and naturalistic groups.",
        [sources[k] for k in ["stage_c", "stage_d_transfer", "natural_qwen", "natural_llama"]],
    )


def _prop_z(p1, n1, p2, n2):
    if not n1 or not n2:
        return math.nan
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2), 0.0))
    return (p1 - p2) / se if se else math.nan


def _f7_rows(sources):
    rows = []
    stage_c = read_json(sources["stage_c"])
    if stage_c:
        vals = stage_c.get("groups", {}).get("parallel_all_correct", {})
        arm = vals.get("arm_l", {}).get("accuracy")
        rnd = vals.get("random_heads", {}).get("accuracy")
        if arm is not None and rnd is not None:
            rows.append(
                {
                    "setting": "stage_c:parallel",
                    "arm_l": arm,
                    "random_heads": rnd,
                    "n_arm_l": 400,
                    "n_random_heads": 400,
                    "z": _prop_z(arm, 400, rnd, 400),
                }
            )
    stage_d = read_json(sources["stage_d_transfer"])
    if stage_d:
        vals = stage_d.get("headline", {}).get("parallel_entities_accuracy", {})
        arm = vals.get("arm_l")
        rnd = vals.get("random_heads")
        if arm is not None and rnd is not None:
            rows.append(
                {
                    "setting": "stage_d:parallel_entities",
                    "arm_l": arm,
                    "random_heads": rnd,
                    "n_arm_l": 100,
                    "n_random_heads": 100,
                    "z": _prop_z(arm, 100, rnd, 100),
                }
            )
    for source_key, prefix in [
        ("natural_transfer", "natural_seed1"),
        ("natural_transfer_seed2", "natural_seed2"),
    ]:
        natural = read_json(sources[source_key])
        if not natural:
            continue
        for family, arms in natural.get("high_i_summary", {}).items():
            arm = arms.get("arm_l", {})
            rnd = arms.get("random_heads", {})
            if "accuracy" not in arm or "accuracy" not in rnd:
                continue
            rows.append(
                {
                    "setting": f"{prefix}:{family}",
                    "arm_l": arm["accuracy"],
                    "random_heads": rnd["accuracy"],
                    "n_arm_l": arm.get("n", 400),
                    "n_random_heads": rnd.get("n", 400),
                    "z": _prop_z(
                        arm["accuracy"],
                        arm.get("n", 400),
                        rnd["accuracy"],
                        rnd.get("n", 400),
                    ),
                }
            )
    return rows


def figure_f7(out_dir, sources):
    rows = _f7_rows(sources)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    x = np.arange(len(rows))
    gaps = [row["arm_l"] - row["random_heads"] for row in rows]
    ax.bar(x, gaps, color="#59a14f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Arm L - Random accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([row["setting"] for row in rows], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        if math.isfinite(row["z"]):
            ax.text(i, gaps[i], f"z={row['z']:.1f}", ha="center", va="bottom", fontsize=7)
    return save_bundle(
        out_dir,
        "F7_arm_l_random_transfer_gap",
        fig,
        {"rows": rows},
        "Arm L vs Random-heads transfer gap with approximate two-proportion z.",
        [sources[k] for k in ["stage_c", "stage_d_transfer", "natural_transfer", "natural_transfer_seed2"]],
    )


def _mean_cells(summary, condition="interference"):
    if not summary:
        return None
    vals = [c["accuracy"] for c in summary.get("cells", []) if c.get("condition") == condition]
    return float(np.mean(vals)) if vals else None


def _mean_within_stale(summary):
    if not summary:
        return None
    vals = [
        c["within_stale_rate_among_errors"]
        for c in summary.get("cells", [])
        if c.get("condition") == "interference"
        and c.get("within_stale_rate_among_errors") is not None
    ]
    return float(np.mean(vals)) if vals else None


def figure_f8(out_dir, sources):
    accuracy_rows = []
    stale_rows = []
    sweep = read_json(sources["openrouter"])
    if sweep:
        for model in sorted(sweep.get("models", []), key=lambda m: (_model_scale_b(m), m)):
            cells = [
                c
                for c in sweep.get("cells", [])
                if c.get("model") == model and c.get("condition") == "interference"
            ]
            if not cells:
                continue
            by_load = {}
            for cell in cells:
                load = cell["interference_load"]
                acc_n = cell["accuracy"] * cell["n"]
                wrong = int(round(cell["n"] * (1.0 - cell["accuracy"])))
                within = (
                    int(round(wrong * cell["within_stale_rate_among_errors"]))
                    if wrong and cell.get("within_stale_rate_among_errors") is not None
                    else 0
                )
                slot = by_load.setdefault(
                    load,
                    {"n": 0, "correct": 0.0, "wrong": 0, "within": 0},
                )
                slot["n"] += cell["n"]
                slot["correct"] += acc_n
                slot["wrong"] += wrong
                slot["within"] += within
            wrong_total = 0
            within_stale_total = 0
            for load, slot in sorted(by_load.items()):
                wrong_total += slot["wrong"]
                within_stale_total += slot["within"]
                accuracy_rows.append(
                    {
                        "model": model,
                        "scale_b": _model_scale_b(model),
                        "interference_load": load,
                        "n": slot["n"],
                        "accuracy": slot["correct"] / slot["n"],
                        "wrong": slot["wrong"],
                        "within_stale_errors": slot["within"],
                    }
                )
            stale_rows.append(
                {
                    "model": model,
                    "scale_b": _model_scale_b(model),
                    "wrong": wrong_total,
                    "within_stale_errors": within_stale_total,
                    "other_errors": wrong_total - within_stale_total,
                    "within_stale_share": within_stale_total / wrong_total if wrong_total else math.nan,
                }
            )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    model_order = [row["model"] for row in stale_rows]
    short_labels = [_short_model_label(m) for m in model_order]
    for model in model_order:
        rows = [r for r in accuracy_rows if r["model"] == model]
        axes[0].plot(
            [r["interference_load"] for r in rows],
            [r["accuracy"] for r in rows],
            marker="o",
            label=_short_model_label(model),
        )
    axes[0].set_xlabel("interference load I")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[0].set_title("threshold shifts with scale")

    x = np.arange(len(stale_rows))
    vals = [row["within_stale_share"] for row in stale_rows]
    axes[1].bar(x, vals, color=COLORS["within_stale"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("within-stale share of errors")
    axes[1].set_title("error signature persists")
    axes[1].grid(axis="y", alpha=0.25)
    for i, row in enumerate(stale_rows):
        label = f"n={row['wrong']}"
        axes[1].text(i, 0.03 if math.isnan(vals[i]) else min(vals[i] + 0.03, 1.0), label, ha="center", fontsize=7)
    return save_bundle(
        out_dir,
        "F8_openrouter_scale_sweep",
        fig,
        {"accuracy_by_i": accuracy_rows, "within_stale_by_model": stale_rows},
        "OpenRouter scale sweep: accuracy vs interference load and within-stale share of errors.",
        [sources["openrouter"]],
    )


def _model_scale_b(model):
    lower = model.lower()
    for token, scale in [("405b", 405), ("72b", 72), ("70b", 70), ("32b", 32), ("14b", 14), ("8b", 8), ("7b", 7)]:
        if token in lower:
            return scale
    if "gpt-4o" in lower:
        return 1000
    return 0


def _short_model_label(model):
    lower = model.lower()
    if "qwen" in lower and "72b" in lower:
        return "Qwen-72B"
    if "qwen" in lower and "7b" in lower:
        return "Qwen-7B"
    if "llama" in lower and "70b" in lower:
        return "Llama-70B"
    if "llama" in lower and "8b" in lower:
        return "Llama-8B"
    if "gpt-4o" in lower:
        return "GPT-4o"
    return model


def default_sources():
    return {
        "qwen_p0": "results/p0_qwen7b.summary.json",
        "llama_p0": "results/stage_e/llama/p0_ai_local_40g.summary.json",
        "qwen_p0_errors": "results/stage_e/qwen_p0_errors.summary.json",
        "llama_p0_errors": "results/stage_e/llama/p0_errors_ai_local_40g.summary.json",
        "natural_qwen": "results/stage_e/natural/qwen_baseline_ai.summary.json",
        "natural_llama": "results/stage_e/natural/llama_baseline_ai_local_relaxed_40g.summary.json",
        "qwen_a1_npz": "results/stage_a/extract_stable.npz",
        "qwen_a1_index": "results/stage_a/extract_stable_index.jsonl",
        "llama_a1_npz": "results/stage_e/llama/extract_stable_ai_local.npz",
        "llama_a1_index": "results/stage_e/llama/extract_stable_index_ai_local.jsonl",
        "qwen_a2": "results/stage_a/a2_stable_summary.json",
        "llama_a2": "results/stage_e/llama/a2_stable_median_ai_summary.json",
        "stage_b": "results/stage_b/summary.json",
        "stage_c": "results/stage_c/summary.json",
        "stage_d_transfer": "results/stage_d/transfer_summary.json",
        "natural_transfer": "results/stage_e/natural/transfer_summary_ai.json",
        "natural_transfer_seed2": "results/stage_e/second_seed/natural/transfer_summary.json",
        "openrouter": "results/stage_e/openrouter/sweep_final.summary.json",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    configure_matplotlib()
    sources = default_sources()
    out_dir = Path(args.out_dir)
    entries = [
        figure_f1(out_dir, sources),
        figure_f2(out_dir, sources),
        figure_f3(out_dir, sources),
        figure_f4(out_dir, sources),
        figure_f5(out_dir, sources),
        figure_f6(out_dir, sources),
        figure_f7(out_dir, sources),
        figure_f8(out_dir, sources),
    ]
    manifest = out_dir / "MANIFEST.md"
    manifest.write_text("# Figure Manifest\n\n" + "\n".join(entries) + "\n")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
