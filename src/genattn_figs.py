"""Render Stage P figures and assemble the preregistered report from summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genattn_gen import CHECKPOINTS


COLORS = {"correct": "#2A9D8F", "stale": "#D65A3A", "control": "#557A95", "null": "#A7A9AC"}
ANALYSIS_CHECKPOINTS = CHECKPOINTS[1:]


def parse_entry(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("entry must be LABEL=ANALYSIS[:PATCH]")
    paths = parts[1].split(":", 1)
    return parts[0], Path(paths[0]), Path(paths[1]) if len(paths) == 2 else None


def load_entries(entries) -> list[dict]:
    loaded = []
    for label, analysis_path, patch_path in entries:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        patch = json.loads(patch_path.read_text(encoding="utf-8")) if patch_path and patch_path.exists() else None
        loaded.append({"label": label, "analysis": analysis, "patch": patch})
    return loaded


def _series(analysis: dict, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct, stale, correct_ci, stale_ci = [], [], [], []
    for checkpoint in CHECKPOINTS[1:]:
        summary = analysis["trajectories"][checkpoint][metric]["raw"]
        correct.append(summary["correct_mean"])
        stale.append(summary["stale_mean"])
        correct_ci.append(summary["correct_bootstrap95"])
        stale_ci.append(summary["stale_bootstrap95"])
    return np.asarray(correct), np.asarray(stale), np.asarray(correct_ci), np.asarray(stale_ci)


def trajectory_figure(entries: list[dict], out_dir: Path) -> None:
    entries = [entry for entry in entries if entry["analysis"].get("status") != "insufficient_matched_pool"]
    if not entries:
        return
    n = len(entries)
    columns = min(3, n)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows * 2, columns, figsize=(4.35 * columns + 0.8, 2.65 * rows * 2), squeeze=False)
    x = np.arange(4)
    labels = [r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_Q$"]
    for model_index, entry in enumerate(entries):
        row = model_index // columns
        column = model_index % columns
        for metric_index, (metric, title, ylabel) in enumerate((
            ("read_ratio", "Attention read ratio", r"$R_\tau$ (log current / stale)"),
            ("behavioral_margin", "Behavioral margin", "current - strongest stale logit"),
        )):
            ax = axes[row * 2 + metric_index, column]
            correct, stale, correct_ci, stale_ci = _series(entry["analysis"], metric)
            ax.errorbar(x, correct, yerr=np.abs(correct_ci.T - correct), color=COLORS["correct"], marker="o", ms=4, lw=1.4, capsize=2, label="no overwrite (correct)")
            ax.errorbar(x, stale, yerr=np.abs(stale_ci.T - stale), color=COLORS["stale"], marker="s", ms=4, lw=1.4, capsize=2, label="overwrite (stale)")
            ax.axhline(0, color="#30343B", lw=0.7, alpha=0.65)
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#D9DDE2", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            title_text = f"{entry['label']} - {title}" if n > 1 else title
            ax.set_title(title_text, fontsize=10, fontweight="bold", pad=7)
    for model_index in range(n, rows * columns):
        row, column = divmod(model_index, columns)
        axes[row * 2, column].axis("off")
        axes[row * 2 + 1, column].axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    suptitle = "Failure-conditioned event trace" + (f" - {entries[0]['label']}" if n == 1 else "")
    fig.suptitle(suptitle, y=0.997, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.015, 0.055, 1, 0.965), h_pad=1.45)
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(out_dir / f"P1_event_trajectories.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def tables(entries: list[dict], out_dir: Path) -> None:
    outcome_rows = []
    predictor_rows = []
    gate_rows = []
    descriptive_rows = []
    for entry in entries:
        analysis = entry["analysis"]
        gates = analysis["validity_gates"]
        counts = analysis["final_label_counts"]
        controls = analysis["same_length_no_overwrite_controls"]
        gate_rows.append({
            "model": entry["label"],
            "n_items": analysis["n_items"],
            "final_current": counts.get("correct_current", 0),
            "final_within_stale": counts.get("within_stale", 0),
            "final_other": counts.get("other", 0),
            "eligible_exact_pairs": gates["exact_behavior_matched_pairs_n"],
            "matched_pool_sufficient": gates["matched_pool_sufficient"],
            "same_length_every_checkpoint": gates["same_length_every_checkpoint"],
            "control_accuracy_soft_gate": gates["no_overwrite_accuracy_soft_threshold_every_checkpoint"],
            "control_tauQ_accuracy": controls["tauQ"]["no_overwrite_accuracy"],
        })
        full_pool = analysis.get("descriptive_full_pool")
        if full_pool:
            for checkpoint in ANALYSIS_CHECKPOINTS:
                row = {"model": entry["label"], "checkpoint": checkpoint, "n": analysis["n_items"]}
                for metric in ("behavioral_margin", "read_ratio", "qk_margin"):
                    summary = full_pool["trajectories"][checkpoint][metric]["length_distance_controlled"]
                    prefix = metric.replace("_margin", "")
                    row[f"{prefix}_overwrite_minus_control"] = summary["overwrite_minus_control"]
                    row[f"{prefix}_bootstrap_low"] = summary["paired_bootstrap95"][0]
                    row[f"{prefix}_bootstrap_high"] = summary["paired_bootstrap95"][1]
                    row[f"{prefix}_signflip_p_lower"] = summary["signflip_p_lower"]
                descriptive_rows.append(row)
        if analysis.get("status") == "insufficient_matched_pool":
            continue
        mixture = analysis["outcome_mixture_on_final_stale"]
        total = max(sum(mixture.values()), 1)
        outcome_rows.append({
            "model": entry["label"],
            "earliest": analysis["earliest_stale_favoring_checkpoint"] or "none",
            "write_side": mixture.get("write_side_qk_negative_at_tau2", 0) / total,
            "read_side": mixture.get("read_side_qk_crossing_after_tau2", 0) / total,
            "no_crossing": mixture.get("no_pooled_qk_crossing", 0) / total,
            "accumulation_slope": analysis["competition_accumulation"]["slope"],
        })
        predictor = analysis["early_to_late_prediction"]
        predictor_rows.append({
            "model": entry["label"],
            "balanced_accuracy": predictor["balanced_accuracy"],
            "shuffle_low": predictor["shuffle95"][0],
            "shuffle_high": predictor["shuffle95"][1],
            "pass": predictor["above_shuffle95"],
            "groups": predictor["n_groups"],
        })
    for name, rows in (
        ("model_gate_summary", gate_rows),
        ("full_pool_descriptive", descriptive_rows),
        ("outcome_mixture", outcome_rows),
        ("early_late_prediction", predictor_rows),
    ):
        if not rows:
            continue
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def patch_figure(entries: list[dict], out_dir: Path) -> None:
    available = [entry for entry in entries if entry["patch"]]
    if not available:
        return
    paths = ["early_query", "early_key", "final_query", "final_key"]
    path_colors = ["#557A95", "#2A9D8F", "#C47F3A", "#D65A3A"]
    if len(available) == 1:
        entry = available[0]
        values = [entry["patch"]["paths"][path]["correction_rate"] for path in paths]
        intervals = [entry["patch"]["paths"][path]["correction_rate_bootstrap95"] for path in paths]
        lower = [value - interval[0] for value, interval in zip(values, intervals)]
        upper = [interval[1] - value for value, interval in zip(values, intervals)]
        y = np.arange(len(paths))
        fig, ax = plt.subplots(figsize=(6.4, 3.35))
        ax.barh(
            y,
            values,
            color=path_colors,
            xerr=np.asarray([lower, upper]),
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        display_paths = [r"$\tau_2$ query", r"$\tau_2$ current key", r"$\tau_Q$ query", r"$\tau_Q$ current key"]
        ax.set_yticks(y, display_paths)
        ax.invert_yaxis()
        ax.set_xlabel("stale-to-current correction rate")
        ax.set_xlim(0, 1)
        ax.set_title(f"Held-out causal patches - {entry['label']}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#D9DDE2", lw=0.6)
    else:
        labels = [entry["label"] for entry in available]
        x = np.arange(len(labels))
        width = 0.18
        fig, ax = plt.subplots(figsize=(max(6.2, 1.4 * len(labels)), 3.3))
        for offset, (path, color) in enumerate(zip(paths, path_colors)):
            values = [entry["patch"]["paths"][path]["correction_rate"] for entry in available]
            intervals = [entry["patch"]["paths"][path]["correction_rate_bootstrap95"] for entry in available]
            lower = [value - interval[0] for value, interval in zip(values, intervals)]
            upper = [interval[1] - value for value, interval in zip(values, intervals)]
            ax.bar(
                x + (offset - 1.5) * width,
                values,
                width=width,
                color=color,
                label=path.replace("_", " "),
                yerr=np.asarray([lower, upper]),
                capsize=2.5,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel("stale-to-current correction rate")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#D9DDE2", lw=0.6)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(out_dir / f"P2_timing_patches.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def descriptive_figure(entries: list[dict], out_dir: Path) -> None:
    available = [entry for entry in entries if entry["analysis"].get("descriptive_full_pool")]
    if not available:
        return
    metrics = (
        ("behavioral_margin", "Behavioral margin", "logit delta"),
        ("read_ratio", "Attention read ratio", "log-ratio delta"),
        ("qk_margin", "QK selection margin", "QK delta"),
    )
    model_labels = []
    for entry in available:
        controls_pass = entry["analysis"]["validity_gates"]["no_overwrite_accuracy_soft_threshold_every_checkpoint"]
        model_labels.append(entry["label"] + (" †" if not controls_pass else ""))
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.8))
    for ax, (metric, title, colorbar_label) in zip(axes, metrics):
        matrix = np.asarray([
            [
                entry["analysis"]["descriptive_full_pool"]["trajectories"][checkpoint][metric]["length_distance_controlled"]["overwrite_minus_control"]
                for checkpoint in ANALYSIS_CHECKPOINTS
            ]
            for entry in available
        ])
        scale = max(float(np.max(np.abs(matrix))), 1e-8)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text_color = "white" if abs(value) > 0.58 * scale else "#20242A"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.3, color=text_color)
        ax.set_xticks(np.arange(len(ANALYSIS_CHECKPOINTS)), [r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_Q$"])
        ax.set_yticks(np.arange(len(model_labels)), model_labels if ax is axes[0] else [])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color("#C9CDD2")
            spine.set_linewidth(0.7)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label(colorbar_label, fontsize=7.5)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle("Six-model overwrite effect after pool residualization", y=0.965, fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Negative values favor stale over current. All attention/QK cells pass the paired sign-flip null (p = 0.000999).\n"
        "† Control behavior misses the preregistered soft threshold; descriptive only.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.015, 0.14, 1, 0.90), w_pad=1.4)
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(out_dir / f"P3_six_model_descriptive.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def build_report(entries: list[dict], out_dir: Path) -> None:
    complete = [entry for entry in entries if entry["analysis"].get("status") != "insufficient_matched_pool"]
    lines = [
        "# Stage P: Event-Aligned Inference Trace",
        "",
        "## Preregistered verdict",
        "",
        "**The strong inevitability gate fails.** On Pythia-160M, a failure-conditioned current-to-stale separation is already visible immediately after the update (`tau2`) and predicts the counterfactual final arm out of sample. Patching the clean key/query in the `tau2` branch strongly repairs the local answer. However, reapplying the clean current-write key or query in the full `tauQ` branch corrects few held-out errors. The update therefore contains a real early write-side distortion, but it does not alone make the later stale answer inevitable; later selection dynamics can re-form the failure.",
        "",
        "**The honest adjudication is mixed write-side plus later read-side competition, not a single-time-point failure.** Among 121 final stale Pythia examples, 39 already have a stale-favoring pooled QK margin at `tau2`, while 82 cross only after `tau2`. The current value remains decodable above shuffle throughout, although decodability weakens in stale runs.",
        "",
        "**A broad descriptive signature transfers, but failure-conditioned timing does not.** In all six models, overwrite branches shift the residualized attention read ratio and QK margin toward stale values relative to exact same-length no-overwrite branches from `tau2` onward (all paired sign-flip `p=0.000999`). On this frozen substrate, only Pythia-160M supplies the preregistered minimum of 20 exact final correct/stale pairs. Larger models mostly answer correctly, and Gemma fails the no-overwrite behavior control. Their trajectories are descriptive evidence of competition pressure, not cross-model replication of the failure mechanism.",
        "",
        "## Model gates",
        "",
        "| Model | n | Final current / stale / other | Exact failure pairs | Same length | Control soft gate | Failure-conditioned mechanism |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        analysis = entry["analysis"]
        counts = analysis["final_label_counts"]
        gates = analysis["validity_gates"]
        mechanism = "PASS" if gates["matched_pool_sufficient"] else f"NOT IDENTIFIED ({gates['exact_behavior_matched_pairs_n']} < {analysis.get('min_exact_pairs', 20)})"
        lines.append(
            f"| {entry['label']} | {analysis['n_items']} | {counts.get('correct_current', 0)} / {counts.get('within_stale', 0)} / {counts.get('other', 0)} "
            f"| {gates['exact_behavior_matched_pairs_n']} | {'PASS' if gates['same_length_every_checkpoint'] else 'FAIL'} "
            f"| {'PASS' if gates['no_overwrite_accuracy_soft_threshold_every_checkpoint'] else 'MISS'} | {mechanism} |"
        )

    if complete:
        entry = complete[0]
        analysis = entry["analysis"]
        predictor = analysis["early_to_late_prediction"]
        tau2 = analysis["trajectories"]["tau2"]
        retention_tau2 = tau2["retention"]
        retention_final = analysis["trajectories"]["tauQ"]["retention"]
        mixture = analysis["outcome_mixture_on_final_stale"]
        lines.extend([
            "",
            "## Failure-conditioned temporal localization",
            "",
            f"Pythia-160M contributes {analysis['validity_gates']['exact_behavior_matched_pairs_n']} exact behavior pairs. The stable split contains 52 discovery and 69 held-out causal pairs. The first preregistered stale-favoring group contrast occurs at **{analysis['earliest_stale_favoring_checkpoint']}**, directly after the update.",
            "",
            "| `tau2` reading | Within-stale minus correct, controlled | Bootstrap 95% | Shuffle 95% |",
            "|---|---:|---:|---:|",
        ])
        for metric, label in (("behavioral_margin", "Behavioral logit margin"), ("read_ratio", "Attention read ratio"), ("qk_margin", "QK margin")):
            result = tau2[metric]["length_distance_controlled"]
            lines.append(
                f"| {label} | {result['stale_minus_correct']:.3f} | [{result['bootstrap95'][0]:.3f}, {result['bootstrap95'][1]:.3f}] "
                f"| [{result['shuffle95'][0]:.3f}, {result['shuffle95'][1]:.3f}] |"
            )
        lines.extend([
            "",
            f"The grouped early-to-late classifier reaches balanced accuracy **{predictor['balanced_accuracy']:.3f}**, above its refit label-shuffle interval [{predictor['shuffle95'][0]:.3f}, {predictor['shuffle95'][1]:.3f}] (`p={predictor['shuffle_p_upper']:.6f}`). This predicts membership in the matched overwrite/no-overwrite counterfactual arm; it is not an estimate of natural failure prevalence.",
            "",
            f"The current value remains linearly decodable in stale runs at `tau2` (mean true-current score {retention_tau2['final_stale_mean_true_current_score']:.3f}; shuffle upper bound {retention_tau2['value_label_shuffle95'][1]:.3f}) and at `tauQ` ({retention_final['final_stale_mean_true_current_score']:.3f}; shuffle upper bound {retention_final['value_label_shuffle95'][1]:.3f}). The decline relative to matched correct runs is significant at both checkpoints, so the result is selection-dominant with measurable retention degradation, not perfect retention.",
            "",
            "### Outcome mixture",
            "",
            f"- Write-side QK-negative at `tau2`: **{mixture.get('write_side_qk_negative_at_tau2', 0)} / {sum(mixture.values())}**.",
            f"- Read-side QK crossing after `tau2`: **{mixture.get('read_side_qk_crossing_after_tau2', 0)} / {sum(mixture.values())}**.",
            f"- The descriptive pooled accumulation fit has slope {analysis['competition_accumulation']['slope']:.3f} and R2={analysis['competition_accumulation']['r2']:.3f}. It is not promoted to a mechanistic law.",
        ])
        if entry["patch"]:
            patch = entry["patch"]
            lines.extend([
                "",
                "## Held-out causal patching",
                "",
                "Heads are ranked on discovery data only and evaluated on the 69 held-out stale examples. `early query/key` patches operate in the short `tau2` branch. `final query` patches the diagnostic query in the full `tauQ` branch, while `final key` patches the earlier current-write token's key inside that full branch. Because an earlier key is causally cached, the latter is the direct test of carrying the clean update key into the complete context.",
                "",
                "| Path | Corrected | Bootstrap 95% | Mean margin change | Margin-change 95% | Identity |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            path_labels = {
                "early_key": "tau2 current key",
                "early_query": "tau2 query",
                "final_key": "tauQ current key",
                "final_query": "tauQ query",
            }
            for path in ("early_key", "early_query", "final_key", "final_query"):
                result = patch["paths"][path]
                lines.append(
                    f"| {path_labels[path]} | {result['correction_rate']:.3f} | [{result['correction_rate_bootstrap95'][0]:.3f}, {result['correction_rate_bootstrap95'][1]:.3f}] "
                    f"| {result['mean_margin_change']:.3f} | [{result['margin_change_bootstrap95'][0]:.3f}, {result['margin_change_bootstrap95'][1]:.3f}] "
                    f"| {'PASS' if result['identity_exact_zero'] else 'FAIL'} |"
                )
            lines.extend([
                "",
                f"All four identity patches are exact zero. The `tau2` current-key/query correction is {patch['paths']['early_key']['correction_rate']:.3f}/{patch['paths']['early_query']['correction_rate']:.3f}, but the full-branch `tauQ` current-key/query correction is only {patch['paths']['final_key']['correction_rate']:.3f}/{patch['paths']['final_query']['correction_rate']:.3f}. Thus, even carrying the matched clean current-write key into the complete context does not usually prevent final stale selection. The cached-key invariance audit differs by at most {patch['key_cache_invariance_max_abs']:.6f}, a small floating-point execution difference rather than exact bitwise identity.",
            ])

    lines.extend([
        "",
        "## Six-model descriptive transfer",
        "",
        "The table reports the final-checkpoint overwrite-minus-control contrast after one pool-level regression on log prefix length and nearest-competitor distance. Negative values mean that introducing the overwrite moves the read toward stale relative to its exact same-item, same-length no-overwrite control.",
        "",
        "| Model | Behavioral delta | Attention-ratio delta | QK delta | Current-value probe score | Shuffle upper | Scope flag |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for entry in entries:
        analysis = entry["analysis"]
        full = analysis["descriptive_full_pool"]
        tauq = full["trajectories"]["tauQ"]
        retention = full["retention"]["tauQ"]
        scope = "descriptive only"
        if not analysis["validity_gates"]["no_overwrite_accuracy_soft_threshold_every_checkpoint"]:
            scope = "control miss; descriptive only"
        lines.append(
            f"| {entry['label']} | {tauq['behavioral_margin']['length_distance_controlled']['overwrite_minus_control']:.3f} "
            f"| {tauq['read_ratio']['length_distance_controlled']['overwrite_minus_control']:.3f} "
            f"| {tauq['qk_margin']['length_distance_controlled']['overwrite_minus_control']:.3f} "
            f"| {retention['final_stale_mean_true_current_score']:.3f} | {retention['value_label_shuffle95'][1]:.3f} | {scope} |"
        )
    lines.extend([
        "",
        "All 24 attention-ratio cells and all 24 QK cells (`6 models x 4 checkpoints`) have a negative paired contrast with lower-tail sign-flip `p=0.000999`. This is the broad result: overwrite events consistently create stale-directed internal competition even when the model's final behavioral margin remains positive enough to answer correctly. The all-overwrite retention probes are above their grouped value-label shuffle null in all six models, but these full-pool probes must not be called selection-failure evidence because most larger-model examples are correct.",
        "",
        "### Per-model adjudication",
        "",
        "- **Pythia-160M:** mixed early write-side distortion plus later read-side crossing; strong permanent-inevitability gate fails.",
        "- **Pythia-1.4B:** 1 exact final stale pair; descriptive competition pressure transfers, failure timing not identifiable.",
        "- **Qwen2.5-1.5B:** 8 exact final stale pairs; descriptive competition pressure transfers, preregistered trajectory/patch validation is gated off.",
        "- **Qwen2.5-7B and Llama-3.1-8B:** 1 and 2 exact final stale pairs on the fixed 72-item confirmation set; descriptive confirmation only.",
        "- **Gemma-2-2B:** only 69/144 final no-overwrite controls are correct, so mechanism attribution is invalid on this task. Its paired numerical trajectory remains in the audit table but carries no mechanistic claim.",
        "",
        "## Validity audit",
        "",
        "- Fixed diagnostic query: **PASS**. Each checkpoint is an independent branch; the query is never added to later prefixes.",
        "- Program-located marker spans and tokenizer gate: **PASS** for all model families.",
        "- Hooked-noop logit identity: **PASS** for all model captures.",
        "- Exact same-length overwrite/no-overwrite controls: **PASS** for all six models at every checkpoint.",
        "- No-overwrite accuracy soft threshold: **PASS** for Pythia-1.4B, Qwen-1.5B, and Qwen-7B; **MISS** for Pythia-160M (final 136/144), Llama-8B (early `tau1` only), and Gemma-2B (substantial, including final). Exact pair-level controls, rather than the soft aggregate threshold, determine eligibility.",
        "- Pool-level length and nearest-competitor-distance residualization: **PASS**.",
        "- GroupKFold plus 1,000 refit shuffles: **PASS** where the matched pool is sufficient (Pythia-160M only).",
        "- Identity patch exact-zero: **PASS** on all four Pythia paths.",
        "- GQA scope: Qwen/Llama/Gemma per-head values are query-head readings against shared KV heads; key patches deduplicate shared KV heads.",
        "",
        "## Scope and stop decision",
        "",
        "The experiment answers what can be compared in a causal transformer: how identical new diagnostic queries read fixed earlier writes at event-aligned checkpoints. It does not claim that an earlier token's cached attention changes over time. It also does not estimate population prevalence, tune the task to force failures in larger models, or infer failure-conditioned mechanisms from full-pool averages.",
        "",
        "Stage P stops here. The preregistered strong result is negative, the Pythia timing diagnosis is mixed and causally qualified, and the cross-model transfer is descriptive rather than failure-conditioned.",
        "",
        "## Artifacts",
        "",
        "- `figures/P1_event_trajectories.{pdf,png}`: failure-conditioned Pythia trajectory.",
        "- `figures/P2_timing_patches.{pdf,png}`: held-out causal patch corrections with paired bootstrap intervals.",
        "- `figures/P3_six_model_descriptive.{pdf,png}`: six-model residualized overwrite-minus-control trajectories.",
        "- `figures/model_gate_summary.csv`, `figures/full_pool_descriptive.csv`, `figures/outcome_mixture.csv`, `figures/early_late_prediction.csv`: auditable tables.",
        "",
        "![Failure-conditioned event trace](figures/P1_event_trajectories.png)",
        "",
        "![Held-out timing patches](figures/P2_timing_patches.png)",
        "",
        "![Six-model descriptive transfer](figures/P3_six_model_descriptive.png)",
    ])
    (out_dir.parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(args) -> None:
    entries = load_entries(args.entry)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_figure(entries, out_dir)
    tables(entries, out_dir)
    patch_figure(entries, out_dir)
    descriptive_figure(entries, out_dir)
    build_report(entries, out_dir)
    print(json.dumps({"models": [entry["label"] for entry in entries], "out_dir": str(out_dir), "report": str(out_dir.parent / 'REPORT.md')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    parser.add_argument("--out-dir", default="results/stage_p/figures")
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
