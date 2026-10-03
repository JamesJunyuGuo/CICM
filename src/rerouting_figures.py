"""Freeze and visualize the Stage-R10 balanced-routing result.

This is a CPU-only paper-asset generator. It reads completed artifacts and
never runs a language model or mutates the frozen run directories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from figure_style import COLORS, TEXT_WIDTH_IN, apply_paper_style


ROOT = Path(__file__).resolve().parents[1]
H100_DIR = ROOT / "results/attention_rerouting/routing/balanced_full_h100"
A100_DIR = ROOT / "results/attention_rerouting/routing/balanced_full_a100"
DATA_PATH = ROOT / "data/cicm/cicm_natural_factorial_otherdist_l0.jsonl"
OUT_DIR = ROOT / "results/attention_rerouting/figures"
FREEZE_DIR = ROOT / "results/attention_rerouting/routing"

SEMANTIC_COLORS = {
    "correct_current": COLORS["current"],
    "within_stale": COLORS["stale"],
    "cross_slot": COLORS["cross"],
    "other": COLORS["other"],
}
DISPLAY_LABELS = {
    "correct_current": "Correct current",
    "within_stale": "Within-slot stale",
    "cross_slot": "Cross-slot",
    "other": "Other",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float, decimals: int = 1) -> str:
    return f"{100.0 * value:.{decimals}f}"


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450)
    plt.close(fig)


def validate_inputs(h100: dict[str, Any], a100: dict[str, Any]) -> None:
    """Fail closed if the frozen result contract no longer matches."""
    for name, summary in (("H100", h100), ("A100", a100)):
        if summary["stage"] != "R10-balanced-natural-dialogue-routing":
            raise ValueError(f"{name}: unexpected stage {summary['stage']}")
        if summary["n_base_rows"] != 1200 or summary["n_calibration_rows"] != 240:
            raise ValueError(f"{name}: unexpected data split")
        if summary["n_confirmation_rows"] != 960 or summary["beta"] != 8.0:
            raise ValueError(f"{name}: unexpected confirmation protocol")
        if summary["route_audit"] != {"accuracy": 1.0, "matches": 1200, "mismatches": [], "n": 1200}:
            raise ValueError(f"{name}: route audit is not exact")
        if not summary["tasks"]["retrieval"]["gate"]["all_pass"]:
            raise ValueError(f"{name}: retrieval gate did not pass")
        if summary["tasks"]["derived_decision"]["gate"]["all_pass"]:
            raise ValueError(f"{name}: derived-decision boundary unexpectedly passed")
        if len(summary["tasks"]["retrieval"]["per_factorial_cell"]) != 6:
            raise ValueError(f"{name}: expected six factorial cells")
        if summary["tasks"]["retrieval"]["random_positions"]["n"] != 16:
            raise ValueError(f"{name}: expected 16 random-position controls")


def write_frozen_summary(h100: dict[str, Any], a100: dict[str, Any]) -> None:
    h_retrieval = h100["tasks"]["retrieval"]["routed"]
    a_retrieval = a100["tasks"]["retrieval"]["routed"]
    h_decision = h100["tasks"]["derived_decision"]["routed"]
    a_decision = a100["tasks"]["derived_decision"]["routed"]
    random_gain = h100["tasks"]["retrieval"]["random_positions"]["net_accuracy_gain"]

    paper_summary = {
        "status": "frozen",
        "verdict": "controlled_natural_dialogue_retrieval_correction_only",
        "canonical_hardware": "NVIDIA H100 80GB HBM3",
        "model": h100["model"],
        "dataset_n": h100["n_base_rows"],
        "calibration_n": h100["n_calibration_rows"],
        "confirmation_n": h100["n_confirmation_rows"],
        "selected_beta": h100["beta"],
        "route_audit_accuracy": h100["route_audit"]["accuracy"],
        "retrieval": {
            "h100": {
                "baseline_accuracy": h_retrieval["baseline_accuracy"],
                "routed_accuracy": h_retrieval["arm_accuracy"],
                "net_accuracy_gain": h_retrieval["net_accuracy_gain"],
                "clustered_bootstrap_ci95": h_retrieval["paired_net_gain"]["ci"],
                "correct_preservation": h_retrieval["correct_preservation"],
                "within_stale_correction": h_retrieval["within_stale_correction"],
                "cross_slot_correction": h_retrieval["by_baseline_mode"]["cross_slot"]["to_correct"],
                "positive_factorial_cells": sum(
                    cell["net_accuracy_gain"] > 0
                    for cell in h100["tasks"]["retrieval"]["per_factorial_cell"].values()
                ),
                "factorial_cells": 6,
                "random_position_gain_mean": random_gain["mean"],
                "random_position_gain_interval95": random_gain["interval95"],
                "opposite_gain": h100["tasks"]["retrieval"]["opposite"]["net_accuracy_gain"],
            },
            "a100_replication": {
                "baseline_accuracy": a_retrieval["baseline_accuracy"],
                "routed_accuracy": a_retrieval["arm_accuracy"],
                "net_accuracy_gain": a_retrieval["net_accuracy_gain"],
                "clustered_bootstrap_ci95": a_retrieval["paired_net_gain"]["ci"],
                "correct_preservation": a_retrieval["correct_preservation"],
            },
        },
        "derived_decision_boundary": {
            "h100_net_accuracy_gain": h_decision["net_accuracy_gain"],
            "h100_clustered_bootstrap_ci95": h_decision["paired_net_gain"]["ci"],
            "a100_net_accuracy_gain": a_decision["net_accuracy_gain"],
            "a100_clustered_bootstrap_ci95": a_decision["paired_net_gain"]["ci"],
        },
        "paper_claim": (
            "A training-free attention-routing intervention improves direct current-binding "
            "retrieval in controlled natural dialogue on Qwen2.5-7B-Instruct while preserving "
            "initially correct retrievals. It does not improve the derived decision task."
        ),
        "excluded_claims": [
            "unrestricted-dialogue deployment",
            "closed-API deployment",
            "arbitrary downstream decision improvement",
            "cross-model correction",
            "uniqueness of the frozen head set",
        ],
    }
    summary_path = FREEZE_DIR / "R10_PAPER_SUMMARY.json"
    summary_path.write_text(json.dumps(paper_summary, indent=2) + "\n", encoding="utf-8")

    markdown = f"""# Frozen Stage R10 paper summary

**Status:** frozen on 2026-09-11
**Verdict:** `controlled_natural_dialogue_retrieval_correction_only`

The canonical H100 run used {h100['n_calibration_rows']} calibration and
{h100['n_confirmation_rows']} held-out confirmation examples from the 1,200-row
natural CICM factorial. Retrieval accuracy increased from
{percent(h_retrieval['baseline_accuracy'])}% to
{percent(h_retrieval['arm_accuracy'])}%: a raw paired gain of
{percent(h_retrieval['net_accuracy_gain'])} percentage points, with a
semantic-ID clustered 95% bootstrap interval of
[{percent(h_retrieval['paired_net_gain']['ci'][0])},
{percent(h_retrieval['paired_net_gain']['ci'][1])}] points. The intervention
preserved {percent(h_retrieval['correct_preservation'])}% of initially correct
retrievals and corrected {percent(h_retrieval['within_stale_correction'])}% of
within-slot stale errors. All six factorial cells improved. Sixteen matched
random-position controls averaged {percent(random_gain['mean'], 2)} points,
whereas reversing the route changed accuracy by
{percent(h100['tasks']['retrieval']['opposite']['net_accuracy_gain'])} points.

The independent A100 run reproduced the result:
{percent(a_retrieval['baseline_accuracy'])}% to
{percent(a_retrieval['arm_accuracy'])}%
({percent(a_retrieval['net_accuracy_gain'])} points). The derived decision task
did not improve (H100 {percent(h_decision['net_accuracy_gain'], 2)} points;
A100 {percent(a_decision['net_accuracy_gain'], 2)} points).
"""
    markdown += r"""

## What beta means

Let \(s_{lh}(q,j)\) denote the pre-softmax query--key attention score from the
final prompt token \(q\) to context token \(j\), in layer \(l\) and head \(h\).
For the frozen set of eight routing heads, balanced routing applies

\[
s'_{lh}(q,j)=
\begin{cases}
s_{lh}(q,j)-\beta, & j\text{ is part of a superseded same-slot value},\\
s_{lh}(q,j)+\beta, & j\text{ is part of the latest same-slot value},\\
s_{lh}(q,j), & \text{otherwise}.
\end{cases}
\]

The attention weights are then computed normally as
\(A'_{lh}(q,\cdot)=\operatorname{softmax}(s'_{lh}(q,\cdot))\). Thus
\(\beta\) is a dimensionless additive bias in attention-logit units, not a
percentage change in attention, a residual-stream steering norm, or a learned
parameter. For one routed current key and one routed stale key, the operation
adds \(2\beta\) to their log-attention odds:

\[
\log\frac{A'_{lh}(q,j_{\mathrm{current}})}
{A'_{lh}(q,j_{\mathrm{stale}})}
=
\log\frac{A_{lh}(q,j_{\mathrm{current}})}
{A_{lh}(q,j_{\mathrm{stale}})}+2\beta.
\]

This pairwise identity describes the local attention operation; it does not
imply an \(e^{2\beta}\) change in the model's output probability, because
multiple keys, heads, layers, and the rest of the network still contribute.
We selected \(\beta\) only on the 240-example calibration split from the fixed
grid \(\{0.5,1,2,4,8\}\), maximizing retrieval gain subject to preserving at
least 95% of initially correct calibration answers. The selected value was
\(\beta=8\), which was then frozen before evaluating the 960 confirmation
examples. Its magnitude should not be compared across model architectures
without recalibration because their attention-score scales can differ.

This choice is calibrated but still heuristic. On calibration, retrieval gain
rose monotonically across the tested grid (3.3, 6.2, 13.8, 20.8, and 37.1
percentage points for \(\beta=0.5,1,2,4,8\), respectively), with 100% correct
preservation at every point. Because \(\beta=8\) is also the upper grid boundary,
the experiment establishes it only as the best eligible value tested; it does
not locate a saturation point, establish an optimum, or justify transferring
the same numerical value to another model.

## How the test-time correction is applied

1. From the raw dialogue, a deterministic detector uses the declared slot
   schema to identify the queried slot, its latest update, and superseded
   same-slot mentions. Tokenizer offsets map those mentions to key positions.
   The detector does not read the gold answer, error label, saved experimental
   metadata, or the model's baseline response.
2. The model weights remain frozen. During prompt prefill, only the final
   answer-query row in the eight preselected heads is changed using the formula
   above; every other query, key, and head is untouched. The \(\beta=0\) arm is
   therefore an exact identity operation.
3. Greedy decoding then proceeds normally with the model's cache. The hook is
   inactive on later one-token decoding steps. A deployed routed answer needs
   one model decode, with no training, gradient update, second judging model, or
   post-hoc check that the baseline answer was wrong.

This is test-time correction in the precise sense that it changes an internal
attention computation for one forward pass while leaving the parameters and
prompt text unchanged. It currently requires an open-weight model with
attention-hook access and a parseable update structure; it is not directly
available through a closed text-only API.

## Paper-facing claim

A training-free attention-routing intervention improves direct current-binding
retrieval in controlled natural dialogue on Qwen2.5-7B-Instruct while
preserving initially correct retrievals. It does not improve the derived
decision task.

## Scope boundary

Do not claim unrestricted-dialogue or closed-API deployment, arbitrary
downstream decision improvement, cross-model correction, or uniqueness of the
frozen head set. H100 is the canonical numerical run; A100 is a hardware
replication.
"""
    (FREEZE_DIR / "R10_PAPER_SUMMARY.md").write_text(markdown, encoding="utf-8")

    evidence_paths = [
        DATA_PATH,
        ROOT / "docs/stage_r_testtime_correction_spec.md",
        ROOT / "src/stale_binding_routing.py",
        ROOT / "src/stale_binding_natural_routing.py",
    ]
    for run_dir in (H100_DIR, A100_DIR):
        evidence_paths.extend(sorted(path for path in run_dir.rglob("*") if path.is_file()))
    manifest = {
        "frozen_on": "2026-09-11",
        "status": "immutable_evidence",
        "canonical_run": str(H100_DIR.relative_to(ROOT)),
        "hardware_replication": str(A100_DIR.relative_to(ROOT)),
        "extension_policy": "write new model-specific directories; do not overwrite these files",
        "files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(evidence_paths))
        },
    }
    (FREEZE_DIR / "R10_FROZEN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def figure_correction_evidence(summary: dict[str, Any]) -> None:
    calibration = summary["calibration"]["curve"]
    task = summary["tasks"]["retrieval"]
    routed = task["routed"]
    opposite = task["opposite"]
    recap = task["recap"]
    random_rows = load_jsonl(H100_DIR / "retrieval/random_position_controls.jsonl")

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 4.65), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.15))
    ax_dose = fig.add_subplot(grid[0, 0])
    ax_controls = fig.add_subplot(grid[0, 1])
    ax_cells = fig.add_subplot(grid[1, :])

    betas = np.asarray([row["beta"] for row in calibration], dtype=float)
    routed_acc = np.asarray([row["arm_accuracy"] for row in calibration], dtype=float)
    baseline = float(calibration[0]["baseline_accuracy"])
    ax_dose.axhline(baseline, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax_dose.plot(
        betas,
        routed_acc,
        color=COLORS["current"],
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ax_dose.annotate(
        r"selected $\beta=8$",
        xy=(8, routed_acc[-1]),
        xytext=(5.15, 0.82),
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.8},
        fontsize=7.6,
        color=COLORS["muted"],
    )
    ax_dose.set_xscale("log", base=2)
    ax_dose.set(xlabel=r"Routing strength ($\beta$)", ylabel="Retrieval accuracy", xlim=(0.43, 9.2), ylim=(0.25, 0.84))
    ax_dose.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax_dose.set_yticks([0.3, 0.5, 0.7])
    ax_dose.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax_dose.set_title("a   Calibration dose response", loc="left")

    target_ci = routed["paired_net_gain"]["ci"]
    opposite_ci = opposite["paired_net_gain"]["ci"]
    recap_ci = recap["paired_net_gain"]["ci"]
    rows = ["Balanced routing", "Random positions", "Reversed routing", "Explicit recap"]
    y = np.arange(len(rows))[::-1]
    ax_controls.axvline(0, color=COLORS["muted"], linewidth=0.8)
    ax_controls.errorbar(
        routed["net_accuracy_gain"],
        y[0],
        xerr=[[routed["net_accuracy_gain"] - target_ci[0]], [target_ci[1] - routed["net_accuracy_gain"]]],
        color=COLORS["current"],
        marker="o",
        markersize=5,
        capsize=2.5,
        linewidth=1.25,
        zorder=4,
    )
    random_gains = np.asarray([row["net_accuracy_gain"] for row in random_rows], dtype=float)
    offsets = np.linspace(-0.11, 0.11, len(random_gains))
    ax_controls.scatter(random_gains, y[1] + offsets, s=11, color=COLORS["other"], alpha=0.9, zorder=3)
    ax_controls.errorbar(
        opposite["net_accuracy_gain"],
        y[2],
        xerr=[[opposite["net_accuracy_gain"] - opposite_ci[0]], [opposite_ci[1] - opposite["net_accuracy_gain"]]],
        color=COLORS["stale"],
        marker="v",
        markersize=5,
        capsize=2.5,
        linewidth=1.1,
    )
    ax_controls.errorbar(
        recap["net_accuracy_gain"],
        y[3],
        xerr=[[recap["net_accuracy_gain"] - recap_ci[0]], [recap_ci[1] - recap["net_accuracy_gain"]]],
        color=COLORS["muted"],
        marker="s",
        markerfacecolor="white",
        markersize=5,
        capsize=2.5,
        linewidth=1.0,
    )
    ax_controls.text(routed["net_accuracy_gain"] + 0.025, y[0], "+31.2 pp", va="center", color=COLORS["current"], fontsize=8.0, fontweight="semibold")
    ax_controls.set_yticks(y, rows)
    ax_controls.set_xlabel("Accuracy change (pp)")
    ax_controls.set_xlim(-0.24, 0.62)
    ax_controls.set_xticks([-0.2, 0, 0.2, 0.4, 0.6], ["−20", "0", "+20", "+40", "+60"])
    ax_controls.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax_controls.set_title("b   Direction and controls", loc="left")

    cells = task["per_factorial_cell"]
    order = [
        "same_far__other_far",
        "same_far__other_mid",
        "same_far__other_near2",
        "same_near__other_far",
        "same_near__other_mid",
        "same_near__other_near2",
    ]
    labels = [
        "old far · other far",
        "old far · other mid",
        "old far · other near",
        "old near · other far",
        "old near · other mid",
        "old near · other near",
    ]
    yy = np.arange(len(order))[::-1]
    for pos, key in zip(yy, order):
        base = cells[key]["baseline_accuracy"]
        arm = cells[key]["arm_accuracy"]
        ax_cells.plot([base, arm], [pos, pos], color=COLORS["grid"], linewidth=2.0, zorder=1)
        ax_cells.scatter(base, pos, color=COLORS["ink"], marker="D", s=21, zorder=3)
        ax_cells.scatter(arm, pos, color=COLORS["current"], marker="o", s=26, edgecolor="white", linewidth=0.6, zorder=4)
    ax_cells.set_yticks(yy, labels)
    ax_cells.set_xlim(0, 1.02)
    ax_cells.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_cells.set_xlabel("Retrieval accuracy")
    ax_cells.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax_cells.set_title("c   Across the identity × recency design (n=160/cell)", loc="left")
    ax_cells.scatter([], [], color=COLORS["ink"], marker="D", s=20, label="Baseline")
    ax_cells.scatter([], [], color=COLORS["current"], marker="o", s=24, label="Balanced routing")
    ax_cells.legend(loc="lower right", frameon=False, ncol=2)

    save_figure(fig, "R10_correction_evidence")


def transition_matrix(transitions: dict[str, int], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for key, count in transitions.items():
        source, target = key.split("->")
        if source in labels and target in labels:
            matrix[labels.index(source), labels.index(target)] += count
    totals = matrix.sum(axis=1)
    normalized = np.divide(matrix, totals[:, None], out=np.zeros_like(matrix), where=totals[:, None] > 0)
    return normalized, totals


def plot_transition_bars(ax: plt.Axes, routed: dict[str, Any], title: str, labels: list[str]) -> None:
    normalized, totals = transition_matrix(routed["transitions"], labels)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for col, target in enumerate(labels):
        values = normalized[:, col]
        ax.barh(y, values, left=left, height=0.58, color=SEMANTIC_COLORS[target], edgecolor="white", linewidth=0.5)
        for row, value in enumerate(values):
            if value >= 0.085:
                ax.text(left[row] + value / 2, y[row], f"{100 * value:.0f}%", ha="center", va="center", fontsize=7.1, color="white" if target != "other" else COLORS["ink"])
        left += values
    row_labels = [f"{DISPLAY_LABELS[label]}  n={int(total)}" for label, total in zip(labels, totals)]
    ax.set_yticks(y, row_labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_xlabel("Response after routing (%)")
    ax.set_title(title, loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def figure_inference_transitions(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.15), layout="constrained")
    retrieval_labels = ["correct_current", "within_stale", "cross_slot", "other"]
    decision_labels = ["correct_current", "within_stale", "other"]
    plot_transition_bars(axes[0], summary["tasks"]["retrieval"]["routed"], "a   Direct retrieval: errors move to the current value", retrieval_labels)
    plot_transition_bars(axes[1], summary["tasks"]["derived_decision"]["routed"], "b   Derived decision: the correction does not transfer", decision_labels)
    handles = [patches.Patch(facecolor=SEMANTIC_COLORS[label], label=DISPLAY_LABELS[label]) for label in retrieval_labels]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save_figure(fig, "R10_inference_transitions")


def choose_case() -> dict[str, Any]:
    baseline = {row["id"]: row for row in load_jsonl(H100_DIR / "retrieval/baseline_rows.jsonl")}
    routed = {row["id"]: row for row in load_jsonl(H100_DIR / "retrieval/routed_rows.jsonl")}
    data = {row["id"]: row for row in load_jsonl(DATA_PATH)}
    candidates = [
        row_id
        for row_id in sorted(baseline)
        if baseline[row_id]["label"] == "within_stale"
        and routed[row_id]["label"] == "correct_current"
        and data[row_id]["slot"] == "diet"
        and data[row_id]["k_overwrites"] >= 4
    ]
    if not candidates:
        raise RuntimeError("no deterministic within-stale-to-correct case found")
    row_id = candidates[0]
    return {"data": data[row_id], "baseline": baseline[row_id], "routed": routed[row_id]}


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, face: str, edge: str, linewidth: float = 0.8) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    return box


def figure_routing_method() -> None:
    case = choose_case()
    row = case["data"]
    stale_values = row["stale_values"]
    old_text = "  →  ".join(stale_values[-3:])

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.55), gridspec_kw={"width_ratios": [1.35, 1.1, 0.9]}, layout="constrained")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("a   A real updated preference", loc="left")
    rounded_box(ax, (0.03, 0.66), 0.94, 0.19, "#FBEDE8", COLORS["stale"])
    ax.text(0.07, 0.80, "Superseded meal styles", fontsize=7.1, color=COLORS["muted"], va="center")
    ax.text(0.07, 0.71, old_text, fontsize=8.0, color=COLORS["stale"], va="center")
    rounded_box(ax, (0.03, 0.38), 0.94, 0.19, "#EAF3ED", COLORS["current"])
    ax.text(0.07, 0.52, "Latest update", fontsize=7.1, color=COLORS["muted"], va="center")
    ax.text(0.07, 0.43, f"meal style  =  {row['current_value']}", fontsize=8.2, color=COLORS["current"], va="center", fontweight="semibold")
    rounded_box(ax, (0.03, 0.08), 0.94, 0.19, "#F2F4F5", COLORS["grid"])
    ax.text(0.07, 0.18, "Query: latest meal style?", fontsize=8.0, va="center")
    ax.text(0.07, 0.11, "Detector reads dialogue only", fontsize=7.0, color=COLORS["muted"], va="center")

    ax = axes[1]
    ax.set_title("b   Route the answer query", loc="left")
    ax.text(0.50, 0.88, "β shifts pre-softmax QK scores", ha="center", fontsize=7.1, color=COLORS["muted"])
    rounded_box(ax, (0.08, 0.59), 0.34, 0.16, "#FBEDE8", COLORS["stale"])
    rounded_box(ax, (0.58, 0.59), 0.34, 0.16, "#EAF3ED", COLORS["current"])
    ax.text(0.25, 0.67, "old keys\n− β", ha="center", va="center", fontsize=8.3, color=COLORS["stale"])
    ax.text(0.75, 0.67, "current key\n+ β", ha="center", va="center", fontsize=8.3, color=COLORS["current"])
    rounded_box(ax, (0.29, 0.23), 0.42, 0.16, "#EEF1F3", COLORS["ink"])
    ax.text(0.50, 0.31, "answer query", ha="center", va="center", fontsize=8.2)
    ax.annotate("", xy=(0.25, 0.58), xytext=(0.38, 0.40), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": COLORS["stale"]})
    ax.annotate("", xy=(0.75, 0.58), xytext=(0.62, 0.40), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": COLORS["current"]})
    ax.text(0.50, 0.10, "β=8 (calibrated) · prefill only · weights frozen", ha="center", fontsize=6.7, color=COLORS["muted"])

    ax = axes[2]
    ax.set_title("c   Output changes", loc="left")
    rounded_box(ax, (0.06, 0.61), 0.88, 0.18, "#FBEDE8", COLORS["stale"])
    ax.text(0.11, 0.73, "Baseline", fontsize=7.1, color=COLORS["muted"])
    ax.text(0.11, 0.64, case["baseline"]["response"], fontsize=9.0, color=COLORS["stale"], fontweight="semibold")
    ax.annotate("", xy=(0.50, 0.48), xytext=(0.50, 0.59), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLORS["muted"]})
    rounded_box(ax, (0.06, 0.25), 0.88, 0.18, "#EAF3ED", COLORS["current"])
    ax.text(0.11, 0.37, "Routed", fontsize=7.1, color=COLORS["muted"])
    ax.text(0.11, 0.28, case["routed"]["response"], fontsize=9.0, color=COLORS["current"], fontweight="semibold")
    ax.text(0.50, 0.10, "Observed confirmation example", ha="center", fontsize=7.1, color=COLORS["muted"])

    save_figure(fig, "R10_attention_routing_method")


def write_captions() -> None:
    captions = """# Stage R10 figure captions

## R10_correction_evidence

**Balanced attention routing improves direct current-binding retrieval.**
**a,** Retrieval calibration selects beta=8 from a fixed dose grid; the dashed
line is the unmodified baseline. **b,** On 960 held-out confirmation examples,
balanced routing increases accuracy by 31.2 percentage points (semantic-ID
clustered 95% bootstrap interval), whereas 16 count-matched random-position
routes remain near zero and reversing the route is harmful. An explicit recap
is shown as a deployment baseline because it places the answer in the prompt.
**c,** The gain is positive in every identity-by-recency cell (n=160 per cell).

## R10_inference_transitions

**The intervention corrects retrieval errors but does not transfer to a derived
decision task.** Rows condition on the baseline response type and colors show
the response after routing. In direct retrieval, all initially correct answers
remain correct, 42.7% of within-slot stale responses are repaired, and 84.7% of
cross-slot responses move to the current value. The same route produces no net
benefit on the derived decision task.

## R10_attention_routing_method

**Training-free attention routing changes which contextual write is read.** A
real held-out example contains several superseded meal-style values followed
by the current value. An automatic dialogue-only detector identifies the
relevant writes; at the answer query, the intervention subtracts beta from
old-write attention logits and adds beta to the current-write logit in eight
frozen heads. Here beta is an additive bias to the pre-softmax query--key score,
selected on the calibration split; the hook acts during prompt prefill only,
after which greedy decoding proceeds normally with unchanged model weights.
The unmodified model returns a stale value, while the routed decode returns the
current value. Panel b depicts the intervention operator, not a measured
attention heatmap.
"""
    (OUT_DIR / "CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    apply_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h100 = load_json(H100_DIR / "summary.json")
    a100 = load_json(A100_DIR / "summary.json")
    validate_inputs(h100, a100)
    write_frozen_summary(h100, a100)
    figure_correction_evidence(h100)
    figure_inference_transitions(h100)
    figure_routing_method()
    write_captions()
    print(json.dumps({"out_dir": str(OUT_DIR), "figures": 3, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
