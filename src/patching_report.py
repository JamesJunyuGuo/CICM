"""Assemble Stage G figures/tables and REPORT.md from completed artifacts."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt


PATCHING_READING = (
    "if (4-stale) or (5) recovers most of Δ → the circuit lesion is in QK matching "
    "(identity-based lookup that cannot separate current from stale). If only (1) at "
    "late layers recovers → resolution is more distributed."
)
PATCHING_R1_READING = (
    "if (4-stale) or (5) recovers a large share → QK-matching lesion confirmed "
    "causally; if not, the story stays \"distributed late resolution + QK geometry "
    "as the repair's mechanism,\" stated exactly so."
)
QK_HYPOTHESIS = "weak/near-chance, or present in keys but ignored — see 3."
QK_STORY = "QK implements identity lookup, not latest-binding lookup"
EXTERNAL_READINGS = [
    'Baseline shows stale-like errors on external tasks → the phenomenon is not an artifact of our generators (kills "self-built data" objection).',
    "Arm L improves (or at minimum does not hurt) external tasks, with the Arm L vs Random gap tracked → transfer evidence on community data.",
    "If a benchmark shows NO stale-like failure or no repair transfer, report it as a boundary — do not tune prompts to force an effect. Any post-hoc prompt adjustment must be labeled as such.",
]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def maybe_json(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p, read_json(p)
    return None, None


def newest_existing(patterns):
    candidates = []
    for pattern in patterns:
        candidates.extend(Path().glob(str(pattern)))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0], read_json(candidates[0])


def find_external_summaries(root):
    root = Path(root)
    summaries = []
    v2_root = root / "full_ai" / "v2"
    patterns = [v2_root.glob("**/summary.json")] if v2_root.exists() else [root.glob("**/summary.json")]
    for pattern in patterns:
        paths = sorted(pattern)
        for path in paths:
            parts = set(path.parts)
            if (
                "identity" in parts
                or "deprecated_v1" in parts
                or any(part.startswith("smoke") for part in parts)
            ):
                continue
            try:
                data = read_json(path)
            except Exception:
                continue
            if "cells" in data and "arm" in data:
                summaries.append((path, data))
    return summaries


def make_f13(summaries, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, summary in summaries:
        arm = summary.get("arm", path.parents[1].name)
        for cell in summary.get("cells", []):
            rows.append(
                {
                    "arm": arm,
                    "benchmark": cell["benchmark"],
                    "task": cell["task"],
                    "length": cell["length"],
                    "headline_metric": cell.get("headline_metric", cell["accuracy"]),
                    "accuracy": cell["accuracy"],
                    "strict_accuracy": cell.get("strict_accuracy"),
                    "mean_recall": cell.get("mean_recall"),
                    "superseded_error_rate": cell["superseded_error_rate"],
                    "omission_error_rate": cell.get("omission_error_rate", 0.0),
                    "cross_chain_inclusion_error_rate": cell.get("cross_chain_inclusion_error_rate", 0.0),
                    "n": cell["n"],
                }
            )
    if not rows:
        return None
    labels = [f"{r['arm']}\n{r['benchmark']}:{r['task']}:{r['length']}" for r in rows]
    acc = [r["headline_metric"] for r in rows]
    stale = [
        r["cross_chain_inclusion_error_rate"] if r["benchmark"] == "ruler_vt" else r["superseded_error_rate"]
        for r in rows
    ]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(rows) * 0.35), 7), sharex=True)
    axes[0].bar(range(len(rows)), acc, color="#476A6F")
    axes[0].set_ylabel("headline metric")
    axes[0].set_ylim(0, 1)
    axes[1].bar(range(len(rows)), stale, color="#C97B5A")
    axes[1].set_ylabel("signature share of errors")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "F13_external_benchmarks.png", dpi=200)
    fig.savefig(out_dir / "F13_external_benchmarks.pdf")
    plt.close(fig)
    with open(out_dir / "F13_external_benchmarks.data.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.3f}"
    return str(x)


def report(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    circuit_path, circuit = newest_existing(
        [
            out_dir / "circuit/full*/summary.json",
            out_dir / "circuit/smoke*/summary.json",
            out_dir / "circuit/summary.json",
        ]
    )
    pair_path, pairs = newest_existing(
        [
            out_dir / "circuit/full*/pairs.summary.json",
            out_dir / "circuit/smoke*/pairs.summary.json",
        ]
    )
    ext_summaries = find_external_summaries(out_dir / "external")
    audit_files = sorted((out_dir / "external/full_ai/audit").glob("*.txt"))
    deprecated = sorted((out_dir / "external/full_ai/deprecated_v1").glob("**/summary.json"))
    f13_rows = make_f13(ext_summaries, out_dir / "figures")

    lines = []
    lines.append("# Stage G Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("Stage G ran the circuit-level stale-binding checks and external benchmark validation requested in `docs/stage_g_spec.md`, including Addendum 1 (2026-07-11).")
    lines.append("")
    lines.append("## Addendum 1 Status")
    lines.append("- v1 external summaries are void for paper use because they used scoring/signature protocols superseded by Addendum 1.")
    lines.append(f"- Deprecated v1 summaries moved under `results/patching_and_external_benchmarks/external/full_ai/deprecated_v1/` (found {len(deprecated)} files). Raw `results.jsonl` rows were not overwritten.")
    lines.append(f"- Raw-output audit hard gate: {len(audit_files)} audit files under `results/patching_and_external_benchmarks/external/full_ai/audit/`.")
    if audit_files:
        lines.append("- Audit outcome: BABILong qa1-0k baseline-vs-adapter gap was a parse-protocol artifact; lenient containment raises Qwen baseline substantially. Llama boxes audit showed generation-level prompt/template failure, so entity-tracking was rerun with a direct boxes prompt.")
    lines.append("")
    lines.append("## Pre-registered Readings")
    lines.append(f"- Path patching: `{PATCHING_READING}`")
    lines.append(f"- R-1 causal patching: `{PATCHING_R1_READING}`")
    lines.append(f"- QK hypothesis: `{QK_HYPOTHESIS}`")
    lines.append(f"- Circuit story: `{QK_STORY}`")
    for item in EXTERNAL_READINGS:
        lines.append(f"- External: `{item}`")
    lines.append("")
    lines.append("## G-1 Circuit")
    if circuit:
        gates = circuit.get("hard_gates", {})
        lines.append(f"- Artifact: `{circuit_path}`")
        if pairs:
            lines.append(
                f"- Pair yield: {pairs.get('n_pairs')} / {pairs.get('candidates')} candidates "
                f"(yield={fmt(pairs.get('yield'))}, target_met={pairs.get('target_met')}); first-token collisions dropped={pairs.get('first_token_collision_drops')}."
            )
        lines.append(f"- Self-patching identity hard gate: {gates.get('self_patching_identity')}.")
        lines.append("- Path-patching component effects:")
        for row in circuit.get("patching", {}).get("component_effects", []):
            lines.append(
                f"  - {row['component']}: mean Δ recovery={fmt(row.get('delta_recovery_mean'))}, "
                f"median={fmt(row.get('delta_recovery_median'))}, n={row.get('n')}"
            )
        geom = circuit.get("geometry", {})
        base = geom.get("baseline", {})
        arm = geom.get("arm_l", {})
        lines.append(
            "- QK geometry baseline: "
            f"mean current-stale margin={fmt(base.get('mean_current_minus_stale_qk'))}, "
            f"query identity align={fmt(base.get('mean_query_identity_alignment'))}, "
            f"query recency align={fmt(base.get('mean_query_recency_alignment'))}."
        )
        if arm:
            lines.append(
                "- QK geometry Arm L: "
                f"mean current-stale margin={fmt(arm.get('mean_current_minus_stale_qk'))}, "
                f"query identity align={fmt(arm.get('mean_query_identity_alignment'))}, "
                f"query recency align={fmt(arm.get('mean_query_recency_alignment'))}."
            )
        ov = circuit.get("ov_sanity", {})
        lines.append(
            "- OV sanity: "
            f"mean attended-value token logit={fmt(ov.get('mean_value_token_logit'))}, "
            f"gold token={fmt(ov.get('mean_gold_token_logit'))}, "
            f"answered-stale token={fmt(ov.get('mean_answered_stale_token_logit'))}."
        )
        if circuit.get("limitations"):
            lines.append("- Limitations: " + " ".join(circuit["limitations"]))
    else:
        lines.append("- No completed circuit summary found.")
    lines.append("")
    lines.append("## G-2 External Benchmarks")
    lines.append("- Protocols:")
    lines.append("  - RULER-VT: per-example per-variable recall (fraction of gold chain variables present in output, case-insensitive) is the headline; strict set equality is secondary. Error labels: omission, cross-chain inclusion, other. No stale/superseded language.")
    lines.append("  - BABILong: normalized gold string contained in normalized response (`target_in_response`). For qa2/qa3 only, superseded iff the prediction equals a strictly earlier state of the queried entity.")
    lines.append("  - Entity-tracking boxes: predicted contents are parsed as a normalized set; order-insensitive set equality is accuracy. Superseded iff the predicted set equals the queried box contents at an earlier timestep.")
    if ext_summaries:
        lines.append(f"- Found {len(ext_summaries)} v2 external summary artifacts.")
        for path, summary in ext_summaries:
            lines.append(f"- `{path}` arm={summary.get('arm')} model={summary.get('model')}")
            for cell in summary.get("cells", []):
                metric = cell.get("headline_metric", cell["accuracy"])
                sig_label = "cross-chain-error-share" if cell["benchmark"] == "ruler_vt" else "superseded-error-share"
                sig_val = cell.get("cross_chain_inclusion_error_rate", 0.0) if cell["benchmark"] == "ruler_vt" else cell.get("superseded_error_rate", 0.0)
                lines.append(
                    f"  - {cell['benchmark']} {cell['task']} {cell['length']}: "
                    f"n={cell['n']}, headline={fmt(metric)}, strict_acc={fmt(cell.get('strict_accuracy'))}, "
                    f"{sig_label}={fmt(sig_val)}, "
                    f"other-error-share={fmt(cell['other_error_rate'])}"
                )
        if f13_rows is not None:
            lines.append("- Figure F13 written to `results/patching_and_external_benchmarks/figures/F13_external_benchmarks.{png,pdf}`.")
    else:
        lines.append("- No completed external summaries found.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- F11/F12: `results/patching_and_external_benchmarks/**/figures/` or `results/patching_and_external_benchmarks/figures/` depending on winning portal.")
    lines.append("- F13: `results/patching_and_external_benchmarks/figures/F13_external_benchmarks.{png,pdf}` when external summaries are present.")
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/patching_and_external_benchmarks")
    args = ap.parse_args()
    report(args)


if __name__ == "__main__":
    main()
