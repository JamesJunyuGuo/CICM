import argparse
import math
from pathlib import Path

import numpy as np

from icf_mech import (
    compare_groups,
    dump_json,
    load_jsonl,
    mean_attention_ratio,
    summarize_attention_groups,
    summarize_label_counts,
)


def summarize_dp_logit_gaps(rows: list[dict], gaps: np.ndarray) -> dict:
    from icf_mech import summarize_dp_logit_gaps as _summarize

    return _summarize(rows, gaps)


def compute_dp_logit_gaps(dp_npz_path: str, dp_index_rows: list[dict], dp_labeled_rows: list[dict]) -> dict:

    data = np.load(dp_npz_path)
    gaps_all = data["new_old_logit_gap"]
    valid_all = data["new_old_logit_gap_valid"].astype(bool)
    by_id = {str(row["id"]): row for row in dp_labeled_rows}
    rows = []
    gaps = []
    dropped_shared = 0
    for i, row in enumerate(dp_index_rows):
        if str(row["id"]) not in by_id:
            continue
        if not valid_all[i]:
            dropped_shared += 1
            continue
        rows.append({**row, **by_id[str(row["id"])]})
        gaps.append(gaps_all[i])
    gap_arr = np.stack(gaps, axis=0) if gaps else np.zeros((0, gaps_all.shape[1]), dtype=np.float32)
    summary = summarize_dp_logit_gaps(rows, gap_arr)
    summary.update(
        {
            "n_total": len(dp_index_rows),
            "n_kept": len(rows),
            "n_dropped_shared_first_token": dropped_shared,
            "drop_rate": dropped_shared / len(dp_index_rows) if dp_index_rows else math.nan,
            "reading": "positive NEW-minus-OLD logit gap on within-stale failures supports retained current binding; negative gap supports retention/retrievability failure.",
        }
    )
    return {"summary": summary, "rows": rows, "gaps": gap_arr}


def _head_mean(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(arr, axis=2)


def analyze_dp_attention(dp_npz_path: str, dp_index_rows: list[dict], dp_labeled_rows: list[dict]) -> dict:
    data = np.load(dp_npz_path)
    by_id = {str(row["id"]): row for row in dp_labeled_rows}
    rows = [{**row, **by_id[str(row["id"])]} for row in dp_index_rows if str(row["id"]) in by_id]
    keep_indices = [i for i, row in enumerate(dp_index_rows) if str(row["id"]) in by_id]
    old_attn = data["attn_old_pref"][keep_indices]
    new_attn = data["attn_new_pref"][keep_indices]
    old_ratio = _head_mean(mean_attention_ratio(old_attn, new_attn))
    new_mass = _head_mean(new_attn)
    old_mass = _head_mean(old_attn)
    return {
        "old_pref_ratio_groups": summarize_attention_groups(rows, old_ratio),
        "old_pref_ratio_within_stale_minus_correct": compare_groups(rows, old_ratio, "within_stale", "correct_current"),
        "old_pref_mass_groups": summarize_attention_groups(rows, old_mass),
        "new_pref_mass_groups": summarize_attention_groups(rows, new_mass),
        "label_counts": summarize_label_counts(rows),
        "reading": "higher OLD/(OLD+NEW) attention on within-stale failures supports stale-selection attention.",
    }


def analyze_if_attention(if_npz_path: str, if_index_rows: list[dict]) -> dict:
    data = np.load(if_npz_path)
    rows = [row for row in if_index_rows if row.get("value_span_found")]
    keep_indices = [i for i, row in enumerate(if_index_rows) if row.get("value_span_found")]
    value_attn = data["attn_forget_value"][keep_indices]
    instr_attn = data["attn_forget_instruction"][keep_indices]
    value_ratio = _head_mean(mean_attention_ratio(value_attn, instr_attn))
    instr_ratio = 1.0 - value_ratio
    instr_mass = _head_mean(instr_attn)
    return {
        "n_total": len(if_index_rows),
        "n_value_span_found": len(rows),
        "label_counts_all": summarize_label_counts(if_index_rows),
        "label_counts_value_span_found": summarize_label_counts(rows),
        "value_vs_instruction_ratio_groups": summarize_attention_groups(rows, value_ratio),
        "instruction_ratio_groups": summarize_attention_groups(rows, instr_ratio),
        "instruction_mass_groups": summarize_attention_groups(rows, instr_mass),
        "value_ratio_within_stale_minus_correct_forget": compare_groups(rows, value_ratio, "within_stale", "correct_forget"),
        "instruction_ratio_within_stale_minus_correct_forget": compare_groups(rows, instr_ratio, "within_stale", "correct_forget"),
        "reading": "higher forbidden-value ratio or lower forget-instruction ratio on within-stale failures supports IF suppression/selection failure.",
    }


def write_report(path: str, summary: dict) -> None:
    dp_k3 = summary["k3_dp_logit_lens"]["groups"]
    dp_attn = summary["k2_dp_attention"]
    if_attn = summary["k2_if_attention"]
    lines = [
        "# Stage K Mechanism Report",
        "",
        "## Scope",
        "",
        "K2/K3 use local `Qwen/Qwen2.5-7B-Instruct` GPU harvests with fp32 eager attention. Outputs are isolated under `results/icf_bench/mech/`; Stage-J and K1d artifacts were not modified.",
        "",
        "## K3 DP Retention vs Selection",
        "",
        "Frozen Stage-A-style linear readout: final-token hidden states are passed through the model final norm and unembedding; the measured gap is NEW aligned choice first-token logit minus OLD aligned choice first-token logit.",
        "",
        f"- n kept after shared-first-token drop: {summary['k3_dp_logit_lens']['n_kept']} / {summary['k3_dp_logit_lens']['n_total']}",
    ]
    for label in ["correct_current", "within_stale", "other"]:
        if label in dp_k3:
            row = dp_k3[label]
            lines.append(
                f"- {label}: n={row['n']}, final NEW-OLD gap={row['final_layer_mean_gap']:.3f}, CI=[{row['final_layer_gap_ci'][0]:.3f}, {row['final_layer_gap_ci'][1]:.3f}]"
            )
    lines.extend(
        [
            "",
            "Pre-registered reading: NEW preference still linearly decodable on within-stale trials -> SELECTION failure replicates; NEW not decodable -> RETENTION/retrievability failure.",
            "",
            "## K2 DP Attention",
            "",
            f"- Labels: {dp_attn['label_counts']}",
            f"- OLD/(OLD+NEW) within_stale minus correct final-layer delta: {dp_attn['old_pref_ratio_within_stale_minus_correct']['delta_final_layer']:.4f}",
            "",
            "## K2 IF Attention",
            "",
            f"- n total: {if_attn['n_total']}; value span found: {if_attn['n_value_span_found']}",
            f"- Labels all: {if_attn['label_counts_all']}",
            f"- forbidden-value ratio within_stale minus correct_forget final-layer delta: {if_attn['value_ratio_within_stale_minus_correct_forget']['delta_final_layer']:.4f}",
            f"- forget-instruction ratio within_stale minus correct_forget final-layer delta: {if_attn['instruction_ratio_within_stale_minus_correct_forget']['delta_final_layer']:.4f}",
            "",
            "## Artifacts",
            "",
            "- `dynamic_preference.npz` / `dynamic_preference_index.jsonl`",
            "- `instructional_forgetting.npz` / `instructional_forgetting_index.jsonl`",
            "- `dynamic_preference_local_labeled.jsonl`",
            "- `k2k3_summary.json`",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-dir", default="results/icf_bench/mech")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="results/icf_bench/mech/k2k3_summary.json")
    parser.add_argument("--report-out", default="results/icf_bench/mech/REPORT.md")
    args = parser.parse_args()

    mech = Path(args.mech_dir)
    dp_index = load_jsonl(mech / "dynamic_preference_index.jsonl")
    dp_labeled = load_jsonl(mech / "dynamic_preference_local_labeled.jsonl")
    if_index = load_jsonl(mech / "instructional_forgetting_index.jsonl")
    k3 = compute_dp_logit_gaps(str(mech / "dynamic_preference.npz"), dp_index, dp_labeled)["summary"]
    summary = {
        "model": args.model,
        "k3_dp_logit_lens": k3,
        "k2_dp_attention": analyze_dp_attention(str(mech / "dynamic_preference.npz"), dp_index, dp_labeled),
        "k2_if_attention": analyze_if_attention(str(mech / "instructional_forgetting.npz"), if_index),
    }
    dump_json(args.out, summary)
    write_report(args.report_out, summary)
    print(f"wrote {args.out}, {args.report_out}")


if __name__ == "__main__":
    main()
