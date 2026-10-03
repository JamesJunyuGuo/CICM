"""Stage F-1: repaired-model p_last gain analysis."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from analyze_write_attention import compute_p_last, load_jsonl


ARMS = {
    "arm_l": {
        "npz": "results/hard_tier_and_transport/mech_loop/arm_l/extract.npz",
        "index": "results/hard_tier_and_transport/mech_loop/arm_l/extract_index.jsonl",
        "adapter_config": "results/head_adapters/adapters_full_ai/arm_l/adapter_config.json",
    },
    "random_heads": {
        "npz": "results/hard_tier_and_transport/mech_loop/random_heads/extract.npz",
        "index": "results/hard_tier_and_transport/mech_loop/random_heads/extract_index.jsonl",
        "adapter_config": "results/head_adapters/adapters_full_ai/random_heads/adapter_config.json",
    },
    "generic": {
        "npz": "results/hard_tier_and_transport/mech_loop/generic/extract.npz",
        "index": "results/hard_tier_and_transport/mech_loop/generic/extract_index.jsonl",
        "adapter_config": "results/head_adapters/adapters_full_ai/generic/adapter_config.json",
    },
}


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_heads(path):
    with open(path) as f:
        summary = json.load(f)
    return [
        {"layer": int(item["layer"]), "head": int(item["head"])}
        for item in summary["top_heads_by_correct_p_last"][:8]
    ]


def load_adapter_heads(path):
    with open(path) as f:
        config = json.load(f)
    return [
        {"layer": int(item["layer"]), "head": int(item["head"])}
        for item in config.get("heads", [])
    ]


def head_set(heads):
    return {(int(item["layer"]), int(item["head"])) for item in heads}


def top_gain_heads(gain, top_k=16):
    flat = np.asarray(gain).reshape(-1)
    n_heads = gain.shape[1]
    out = []
    for idx in np.argsort(flat)[::-1][:top_k]:
        out.append(
            {
                "layer": int(idx // n_heads),
                "head": int(idx % n_heads),
                "gain": float(flat[idx]),
            }
        )
    return out


def mean_head_values(values, heads):
    if not heads:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    selected = np.stack(
        [values[:, int(item["layer"]), int(item["head"])] for item in heads],
        axis=1,
    )
    return selected.mean(axis=1)


def align_by_id(rows, p_last, target_ids):
    pos = {row["id"]: i for i, row in enumerate(rows)}
    missing = [row_id for row_id in target_ids if row_id not in pos]
    if missing:
        raise ValueError(f"missing {len(missing)} ids, first={missing[0]}")
    idx = [pos[row_id] for row_id in target_ids]
    return [rows[i] for i in idx], p_last[idx]


def load_p_last(npz_path, index_path):
    rows = load_jsonl(index_path)
    data = np.load(npz_path)
    write_counts = np.array([row["write_count"] for row in rows], dtype=np.int64)
    return rows, compute_p_last(data["attn_writes"], write_counts)


def mean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def corr_or_nan(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.unique(y[mask]).size < 2:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def classify_mechanism_reading(arm_summaries):
    arm_l = arm_summaries.get("arm_l", {})
    random_arm = arm_summaries.get("random_heads", {})
    all_original = all(
        arm_summaries.get(arm, {}).get("original_top8_overlap_in_top16", 0) >= 4
        for arm in ["arm_l", "random_heads", "generic"]
    )
    if all_original:
        return "circuit-restoration"
    if (
        arm_l.get("original_top8_overlap_in_top16", 0) >= 4
        and random_arm.get("own_head_overlap_in_top16", 0) >= 4
        and random_arm.get("original_top8_overlap_in_top16", 0) < 4
    ):
        return "grafting"
    return "mixed/diffuse"


def build_subset(args):
    rows = load_jsonl(args.stable_index)
    stable_wrong = [row for row in rows if row.get("stable_label") == "stable_wrong"]
    stable_correct = [row for row in rows if row.get("stable_label") == "stable_correct"]
    selected_ids = {row["id"] for row in stable_wrong}
    selected = list(stable_wrong)
    for row in stable_correct:
        if len([x for x in selected if x.get("stable_label") == "stable_correct"]) >= args.n_correct:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    selected.sort(key=lambda row: int(row["row_index"]))
    write_jsonl(args.out, selected)
    write_json(
        os.path.splitext(args.out)[0] + ".summary.json",
        {
            "stable_index": args.stable_index,
            "out": args.out,
            "n": len(selected),
            "stable_wrong": len(stable_wrong),
            "stable_correct_reference": sum(
                1 for row in selected if row.get("stable_label") == "stable_correct"
            ),
        },
    )
    print(f"wrote {args.out}")


def verify_identity(args):
    baseline_all = load_jsonl(args.baseline_index)
    baseline_by_id = {row.get("id"): row for row in baseline_all}
    candidate_rows = load_jsonl(args.candidate_index)[: args.n]
    mismatches = []
    for cand in candidate_rows:
        base = baseline_by_id.get(cand.get("id"))
        if base is None:
            mismatches.append({"id": cand.get("id"), "error": "missing baseline id"})
            continue
        fields_match = (
            base.get("id") == cand.get("id")
            and base.get("pred") == cand.get("pred")
            and base.get("raw") == cand.get("raw")
            and base.get("correct") == cand.get("correct")
            and base.get("answer_type") == cand.get("answer_type")
        )
        if not fields_match:
            mismatches.append(
                {
                    "id": base.get("id"),
                    "baseline": {
                        "pred": base.get("pred"),
                        "raw": base.get("raw"),
                        "correct": base.get("correct"),
                        "answer_type": base.get("answer_type"),
                    },
                    "candidate": {
                        "pred": cand.get("pred"),
                        "raw": cand.get("raw"),
                        "correct": cand.get("correct"),
                        "answer_type": cand.get("answer_type"),
                    },
                }
            )
    summary = {
        "baseline_index": args.baseline_index,
        "candidate_index": args.candidate_index,
        "n_checked": min(len(candidate_rows), args.n),
        "passed": not mismatches and len(candidate_rows) >= args.n,
        "mismatches": mismatches[:10],
    }
    write_json(args.out, summary)
    if not summary["passed"]:
        raise SystemExit(f"adapter-off identity gate failed; wrote {args.out}")
    print(f"identity gate passed; wrote {args.out}")


def summarize_arm(arm, baseline_rows, baseline_p_last, arm_rows, arm_p_last, original_heads, own_heads):
    ids = [row["id"] for row in arm_rows]
    base_rows, base_p = align_by_id(baseline_rows, baseline_p_last, ids)
    gain_by_example = arm_p_last - base_p
    gain = gain_by_example.mean(axis=0)
    top16 = top_gain_heads(gain, top_k=16)
    top16_set = head_set(top16)
    original_set = head_set(original_heads)
    own_set = head_set(own_heads)

    original_wrong = np.array([row.get("stable_label") == "stable_wrong" for row in base_rows])
    flipped = np.array(
        [
            bool(original_wrong[i])
            and not bool(base_rows[i].get("correct"))
            and bool(arm_rows[i].get("correct"))
            for i in range(len(base_rows))
        ],
        dtype=bool,
    )
    wrong_idx = original_wrong

    original_before = mean_head_values(base_p, original_heads)
    original_after = mean_head_values(arm_p_last, original_heads)
    own_before = mean_head_values(base_p, own_heads)
    own_after = mean_head_values(arm_p_last, own_heads)

    top16_linkage = []
    for head in top16:
        layer = head["layer"]
        h = head["head"]
        per_example_gain = gain_by_example[:, layer, h]
        top16_linkage.append(
            {
                "layer": layer,
                "head": h,
                "mean_gain_all": float(gain[layer, h]),
                "mean_gain_original_wrong": mean_or_nan(per_example_gain[wrong_idx]),
                "mean_gain_flipped": mean_or_nan(per_example_gain[flipped]),
                "flip_correlation": corr_or_nan(per_example_gain[wrong_idx], flipped[wrong_idx]),
            }
        )

    return {
        "arm": arm,
        "n": len(arm_rows),
        "accuracy": mean_or_nan([row.get("correct") for row in arm_rows]),
        "original_stable_wrong_n": int(original_wrong.sum()),
        "flipped_to_correct_n": int(flipped.sum()),
        "flipped_to_correct_rate": float(flipped.sum() / original_wrong.sum())
        if original_wrong.any()
        else math.nan,
        "top16_gain_heads": top16,
        "original_top8_overlap_in_top16": len(top16_set & original_set),
        "own_head_overlap_in_top16": len(top16_set & own_set) if own_heads else None,
        "mean_gain": {
            "all_heads": float(gain.mean()),
            "original_top8": mean_or_nan([gain[h["layer"], h["head"]] for h in original_heads]),
            "own_heads": mean_or_nan([gain[h["layer"], h["head"]] for h in own_heads])
            if own_heads
            else math.nan,
        },
        "behavior_linkage": {
            "original_top8_before_original_wrong": mean_or_nan(original_before[wrong_idx]),
            "original_top8_after_original_wrong": mean_or_nan(original_after[wrong_idx]),
            "original_top8_before_flipped": mean_or_nan(original_before[flipped]),
            "original_top8_after_flipped": mean_or_nan(original_after[flipped]),
            "own_heads_before_flipped": mean_or_nan(own_before[flipped]) if own_heads else math.nan,
            "own_heads_after_flipped": mean_or_nan(own_after[flipped]) if own_heads else math.nan,
        },
        "top16_behavior_linkage": top16_linkage,
        "gain_heatmap": gain,
    }


def plot_f9(arm_summaries, path_png, path_pdf):
    fig, axes = plt.subplots(1, len(arm_summaries), figsize=(14, 4), constrained_layout=True)
    if len(arm_summaries) == 1:
        axes = [axes]
    vmax = max(float(np.nanmax(np.abs(summary["gain_heatmap"]))) for summary in arm_summaries)
    vmax = max(vmax, 1e-6)
    for ax, summary in zip(axes, arm_summaries):
        im = ax.imshow(summary["gain_heatmap"], aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(summary["arm"])
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.8, label="p_last gain vs baseline")
    Path(path_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def analyze(args):
    baseline_rows, baseline_p_last = load_p_last(args.baseline_npz, args.baseline_index)
    original_heads = load_heads(args.a1_summary)
    arm_outputs = {}
    plot_summaries = []
    for arm, defaults in ARMS.items():
        npz_path = getattr(args, f"{arm}_npz") or defaults["npz"]
        index_path = getattr(args, f"{arm}_index") or defaults["index"]
        config_path = getattr(args, f"{arm}_adapter_config") or defaults["adapter_config"]
        own_heads = [] if arm == "generic" else load_adapter_heads(config_path)
        arm_rows, arm_p_last = load_p_last(npz_path, index_path)
        summary = summarize_arm(
            arm,
            baseline_rows,
            baseline_p_last,
            arm_rows,
            arm_p_last,
            original_heads,
            own_heads,
        )
        plot_summaries.append(summary)
        summary_json = dict(summary)
        summary_json.pop("gain_heatmap")
        arm_outputs[arm] = summary_json
    reading = classify_mechanism_reading(arm_outputs)
    plot_f9(plot_summaries, args.f9_png, args.f9_pdf)
    sidecar = {
        arm: summary["top16_gain_heads"]
        for arm, summary in zip([s["arm"] for s in plot_summaries], plot_summaries)
    }
    write_json(args.f9_data, sidecar)
    out = {
        "task": "F-1",
        "baseline_npz": args.baseline_npz,
        "baseline_index": args.baseline_index,
        "a1_summary": args.a1_summary,
        "arms": arm_outputs,
        "mechanism_reading": reading,
        "figures": {"F9_png": args.f9_png, "F9_pdf": args.f9_pdf, "F9_data": args.f9_data},
    }
    write_json(args.out, out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-subset")
    p.add_argument("--stable-index", default="results/overwrite_attention_extraction/extract_stable_index.jsonl")
    p.add_argument("--n-correct", type=int, default=200)
    p.add_argument("--out", default="results/hard_tier_and_transport/mech_loop/f1_subset_index.jsonl")
    p.set_defaults(func=build_subset)

    p = sub.add_parser("verify-identity")
    p.add_argument("--baseline-index", required=True)
    p.add_argument("--candidate-index", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="results/hard_tier_and_transport/mech_loop/adapter_off_identity.json")
    p.set_defaults(func=verify_identity)

    p = sub.add_parser("analyze")
    p.add_argument("--baseline-npz", default="results/overwrite_attention_extraction/extract_stable.npz")
    p.add_argument("--baseline-index", default="results/overwrite_attention_extraction/extract_stable_index.jsonl")
    p.add_argument("--a1-summary", default="results/overwrite_attention_extraction/a1_stable_summary.json")
    for arm in ARMS:
        p.add_argument(f"--{arm}-npz", default=None)
        p.add_argument(f"--{arm}-index", default=None)
        p.add_argument(f"--{arm}-adapter-config", default=None)
    p.add_argument("--out", default="results/hard_tier_and_transport/mech_loop_summary.json")
    p.add_argument("--f9-png", default="results/figures/F9_repaired_p_last_gain.png")
    p.add_argument("--f9-pdf", default="results/figures/F9_repaired_p_last_gain.pdf")
    p.add_argument("--f9-data", default="results/figures/F9_repaired_p_last_gain.data.json")
    p.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
