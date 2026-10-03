"""Stage A1: attention selection analysis."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_p_last(attn_writes, write_counts):
    out = np.zeros(attn_writes.shape[:3], dtype=np.float32)
    for i, write_count in enumerate(write_counts):
        used = attn_writes[i, :, :, :write_count]
        denom = used.sum(axis=-1)
        last = attn_writes[i, :, :, write_count - 1]
        out[i] = np.divide(last, denom, out=np.zeros_like(last), where=denom > 0)
    return out


def stale_answer_ratios(attn_writes, rows):
    answered = []
    last = []
    row_indices = []
    for i, row in enumerate(rows):
        if row["answer_type"] != "stale" or row["answered_write_idx"] is None:
            continue
        write_count = row["write_count"]
        used = attn_writes[i, :, :, :write_count]
        denom = used.sum(axis=-1)
        ans = attn_writes[i, :, :, row["answered_write_idx"]]
        cur = attn_writes[i, :, :, write_count - 1]
        answered.append(np.divide(ans, denom, out=np.zeros_like(ans), where=denom > 0))
        last.append(np.divide(cur, denom, out=np.zeros_like(cur), where=denom > 0))
        row_indices.append(i)
    if not answered:
        shape = (0,) + attn_writes.shape[1:3]
        return {"answered": np.zeros(shape), "last": np.zeros(shape), "row_indices": []}
    return {
        "answered": np.stack(answered, axis=0),
        "last": np.stack(last, axis=0),
        "row_indices": row_indices,
    }


def _mean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def _ols_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "intercept": math.nan, "slope": math.nan}
    coef = np.polyfit(x[mask], y[mask], deg=1)
    return {"n": int(mask.sum()), "intercept": float(coef[1]), "slope": float(coef[0])}


def wrong_mask_for_rows(rows, wrong_filter):
    mask = np.array([not bool(row["correct"]) for row in rows], dtype=bool)
    if wrong_filter == "any":
        return mask
    if wrong_filter == "within_stale":
        return np.array(
            [
                (not bool(row["correct"])) and row.get("answer_type") == "stale"
                for row in rows
            ],
            dtype=bool,
        )
    raise ValueError(f"unknown wrong_filter={wrong_filter}")


def _plot_heatmap(data, path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="mean p_last")
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_curves(correct_curve, wrong_curve, path):
    x = np.arange(len(correct_curve))
    plt.figure(figsize=(8, 5))
    plt.plot(x, correct_curve, label="correct")
    plt.plot(x, wrong_curve, label="wrong")
    plt.xlabel("layer")
    plt.ylabel("mean p_last")
    plt.title("Last-write preference by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_content_position(rows, p_last_mean, path):
    content_x = []
    content_y = []
    for load in [2, 4, 8]:
        vals = [
            1 - row["correct"]
            for row in rows
            if row["condition"] == "interference"
            and row["n_lines"] == 40
            and row["interference_load"] == load
        ]
        content_x.append(load)
        content_y.append(_mean_or_nan(vals))

    pos_x = []
    pos_y = []
    for n_lines in [20, 40, 80]:
        vals = [
            1 - row["correct"]
            for row in rows
            if row["condition"] == "interference"
            and row["interference_load"] == 4
            and row["n_lines"] == n_lines
        ]
        pos_x.append(n_lines)
        pos_y.append(_mean_or_nan(vals))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(content_x, content_y, marker="o")
    axes[0].set_xlabel("stale writes I at n_lines=40")
    axes[0].set_ylabel("error rate")
    axes[0].set_title("Content load")
    axes[1].plot(pos_x, pos_y, marker="o")
    axes[1].set_xlabel("n_lines at I=4")
    axes[1].set_ylabel("error rate")
    axes[1].set_title("Length / position")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/overwrite_attention_extraction/extract.npz")
    ap.add_argument("--index", default="results/overwrite_attention_extraction/extract_index.jsonl")
    ap.add_argument("--out", default="results/overwrite_attention_extraction/a1_summary.json")
    ap.add_argument("--fig-dir", default="results/overwrite_attention_extraction/figs")
    ap.add_argument(
        "--wrong-filter",
        choices=["any", "within_stale"],
        default="any",
        help="Which wrong examples to use for correct-vs-wrong contrasts.",
    )
    args = ap.parse_args()

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.index)
    data = np.load(args.npz)
    attn_writes = data["attn_writes"]
    write_counts = np.array([row["write_count"] for row in rows], dtype=np.int64)
    p_last = compute_p_last(attn_writes, write_counts)
    p_last_mean = p_last.mean(axis=(1, 2))

    correct = np.array([bool(row["correct"]) for row in rows])
    wrong = wrong_mask_for_rows(rows, args.wrong_filter)
    correct_heatmap = p_last[correct].mean(axis=0)
    wrong_heatmap = p_last[wrong].mean(axis=0) if wrong.any() else None

    heatmap_path = os.path.join(args.fig_dir, "a1_p_last_heatmap.png")
    curves_path = os.path.join(args.fig_dir, "a1_p_last_correct_vs_wrong.png")
    content_path = os.path.join(args.fig_dir, "a1_content_vs_position.png")

    _plot_heatmap(correct_heatmap, heatmap_path, "Mean p_last on correct examples")
    _plot_curves(
        p_last[correct].mean(axis=(0, 2)),
        p_last[wrong].mean(axis=(0, 2)) if wrong.any() else np.zeros(p_last.shape[1]),
        curves_path,
    )
    _plot_content_position(rows, p_last_mean, content_path)

    stale_ratios = stale_answer_ratios(attn_writes, rows)
    flat = correct_heatmap.reshape(-1)
    top_indices = np.argsort(flat)[::-1][:20]
    n_heads = correct_heatmap.shape[1]
    top_heads = [
        {
            "layer": int(idx // n_heads),
            "head": int(idx % n_heads),
            "mean_p_last_correct": float(flat[idx]),
        }
        for idx in top_indices
    ]

    by_cell = []
    for condition in ["simple", "interference"]:
        for n_lines in sorted({row["n_lines"] for row in rows if row["condition"] == condition}):
            loads = sorted(
                {
                    row["interference_load"]
                    for row in rows
                    if row["condition"] == condition and row["n_lines"] == n_lines
                }
            )
            for load in loads:
                idx = [
                    i
                    for i, row in enumerate(rows)
                    if row["condition"] == condition
                    and row["n_lines"] == n_lines
                    and row["interference_load"] == load
                ]
                by_cell.append(
                    {
                        "condition": condition,
                        "n_lines": n_lines,
                        "interference_load": load,
                        "n": len(idx),
                        "accuracy": _mean_or_nan([rows[i]["correct"] for i in idx]),
                        "mean_p_last": _mean_or_nan(p_last_mean[idx]),
                    }
                )

    top_head_by_i = []
    for load in [2, 4, 8]:
        idx_correct = [
            i
            for i, row in enumerate(rows)
            if row["condition"] == "interference"
            and row["n_lines"] == 40
            and row["interference_load"] == load
            and row["correct"]
        ]
        idx_wrong = [
            i
            for i, row in enumerate(rows)
            if row["condition"] == "interference"
            and row["n_lines"] == 40
            and row["interference_load"] == load
            and wrong[i]
        ]
        for head in top_heads[:10]:
            layer = head["layer"]
            h = head["head"]
            top_head_by_i.append(
                {
                    "interference_load": load,
                    "layer": layer,
                    "head": h,
                    "p_last_correct": _mean_or_nan(p_last[idx_correct, layer, h]),
                    "p_last_wrong": _mean_or_nan(p_last[idx_wrong, layer, h]),
                }
            )

    fixed_len = [
        i
        for i, row in enumerate(rows)
        if row["condition"] == "interference" and row["n_lines"] == 40
    ]
    fixed_i = [
        i
        for i, row in enumerate(rows)
        if row["condition"] == "interference" and row["interference_load"] == 4
    ]
    content_x = [rows[i]["interference_load"] for i in fixed_len]
    pos_x = [rows[i]["target_write_positions"][-1] / rows[i]["n_lines"] for i in fixed_i]
    error_y_len = [1 - rows[i]["correct"] for i in fixed_len]
    error_y_pos = [1 - rows[i]["correct"] for i in fixed_i]

    summary = {
        "n": len(rows),
        "model": str(data["model"]) if "model" in data else None,
        "wrong_filter": args.wrong_filter,
        "wrong_counts": {
            "total_wrong": int((~correct).sum()),
            "used_wrong": int(wrong.sum()),
            "excluded_wrong": int((~correct).sum() - wrong.sum()),
        },
        "figures": {
            "p_last_heatmap": heatmap_path,
            "p_last_correct_vs_wrong": curves_path,
            "content_vs_position": content_path,
        },
        "by_cell": by_cell,
        "stale_wrong_ratio_mean": {
            "answered_write": float(stale_ratios["answered"].mean())
            if stale_ratios["answered"].size
            else math.nan,
            "last_write": float(stale_ratios["last"].mean())
            if stale_ratios["last"].size
            else math.nan,
            "n": len(stale_ratios["row_indices"]),
        },
        "top_heads_by_correct_p_last": top_heads,
        "top_head_correct_wrong_by_I": top_head_by_i,
        "regressions": {
            "error_vs_interference_load_at_n40": _ols_slope(content_x, error_y_len),
            "p_last_vs_interference_load_at_n40": _ols_slope(
                content_x, p_last_mean[fixed_len]
            ),
            "error_vs_last_write_position_at_I4": _ols_slope(pos_x, error_y_pos),
            "p_last_vs_last_write_position_at_I4": _ols_slope(
                pos_x, p_last_mean[fixed_i]
            ),
        },
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
