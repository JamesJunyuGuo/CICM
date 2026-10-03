"""Stage F-2: per-example transport-share law."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from analyze_write_attention import compute_p_last, load_jsonl


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def auc_score(y_true, scores):
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sorted_s = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_s[end] == sorted_s[start]:
            end += 1
        if end - start > 1:
            avg = ranks[order[start:end]].mean()
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic_1d(x, y, max_iter=50, ridge=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std()) if float(x.std()) > 1e-12 else 1.0
    z = (x - mean) / std
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = X.T @ (y - p) - ridge * np.array([0.0, beta[1]])
        hess = -(X.T * w) @ X - ridge * np.diag([0.0, 1.0])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break

    def predict(values):
        arr = np.asarray(values, dtype=np.float64)
        zz = (arr - mean) / std
        return sigmoid(beta[0] + beta[1] * zz)

    probs = predict(x)
    return {
        "intercept": float(beta[0] - beta[1] * mean / std),
        "coef": float(beta[1] / std),
        "standardized_intercept": float(beta[0]),
        "standardized_coef": float(beta[1]),
        "x_mean": mean,
        "x_std": std,
        "auc": auc_score(y, probs),
        "predict": predict,
    }


def decile_points(x, y, n_bins=10):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    bins = np.array_split(order, n_bins)
    points = []
    for bin_idx, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        points.append(
            {
                "bin": int(bin_idx),
                "n": int(len(idx)),
                "x_mean": float(x[idx].mean()),
                "x_min": float(x[idx].min()),
                "x_max": float(x[idx].max()),
                "accuracy": float(y[idx].mean()),
            }
        )
    return points


def load_top_heads(path, k=8):
    with open(path) as f:
        summary = json.load(f)
    return [
        (int(item["layer"]), int(item["head"]))
        for item in summary["top_heads_by_correct_p_last"][:k]
    ]


def extract_model_examples(npz_path, index_path, top_heads):
    rows = load_jsonl(index_path)
    data = np.load(npz_path)
    write_counts = np.array([row["write_count"] for row in rows], dtype=np.int64)
    p_last = compute_p_last(data["attn_writes"], write_counts)
    selected = []
    for i, row in enumerate(rows):
        if (
            row.get("condition") == "interference"
            and row.get("n_lines") == 40
            and row.get("interference_load") in {2, 4, 8}
        ):
            selected.append(i)
    idx = np.array(selected, dtype=np.int64)
    all_heads = p_last[idx].mean(axis=(1, 2))
    top8 = np.stack([p_last[idx, layer, head] for layer, head in top_heads], axis=1).mean(axis=1)
    y = np.array([int(rows[i]["correct"]) for i in idx], dtype=np.int64)
    loads = np.array([int(rows[i]["interference_load"]) for i in idx], dtype=np.float64)
    return {
        "rows": [rows[i] for i in idx],
        "correct": y,
        "interference_load": loads,
        "p_last_all_heads": all_heads,
        "p_last_top8": top8,
    }


def summarize_predictor(x, y):
    fit = fit_logistic_1d(x, y)
    return {
        "n": int(len(y)),
        "coef": fit["coef"],
        "intercept": fit["intercept"],
        "auc": fit["auc"],
        "deciles": decile_points(x, y),
    }


def auc_by_load(x, y, loads):
    out = []
    for load in sorted(set(int(v) for v in loads)):
        mask = loads == load
        out.append(
            {
                "interference_load": int(load),
                "n": int(mask.sum()),
                "auc": auc_score(y[mask], x[mask]),
            }
        )
    return out


def plot_f10(model_summaries, path_png, path_pdf):
    fig, axes = plt.subplots(1, len(model_summaries), figsize=(12, 4), constrained_layout=True)
    if len(model_summaries) == 1:
        axes = [axes]
    for ax, summary in zip(axes, model_summaries):
        x = np.asarray(summary["_x"], dtype=np.float64)
        y = np.asarray(summary["_y"], dtype=np.float64)
        fit = fit_logistic_1d(x, y)
        deciles = summary["predictors"]["top8"]["deciles"]
        ax.scatter(
            [d["x_mean"] for d in deciles],
            [d["accuracy"] for d in deciles],
            s=[max(20, d["n"] * 0.6) for d in deciles],
            label="empirical deciles",
        )
        grid = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(grid, fit["predict"](grid), color="black", label="logistic fit")
        ax.set_title(summary["model_label"])
        ax.set_xlabel("top-8 mean p_last")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
    Path(path_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def analyze(args):
    configs = [
        {
            "model_label": "Qwen2.5-7B",
            "npz": args.qwen_npz,
            "index": args.qwen_index,
            "a1": args.qwen_a1,
        },
        {
            "model_label": "Llama3.1-8B",
            "npz": args.llama_npz,
            "index": args.llama_index,
            "a1": args.llama_a1,
        },
    ]
    outputs = []
    for config in configs:
        top_heads = load_top_heads(config["a1"])
        examples = extract_model_examples(config["npz"], config["index"], top_heads)
        y = examples["correct"]
        loads = examples["interference_load"]
        p_all = examples["p_last_all_heads"]
        p_top8 = examples["p_last_top8"]
        i_only = summarize_predictor(loads, y)
        model_out = {
            "model_label": config["model_label"],
            "npz": config["npz"],
            "index": config["index"],
            "a1": config["a1"],
            "n": int(len(y)),
            "accuracy": float(y.mean()),
            "predictors": {
                "all_heads": summarize_predictor(p_all, y),
                "top8": summarize_predictor(p_top8, y),
                "I_only": i_only,
            },
            "within_I_auc": {
                "all_heads": auc_by_load(p_all, y, loads),
                "top8": auc_by_load(p_top8, y, loads),
            },
            "_x": p_top8.tolist(),
            "_y": y.tolist(),
        }
        outputs.append(model_out)
    plot_f10(outputs, args.f10_png, args.f10_pdf)
    sidecar = [
        {
            "model_label": out["model_label"],
            "top8_deciles": out["predictors"]["top8"]["deciles"],
            "top8_auc": out["predictors"]["top8"]["auc"],
            "all_heads_auc": out["predictors"]["all_heads"]["auc"],
            "I_only_auc": out["predictors"]["I_only"]["auc"],
        }
        for out in outputs
    ]
    write_json(args.f10_data, sidecar)
    for out in outputs:
        out.pop("_x")
        out.pop("_y")
    summary = {
        "task": "F-2",
        "models": outputs,
        "figures": {"F10_png": args.f10_png, "F10_pdf": args.f10_pdf, "F10_data": args.f10_data},
    }
    write_json(args.out, summary)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-npz", default="results/overwrite_attention_extraction/extract_stable.npz")
    ap.add_argument("--qwen-index", default="results/overwrite_attention_extraction/extract_stable_index.jsonl")
    ap.add_argument("--qwen-a1", default="results/overwrite_attention_extraction/a1_stable_summary.json")
    ap.add_argument("--llama-npz", default="results/llama_and_scale/llama/extract_stable_ai_local.npz")
    ap.add_argument("--llama-index", default="results/llama_and_scale/llama/extract_stable_index_ai_local.jsonl")
    ap.add_argument("--llama-a1", default="results/llama_and_scale/llama/a1_stable_ai_local_summary.json")
    ap.add_argument("--out", default="results/hard_tier_and_transport/transport_law_summary.json")
    ap.add_argument("--f10-png", default="results/figures/F10_transport_share_law.png")
    ap.add_argument("--f10-pdf", default="results/figures/F10_transport_share_law.pdf")
    ap.add_argument("--f10-data", default="results/figures/F10_transport_share_law.data.json")
    args = ap.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
