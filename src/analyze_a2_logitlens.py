"""Stage A2: logit-lens analysis over saved final-token hidden states."""

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


def should_drop_shared_first_token(gold_token_id, stale_token_ids):
    return gold_token_id in set(stale_token_ids)


def first_value_token_id(tokenizer, value):
    value = str(value)
    candidates = [value, " " + value]
    for text in candidates:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        decoded = tokenizer.decode(ids)
        if decoded == value or decoded.strip() == value:
            return ids[0], decoded
    raise ValueError(f"could not tokenize value with verifiable decode: {value}")


def _mean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def _median_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)) if arr.size else math.nan


def _plot_gap_by_layer(summary_rows, path):
    plt.figure(figsize=(8, 5))
    for key, values in summary_rows.items():
        plt.plot(np.arange(len(values)), values, label=key)
    plt.xlabel("layer")
    plt.ylabel("mean gold - best-stale logit gap")
    plt.title("Logit-lens gap by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _get_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise AttributeError("could not find final norm module")


def wrong_mask_for_rows(rows, wrong_filter):
    if wrong_filter == "any":
        return [not bool(row["correct"]) for row in rows]
    if wrong_filter == "within_stale":
        return [
            (not bool(row["correct"])) and row.get("answer_type") == "stale"
            for row in rows
        ]
    raise ValueError(f"unknown wrong_filter={wrong_filter}")


def compute_logit_gaps(hidden, rows, tokenizer, model, device, batch_size=128):
    import torch

    kept = []
    dropped_shared = []
    skipped_no_stale = []
    gold_token_ids = []
    stale_token_ids = []

    for i, row in enumerate(rows):
        if not row["stale_values"]:
            skipped_no_stale.append(i)
            continue
        gold_id, _ = first_value_token_id(tokenizer, row["gold"])
        stale_ids = [first_value_token_id(tokenizer, v)[0] for v in row["stale_values"]]
        if should_drop_shared_first_token(gold_id, stale_ids):
            dropped_shared.append(i)
            continue
        kept.append(i)
        gold_token_ids.append(gold_id)
        stale_token_ids.append(stale_ids)

    if not kept:
        raise ValueError("no examples left after logit-lens filtering")

    norm = _get_norm_module(model)
    lm_head = model.lm_head
    norm.eval()
    lm_head.eval()
    n_layers = hidden.shape[1]
    gaps = np.zeros((len(kept), n_layers), dtype=np.float32)

    token_pool = sorted(
        set(gold_token_ids).union({tok for toks in stale_token_ids for tok in toks})
    )
    token_to_col = {tok: j for j, tok in enumerate(token_pool)}
    weight = lm_head.weight.detach()
    selected_weight = weight[token_pool].to(device)

    for layer in range(n_layers):
        layer_gaps = []
        for start in range(0, len(kept), batch_size):
            end = min(start + batch_size, len(kept))
            idx = kept[start:end]
            h = torch.from_numpy(hidden[idx, layer, :]).to(device=device, dtype=weight.dtype)
            with torch.no_grad():
                normed = norm(h)
                logits = normed @ selected_weight.T
            logits_np = logits.detach().float().cpu().numpy()
            for local_i, global_i in enumerate(range(start, end)):
                gold_col = token_to_col[gold_token_ids[global_i]]
                stale_cols = [token_to_col[tok] for tok in stale_token_ids[global_i]]
                layer_gaps.append(
                    float(logits_np[local_i, gold_col] - logits_np[local_i, stale_cols].max())
                )
        gaps[:, layer] = np.array(layer_gaps, dtype=np.float32)

    return {
        "kept_indices": kept,
        "dropped_shared_indices": dropped_shared,
        "skipped_no_stale_indices": skipped_no_stale,
        "gaps": gaps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/stage_a/extract.npz")
    ap.add_argument("--index", default="results/stage_a/extract_index.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/stage_a/a2_summary.json")
    ap.add_argument("--fig-dir", default="results/stage_a/figs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--wrong-filter",
        choices=["any", "within_stale"],
        default="any",
        help="Which wrong examples to use for correct-vs-wrong contrasts.",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.index)
    data = np.load(args.npz)
    hidden = data["hidden"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    result = compute_logit_gaps(
        hidden, rows, tokenizer, model, device, batch_size=args.batch_size
    )
    kept_rows = [rows[i] for i in result["kept_indices"]]
    gaps = result["gaps"]
    wrong_mask = wrong_mask_for_rows(kept_rows, args.wrong_filter)
    total_wrong = sum(not bool(row["correct"]) for row in kept_rows)
    used_wrong = sum(wrong_mask)

    curves = {}
    median_curves = {}
    by_cell = []
    for load in [2, 4, 8]:
        for correctness, label in [(1, "correct"), (0, "wrong")]:
            idx = [
                i
                for i, row in enumerate(kept_rows)
                if row["condition"] == "interference"
                and row["n_lines"] == 40
                and row["interference_load"] == load
                and (
                    row["correct"] == correctness
                    if correctness
                    else wrong_mask[i]
                )
            ]
            key = f"I={load} {label}"
            if idx:
                cell_gaps = gaps[idx]
                curve = cell_gaps.mean(axis=0)
                median_curve = np.median(cell_gaps, axis=0)
                curves[key] = curve.tolist()
                median_curves[key] = median_curve.tolist()
                by_cell.append(
                    {
                        "interference_load": load,
                        "correct": bool(correctness),
                        "n": len(idx),
                        "final_layer_gap": float(curve[-1]),
                        "median_final_layer_gap": float(median_curve[-1]),
                        "max_gap": float(curve.max()),
                        "median_max_gap": float(median_curve.max()),
                        "first_positive_layer": int(np.argmax(curve > 0))
                        if np.any(curve > 0)
                        else None,
                        "median_first_positive_layer": int(np.argmax(median_curve > 0))
                        if np.any(median_curve > 0)
                        else None,
                    }
                )

    fig_path = str(Path(args.fig_dir) / "a2_logitlens_gap_by_layer.png")
    _plot_gap_by_layer(curves, fig_path)

    summary = {
        "model": args.model,
        "wrong_filter": args.wrong_filter,
        "n_total": len(rows),
        "n_kept": len(result["kept_indices"]),
        "wrong_counts": {
            "total_wrong_kept": int(total_wrong),
            "used_wrong": int(used_wrong),
            "excluded_wrong": int(total_wrong - used_wrong),
        },
        "n_dropped_shared_first_token": len(result["dropped_shared_indices"]),
        "n_skipped_no_stale": len(result["skipped_no_stale_indices"]),
        "drop_rate_among_stale_examples": len(result["dropped_shared_indices"])
        / max(1, len(rows) - len(result["skipped_no_stale_indices"])),
        "figure": fig_path,
        "by_cell": by_cell,
        "mean_gap_by_layer": curves,
        "median_gap_by_layer": median_curves,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
