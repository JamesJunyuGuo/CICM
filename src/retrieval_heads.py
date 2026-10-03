"""Simplified retrieval-head detector for Stage E E2f.

The detector follows the Addendum 2 operational definition: during greedy
answer generation, a head gets credit when its argmax attention over the prompt
context lands inside the target value span and the generated token equals the
context token at that attended position.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from analyze_write_attention import compute_p_last, load_jsonl
from gen_tasks import make_example
from span_utils import build_chat_prompt, map_target_value_spans


def parse_int(text):
    import re

    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def classify_answer(row, pred):
    if pred == row["gold"]:
        return "correct"
    if pred in row.get("stale_values", []):
        return "stale"
    return "other"


def resolve_torch_dtype(dtype_name):
    import torch

    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_simple_rows(n_per_length, seed, lengths=(40, 80)):
    rng = random.Random(seed)
    rows = []
    uid = 0
    for n_lines in lengths:
        for _ in range(n_per_length):
            row = make_example("simple", n_lines, 0, rng)
            row["id"] = uid
            uid += 1
            rows.append(row)
    return rows


def rankdata(values):
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_p_last_scores(npz_path, index_path):
    rows = load_jsonl(index_path)
    data = np.load(npz_path)
    p_last = compute_p_last(data["attn_writes"], np.array([r["write_count"] for r in rows]))
    correct = np.array([bool(r["correct"]) for r in rows])
    scores = p_last[correct].mean(axis=0)
    return scores


def top_heads_from_matrix(matrix, top_k):
    flat = matrix.reshape(-1)
    n_heads = matrix.shape[1]
    order = np.argsort(flat)[::-1][:top_k]
    return [
        {"layer": int(idx // n_heads), "head": int(idx % n_heads), "score": float(flat[idx])}
        for idx in order
    ]


def verdict_from_overlap(overlap_count, reference_count):
    if overlap_count == reference_count:
        return "same heads"
    if overlap_count == 0:
        return "disjoint"
    return "partial overlap"


def _tokenizer_call(tokenizer, text):
    return tokenizer(text, return_tensors="pt")


def detect_retrieval_scores(model, tokenizer, rows, max_answer_tokens, device):
    import torch

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hit = np.zeros((n_layers, n_heads), dtype=np.float64)
    denom = 0
    per_example = []

    for ex_idx, row in enumerate(rows, 1):
        templated = build_chat_prompt(row["prompt"], tokenizer)
        encoded = _tokenizer_call(tokenizer, templated)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = int(input_ids.shape[1])
        span = map_target_value_spans(row, tokenizer)["current"]
        span_start = int(span["token_start"])
        span_end = int(span["token_end"])
        context_ids = input_ids[0, :prompt_len].detach().cpu().tolist()
        needle_ids = set(context_ids[span_start:span_end])

        generated = []
        example_denom = 0
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_answer_tokens):
                if past_key_values is None:
                    step_input_ids = input_ids
                else:
                    step_input_ids = input_ids[:, -1:]
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1]
                next_id = int(torch.argmax(logits).detach().cpu())
                generated.append(next_id)

                if next_id in needle_ids:
                    denom += 1
                    example_denom += 1
                    for layer_idx, layer_attn in enumerate(outputs.attentions):
                        final_attn = layer_attn[0, :, -1, :prompt_len].detach().float().cpu().numpy()
                        argmax_pos = np.argmax(final_attn, axis=-1)
                        for head_idx, pos in enumerate(argmax_pos):
                            if span_start <= int(pos) < span_end and context_ids[int(pos)] == next_id:
                                hit[layer_idx, head_idx] += 1.0

                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype, device=device)],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                        ],
                        dim=1,
                    )
                parsed = parse_int(tokenizer.decode(generated, skip_special_tokens=True))
                if parsed == row["gold"] and len(generated) >= max(1, span_end - span_start):
                    break

        pred = parse_int(tokenizer.decode(generated, skip_special_tokens=True))
        per_example.append(
            {
                "id": row["id"],
                "n_lines": row["n_lines"],
                "gold": row["gold"],
                "pred": pred,
                "answer_type": classify_answer(row, pred),
                "copied_answer_tokens": example_denom,
            }
        )
        if ex_idx == 1 or ex_idx % 25 == 0 or ex_idx == len(rows):
            print(f"processed {ex_idx}/{len(rows)} copied_tokens={denom}", flush=True)

    scores = hit / max(1, denom)
    return scores, int(denom), per_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--n-per-length", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-answer-tokens", type=int, default=4)
    ap.add_argument("--out-dir", default="results/llama_and_scale/retrieval_heads")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--examples-out", default=None)
    ap.add_argument("--stage-a-npz", default="results/overwrite_attention_extraction/extract_stable.npz")
    ap.add_argument("--stage-a-index", default="results/overwrite_attention_extraction/extract_stable_index.jsonl")
    ap.add_argument("--stage-a-a1", default="results/overwrite_attention_extraction/a1_stable_summary.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    examples_path = Path(args.examples_out) if args.examples_out else out_dir / "examples.jsonl"

    rows = build_simple_rows(args.n_per_length, args.seed)
    if args.limit is not None:
        rows = rows[: args.limit]
    write_jsonl(out_dir / "fresh_simple_seed30.jsonl", rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_torch_dtype(args.dtype),
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    retrieval_scores, copied_tokens, per_example = detect_retrieval_scores(
        model, tokenizer, rows, args.max_answer_tokens, device
    )
    write_jsonl(examples_path, per_example)

    p_last_scores = load_p_last_scores(args.stage_a_npz, args.stage_a_index)
    retrieval_top16 = top_heads_from_matrix(retrieval_scores, 16)
    p_last_top8 = top_heads_from_matrix(p_last_scores, 8)
    p_last_top8_set = {(h["layer"], h["head"]) for h in p_last_top8}
    retrieval_top16_set = {(h["layer"], h["head"]) for h in retrieval_top16}
    overlap = sorted(p_last_top8_set & retrieval_top16_set)
    rho = spearman(retrieval_scores.reshape(-1), p_last_scores.reshape(-1))

    correct = sum(1 for row in per_example if row["answer_type"] == "correct")
    summary = {
        "stage": "E",
        "task": "E2f_retrieval_head_overlap",
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "n_examples": len(rows),
        "n_per_length": args.n_per_length,
        "limit": args.limit,
        "lengths": [40, 80],
        "copied_answer_tokens": copied_tokens,
        "accuracy": correct / len(per_example) if per_example else None,
        "retrieval_top16": retrieval_top16,
        "p_last_top8": p_last_top8,
        "overlap_heads": [{"layer": int(l), "head": int(h)} for l, h in overlap],
        "overlap_count": len(overlap),
        "spearman_retrieval_vs_p_last_all_heads": rho,
        "verdict": verdict_from_overlap(len(overlap), len(p_last_top8_set)),
        "artifacts": {
            "examples": str(examples_path),
            "fresh_simple_data": str(out_dir / "fresh_simple_seed30.jsonl"),
            "stage_a_npz": args.stage_a_npz,
            "stage_a_index": args.stage_a_index,
            "stage_a_a1": args.stage_a_a1,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
