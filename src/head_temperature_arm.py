"""Stage C Arm T: label-free per-head attention temperature."""

import argparse
import contextlib
import contextvars
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervene_attention_bias import build_example_sets, build_scope_heads
from adapter_eval import load_jsonl, score_single_int
from span_utils import build_chat_prompt

_TEMPERATURE_CONTEXT = contextvars.ContextVar("stage_c_temperature_context", default=None)
_QWEN2_PATCHED = False


@dataclass(frozen=True)
class TemperatureContext:
    tau: float
    heads_by_layer: dict


def apply_temperature(attn_scores, layer_idx, ctx):
    if ctx is None or ctx.tau == 1.0:
        return attn_scores
    if ctx.tau <= 0:
        raise ValueError(f"tau must be positive: {ctx.tau}")
    heads = ctx.heads_by_layer.get(layer_idx)
    if not heads:
        return attn_scores
    out = attn_scores.clone()
    head_idx = torch.tensor(tuple(heads), dtype=torch.long, device=attn_scores.device)
    out[:, head_idx, :, :] = out[:, head_idx, :, :] / float(ctx.tau)
    return out


def install_qwen2_temperature_patch():
    global _QWEN2_PATCHED
    if _QWEN2_PATCHED:
        return
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    def stage_c_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = qwen2.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen2.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = apply_temperature(
            attn_weights, module.layer_idx, _TEMPERATURE_CONTEXT.get()
        )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    qwen2.eager_attention_forward = stage_c_eager_attention_forward
    _QWEN2_PATCHED = True


@contextlib.contextmanager
def temperature_context(ctx):
    token = _TEMPERATURE_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _TEMPERATURE_CONTEXT.reset(token)


def generate_one(model, tokenizer, prompt, max_new_tokens, ctx=None):
    templated = build_chat_prompt(prompt, tokenizer)
    enc = tokenizer(templated, return_tensors="pt").to(model.device)
    with torch.no_grad():
        if ctx is None:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            with temperature_context(ctx):
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def resolve_identity_baseline(mode, model_name):
    if mode != "auto":
        return mode
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        return "stage_a"
    return "same_model"


def make_stage_b_sets(extract_index, stable_labels, seed=1, limit=None):
    sets = build_example_sets(
        load_jsonl(extract_index), load_jsonl(stable_labels), correct_subsample_n=200, seed=seed
    )
    if limit:
        return {key: rows[:limit] for key, rows in sets.items()}
    return sets


def load_fresh_single_int(path, limit=None):
    rows = []
    for row in load_jsonl(path):
        if isinstance(row.get("gold"), int):
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def evaluate_arm_t(args):
    install_qwen2_temperature_patch()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    top_summary = json.load(open(args.a1_summary))
    heads_by_layer = build_scope_heads(
        "topk", model.config.num_hidden_layers, model.config.num_attention_heads, top_summary
    )
    data_by_id = {row["id"]: row for row in load_jsonl(args.data)}
    sets = make_stage_b_sets(args.extract_index, args.stable_labels, seed=args.seed, limit=args.limit)
    fresh = load_fresh_single_int(args.fresh_eval, args.fresh_limit)
    identity_baseline = resolve_identity_baseline(args.identity_baseline, args.model)
    baseline_cache = {}

    records = []
    for tau in args.taus:
        ctx = TemperatureContext(tau=float(tau), heads_by_layer=heads_by_layer)
        for set_name, rows in sets.items():
            for row in rows:
                data_row = data_by_id[row["id"]]
                if identity_baseline == "same_model":
                    if row["id"] not in baseline_cache:
                        base_raw = generate_one(
                            model, tokenizer, data_row["prompt"], args.max_new_tokens, ctx=None
                        )
                        baseline_cache[row["id"]] = score_single_int(data_row, base_raw)
                        baseline_cache[row["id"]]["raw"] = base_raw
                    baseline_pred = baseline_cache[row["id"]]["pred"]
                    baseline_correct = baseline_cache[row["id"]]["correct"]
                    baseline_source = "same_model_no_temperature"
                else:
                    baseline_pred = row["pred"]
                    baseline_correct = row["correct"]
                    baseline_source = "stage_a_extract"
                raw = generate_one(model, tokenizer, data_row["prompt"], args.max_new_tokens, ctx)
                scored = score_single_int(data_row, raw)
                records.append(
                    {
                        "eval_set": set_name,
                        "id": row["id"],
                        "tau": float(tau),
                        "baseline_pred": baseline_pred,
                        "baseline_correct": baseline_correct,
                        "baseline_source": baseline_source,
                        "gold": row["gold"],
                        "raw": raw,
                        **scored,
                    }
                )
        for row in fresh:
            raw = generate_one(model, tokenizer, row["prompt"], args.max_new_tokens, ctx)
            scored = score_single_int(row, raw)
            records.append(
                {
                    "eval_set": "fresh",
                    "id": row["id"],
                    "tau": float(tau),
                    "condition": row.get("condition"),
                    "n_lines": row.get("n_lines"),
                    "interference_load": row.get("interference_load"),
                    "gold": row["gold"],
                    "raw": raw,
                    **scored,
                }
            )
        print(f"tau={tau} done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    groups = {}
    for row in records:
        key = (row["eval_set"], row["tau"], row.get("condition"), row.get("interference_load"))
        groups.setdefault(key, []).append(row)
    cells = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        eval_set, tau, condition, load = key
        cells.append(
            {
                "eval_set": eval_set,
                "tau": tau,
                "condition": condition,
                "interference_load": load,
                "n": len(rows),
                "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "flip_to_correct": sum(
                    1
                    for r in rows
                    if r.get("baseline_correct") == 0 and r["correct"] == 1
                )
                / len(rows),
                "harm": sum(
                    1
                    for r in rows
                    if r.get("baseline_correct") == 1 and r["correct"] == 0
                )
                / len(rows),
            }
        )
    summary = {
        "arm": "T",
        "model": args.model,
        "target_heads": [
            {"layer": layer, "head": head}
            for layer, heads in heads_by_layer.items()
            for head in heads
        ],
        "taus": args.taus,
        "identity_baseline": identity_baseline,
        "n_records": len(records),
        "cells": cells,
        "identity_gate": {
            "tau": 1.0,
            "n": sum(
                1
                for row in records
                if row["tau"] == 1.0 and row["eval_set"] in {"W", "C", "S"}
            ),
            "n_mismatch": sum(
                1
                for row in records
                if row["tau"] == 1.0
                and row["eval_set"] in {"W", "C", "S"}
                and row["pred"] != row["baseline_pred"]
            ),
        },
    }
    summary["identity_gate"]["pass"] = summary["identity_gate"]["n_mismatch"] == 0
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extraction_task.jsonl")
    ap.add_argument("--extract-index", default="results/overwrite_attention_extraction/extract_index.jsonl")
    ap.add_argument("--stable-labels", default="results/overwrite_attention_extraction/stable_labels.jsonl")
    ap.add_argument("--a1-summary", default="results/overwrite_attention_extraction/a1_stable_summary.json")
    ap.add_argument("--fresh-eval", default="data/head_adapters/eval_battery.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/head_adapters/arm_t.jsonl")
    ap.add_argument("--summary", default="results/head_adapters/arm_t.summary.json")
    ap.add_argument("--taus", default="1,0.7,0.5,0.3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh-limit", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--identity-baseline",
        choices=["auto", "stage_a", "same_model"],
        default="auto",
    )
    args = ap.parse_args()
    args.taus = [float(x) for x in args.taus.split(",") if x]
    evaluate_arm_t(args)


if __name__ == "__main__":
    main()
