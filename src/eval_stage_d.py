"""Stage D behavioral inference and scoring."""

import argparse
import collections
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_eval import parse_int
from stage_c_eval import build_chat_prompt, load_jsonl, write_jsonl

_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(-?\d+)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


def normalize_text(text):
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def parse_string(raw):
    tokens = _TOKEN_RE.findall(str(raw).lower())
    return tokens[0] if tokens else ""


def parse_list(raw):
    return [tok.lower() for tok in _TOKEN_RE.findall(str(raw)) if not tok.isdigit()]


def parse_dict(raw):
    return {key: int(val) for key, val in _PAIR_RE.findall(str(raw))}


def _stale_hit(row, pred):
    stale = row.get("stale_values", [])
    if row["answer_type"] == "int":
        return any(pred == int(v) for v in stale if isinstance(v, int))
    if row["answer_type"] == "string":
        return any(str(pred) == normalize_text(v) for v in stale)
    if row["answer_type"] == "list":
        pred_set = set(pred)
        return any(normalize_text(v) in pred_set for v in stale)
    if row["answer_type"] == "dict":
        return any(v in pred.values() for v in stale if isinstance(v, int))
    return False


def score_stage_d_row(row, raw):
    answer_type = row["answer_type"]
    gold = row["gold"]
    scored = {}
    if answer_type == "int":
        pred = parse_int(raw)
        correct = int(pred == int(gold))
    elif answer_type == "string":
        pred = parse_string(raw)
        correct = int(pred == normalize_text(gold))
    elif answer_type == "list":
        pred = parse_list(raw)
        gold_norm = [normalize_text(x) for x in gold]
        if row.get("stage_d_cell") == "latent_list":
            correct = int(pred == gold_norm)
        else:
            correct = int(set(pred) == set(gold_norm))
    elif answer_type == "dict":
        pred = parse_dict(raw)
        gold_norm = {str(k): int(v) for k, v in gold.items()}
        per_key = {key: int(pred.get(key) == val) for key, val in gold_norm.items()}
        scored["n_items"] = len(gold_norm)
        scored["n_correct"] = sum(per_key.values())
        scored["per_item_correct"] = per_key
        correct = int(scored["n_correct"] == len(gold_norm))
    else:
        raise ValueError(f"unknown answer_type={answer_type}")
    error_type = "correct" if correct else "stale" if _stale_hit(row, pred) else "other"
    scored.update({"pred": pred, "correct": correct, "error_type": error_type})
    return scored


def build_adapter_args(adapter_dir, adapter_kind):
    if not adapter_dir:
        return {}
    if not adapter_kind:
        raise ValueError("--adapter-kind is required when --adapter-dir is set")
    return {"adapter_dir": adapter_dir, "adapter_kind": adapter_kind}


def load_model_and_tokenizer(model_name, dtype="bfloat16", adapter_dir=None, adapter_kind=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if adapter_dir:
        if adapter_kind == "generic":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        elif adapter_kind in {"head_sliced", "random_head_sliced"}:
            from stage_c_adapters import load_head_sliced_adapter

            load_head_sliced_adapter(model, adapter_dir)
        else:
            raise ValueError(f"unknown adapter_kind={adapter_kind}")
    model.eval()
    return model, tokenizer


def evaluate_rows(model, tokenizer, rows, batch_size, max_new_tokens):
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [build_chat_prompt(tokenizer, row["prompt"]) for row in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for row, raw in zip(batch, decoded):
            raw = raw.strip()
            scored = score_stage_d_row(row, raw)
            result = {
                "id": row["id"],
                "task_family": row["task_family"],
                "stage_d_cell": row["stage_d_cell"],
                "condition": row.get("condition"),
                "interference_load": row.get("interference_load"),
                "length_variant": row.get("length_variant"),
                "answer_type": row["answer_type"],
                "gold": row["gold"],
                "raw": raw,
            }
            result.update(scored)
            results.append(result)
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def summarize_results(results, metadata):
    groups = collections.defaultdict(lambda: {"n": 0, "correct": 0, "stale": 0, "other": 0})
    for row in results:
        key = (
            row["task_family"],
            row["stage_d_cell"],
            row["answer_type"],
            row.get("condition"),
            row.get("interference_load"),
            row.get("length_variant"),
        )
        groups[key]["n"] += 1
        groups[key]["correct"] += row["correct"]
        groups[key]["stale"] += int(row["error_type"] == "stale")
        groups[key]["other"] += int(row["error_type"] == "other")
    cells = []
    for key, vals in sorted(groups.items(), key=lambda item: str(item[0])):
        family, cell, answer_type, condition, load, length_variant = key
        n = vals["n"]
        n_errors = n - vals["correct"]
        cells.append(
            {
                "task_family": family,
                "stage_d_cell": cell,
                "answer_type": answer_type,
                "condition": condition,
                "interference_load": load,
                "length_variant": length_variant,
                "n": n,
                "accuracy": vals["correct"] / n,
                "stale_rate": vals["stale"] / n,
                "other_error_rate": vals["other"] / n,
                "within_stale_rate_among_errors": vals["stale"] / n_errors
                if n_errors
                else 0.0,
            }
        )
    return {**metadata, "n": len(results), "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage_d/eval_all.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--adapter-kind", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.data, args.limit)
    print(f"loaded {len(rows)} examples from {args.data}")
    adapter_args = build_adapter_args(args.adapter_dir, args.adapter_kind)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype, **adapter_args)
    results = evaluate_rows(model, tokenizer, rows, args.batch_size, args.max_new_tokens)
    write_jsonl(args.out, results)
    summary = summarize_results(
        results,
        {
            "stage": "D",
            "model": args.model,
            "data": args.data,
            "dtype": args.dtype,
            "limit": args.limit,
            "adapter_dir": args.adapter_dir,
            "adapter_kind": args.adapter_kind,
        },
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}")


if __name__ == "__main__":
    main()
