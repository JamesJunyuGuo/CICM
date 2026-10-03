"""Unified Stage C evaluation helpers and runner."""

import argparse
import collections
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_eval import parse_int

_PAIR_RE = re.compile(r"([A-Z])\s*=\s*(-?\d+)")


def load_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def score_single_int(row, raw):
    pred = parse_int(raw)
    return {"pred": pred, "correct": int(pred == row["gold"])}


def parse_pairs(text):
    return {m.group(1): int(m.group(2)) for m in _PAIR_RE.finditer(text)}


def score_parallel(row, raw):
    pred = parse_pairs(raw)
    gold = {k: int(v) for k, v in row["gold"].items()}
    per_stream = {s: int(pred.get(s) == v) for s, v in gold.items()}
    n_correct = sum(per_stream.values())
    return {
        "pred": pred,
        "n_streams": len(gold),
        "n_correct": n_correct,
        "per_stream_correct": per_stream,
        "all_correct": int(n_correct == len(gold)),
        "correct": int(n_correct == len(gold)),
    }


def task_family(row):
    if row.get("task_family"):
        return row["task_family"]
    if "K" in row and "U" in row:
        return "parallel"
    return "overwrite"


def score_row(row, raw):
    if task_family(row) == "parallel":
        return score_parallel(row, raw)
    return score_single_int(row, raw)


def build_chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(model_name, adapter_dir=None, adapter_kind=None, dtype="bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto"
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
            scored = score_row(row, raw.strip())
            result = {
                "id": row["id"],
                "task_family": task_family(row),
                "condition": row.get("condition"),
                "n_lines": row.get("n_lines"),
                "interference_load": row.get("interference_load"),
                "K": row.get("K"),
                "U": row.get("U"),
                "gold": row["gold"],
                "raw": raw.strip(),
            }
            result.update(scored)
            results.append(result)
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def summarize_results(results, metadata):
    groups = collections.defaultdict(lambda: {"n": 0, "correct": 0, "streams": 0, "stream_correct": 0})
    for row in results:
        key = (
            row["task_family"],
            row.get("condition"),
            row.get("n_lines"),
            row.get("interference_load"),
            row.get("K"),
            row.get("U"),
        )
        groups[key]["n"] += 1
        groups[key]["correct"] += row.get("correct", 0)
        if row["task_family"] == "parallel":
            groups[key]["streams"] += row["n_streams"]
            groups[key]["stream_correct"] += row["n_correct"]

    cells = []
    for key, vals in sorted(groups.items(), key=lambda item: str(item[0])):
        family, condition, n_lines, load, k, u = key
        cell = {
            "task_family": family,
            "condition": condition,
            "n_lines": n_lines,
            "interference_load": load,
            "K": k,
            "U": u,
            "n": vals["n"],
            "accuracy": vals["correct"] / vals["n"],
        }
        if family == "parallel":
            cell["per_stream_acc"] = vals["stream_correct"] / vals["streams"]
            cell["all_correct_rate"] = vals["correct"] / vals["n"]
        cells.append(cell)
    return {**metadata, "n": len(results), "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--adapter-kind", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--arm", default="baseline")
    args = ap.parse_args()

    rows = load_jsonl(args.data, args.limit)
    print(f"loaded {len(rows)} examples from {args.data}")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.adapter_dir, args.adapter_kind, dtype=args.dtype
    )
    results = evaluate_rows(model, tokenizer, rows, args.batch_size, args.max_new_tokens)
    write_jsonl(args.out, results)
    summary = summarize_results(
        results,
        {
            "arm": args.arm,
            "model": args.model,
            "data": args.data,
            "adapter_dir": args.adapter_dir,
            "adapter_kind": args.adapter_kind,
            "dtype": args.dtype,
        },
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}")


if __name__ == "__main__":
    main()
