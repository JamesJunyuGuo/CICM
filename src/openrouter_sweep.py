"""Stage E OpenRouter cross-scale behavioral sweep.

Builds a fixed subsample, estimates the total prompt-token budget, then can run
OpenAI-compatible OpenRouter requests at temperature 0.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import subprocess
import time
from pathlib import Path

from run_eval import parse_int


DEFAULT_MODELS = [
    "qwen/qwen-2.5-14b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen2.5-72b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "openai/gpt-4o",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def cell_key(row, source):
    return (
        source,
        row.get("condition"),
        row.get("n_lines"),
        row.get("interference_load"),
    )


def deterministic_subsample(rows, per_cell, seed, source, keep_fn):
    grouped = collections.defaultdict(list)
    for row in rows:
        if keep_fn(row):
            grouped[cell_key(row, source)].append(row)

    rng = random.Random(seed)
    selected = []
    for key in sorted(grouped, key=str):
        cell_rows = list(grouped[key])
        rng.shuffle(cell_rows)
        for row in cell_rows[:per_cell]:
            out = dict(row)
            out["openrouter_source"] = source
            selected.append(out)
    return selected


def build_eval_rows(p0_rows, stage_c_rows, per_cell=50, seed=41):
    p0_selected = deterministic_subsample(
        p0_rows,
        per_cell=per_cell,
        seed=seed,
        source="p0",
        keep_fn=lambda row: row.get("condition") in {"simple", "interference"},
    )
    ood_selected = deterministic_subsample(
        stage_c_rows,
        per_cell=per_cell,
        seed=seed + 1,
        source="stage_c_ood",
        keep_fn=lambda row: row.get("condition") == "interference"
        and row.get("interference_load") in {12, 16},
    )
    return p0_selected + ood_selected


def load_tokenizer(model_name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    return tok


def estimate_budget(rows, models, tokenizer_name, max_tokens):
    tok = load_tokenizer(tokenizer_name)
    prompt_tokens = [len(tok(row["prompt"], add_special_tokens=False)["input_ids"]) for row in rows]
    total_input = sum(prompt_tokens) * len(models)
    total_output_cap = len(rows) * len(models) * max_tokens
    return {
        "tokenizer_for_estimate": tokenizer_name,
        "n_rows": len(rows),
        "n_models": len(models),
        "n_calls": len(rows) * len(models),
        "input_tokens_per_dataset_pass": int(sum(prompt_tokens)),
        "estimated_total_input_tokens": int(total_input),
        "max_total_output_tokens": int(total_output_cap),
        "max_input_tokens_single_prompt": int(max(prompt_tokens) if prompt_tokens else 0),
        "mean_input_tokens": float(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0,
    }


def classify(row, pred):
    if pred == row["gold"]:
        return "correct"
    if pred in row.get("stale_values", []):
        return "within_stale"
    if pred is None:
        return "other"
    return "other"


def summarize(results, metadata):
    groups = collections.defaultdict(lambda: collections.Counter())
    for row in results:
        key = (
            row["model"],
            row["openrouter_source"],
            row["condition"],
            row["n_lines"],
            row["interference_load"],
        )
        groups[key]["n"] += 1
        groups[key]["correct"] += int(row["correct"])
        if not row["correct"]:
            groups[key]["wrong"] += 1
            groups[key][row["answer_type"]] += 1

    cells = []
    for (model, source, condition, n_lines, load), counter in sorted(groups.items(), key=lambda item: str(item[0])):
        wrong = counter["wrong"]
        cells.append(
            {
                "model": model,
                "openrouter_source": source,
                "condition": condition,
                "n_lines": n_lines,
                "interference_load": load,
                "n": counter["n"],
                "accuracy": counter["correct"] / counter["n"],
                "within_stale_rate_among_errors": counter["within_stale"] / wrong if wrong else None,
                "other_rate_among_errors": counter["other"] / wrong if wrong else None,
            }
        )
    return {**metadata, "n": len(results), "cells": cells}


def call_one(model, row, max_tokens, retries, timeout):
    import requests

    api_key = os.environ["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Contextual-management Stage E sweep",
    }
    messages = [{"role": "user", "content": row["prompt"]}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            body = ""
            try:
                body = resp.text[:300]
            except Exception:
                pass
            last_error = f"{exc} | body: {body}"
            if attempt >= retries:
                # Record the failure as a row instead of killing the whole sweep.
                return f"__CALL_ERROR__ {last_error}"
            time.sleep(min(30.0, 2.0**attempt))
    return f"__CALL_ERROR__ unreachable retry state: {last_error}"


def run_requests(rows, models, max_tokens, concurrency, retries, timeout, out_path=None):
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    jobs = [(model, row) for model in models for row in rows]
    results = []
    out_f = None
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w")
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            future_to_job = {
                pool.submit(call_one, model, row, max_tokens, retries, timeout): (model, row)
                for model, row in jobs
            }
            for i, fut in enumerate(concurrent.futures.as_completed(future_to_job), 1):
                model, row = future_to_job[fut]
                raw = fut.result().strip()
                if raw.startswith("__CALL_ERROR__"):
                    pred = None
                    answer_type = "call_error"
                    print(f"WARN call failed: model={model} id={row['id']}: {raw[:160]}", flush=True)
                else:
                    pred = parse_int(raw)
                    answer_type = classify(row, pred)
                record = {
                    "id": row["id"],
                    "model": model,
                    "openrouter_source": row["openrouter_source"],
                    "condition": row.get("condition"),
                    "n_lines": row.get("n_lines"),
                    "interference_load": row.get("interference_load"),
                    "gold": row["gold"],
                    "pred": pred,
                    "raw": raw,
                    "correct": int(answer_type == "correct"),
                    "answer_type": answer_type,
                }
                results.append(record)
                if out_f is not None:
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                if i == 1 or i % 50 == 0 or i == len(jobs):
                    print(f"completed {i}/{len(jobs)}", flush=True)
        finally:
            if out_f is not None:
                out_f.close()
    return results


def check_openrouter_dns(timeout):
    try:
        proc = subprocess.run(
            ["getent", "hosts", "openrouter.ai"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("DNS preflight timed out for openrouter.ai") from exc
    if proc.returncode != 0:
        raise RuntimeError("DNS preflight failed for openrouter.ai")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-data", default="data/p0.jsonl")
    ap.add_argument("--stage-c-data", default="data/stage_c/eval_battery.jsonl")
    ap.add_argument("--out", default="results/stage_e/openrouter/sweep.jsonl")
    ap.add_argument("--summary", default="results/stage_e/openrouter/sweep.summary.json")
    ap.add_argument("--budget", default="results/stage_e/openrouter/budget.json")
    ap.add_argument("--subsample-out", default="results/stage_e/openrouter/subsample.jsonl")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tokenizer", default="models/llama31_8b_instruct_local")
    ap.add_argument("--max-input-tokens-total", type=int, default=30_000_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = build_eval_rows(
        load_jsonl(args.p0_data),
        load_jsonl(args.stage_c_data),
        per_cell=args.per_cell,
        seed=args.seed,
    )
    budget = estimate_budget(rows, models, args.tokenizer, args.max_tokens)
    budget.update(
        {
            "models": models,
            "p0_data": args.p0_data,
            "stage_c_data": args.stage_c_data,
            "seed": args.seed,
            "per_cell": args.per_cell,
            "openrouter_key_env": "OPENROUTER_API_KEY",
            "temperature": 0,
        }
    )
    Path(args.budget).parent.mkdir(parents=True, exist_ok=True)
    with open(args.budget, "w") as f:
        json.dump(budget, f, indent=2)
    write_jsonl(args.subsample_out, rows)
    print(json.dumps(budget, indent=2), flush=True)
    if budget["estimated_total_input_tokens"] > args.max_input_tokens_total:
        raise SystemExit(
            "budget guard tripped: "
            f"{budget['estimated_total_input_tokens']} > {args.max_input_tokens_total}"
        )
    if args.dry_run:
        print("dry run only; no OpenRouter requests sent", flush=True)
        return

    check_openrouter_dns(min(args.timeout, 15.0))
    results = run_requests(
        rows,
        models,
        args.max_tokens,
        args.concurrency,
        args.retries,
        args.timeout,
        out_path=args.out,
    )
    summary = summarize(
        results,
        {
            "models": models,
            "budget": budget,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "openrouter_key_env": "OPENROUTER_API_KEY",
        },
    )
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}", flush=True)


if __name__ == "__main__":
    main()
