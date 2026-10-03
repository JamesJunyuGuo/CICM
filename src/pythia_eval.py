"""Program-verifiable behavioral evaluation and Stage N fixed-k gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from pythia_gen import dump_jsonl, load_jsonl, make_clean_counterfactual


def dump_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_token(row: dict, token_id: int) -> str:
    if token_id == int(row["gold_token_id"]):
        return "correct_current"
    if token_id in {int(x) for x in row.get("stale_token_ids", [])}:
        return "within_stale"
    if token_id in {int(x) for x in row.get("cross_token_ids", [])}:
        return "cross_variable"
    return "other"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - half, center + half]


def load_model(model_name: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict_rows(model, tokenizer, device, rows: list[dict], batch_size: int) -> list[dict]:
    import torch

    out_rows = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        enc = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            output = model(**enc, use_cache=False)
        last = enc["attention_mask"].sum(dim=1) - 1
        logits = output.logits[torch.arange(len(batch), device=device), last]
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for local_index, (row, pred_id) in enumerate(zip(batch, pred_ids)):
            gold_id = int(row["gold_token_id"])
            candidate_ids = [gold_id] + [int(x) for x in row.get("stale_token_ids", [])]
            stale_logits = [float(logits[local_index, sid].item()) for sid in row.get("stale_token_ids", [])]
            strongest_stale_id = (
                int(row["stale_token_ids"][max(range(len(stale_logits)), key=stale_logits.__getitem__)])
                if stale_logits else None
            )
            competitor_id = strongest_stale_id if strongest_stale_id is not None else gold_id
            out_rows.append(
                {
                    **row,
                    "pred_token_id": int(pred_id),
                    "prediction": tokenizer.decode([pred_id]),
                    "label": classify_token(row, int(pred_id)),
                    "gold_logit": float(logits[local_index, gold_id].item()),
                    "strongest_stale_token_id": strongest_stale_id,
                    "strongest_stale_logit": max(stale_logits) if stale_logits else math.nan,
                    "current_minus_stale_logit": float(
                        logits[local_index, gold_id].item() - logits[local_index, competitor_id].item()
                    ),
                    "candidate_token_ids": candidate_ids,
                }
            )
        print(f"evaluated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return out_rows


def summarize_group(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    n = len(rows)
    errors = n - counts["correct_current"]
    return {
        "n": n,
        "counts": dict(counts),
        "accuracy": counts["correct_current"] / n if n else math.nan,
        "accuracy_wilson95": wilson(counts["correct_current"], n),
        "within_stale_rate_all": counts["within_stale"] / n if n else math.nan,
        "within_stale_share_errors": counts["within_stale"] / errors if errors else math.nan,
        "within_stale_share_errors_wilson95": wilson(counts["within_stale"], errors),
        "out_of_context_rate": counts["other"] / n if n else math.nan,
    }


def behavior_summary(rows: list[dict], model_name: str, dtype: str) -> dict:
    by_k = defaultdict(list)
    by_cell = defaultdict(list)
    by_seed = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(row)
        by_cell[(int(row["k"]), row["variant"], row["template"])].append(row)
        by_seed[(int(row["k"]), int(row["seed"]))].append(row)
    return {
        "stage": "N-step0-behavior",
        "model": model_name,
        "dtype": dtype,
        "decoding": "one-token greedy over full vocabulary",
        "n": len(rows),
        "by_k": {str(k): summarize_group(group) for k, group in sorted(by_k.items())},
        "by_cell": {
            f"k{k}__{variant}__{template}": summarize_group(group)
            for (k, variant, template), group in sorted(by_cell.items())
        },
        "by_seed": {
            f"k{k}__seed{seed}": summarize_group(group)
            for (k, seed), group in sorted(by_seed.items())
        },
    }


def select_fixed_k(rows: list[dict]) -> dict:
    candidates = []
    for k in sorted({int(row["k"]) for row in rows if int(row["k"]) > 0}):
        subset = [row for row in rows if int(row["k"]) == k]
        counts = Counter(row["label"] for row in subset)
        circuit_subset = [row for row in subset if row["variant"] == "single"]
        circuit_counts = Counter(row["label"] for row in circuit_subset)
        strata = defaultdict(list)
        for row in subset:
            strata[(row["variant"], row["template"], int(row["seed"]))].append(row)
        per_template = defaultdict(Counter)
        per_template_circuit = defaultdict(Counter)
        per_variant = defaultdict(Counter)
        for row in subset:
            per_template[row["template"]][row["label"]] += 1
            per_variant[row["variant"]][row["label"]] += 1
            if row["variant"] == "single":
                per_template_circuit[row["template"]][row["label"]] += 1
        errors = len(subset) - counts["correct_current"]
        gate = (
            circuit_counts["correct_current"] >= 150
            # 120 leaves a 20% validation reserve above the preregistered
            # 96-pair causal target while remaining a substantial failure pool.
            and circuit_counts["within_stale"] >= 120
            and len(per_template_circuit) == 3
            and all(
                c["correct_current"] >= 25 and c["within_stale"] >= 25
                for c in per_template_circuit.values()
            )
        )
        candidates.append(
            {
                "k": k,
                "n": len(subset),
                "counts": dict(counts),
                "circuit_single_variable_counts": dict(circuit_counts),
                "within_stale_share_errors": counts["within_stale"] / errors if errors else math.nan,
                "per_template": {key: dict(value) for key, value in per_template.items()},
                "per_template_circuit": {
                    key: dict(value) for key, value in per_template_circuit.items()
                },
                "per_variant": {key: dict(value) for key, value in per_variant.items()},
                "stratum_count": len(strata),
                "pool_gate": gate,
                "balance_objective": min(
                    circuit_counts["correct_current"], circuit_counts["within_stale"]
                ),
            }
        )
    passing = [candidate for candidate in candidates if candidate["pool_gate"]]
    chosen = max(passing, key=lambda row: (row["balance_objective"], -row["k"])) if passing else None
    return {
        "selection_rule": (
            "Among k>0, require the clean single-variable circuit substrate to have "
            "correct>=150 and within-stale>=120 (96 target causal pairs plus 20% validation reserve), "
            "with every surface template contributing "
            "correct/stale>=25; maximize min(correct, stale), breaking ties toward smaller k. "
            "The multi-variable arm is retained for taxonomy but is not mixed into the circuit pool."
        ),
        "candidates": candidates,
        "gate_pass": chosen is not None,
        "chosen_k": chosen["k"] if chosen else None,
        "chosen": chosen,
    }


def matched_pool(rows: list[dict], chosen_k: int, seed: int, per_stratum_cap: int) -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["k"]) != int(chosen_k) or row["variant"] != "single":
            continue
        key = (row["variant"], row["template"], int(row["seed"]))
        if row["label"] in {"correct_current", "within_stale"}:
            strata[key][row["label"]].append(row)
    selected = []
    for key in sorted(strata):
        correct = strata[key]["correct_current"]
        stale = strata[key]["within_stale"]
        rng.shuffle(correct)
        rng.shuffle(stale)
        n = min(len(correct), len(stale), per_stratum_cap)
        selected.extend(correct[:n])
        selected.extend(stale[:n])
    rng.shuffle(selected)
    return selected


def evaluate(args) -> None:
    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer, device = load_model(args.model, args.dtype)
    evaluated = predict_rows(model, tokenizer, device, rows, args.batch_size)
    dump_jsonl(args.out, evaluated)
    summary = behavior_summary(evaluated, args.model, args.dtype)
    gate = select_fixed_k(evaluated)
    summary["fixed_k_gate"] = gate
    dump_json(args.summary, summary)
    if gate["gate_pass"]:
        pool = matched_pool(evaluated, gate["chosen_k"], args.seed, args.per_stratum_cap)
        dump_jsonl(args.matched_pool, pool)
        summary["matched_pool"] = {
            "path": args.matched_pool,
            "n": len(pool),
            "counts": dict(Counter(row["label"] for row in pool)),
            "strata": len({(r["variant"], r["template"], r["seed"]) for r in pool}),
        }
        dump_json(args.summary, summary)
    print(json.dumps({"summary": args.summary, "gate": gate}, indent=2))


def rescore(args) -> None:
    rows = load_jsonl(args.rows)
    summary = behavior_summary(rows, args.model, args.dtype)
    gate = select_fixed_k(rows)
    summary["fixed_k_gate"] = gate
    if gate["gate_pass"]:
        pool = matched_pool(rows, gate["chosen_k"], args.seed, args.per_stratum_cap)
        dump_jsonl(args.matched_pool, pool)
        summary["matched_pool"] = {
            "path": args.matched_pool,
            "n": len(pool),
            "counts": dict(Counter(row["label"] for row in pool)),
            "strata": len({(r["variant"], r["template"], r["seed"]) for r in pool}),
        }
    dump_json(args.summary, summary)
    print(json.dumps({"summary": args.summary, "gate": gate}, indent=2))


def build_pairs(args) -> None:
    rows = [row for row in load_jsonl(args.rows) if row["label"] == "within_stale"]
    model, tokenizer, device = load_model(args.model, args.dtype)
    clean_rows = [make_clean_counterfactual(row, tokenizer) for row in rows[: args.max_candidates]]
    evaluated_clean = predict_rows(model, tokenizer, device, clean_rows, args.batch_size)
    corrupt_by_id = {row["id"]: row for row in rows}
    pairs = []
    for clean in evaluated_clean:
        pair_id = clean["pair_id"]
        corrupt = corrupt_by_id[pair_id]
        if clean["label"] != "correct_current":
            continue
        clean_ids = tokenizer(clean["prompt"], add_special_tokens=False)["input_ids"]
        corrupt_ids = tokenizer(corrupt["prompt"], add_special_tokens=False)["input_ids"]
        if len(clean_ids) != len(corrupt_ids):
            continue
        pairs.append({"id": pair_id, "clean": clean, "corrupted": corrupt})
        if len(pairs) >= args.n_pairs:
            break
    dump_jsonl(args.out, pairs)
    summary = {
        "model": args.model,
        "candidate_stale_failures": len(rows),
        "clean_evaluated": len(evaluated_clean),
        "n_pairs": len(pairs),
        "target_n_pairs": args.n_pairs,
        "pair_gate": len(pairs) >= args.min_pairs,
        "requirements": "corrupted=within_stale, clean counterfactual=correct_current, equal token length",
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


def self_test(_args) -> None:
    row = {"gold_token_id": 1, "stale_token_ids": [2, 3], "cross_token_ids": [4]}
    assert classify_token(row, 1) == "correct_current"
    assert classify_token(row, 2) == "within_stale"
    assert classify_token(row, 4) == "cross_variable"
    assert classify_token(row, 9) == "other"
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    synthetic = []
    for k in (1, 2):
        for variant in ("single", "multi"):
            for template in ("arrow", "current", "latest"):
                for seed in (11, 29, 47):
                    for i in range(60):
                        label = "correct_current" if i < 30 else "within_stale"
                        synthetic.append({"k": k, "variant": variant, "template": template, "seed": seed, "label": label})
    assert select_fixed_k(synthetic)["gate_pass"]
    print("pythia_eval self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    run = sub.add_parser("run")
    run.add_argument("--data", default="results/stage_n/behavior/tasks.jsonl")
    run.add_argument("--model", default="EleutherAI/pythia-160m")
    run.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    run.add_argument("--out", default="results/stage_n/behavior/rows.jsonl")
    run.add_argument("--summary", default="results/stage_n/behavior/summary.json")
    run.add_argument("--matched-pool", default="results/stage_n/behavior/matched_pool.jsonl")
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=20260807)
    run.add_argument("--per-stratum-cap", type=int, default=48)
    run.set_defaults(func=evaluate)
    rescore_parser = sub.add_parser("rescore")
    rescore_parser.add_argument("--rows", required=True)
    rescore_parser.add_argument("--summary", required=True)
    rescore_parser.add_argument("--matched-pool", required=True)
    rescore_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    rescore_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    rescore_parser.add_argument("--seed", type=int, default=20260807)
    rescore_parser.add_argument("--per-stratum-cap", type=int, default=48)
    rescore_parser.set_defaults(func=rescore)
    pairs = sub.add_parser("build-pairs")
    pairs.add_argument("--rows", default="results/stage_n/behavior/matched_pool.jsonl")
    pairs.add_argument("--model", default="EleutherAI/pythia-160m")
    pairs.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    pairs.add_argument("--out", default="results/stage_n/behavior/pairs.jsonl")
    pairs.add_argument("--summary", default="results/stage_n/behavior/pairs.summary.json")
    pairs.add_argument("--n-pairs", type=int, default=96)
    pairs.add_argument("--min-pairs", type=int, default=48)
    pairs.add_argument("--max-candidates", type=int, default=300)
    pairs.add_argument("--batch-size", type=int, default=128)
    pairs.set_defaults(func=build_pairs)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
