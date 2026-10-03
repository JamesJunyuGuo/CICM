"""Stage O Part B: graded output suppression of the frozen 160M head set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pythia_circuit as core
from pythia_eval import classify_token, dump_json, load_model
from pythia_gen import dump_jsonl, load_jsonl


GAMMAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
CORRECT = "correct_current"
STALE = "within_stale"


def apply_head_scale(hidden, query, heads, gamma: float, n_heads: int):
    """Scale selected pre-output head slices at each row's decision position."""
    import torch

    if hidden.shape[-1] % n_heads:
        raise ValueError("hidden size must be divisible by query-head count")
    if float(gamma) == 1.0:
        return hidden
    value = hidden.clone()
    head_dim = hidden.shape[-1] // n_heads
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    for head in heads:
        sl = slice(int(head) * head_dim, (int(head) + 1) * head_dim)
        if query is None:
            value[..., sl] *= gamma
        else:
            value[batch, query, sl] *= gamma
    return value


def scaled_logits(model, enc, query, heads: list[tuple[int, int]], gamma: float):
    by_layer = defaultdict(list)
    for layer, head in heads:
        by_layer[int(layer)].append(int(head))
    n_heads = int(model.config.num_attention_heads)
    handles = []
    for layer, selected in by_layer.items():
        def hook(_module, inputs, selected=tuple(selected)):
            return (apply_head_scale(inputs[0], query, selected, gamma, n_heads),)

        handles.append(
            model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook)
        )
    try:
        return core.run_logits(model, enc)
    finally:
        for handle in handles:
            handle.remove()


def evaluate_rows(model, tokenizer, device, rows, heads, gamma, batch_size):
    labels = []
    gaps = []
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc = core.batch_encode(tokenizer, device, batch_rows)
        query = core.query_positions(enc)
        logits = scaled_logits(model, enc, query, heads, gamma)
        batch_gaps, batch_predictions = core.gaps_from_logits(
            logits, batch_rows, query
        )
        gaps.extend(batch_gaps.tolist())
        predictions.extend(batch_predictions)
        labels.extend(
            classify_token(row, prediction)
            for row, prediction in zip(batch_rows, batch_predictions)
        )
    return {
        "labels": np.asarray(labels, dtype=object),
        "gaps": np.asarray(gaps, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.int64),
    }


def prepare_lm(args) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=args.split
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    token_ids = np.asarray(
        tokenizer.encode(text, add_special_tokens=False), dtype=np.int64
    )
    required = args.n_segments * args.segment_length
    if len(token_ids) < required:
        raise ValueError(f"corpus has {len(token_ids)} tokens; need {required}")
    rng = np.random.default_rng(args.seed)
    offset = int(rng.integers(0, len(token_ids) - required + 1))
    segments = token_ids[offset : offset + required].reshape(
        args.n_segments, args.segment_length
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, segments)
    summary = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": args.split,
        "model_tokenizer": args.model,
        "seed": args.seed,
        "corpus_tokens": int(len(token_ids)),
        "offset": offset,
        "n_segments": args.n_segments,
        "segment_length": args.segment_length,
        "selection": "one seeded offset followed by contiguous non-overlapping segments",
        "out": str(out),
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_lm(model, device, segments, heads, gamma, batch_size):
    import torch
    import torch.nn.functional as functional

    losses = []
    for start in range(0, len(segments), batch_size):
        ids = torch.as_tensor(
            segments[start : start + batch_size], device=device, dtype=torch.long
        )
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        # Every prediction site is treated as a query for this side-effect assay.
        logits = scaled_logits(model, enc, None, heads, gamma)
        token_loss = functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            reduction="none",
        )
        losses.extend(token_loss.detach().cpu().tolist())
    values = np.asarray(losses, dtype=np.float64)
    return {"mean": float(values.mean()), "n_tokens": int(len(values))}


def _paired_binary(labels, target):
    return (np.asarray(labels, dtype=object) == target).astype(np.float64)


def _random_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "quantile95": np.quantile(values, [0.025, 0.975]).tolist(),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_arm(
    failure_result,
    correct_result,
    k0_result,
    lm_result,
    baseline_failure,
    baseline_correct,
    baseline_k0,
    baseline_lm,
    n_boot,
    seed,
):
    correction = _paired_binary(failure_result["labels"], CORRECT) - _paired_binary(
        baseline_failure["labels"], CORRECT
    )
    preservation = _paired_binary(correct_result["labels"], CORRECT)
    return {
        "correction": core.paired_bootstrap(correction, n_boot, seed),
        "correct_preservation": float(preservation.mean()),
        "correct_prediction_changes": int(
            np.sum(correct_result["predictions"] != baseline_correct["predictions"])
        ),
        "k0_accuracy": float(_paired_binary(k0_result["labels"], CORRECT).mean()),
        "k0_prediction_changes": int(
            np.sum(k0_result["predictions"] != baseline_k0["predictions"])
        ),
        "lm_loss": float(lm_result["mean"]),
        "lm_delta": float(lm_result["mean"] - baseline_lm["mean"]),
        "max_abs_gap_change": float(
            max(
                np.max(np.abs(failure_result["gaps"] - baseline_failure["gaps"])),
                np.max(np.abs(correct_result["gaps"] - baseline_correct["gaps"])),
                np.max(np.abs(k0_result["gaps"] - baseline_k0["gaps"])),
            )
        ),
    }


def adjudicate_gamma(row: dict, k0_baseline: float) -> dict:
    targeted = row["targeted"]
    random = row["random"]
    correction_specific = bool(
        targeted["correction"]["mean"] > random["correction"]["quantile95"][1]
    )
    k0_pass = bool(targeted["k0_accuracy"] >= k0_baseline - 0.02)
    preservation_pass = bool(targeted["correct_preservation"] >= 0.95)
    return {
        "gamma_below_one": bool(float(row["gamma"]) < 1.0),
        "correction_exceeds_random95": correction_specific,
        "k0_drop_at_most_0.02": k0_pass,
        "correct_preservation_at_least_0.95": preservation_pass,
        "three_hard_controls_pass": bool(
            float(row["gamma"]) < 1.0
            and correction_specific
            and k0_pass
            and preservation_pass
        ),
        "lm_targeted_minus_random_mean": float(
            targeted["lm_delta"] - random["lm_delta"]["mean"]
        ),
        "lm_materiality": "Report numerically; no post-hoc threshold is imposed.",
    }


def run(args) -> None:
    frozen = json.loads(Path(args.head_summary).read_text(encoding="utf-8"))
    heads = [
        (int(row["layer"]), int(row["head"]))
        for row in frozen["stale_promoting"]["heads"]
    ]
    if len(heads) != 12:
        raise ValueError(f"expected frozen 12-head set, found {len(heads)}")
    pool = load_jsonl(args.pool)
    failures = [
        row
        for row in pool
        if row["label"] == STALE and core.stable_split(row["semantic_id"]) == "evaluation"
    ]
    correct = [
        row
        for row in pool
        if row["label"] == CORRECT
        and int(row["k"]) == args.correct_k
        and core.stable_split(row["semantic_id"]) == "evaluation"
    ]
    k0 = [
        row
        for row in load_jsonl(args.behavior_rows)
        if row["variant"] == "single" and int(row["k"]) == 0
    ]
    if not args.smoke and len(failures) != int(frozen["stale_promoting"]["evaluation_n"]):
        raise ValueError("held-out failure pool drifted from Phase 1b")
    segments = np.load(args.lm_segments)
    if args.smoke:
        failures = failures[: args.smoke_limit]
        correct = correct[: args.smoke_limit]
        k0 = k0[: args.smoke_limit]
        segments = segments[: args.smoke_lm_segments]
    model, tokenizer, device = load_model(args.model, args.dtype)
    baseline_failure = evaluate_rows(
        model, tokenizer, device, failures, [], 1.0, args.batch_size
    )
    baseline_correct = evaluate_rows(
        model, tokenizer, device, correct, [], 1.0, args.batch_size
    )
    baseline_k0 = evaluate_rows(model, tokenizer, device, k0, [], 1.0, args.batch_size)
    baseline_lm = evaluate_lm(
        model, device, segments, [], 1.0, args.lm_batch_size
    )
    if np.any(baseline_failure["labels"] != STALE):
        raise RuntimeError("held-out failure baseline no longer reproduces stale labels")
    if np.any(baseline_correct["labels"] != CORRECT):
        raise RuntimeError("held-out correct baseline no longer reproduces correct labels")
    k0_baseline = float(_paired_binary(baseline_k0["labels"], CORRECT).mean())
    random_sets = core.layer_matched_random_sets(
        heads,
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        args.n_random,
        args.seed,
    )
    curve = []
    random_records = []
    targeted_records = []
    for gamma_index, gamma in enumerate(args.gammas):
        targeted_results = (
            evaluate_rows(model, tokenizer, device, failures, heads, gamma, args.batch_size),
            evaluate_rows(model, tokenizer, device, correct, heads, gamma, args.batch_size),
            evaluate_rows(model, tokenizer, device, k0, heads, gamma, args.batch_size),
            evaluate_lm(model, device, segments, heads, gamma, args.lm_batch_size),
        )
        targeted = summarize_arm(
            *targeted_results,
            baseline_failure,
            baseline_correct,
            baseline_k0,
            baseline_lm,
            args.n_boot,
            args.seed + gamma_index * 1000,
        )
        for group, group_rows, result, baseline in (
            ("heldout_failure", failures, targeted_results[0], baseline_failure),
            ("heldout_correct_k4", correct, targeted_results[1], baseline_correct),
            ("k0_control", k0, targeted_results[2], baseline_k0),
        ):
            targeted_records.extend(
                {
                    "gamma": gamma,
                    "group": group,
                    "id": row["id"],
                    "semantic_id": row["semantic_id"],
                    "baseline_label": str(baseline["labels"][index]),
                    "intervention_label": str(result["labels"][index]),
                    "baseline_prediction": int(baseline["predictions"][index]),
                    "intervention_prediction": int(result["predictions"][index]),
                    "baseline_gap": float(baseline["gaps"][index]),
                    "intervention_gap": float(result["gaps"][index]),
                }
                for index, row in enumerate(group_rows)
            )
        random_metrics = []
        for random_index, random_heads in enumerate(random_sets):
            results = (
                evaluate_rows(
                    model, tokenizer, device, failures, random_heads, gamma, args.batch_size
                ),
                evaluate_rows(
                    model, tokenizer, device, correct, random_heads, gamma, args.batch_size
                ),
                evaluate_rows(model, tokenizer, device, k0, random_heads, gamma, args.batch_size),
                evaluate_lm(
                    model, device, segments, random_heads, gamma, args.lm_batch_size
                ),
            )
            metric = summarize_arm(
                *results,
                baseline_failure,
                baseline_correct,
                baseline_k0,
                baseline_lm,
                args.n_boot,
                args.seed + gamma_index * 1000 + random_index + 1,
            )
            random_metrics.append(metric)
            random_records.append(
                {
                    "gamma": gamma,
                    "random_index": random_index,
                    "heads": [
                        {"layer": layer, "head": head} for layer, head in random_heads
                    ],
                    **metric,
                }
            )
            if (random_index + 1) % 8 == 0:
                print(
                    f"gamma={gamma:.1f} random={random_index + 1}/{len(random_sets)}",
                    flush=True,
                )
        random_summary = {
            "correction": _random_summary(
                [row["correction"]["mean"] for row in random_metrics]
            ),
            "correct_preservation": _random_summary(
                [row["correct_preservation"] for row in random_metrics]
            ),
            "k0_accuracy": _random_summary([row["k0_accuracy"] for row in random_metrics]),
            "lm_delta": _random_summary([row["lm_delta"] for row in random_metrics]),
        }
        row = {"gamma": gamma, "targeted": targeted, "random": random_summary}
        row["gate"] = adjudicate_gamma(row, k0_baseline)
        curve.append(row)
        print(json.dumps(row, indent=2), flush=True)
    identity = next(row for row in curve if float(row["gamma"]) == 1.0)
    zero = next(row for row in curve if float(row["gamma"]) == 0.0)
    phase1_effect = float(frozen["stale_promoting"]["paired_effect"]["mean"])
    identity_gate = {
        "targeted_max_abs_gap_change": identity["targeted"]["max_abs_gap_change"],
        "targeted_lm_delta": identity["targeted"]["lm_delta"],
        "pass": bool(
            identity["targeted"]["max_abs_gap_change"] == 0.0
            and identity["targeted"]["lm_delta"] == 0.0
        ),
    }
    consistency = (
        {
            "phase1b": phase1_effect,
            "gamma0": zero["targeted"]["correction"]["mean"],
            "absolute_difference": abs(
                zero["targeted"]["correction"]["mean"] - phase1_effect
            ),
            "pass_exact": bool(
                zero["targeted"]["correction"]["mean"] == phase1_effect
            ),
        }
        if not args.smoke
        else {
            "phase1b": phase1_effect,
            "gamma0_smoke": zero["targeted"]["correction"]["mean"],
            "pass_exact": None,
            "note": "Full-pool consistency is not adjudicated on a smoke subset.",
        }
    )
    if not identity_gate["pass"]:
        raise RuntimeError(f"gamma=1 identity gate failed: {identity_gate}")
    if not args.smoke and not consistency["pass_exact"]:
        raise RuntimeError(f"gamma=0 consistency gate failed: {consistency}")
    summary = {
        "stage": "O-Part-B-smoke" if args.smoke else "O-Part-B",
        "model": args.model,
        "dtype": args.dtype,
        "gammas": args.gammas,
        "heads": [{"layer": layer, "head": head} for layer, head in heads],
        "evaluation": {
            "heldout_failures": len(failures),
            "heldout_correct_k4": len(correct),
            "k0_single": len(k0),
            "k0_baseline_accuracy": k0_baseline,
            "lm_segments": int(len(segments)),
            "lm_tokens_per_segment": int(segments.shape[1]),
            "lm_baseline_loss": baseline_lm["mean"],
        },
        "identity_gate": identity_gate,
        "gamma0_consistency": consistency,
        "curve": curve,
        "hard_control_candidates": [
            row["gamma"] for row in curve if row["gate"]["three_hard_controls_pass"]
        ],
        "claim_boundary": (
            "A 160M inference-time existence test, not a deployable or cross-scale repair. "
            "LM loss applies the query-position rule at every next-token prediction site."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "targeted_rows.jsonl", targeted_records)
    dump_jsonl(out_dir / "random_controls.jsonl", random_records)
    dump_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "REPORT.md")


def write_report(summary, path):
    consistency = summary["gamma0_consistency"]
    consistency_text = (
        f"**{'PASS' if consistency['pass_exact'] else 'FAIL'}** "
        f"({consistency['gamma0']:.3f} vs Phase-1b {consistency['phase1b']:.3f})."
        if consistency["pass_exact"] is not None
        else "not adjudicated on the smoke subset."
    )
    lines = [
        "# Stage O Part B: graded circuit correction",
        "",
        summary["claim_boundary"],
        "",
        f"Identity gate: **{'PASS' if summary['identity_gate']['pass'] else 'FAIL'}**. ",
        f"Gamma-zero consistency: {consistency_text}",
        "",
        "| gamma | correction | random 95% | preserve correct | k=0 acc | LM delta | random LM mean | hard controls |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["curve"]:
        target = row["targeted"]
        random = row["random"]
        lines.append(
            f"| {row['gamma']:.1f} | {target['correction']['mean']:.3f} | "
            f"[{random['correction']['quantile95'][0]:.3f}, {random['correction']['quantile95'][1]:.3f}] | "
            f"{target['correct_preservation']:.3f} | {target['k0_accuracy']:.3f} | "
            f"{target['lm_delta']:.4f} | {random['lm_delta']['mean']:.4f} | "
            f"{'pass' if row['gate']['three_hard_controls_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The operating curve is the result; no gamma is selected post hoc. The LM-loss "
            "difference is reported numerically without inventing a materiality threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_floats(value):
    return [float(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("prepare-lm")
    corpus.add_argument("--model", default="EleutherAI/pythia-160m")
    corpus.add_argument("--split", default="train")
    corpus.add_argument("--n-segments", type=int, default=200)
    corpus.add_argument("--segment-length", type=int, default=512)
    corpus.add_argument("--seed", type=int, default=20260818)
    corpus.add_argument("--out", required=True)
    corpus.add_argument("--summary", required=True)
    corpus.set_defaults(func=prepare_lm)
    sweep = sub.add_parser("run")
    sweep.add_argument("--model", default="EleutherAI/pythia-160m")
    sweep.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    sweep.add_argument(
        "--head-summary",
        default="results/pythia_circuit/phase1b/a_full_cpu/ablation_heldout.summary.json",
    )
    sweep.add_argument(
        "--pool", default="results/pythia_circuit/behavior_full_cpu/matched_pool.jsonl"
    )
    sweep.add_argument(
        "--behavior-rows", default="results/pythia_circuit/behavior_full_cpu/rows.jsonl"
    )
    sweep.add_argument("--lm-segments", required=True)
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--gammas", type=parse_floats, default=list(GAMMAS))
    sweep.add_argument("--correct-k", type=int, default=4)
    sweep.add_argument("--batch-size", type=int, default=128)
    sweep.add_argument("--lm-batch-size", type=int, default=8)
    sweep.add_argument("--n-random", type=int, default=64)
    sweep.add_argument("--n-boot", type=int, default=2000)
    sweep.add_argument("--seed", type=int, default=20260818)
    sweep.add_argument("--smoke", action="store_true")
    sweep.add_argument("--smoke-limit", type=int, default=4)
    sweep.add_argument("--smoke-lm-segments", type=int, default=2)
    sweep.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
