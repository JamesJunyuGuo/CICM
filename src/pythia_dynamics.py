"""Stage N Phase 2: Pythia-160m training-time stale-binding dynamics.

The checkpoint grid, task, circuit heads, failure-pool gate, and change-point
rules are constants so the analysis cannot be tuned after inspecting curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from pythia_eval import classify_token, predict_rows, summarize_group
from pythia_gen import dump_jsonl, load_jsonl


MODEL = "EleutherAI/pythia-160m"
CHECKPOINT_STEPS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000,
    4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000,
    96000, 143000,
)
TOKENS_PER_STEP = 2_097_152
CIRCUIT_HEADS = ((8, 10), (8, 2))
TASK_SEEDS = (11, 29, 47)
TASK_TEMPLATES = ("arrow", "current", "latest")
K_VALUES = (0, 1, 2, 3, 4, 5)
QK_K = 4
STALE_MIN_ERRORS = 30
STALE_MIN_ERRORS_PER_SEED = 5
DEFAULT_SEED = 20260809
TAXONOMY = ("correct_current", "within_stale", "cross_variable", "other")


def dump_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[str, int]:
    revision = value if value.startswith("step") else f"step{value}"
    try:
        step = int(revision[4:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}") from exc
    if step not in CHECKPOINT_STEPS:
        raise argparse.ArgumentTypeError(f"checkpoint not preregistered: {revision}")
    return revision, step


def checkpoint_metadata(revision: str, step: int) -> dict:
    return {
        "revision": revision,
        "step": step,
        "tokens_seen": step * TOKENS_PER_STEP,
        "tokens_per_step": TOKENS_PER_STEP,
    }


def filter_and_validate_task(rows: Iterable[dict]) -> list[dict]:
    selected = [row for row in rows if row.get("variant") == "single"]
    expected_n = len(TASK_SEEDS) * len(TASK_TEMPLATES) * len(K_VALUES) * 72
    if len(selected) != expected_n:
        raise ValueError(f"expected {expected_n} frozen single-variable rows, got {len(selected)}")
    if {int(row["seed"]) for row in selected} != set(TASK_SEEDS):
        raise ValueError("task seeds differ from Phase 1")
    if {row["template"] for row in selected} != set(TASK_TEMPLATES):
        raise ValueError("task templates differ from Phase 1")
    if {int(row["k"]) for row in selected} != set(K_VALUES):
        raise ValueError("task k grid differs from Phase 1")
    counts = Counter((int(r["seed"]), r["template"], int(r["k"])) for r in selected)
    if set(counts.values()) != {72}:
        raise ValueError(f"unbalanced frozen task cells: {sorted(set(counts.values()))}")
    return selected


def prepare(args) -> None:
    source = Path(args.source)
    rows = filter_and_validate_task(load_jsonl(source))
    dump_jsonl(args.out, rows)
    qk_rows = [row for row in rows if int(row["k"]) == QK_K]
    dump_jsonl(args.qk_probe, qk_rows)
    manifest = {
        "stage": "N-phase2-preregistered-inputs",
        "model": MODEL,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_revisions": [f"step{step}" for step in CHECKPOINT_STEPS],
        "tokens_per_step": TOKENS_PER_STEP,
        "task": {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "filtered_path": str(args.out),
            "filtered_sha256": file_sha256(args.out),
            "variant": "single",
            "n": len(rows),
            "seeds": list(TASK_SEEDS),
            "templates": list(TASK_TEMPLATES),
            "k_values": list(K_VALUES),
            "n_per_seed_template_k": 72,
        },
        "qk_probe": {
            "path": str(args.qk_probe),
            "sha256": file_sha256(args.qk_probe),
            "selection": f"all frozen single-variable rows at k={QK_K}",
            "n": len(qk_rows),
            "heads": [{"layer": layer, "head": head} for layer, head in CIRCUIT_HEADS],
        },
        "induction_probe": {
            "definition": (
                "repeat contiguous token windows sampled from the frozen task prompts; "
                "score second-copy token i attention to first-copy token i+1 minus "
                "mean attention to non-target first-copy positions"
            ),
            "n_sequences_per_seed": args.n_induction_per_seed,
            "sequence_length_per_copy": args.induction_length,
            "seeds": list(TASK_SEEDS),
            "global_score": "mean of the five highest per-head induction-minus-baseline means",
        },
        "transition_rules": {
            "estimator": "right endpoint of the largest positive adjacent-checkpoint level change",
            "bootstrap": "hierarchical resampling of seed, then items within seed; 2000 replicates",
            "stale_pool_gate": {
                "scope": "all k>=1 single-variable rows",
                "minimum_total_errors": STALE_MIN_ERRORS,
                "minimum_errors_each_seed": STALE_MIN_ERRORS_PER_SEED,
            },
            "immediately_follows": "same or next checkpoint in the preregistered ordered grid",
            "abrupt_transition_reliability": (
                "bootstrap detection fraction >=0.95 and >=0.50 of detected bootstrap "
                "steps within one checkpoint index of the point estimate"
            ),
            "taxonomy_requirement": "within-stale share rises and other-error share falls across S_stale",
        },
    }
    dump_json(args.manifest, manifest)
    print(json.dumps({"manifest": args.manifest, "task_n": len(rows), "qk_n": len(qk_rows)}, indent=2))


def load_checkpoint(model_name: str, revision: str, dtype: str, online: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    local_only = not online
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=local_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        local_files_only=local_only,
        attn_implementation="eager",
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def behavior_checkpoint(args) -> None:
    revision, step = parse_checkpoint(args.checkpoint)
    rows = filter_and_validate_task(load_jsonl(args.task))
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer, device = load_checkpoint(args.model, revision, args.dtype, args.online)
    evaluated = predict_rows(model, tokenizer, device, rows, args.batch_size)
    out_dir = Path(args.out_root) / revision
    dump_jsonl(out_dir / "behavior_rows.jsonl", evaluated)
    by_k = {}
    by_seed = {}
    by_k_seed = {}
    for k in K_VALUES:
        subset = [r for r in evaluated if int(r["k"]) == k]
        by_k[str(k)] = summarize_group(subset)
        for seed in TASK_SEEDS:
            cell = [r for r in subset if int(r["seed"]) == seed]
            by_k_seed[f"k{k}__seed{seed}"] = summarize_group(cell)
    for seed in TASK_SEEDS:
        by_seed[str(seed)] = summarize_group([r for r in evaluated if int(r["seed"]) == seed])
    errors = [r for r in evaluated if int(r["k"]) >= 1 and r["label"] != "correct_current"]
    error_counts_by_seed = {
        str(seed): sum(int(r["seed"]) == seed for r in errors) for seed in TASK_SEEDS
    }
    stale_gate = len(errors) >= STALE_MIN_ERRORS and all(
        count >= STALE_MIN_ERRORS_PER_SEED for count in error_counts_by_seed.values()
    )
    k4 = [r for r in evaluated if int(r["k"]) == QK_K]
    summary = {
        "stage": "N-phase2-behavior-checkpoint",
        "model": args.model,
        **checkpoint_metadata(revision, step),
        "dtype": args.dtype,
        "device": str(device),
        "decoding": "one-token greedy over full vocabulary",
        "task_path": str(args.task),
        "task_sha256": file_sha256(args.task),
        "n": len(evaluated),
        "by_k": by_k,
        "by_seed": by_seed,
        "by_k_seed": by_k_seed,
        "stale_onset_pool": {
            "scope": "k>=1 errors",
            "n_errors": len(errors),
            "errors_by_seed": error_counts_by_seed,
            "counts": dict(Counter(r["label"] for r in errors)),
            "within_stale_share": (
                sum(r["label"] == "within_stale" for r in errors) / len(errors)
                if errors else math.nan
            ),
            "cross_variable_share": (
                sum(r["label"] == "cross_variable" for r in errors) / len(errors)
                if errors else math.nan
            ),
            "other_share": (
                sum(r["label"] == "other" for r in errors) / len(errors)
                if errors else math.nan
            ),
            "gate_pass": stale_gate,
        },
        "k4": summarize_group(k4),
        "taxonomy_all": dict(Counter(r["label"] for r in evaluated)),
    }
    dump_json(out_dir / "behavior.summary.json", summary)
    print(json.dumps({"summary": str(out_dir / 'behavior.summary.json'), "stale_gate": stale_gate}, indent=2))


def make_induction_sequences(tokenizer, task_rows: list[dict], n_per_seed: int, length: int) -> list[dict]:
    by_seed = defaultdict(list)
    for row in task_rows:
        by_seed[int(row["seed"])].append(row)
    sequences = []
    for seed in TASK_SEEDS:
        rng = random.Random(DEFAULT_SEED + seed)
        candidates = []
        for row in by_seed[seed]:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            if len(ids) >= length:
                candidates.append((row["id"], ids))
        for index in range(n_per_seed):
            source_id, ids = candidates[rng.randrange(len(candidates))]
            start = rng.randrange(len(ids) - length + 1)
            first = ids[start : start + length]
            sequences.append({
                "id": f"ind_s{seed}_{index:04d}",
                "seed": seed,
                "source_id": source_id,
                "source_start": start,
                "token_ids": first,
            })
    return sequences


def induction_checkpoint(
    model, tokenizer, device, task_rows, n_per_seed: int, length: int, batch_size: int
) -> tuple[list[dict], dict]:
    import torch

    sequences = make_induction_sequences(tokenizer, task_rows, n_per_seed, length)
    raw = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        ids = torch.as_tensor(
            [sequence["token_ids"] + sequence["token_ids"] for sequence in batch], device=device
        )
        with torch.no_grad():
            output = model(input_ids=ids, use_cache=False, output_attentions=True, return_dict=True)
        for layer, attention in enumerate(output.attentions):
            weights = attention.detach().float().cpu().numpy()
            targets = []
            baselines = []
            prefixes = []
            for offset in range(length - 1):
                query = length + offset
                target = offset + 1
                non_targets = [position for position in range(length) if position != target]
                targets.append(weights[:, :, query, target])
                baselines.append(weights[:, :, query, non_targets].mean(axis=2))
                prefixes.append(weights[:, :, query, offset])
            target_mean = np.mean(targets, axis=0)
            baseline_mean = np.mean(baselines, axis=0)
            prefix_mean = np.mean(prefixes, axis=0)
            for local, sequence in enumerate(batch):
                for head in range(weights.shape[1]):
                    raw.append({
                        "id": sequence["id"],
                        "seed": sequence["seed"],
                        "layer": layer,
                        "head": head,
                        "induction_score": float(target_mean[local, head]),
                        "baseline": float(baseline_mean[local, head]),
                        "induction_minus_baseline": float(
                            target_mean[local, head] - baseline_mean[local, head]
                        ),
                        "prefix_score": float(prefix_mean[local, head]),
                    })
        print(f"induction {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)
    head_rows = []
    for layer in range(int(model.config.num_hidden_layers)):
        for head in range(int(model.config.num_attention_heads)):
            subset = [r for r in raw if r["layer"] == layer and r["head"] == head]
            values = np.asarray([r["induction_minus_baseline"] for r in subset])
            head_rows.append({
                "layer": layer,
                "head": head,
                "n": len(subset),
                "induction_minus_baseline_mean": float(values.mean()),
                "induction_score_mean": float(np.mean([r["induction_score"] for r in subset])),
                "per_seed": {
                    str(seed): float(np.mean([r["induction_minus_baseline"] for r in subset if r["seed"] == seed]))
                    for seed in TASK_SEEDS
                },
            })
    top = sorted(head_rows, key=lambda row: row["induction_minus_baseline_mean"], reverse=True)[:5]
    global_score = float(np.mean([row["induction_minus_baseline_mean"] for row in top]))
    return raw, {
        "probe": "repeated contiguous windows from frozen task prompts",
        "n_sequences": len(sequences),
        "n_sequences_per_seed": n_per_seed,
        "sequence_length_per_copy": length,
        "definition": "second-copy i to first-copy i+1 attention minus non-target first-copy mean",
        "global_definition": "mean of five highest per-head means at this checkpoint",
        "global_score": global_score,
        "top5_heads": top,
        "circuit_heads": [
            next(row for row in head_rows if (row["layer"], row["head"]) == circuit)
            for circuit in CIRCUIT_HEADS
        ],
        "all_heads": head_rows,
    }


def qk_checkpoint(model, tokenizer, device, rows: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    import torch
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    raw = []
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        enc = tokenizer(
            [row["prompt"] for row in batch], return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            output = model(
                **enc, use_cache=False, output_hidden_states=True,
                output_attentions=True, return_dict=True,
            )
        query_positions = enc["attention_mask"].sum(dim=1) - 1
        predictions = output.logits[
            torch.arange(len(batch), device=device), query_positions
        ].argmax(dim=-1).detach().cpu().tolist()
        for layer, head in CIRCUIT_HEADS:
            block = model.gpt_neox.layers[layer]
            hidden = output.hidden_states[layer]
            normalized = block.input_layernorm(hidden)
            qkv = block.attention.query_key_value(normalized).view(
                len(batch), normalized.shape[1], model.config.num_attention_heads, 3 * head_dim
            ).transpose(1, 2)
            query_states, key_states, _ = qkv.chunk(3, dim=-1)
            position_ids = torch.arange(normalized.shape[1], device=device).unsqueeze(0).expand(len(batch), -1)
            cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            attention = output.attentions[layer]
            for local, row in enumerate(batch):
                query = int(query_positions[local].item())
                q = query_states[local, head, query]
                scores = torch.matmul(key_states[local, head], q) / math.sqrt(head_dim)
                current = int(row["current_value_span"][0])
                stale = [int(position) for position in row["stale_value_spans"]]
                current_score = float(scores[current].item())
                stale_scores = [float(scores[position].item()) for position in stale]
                current_attention = float(attention[local, head, query, current].item())
                stale_attention = float(attention[local, head, query, stale].sum().item())
                raw.append({
                    "id": row["id"],
                    "seed": int(row["seed"]),
                    "template": row["template"],
                    "k": int(row["k"]),
                    "layer": layer,
                    "head": head,
                    "label": classify_token(row, int(predictions[local])),
                    "qk_current": current_score,
                    "qk_stale_max": max(stale_scores),
                    "qk_latest_margin": current_score - max(stale_scores),
                    "attention_current": current_attention,
                    "attention_stale_sum": stale_attention,
                    "attention_latest_margin": current_attention - stale_attention,
                })
        print(f"QK {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    heads = []
    for layer, head in CIRCUIT_HEADS:
        subset = [r for r in raw if r["layer"] == layer and r["head"] == head]
        by_label = {}
        for label in TAXONOMY:
            label_rows = [r for r in subset if r["label"] == label]
            if label_rows:
                by_label[label] = {
                    "n": len(label_rows),
                    "qk_latest_margin_mean": float(np.mean([r["qk_latest_margin"] for r in label_rows])),
                    "attention_current_mean": float(np.mean([r["attention_current"] for r in label_rows])),
                    "attention_stale_sum_mean": float(np.mean([r["attention_stale_sum"] for r in label_rows])),
                    "attention_latest_margin_mean": float(np.mean([r["attention_latest_margin"] for r in label_rows])),
                }
        heads.append({
            "layer": layer,
            "head": head,
            "n": len(subset),
            "qk_latest_margin_mean": float(np.mean([r["qk_latest_margin"] for r in subset])),
            "attention_latest_margin_mean": float(np.mean([r["attention_latest_margin"] for r in subset])),
            "by_label": by_label,
        })
    return raw, {
        "probe": f"all frozen single-variable k={QK_K} rows",
        "n_rows": len(rows),
        "heads": heads,
        "qk_definition": "answer-position rotated query dot write-position rotated key / sqrt(head_dim)",
        "attention_definition": "query attention to current write minus summed attention to stale writes",
    }


def mechanism_checkpoint(args) -> None:
    revision, step = parse_checkpoint(args.checkpoint)
    task_rows = filter_and_validate_task(load_jsonl(args.task))
    qk_rows = load_jsonl(args.qk_probe)
    if len(qk_rows) != len(TASK_SEEDS) * len(TASK_TEMPLATES) * 72:
        raise ValueError("fixed QK probe is incomplete")
    if args.qk_limit:
        qk_rows = qk_rows[: args.qk_limit]
    model, tokenizer, device = load_checkpoint(args.model, revision, args.dtype, args.online)
    induction_rows, induction_summary = induction_checkpoint(
        model, tokenizer, device, task_rows, args.n_induction_per_seed,
        args.induction_length, args.batch_size,
    )
    qk_rows_out, qk_summary = qk_checkpoint(model, tokenizer, device, qk_rows, args.batch_size)
    out_dir = Path(args.out_root) / revision
    dump_jsonl(out_dir / "induction_rows.jsonl", induction_rows)
    dump_jsonl(out_dir / "qk_rows.jsonl", qk_rows_out)
    summary = {
        "stage": "N-phase2-mechanism-checkpoint",
        "model": args.model,
        **checkpoint_metadata(revision, step),
        "dtype": args.dtype,
        "device": str(device),
        "task_sha256": file_sha256(args.task),
        "qk_probe_sha256": file_sha256(args.qk_probe),
        "induction": induction_summary,
        "qk_attention": qk_summary,
    }
    dump_json(out_dir / "mechanism.summary.json", summary)
    print(json.dumps({"summary": str(out_dir / 'mechanism.summary.json')}, indent=2))


def behavior_bootstrap(rows: list[dict], n_boot: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Hierarchical seed-then-item bootstrap for all behavioral curves."""
    per_seed = {seed: [row for row in rows if int(row["seed"]) == seed] for seed in TASK_SEEDS}
    seed_draws = rng.integers(0, len(TASK_SEEDS), size=(n_boot, len(TASK_SEEDS)))
    seed_outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    seed_error_counts = []
    for seed in TASK_SEEDS:
        group = per_seed[seed]
        for k in K_VALUES:
            subset = [row for row in group if int(row["k"]) == k]
            correct = np.asarray([row["label"] == "correct_current" for row in subset], dtype=np.int16)
            indices = rng.integers(0, len(correct), size=(n_boot, len(correct)))
            seed_outputs[f"accuracy_k{k}"].append(correct[indices].mean(axis=1))
        overwrite = [row for row in group if int(row["k"]) >= 1]
        labels = np.asarray([row["label"] for row in overwrite], dtype=object)
        indices = rng.integers(0, len(labels), size=(n_boot, len(labels)))
        sampled = labels[indices]
        errors = sampled != "correct_current"
        error_counts = errors.sum(axis=1)
        seed_error_counts.append(error_counts)
        stale_counts = ((sampled == "within_stale") & errors).sum(axis=1)
        cross_counts = ((sampled == "cross_variable") & errors).sum(axis=1)
        other_counts = ((sampled == "other") & errors).sum(axis=1)
        seed_outputs["error_count"].append(error_counts)
        seed_outputs["stale_count"].append(stale_counts)
        seed_outputs["cross_count"].append(cross_counts)
        seed_outputs["other_count"].append(other_counts)

    result = {}
    row_index = np.arange(n_boot)[:, None]
    for k in K_VALUES:
        matrix = np.stack(seed_outputs[f"accuracy_k{k}"], axis=1)
        result[f"accuracy_k{k}"] = matrix[row_index, seed_draws].mean(axis=1)
    selected_counts = {}
    for key in ("error_count", "stale_count", "cross_count", "other_count"):
        matrix = np.stack(seed_outputs[key], axis=1)
        selected_counts[key] = matrix[row_index, seed_draws]
    total_errors = selected_counts["error_count"].sum(axis=1)
    denominator = np.maximum(total_errors, 1)
    result["stale_share"] = selected_counts["stale_count"].sum(axis=1) / denominator
    result["cross_share"] = selected_counts["cross_count"].sum(axis=1) / denominator
    result["other_share"] = selected_counts["other_count"].sum(axis=1) / denominator
    result["stale_eligible"] = (
        (total_errors >= STALE_MIN_ERRORS)
        & np.all(selected_counts["error_count"] >= STALE_MIN_ERRORS_PER_SEED, axis=1)
    )
    result["bind"] = result["accuracy_k0"]
    return result


def induction_bootstrap(rows: list[dict], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample seed and whole repeated sequences, preserving all heads per sequence."""
    head_keys = sorted({(int(row["layer"]), int(row["head"])) for row in rows})
    seed_matrices = []
    for seed in TASK_SEEDS:
        subset = [row for row in rows if int(row["seed"]) == seed]
        ids = sorted({row["id"] for row in subset})
        lookup = {(row["id"], int(row["layer"]), int(row["head"])): row for row in subset}
        matrix = np.asarray([
            [lookup[(identifier, layer, head)]["induction_minus_baseline"] for layer, head in head_keys]
            for identifier in ids
        ], dtype=np.float64)
        indices = rng.integers(0, len(matrix), size=(n_boot, len(matrix)))
        seed_matrices.append(matrix[indices].mean(axis=1))
    per_seed_means = np.stack(seed_matrices, axis=1)
    seed_draws = rng.integers(0, len(TASK_SEEDS), size=(n_boot, len(TASK_SEEDS)))
    selected = per_seed_means[np.arange(n_boot)[:, None], seed_draws].mean(axis=1)
    return np.sort(selected, axis=1)[:, -5:].mean(axis=1)


def scalar_bootstrap(rows: list[dict], key: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    per_seed = []
    for seed in TASK_SEEDS:
        values = np.asarray([float(row[key]) for row in rows if int(row["seed"]) == seed])
        indices = rng.integers(0, len(values), size=(n_boot, len(values)))
        per_seed.append(values[indices].mean(axis=1))
    means = np.stack(per_seed, axis=1)
    seed_draws = rng.integers(0, len(TASK_SEEDS), size=(n_boot, len(TASK_SEEDS)))
    return means[np.arange(n_boot)[:, None], seed_draws].mean(axis=1)


def largest_positive_jump(steps: list[int], values: list[float], eligible: list[bool] | None = None) -> dict:
    if len(steps) != len(values) or len(steps) < 2:
        raise ValueError("change-point arrays must have equal length >=2")
    eligible = eligible or [True] * len(steps)
    candidates = []
    for index in range(1, len(steps)):
        if not (eligible[index - 1] and eligible[index]):
            continue
        left, right = values[index - 1], values[index]
        if not (math.isfinite(left) and math.isfinite(right)):
            continue
        candidates.append((right - left, index))
    if not candidates:
        return {"detected": False, "step": None, "left_step": None, "jump": math.nan}
    jump, index = max(candidates, key=lambda item: (item[0], -item[1]))
    return {
        "detected": bool(jump > 0),
        "step": steps[index] if jump > 0 else None,
        "left_step": steps[index - 1] if jump > 0 else None,
        "jump": float(jump),
    }


def transition_ci(samples: list[int | None]) -> list[int] | None:
    valid = np.asarray([sample for sample in samples if sample is not None], dtype=int)
    if not len(valid):
        return None
    return [int(x) for x in np.quantile(valid, [0.025, 0.975], method="nearest")]


def behavior_metric(rows: list[dict], metric: str) -> tuple[float, bool]:
    if metric == "bind":
        subset = [row for row in rows if int(row["k"]) == 0]
        return float(np.mean([row["label"] == "correct_current" for row in subset])), True
    errors = [row for row in rows if int(row["k"]) >= 1 and row["label"] != "correct_current"]
    per_seed = Counter(int(row["seed"]) for row in errors)
    eligible = len(errors) >= STALE_MIN_ERRORS and all(per_seed[seed] >= STALE_MIN_ERRORS_PER_SEED for seed in TASK_SEEDS)
    if not errors:
        return math.nan, False
    return float(np.mean([row["label"] == "within_stale" for row in errors])), eligible


def induction_metric(rows: list[dict]) -> float:
    head_values = defaultdict(list)
    for row in rows:
        head_values[(int(row["layer"]), int(row["head"]))].append(float(row["induction_minus_baseline"]))
    means = sorted((float(np.mean(values)) for values in head_values.values()), reverse=True)
    return float(np.mean(means[:5]))


def load_checkpoint_artifacts(root: Path, filename: str) -> dict[int, list[dict]]:
    found = {}
    for step in CHECKPOINT_STEPS:
        path = root / f"step{step}" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        found[step] = load_jsonl(path)
    return found


def write_report(root: Path, summary: dict, curves: list[dict]) -> None:
    transitions = summary["transitions"]
    alignment = summary["alignment"]
    reading = alignment["reading"]
    if reading == "core_supported":
        verdict = (
            "**SUPPORTED on this preregistered Pythia-160m trajectory.** The stale-error "
            "transition coincides with or immediately follows the induction transition, and "
            "the error taxonomy shifts from non-stale errors toward within-variable stale values."
        )
    elif reading == "null_temporal_misalignment":
        verdict = (
            "**NOT SUPPORTED (temporal null).** The induction and stale-error transitions are "
            "not both reliably detected at the same or immediately adjacent checkpoint."
        )
    else:
        verdict = (
            "**NOT SUPPORTED (taxonomy gate failed).** Temporal proximity alone is insufficient: "
            "the preregistered non-stale-to-stale taxonomy shift is absent."
        )

    def transition_row(name: str) -> str:
        row = transitions[name]
        ci = row["bootstrap_ci"]
        ci_text = f"[{ci[0]:,}, {ci[1]:,}]" if ci else "not detected"
        step = f"{row['step']:,}" if row["step"] is not None else "not detected"
        return (
            f"| `{name}` | {step} | {ci_text} | {row['jump']:.4f} | "
            f"{row['bootstrap_local_fraction']:.3f} | {str(row['abrupt_reliable']).lower()} |"
        )

    bind_step = transitions["S_bind"]["step"]
    stale_step = transitions["S_stale"]["step"]
    if bind_step is None or stale_step is None:
        ordering = "At least one behavioral transition was not reliably locatable."
    elif bind_step == stale_step:
        ordering = "Basic binding and stale-binding onset share the same point estimate."
    elif bind_step < stale_step:
        ordering = "Basic binding appears before stale-binding onset."
    else:
        ordering = "The stale-error composition changes before the sharpest k=0 binding gain."

    taxonomy = alignment.get("taxonomy_shift")
    taxonomy_text = "No eligible adjacent taxonomy comparison was available."
    if taxonomy:
        taxonomy_text = (
            f"Across step {taxonomy['left_step']:,} to {taxonomy['right_step']:,}, the "
            f"within-stale error share changes by {taxonomy['within_stale_change']:+.3f} and "
            f"the `other` share by {taxonomy['other_change']:+.3f}."
        )
    all_pools = sum(bool(row["stale_pool_gate"]) for row in curves)
    checkpoint_text = ", ".join(str(step) for step in CHECKPOINT_STEPS)
    qk_final = [
        row for row in summary["qk_attention_trajectories"] if row["step"] == CHECKPOINT_STEPS[-1]
    ]
    qk_lines = []
    for row in qk_final:
        qk_lines.append(
            f"- L{row['layer']}H{row['head']}: final mean QK latest margin "
            f"{row['qk_latest_margin_mean']:.3f}; attention latest margin "
            f"{row['attention_latest_margin_mean']:.3f}."
        )
    report = f"""# Stage N Phase 2 Report: Training Dynamics of Stale Binding

## Pre-registered question and answer

Does stale-binding emerge at the induction-head phase transition?

{verdict}

This is a one-model, single-variable developmental result. It does not establish a
cross-model training law.

## Transition estimates

The point estimator is the right endpoint of the largest positive adjacent-checkpoint
level change. Intervals use 2,000 hierarchical seed-then-item/sequence bootstrap
replicates. A transition is called abrupt/reliable only when at least 95% of bootstrap
replicates detect a positive jump and at least 50% place it within one grid index of the
point estimate.

| transition | step | bootstrap 95% interval | jump | local bootstrap mass | reliable |
|---|---:|---:|---:|---:|:---:|
{transition_row('S_ind')}
{transition_row('S_bind')}
{transition_row('S_stale')}

`S_ind` is the global in-distribution induction score, `S_bind` is k=0 accuracy,
and `S_stale` is the within-stale share among all k>=1 errors after applying the
failure-pool gate. “Immediately follows” was fixed as the next checkpoint in the
ordered preregistered grid.

## Alignment and ordering

- Temporal gate: **{str(alignment['temporal_gate']).upper()}**.
- Taxonomy gate: **{str(alignment['taxonomy_gate']).upper()}**.
- Core gate: **{str(alignment['core_supported']).upper()}**.
- {ordering}
- {taxonomy_text}

## Behavioral and mechanism checks

- All {all_pools}/25 checkpoint-level stale-onset pools pass the preregistered minimum
  of 30 total k>=1 errors and at least 5 errors per seed.
- The behavior substrate is unchanged from Phase 1: single-variable arm, templates
  `arrow/current/latest`, seeds 11/29/47, digit values, and k=0..5, with 3,888 rows
  per checkpoint.
- The induction probe repeats contiguous 24-token windows sampled from those frozen
  prompts; it does not use the invalid uniform-token pilot.
- QK/attention trajectories use all 648 frozen k=4 prompts and only the preregistered
  deciding heads L8H10 and L8H2.
{chr(10).join(qk_lines)}

## Artifacts and scope

![Training dynamics](figures/N7_training_dynamics.png)

Exact initial checkpoint set: {checkpoint_text}.
Each step corresponds to 2,097,152 training tokens. No post-result task tuning was
performed. The initial grid was not automatically densified; any densification around
an uncertain transition is a review-gated follow-up and must preserve the same task and
estimators.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")


def analyze(args) -> None:
    import matplotlib.pyplot as plt

    root = Path(args.out_root)
    behavior = load_checkpoint_artifacts(root, "behavior_rows.jsonl")
    induction = load_checkpoint_artifacts(root, "induction_rows.jsonl")
    steps = list(CHECKPOINT_STEPS)
    bind_values, stale_values, stale_eligible, induction_values = [], [], [], []
    behavior_boot = []
    induction_boot = []
    dose = {k: [] for k in K_VALUES}
    k4_stale_share = []
    taxonomy = {label: [] for label in TAXONOMY if label != "correct_current"}
    rng = np.random.default_rng(args.seed)
    for step in steps:
        rows = behavior[step]
        bind, _ = behavior_metric(rows, "bind")
        stale, eligible = behavior_metric(rows, "stale")
        bind_values.append(bind)
        stale_values.append(stale)
        stale_eligible.append(eligible)
        induction_values.append(induction_metric(induction[step]))
        behavior_boot.append(behavior_bootstrap(rows, args.n_boot, rng))
        induction_boot.append(induction_bootstrap(induction[step], args.n_boot, rng))
        for k in K_VALUES:
            subset = [row for row in rows if int(row["k"]) == k]
            dose[k].append(float(np.mean([row["label"] == "correct_current" for row in subset])))
            if k == QK_K:
                k4_errors = [row for row in subset if row["label"] != "correct_current"]
                k4_stale_share.append(
                    float(np.mean([row["label"] == "within_stale" for row in k4_errors]))
                    if k4_errors else math.nan
                )
        errors = [row for row in rows if int(row["k"]) >= 1 and row["label"] != "correct_current"]
        for label in taxonomy:
            taxonomy[label].append(float(np.mean([row["label"] == label for row in errors])) if errors else math.nan)

    point = {
        "S_ind": largest_positive_jump(steps, induction_values),
        "S_bind": largest_positive_jump(steps, bind_values),
        "S_stale": largest_positive_jump(steps, stale_values, stale_eligible),
    }
    boot_steps = {key: [] for key in point}
    bind_boot_matrix = np.stack([item["bind"] for item in behavior_boot], axis=1)
    stale_boot_matrix = np.stack([item["stale_share"] for item in behavior_boot], axis=1)
    stale_ok_matrix = np.stack([item["stale_eligible"] for item in behavior_boot], axis=1)
    induction_boot_matrix = np.stack(induction_boot, axis=1)
    for bootstrap_index in range(args.n_boot):
        boot_steps["S_bind"].append(largest_positive_jump(steps, bind_boot_matrix[bootstrap_index].tolist())["step"])
        boot_steps["S_stale"].append(largest_positive_jump(
            steps, stale_boot_matrix[bootstrap_index].tolist(), stale_ok_matrix[bootstrap_index].tolist()
        )["step"])
        boot_steps["S_ind"].append(largest_positive_jump(steps, induction_boot_matrix[bootstrap_index].tolist())["step"])
    transitions = {}
    for key in point:
        valid = [value for value in boot_steps[key] if value is not None]
        point_index = steps.index(point[key]["step"]) if point[key]["step"] in steps else None
        local_fraction = (
            float(np.mean([abs(steps.index(value) - point_index) <= 1 for value in valid]))
            if valid and point_index is not None else 0.0
        )
        detected_fraction = len(valid) / args.n_boot
        transitions[key] = {
            **point[key],
            "bootstrap_ci": transition_ci(valid),
            "bootstrap_detected_fraction": detected_fraction,
            "bootstrap_mode": Counter(valid).most_common(1)[0][0] if valid else None,
            "bootstrap_local_fraction": local_fraction,
            "abrupt_reliable": bool(detected_fraction >= 0.95 and local_fraction >= 0.50),
        }

    ind_step = transitions["S_ind"]["step"]
    stale_step = transitions["S_stale"]["step"]
    ind_index = steps.index(ind_step) if ind_step in steps else None
    stale_index = steps.index(stale_step) if stale_step in steps else None
    temporal_gate = bool(
        transitions["S_ind"]["abrupt_reliable"]
        and transitions["S_stale"]["abrupt_reliable"]
        and ind_index is not None
        and stale_index in {ind_index, ind_index + 1}
    )
    taxonomy_gate = False
    taxonomy_shift = None
    if stale_index is not None and stale_index > 0:
        taxonomy_shift = {
            "left_step": steps[stale_index - 1],
            "right_step": steps[stale_index],
            "within_stale_change": taxonomy["within_stale"][stale_index] - taxonomy["within_stale"][stale_index - 1],
            "other_change": taxonomy["other"][stale_index] - taxonomy["other"][stale_index - 1],
        }
        taxonomy_gate = taxonomy_shift["within_stale_change"] > 0 and taxonomy_shift["other_change"] < 0
    core_supported = bool(temporal_gate and taxonomy_gate)

    qk_summaries = []
    qk_bootstrap = defaultdict(dict)
    mechanism_summaries = []
    for step in steps:
        summary_path = root / f"step{step}" / "mechanism.summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        mechanism_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        qk_raw = load_jsonl(root / f"step{step}" / "qk_rows.jsonl")
        for head in mechanism_summaries[-1]["qk_attention"]["heads"]:
            qk_summaries.append({"step": step, **head})
            subset = [row for row in qk_raw if row["layer"] == head["layer"] and row["head"] == head["head"]]
            qk_bootstrap[(head["layer"], head["head"])][step] = {
                "qk": scalar_bootstrap(subset, "qk_latest_margin", args.n_boot, rng),
                "attention": scalar_bootstrap(subset, "attention_latest_margin", args.n_boot, rng),
            }

    curve_rows = []
    for index, step in enumerate(steps):
        row = {
            "step": step,
            "tokens_seen": step * TOKENS_PER_STEP,
            "induction_global": induction_values[index],
            "k0_accuracy": bind_values[index],
            "within_stale_share_kge1_errors": stale_values[index],
            "stale_pool_gate": stale_eligible[index],
            "induction_global_ci": np.quantile(induction_boot_matrix[:, index], [0.025, 0.975]).tolist(),
            "k0_accuracy_ci": np.quantile(bind_boot_matrix[:, index], [0.025, 0.975]).tolist(),
            "within_stale_share_ci": np.quantile(stale_boot_matrix[:, index], [0.025, 0.975]).tolist(),
            **{f"accuracy_k{k}": dose[k][index] for k in K_VALUES},
            "k4_within_stale_share_errors": k4_stale_share[index],
            **{f"error_share_{label}": taxonomy[label][index] for label in taxonomy},
            "error_share_within_stale_ci": np.quantile(
                stale_boot_matrix[:, index], [0.025, 0.975]
            ).tolist(),
            "error_share_cross_variable_ci": np.quantile(
                np.stack([item["cross_share"] for item in behavior_boot], axis=1)[:, index],
                [0.025, 0.975],
            ).tolist(),
            "error_share_other_ci": np.quantile(
                np.stack([item["other_share"] for item in behavior_boot], axis=1)[:, index],
                [0.025, 0.975],
            ).tolist(),
        }
        curve_rows.append(row)
    dump_jsonl(root / "curves.jsonl", curve_rows)
    analysis_summary = {
        "stage": "N-phase2-analysis",
        "checkpoint_steps": steps,
        "tokens_per_step": TOKENS_PER_STEP,
        "transition_estimator": "right endpoint of largest positive adjacent-checkpoint level change",
        "n_boot": args.n_boot,
        "transitions": transitions,
        "alignment": {
            "immediately_follows_definition": "S_stale is S_ind or the next preregistered checkpoint",
            "temporal_gate": temporal_gate,
            "taxonomy_shift": taxonomy_shift,
            "taxonomy_gate": taxonomy_gate,
            "core_supported": core_supported,
            "reading": (
                "core_supported" if core_supported else
                "null_temporal_misalignment" if not temporal_gate else
                "null_taxonomy_shift_missing"
            ),
        },
        "qk_attention_trajectories": qk_summaries,
    }
    dump_json(root / "transitions.summary.json", analysis_summary)

    x = np.log10(np.asarray(steps, dtype=float) + 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, induction_values, marker="o", ms=3.5, lw=1.5, label="global induction")
    ax.plot(x, bind_values, marker="s", ms=3.2, lw=1.4, label="k=0 binding accuracy")
    ax.plot(x, stale_values, marker="^", ms=3.2, lw=1.4, label="stale share among errors")
    for values, boot, color in (
        (induction_values, induction_boot_matrix, "#0072B2"),
        (bind_values, bind_boot_matrix, "#009E73"),
        (stale_values, stale_boot_matrix, "#D55E00"),
    ):
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        ax.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
    for key, color in (("S_ind", "#0072B2"), ("S_bind", "#009E73"), ("S_stale", "#D55E00")):
        if transitions[key]["step"] is not None:
            ax.axvline(math.log10(transitions[key]["step"] + 1), color=color, ls="--", lw=1, alpha=0.8)
    ax.set_title("a. Preregistered developmental curves")
    ax.set_ylabel("score / proportion")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[0, 1]
    for k in K_VALUES:
        ax.plot(x, dose[k], marker="o", ms=2.6, lw=1.1, label=f"k={k}")
        boot = np.stack([item[f"accuracy_k{k}"] for item in behavior_boot], axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        ax.fill_between(x, lower, upper, alpha=0.06, linewidth=0)
    ax.set_title("b. Overwrite dose across training")
    ax.set_ylabel("accuracy")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    bottom = np.zeros(len(steps))
    colors = {"within_stale": "#D55E00", "cross_variable": "#0072B2", "other": "#999999"}
    for label in ("within_stale", "cross_variable", "other"):
        values = np.nan_to_num(np.asarray(taxonomy[label]), nan=0.0)
        ax.fill_between(x, bottom, bottom + values, label=label.replace("_", " "), color=colors[label], alpha=0.78)
        bottom += values
    ax.set_title("c. Error taxonomy, k>=1")
    ax.set_ylabel("share of errors")
    ax.set_xlabel("log10(training step + 1)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    for layer, head in CIRCUIT_HEADS:
        subset = sorted([r for r in qk_summaries if r["layer"] == layer and r["head"] == head], key=lambda r: r["step"])
        ax.plot(x, [r["qk_latest_margin_mean"] for r in subset], marker="o", ms=3, lw=1.3, label=f"L{layer}H{head} QK")
        ax.plot(x, [r["attention_latest_margin_mean"] for r in subset], ls="--", lw=1.2, label=f"L{layer}H{head} attn")
        for metric, color in (("qk", "#0072B2"), ("attention", "#D55E00")):
            boot = np.stack([qk_bootstrap[(layer, head)][step][metric] for step in steps], axis=1)
            lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
            ax.fill_between(x, lower, upper, color=color, alpha=0.045, linewidth=0)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_title("d. Deciding-head latest-write trajectory")
    ax.set_xlabel("log10(training step + 1)")
    ax.set_ylabel("current - stale margin")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.22)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "N7_training_dynamics.png", dpi=220)
    fig.savefig(figure_dir / "N7_training_dynamics.pdf")
    plt.close(fig)
    write_report(root, analysis_summary, curve_rows)
    print(json.dumps({"summary": str(root / 'transitions.summary.json'), "core_supported": core_supported}, indent=2))


def self_test(_args) -> None:
    assert parse_checkpoint("64") == ("step64", 64)
    assert parse_checkpoint("step143000") == ("step143000", 143000)
    jump = largest_positive_jump([0, 1, 2, 4], [0.1, 0.2, 0.8, 0.7])
    assert jump["detected"] and jump["step"] == 2 and math.isclose(jump["jump"], 0.6)
    gated = largest_positive_jump([0, 1, 2], [0.1, 0.9, 0.5], [False, False, True])
    assert not gated["detected"]
    rows = []
    for seed in TASK_SEEDS:
        for index in range(10):
            rows.append({"seed": seed, "k": 0, "label": "correct_current" if index < 7 else "other"})
    value, eligible = behavior_metric(rows, "bind")
    assert abs(value - 0.7) < 1e-12 and eligible
    bootstrap_rows = []
    for seed in TASK_SEEDS:
        for k in K_VALUES:
            for index in range(10):
                bootstrap_rows.append({
                    "seed": seed,
                    "k": k,
                    "label": "correct_current" if index < 7 else ("within_stale" if k else "other"),
                })
    boot = behavior_bootstrap(bootstrap_rows, 20, np.random.default_rng(0))
    assert boot["bind"].shape == (20,) and boot["stale_share"].shape == (20,)
    induction_rows = []
    for seed in TASK_SEEDS:
        for item in range(3):
            for head in range(2):
                induction_rows.append({
                    "id": f"s{seed}_{item}", "seed": seed, "layer": 0, "head": head,
                    "induction_minus_baseline": float(item + head),
                })
    induction_boot = induction_bootstrap(induction_rows, 20, np.random.default_rng(1))
    assert induction_boot.shape == (20,) and np.isfinite(induction_boot).all()
    print("pythia_dynamics self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)

    prep = sub.add_parser("prepare")
    prep.add_argument("--source", default="results/stage_n/behavior_full_cpu/tasks.jsonl")
    prep.add_argument("--out", default="results/stage_n/dynamics/task_single.jsonl")
    prep.add_argument("--qk-probe", default="results/stage_n/dynamics/qk_probe_k4.jsonl")
    prep.add_argument("--manifest", default="results/stage_n/dynamics/preregistered_inputs.json")
    prep.add_argument("--n-induction-per-seed", type=int, default=64)
    prep.add_argument("--induction-length", type=int, default=24)
    prep.set_defaults(func=prepare)

    behavior = sub.add_parser("behavior")
    behavior.add_argument("--checkpoint", required=True)
    behavior.add_argument("--task", default="results/stage_n/dynamics/task_single.jsonl")
    behavior.add_argument("--out-root", default="results/stage_n/dynamics")
    behavior.add_argument("--model", default=MODEL)
    behavior.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    behavior.add_argument("--batch-size", type=int, default=128)
    behavior.add_argument("--limit", type=int)
    behavior.add_argument("--online", action="store_true")
    behavior.set_defaults(func=behavior_checkpoint)

    mechanism = sub.add_parser("mechanism")
    mechanism.add_argument("--checkpoint", required=True)
    mechanism.add_argument("--task", default="results/stage_n/dynamics/task_single.jsonl")
    mechanism.add_argument("--qk-probe", default="results/stage_n/dynamics/qk_probe_k4.jsonl")
    mechanism.add_argument("--out-root", default="results/stage_n/dynamics")
    mechanism.add_argument("--model", default=MODEL)
    mechanism.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    mechanism.add_argument("--batch-size", type=int, default=24)
    mechanism.add_argument("--qk-limit", type=int)
    mechanism.add_argument("--n-induction-per-seed", type=int, default=64)
    mechanism.add_argument("--induction-length", type=int, default=24)
    mechanism.add_argument("--online", action="store_true")
    mechanism.set_defaults(func=mechanism_checkpoint)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--out-root", default="results/stage_n/dynamics")
    analysis.add_argument("--n-boot", type=int, default=2000)
    analysis.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analysis.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
