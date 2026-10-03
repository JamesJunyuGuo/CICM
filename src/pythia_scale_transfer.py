"""Stage N descriptive mechanism transfer across final Pythia model scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pythia_circuit import delta_summary, fit_probe_layer, value_shuffle_null
from pythia_eval import dump_json, load_model, matched_pool
from pythia_gen import dump_jsonl, load_jsonl


CORRECT = "correct_current"
FAILURE = "within_stale"


def fixed_k_pool_gate(rows: list[dict], fixed_k: int = 4) -> dict:
    subset = [
        row for row in rows
        if int(row["k"]) == fixed_k and row["variant"] == "single"
    ]
    counts = Counter(row["label"] for row in subset)
    per_template = defaultdict(Counter)
    for row in subset:
        per_template[row["template"]][row["label"]] += 1
    expected_templates = {"arrow", "current", "latest"}
    gate_pass = (
        counts[CORRECT] >= 150
        and counts[FAILURE] >= 120
        and set(per_template) == expected_templates
        and all(
            per_template[template][CORRECT] >= 25
            and per_template[template][FAILURE] >= 25
            for template in expected_templates
        )
    )
    return {
        "fixed_k": fixed_k,
        "n_single_variable": len(subset),
        "counts": dict(counts),
        "per_template": {
            template: dict(per_template[template]) for template in sorted(per_template)
        },
        "gate_pass": bool(gate_pass),
        "rule": (
            "At fixed k=4, single-variable correct>=150 and within-stale>=120, "
            "with correct>=25 and within-stale>=25 in each of arrow/current/latest."
        ),
    }


def pooled_residualize_matrix(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] != len(lengths):
        raise ValueError("values must have examples on axis 0 and at least one feature axis")
    flat = values.reshape(len(values), -1)
    finite_rows = np.isfinite(lengths) & (lengths > 0) & np.all(np.isfinite(flat), axis=1)
    residuals = np.full_like(flat, np.nan, dtype=np.float64)
    if finite_rows.sum() < 2:
        return residuals.reshape(values.shape)
    design = np.column_stack([
        np.ones(finite_rows.sum(), dtype=np.float64),
        np.log(lengths[finite_rows]),
    ])
    coefficients, *_ = np.linalg.lstsq(design, flat[finite_rows], rcond=None)
    residuals[finite_rows] = flat[finite_rows] - design @ coefficients
    return residuals.reshape(values.shape)


def _group_delta(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    failure = labels == FAILURE
    correct = labels == CORRECT
    if not failure.any() or not correct.any():
        raise ValueError("both correct_current and within_stale labels are required")
    return np.nanmean(values[failure], axis=0) - np.nanmean(values[correct], axis=0)


def _shuffle_labels_within_strata(
    labels: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = labels.copy()
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def _bootstrap_delta_ci(
    values: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    failure = np.flatnonzero(labels == FAILURE)
    correct = np.flatnonzero(labels == CORRECT)
    samples = np.empty((n_boot,) + values.shape[1:], dtype=np.float32)
    for iteration in range(n_boot):
        f = rng.choice(failure, size=len(failure), replace=True)
        c = rng.choice(correct, size=len(correct), replace=True)
        samples[iteration] = np.nanmean(values[f], axis=0) - np.nanmean(values[c], axis=0)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def _head_records(
    qk_delta: np.ndarray,
    attention_delta: np.ndarray,
    qk_ci: tuple[np.ndarray, np.ndarray],
    attention_ci: tuple[np.ndarray, np.ndarray],
) -> list[dict]:
    records = []
    for layer in range(qk_delta.shape[0]):
        for head in range(qk_delta.shape[1]):
            records.append({
                "layer": layer,
                "head": head,
                "qk_failure_minus_correct": float(qk_delta[layer, head]),
                "qk_bootstrap95": [
                    float(qk_ci[0][layer, head]),
                    float(qk_ci[1][layer, head]),
                ],
                "attention_failure_minus_correct": float(attention_delta[layer, head]),
                "attention_bootstrap95": [
                    float(attention_ci[0][layer, head]),
                    float(attention_ci[1][layer, head]),
                ],
            })
    return records


def summarize_head_transfer(
    qk_latest_margin: np.ndarray,
    stale_attention_ratio: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    strata: np.ndarray,
    n_shuffle: int = 2000,
    n_boot: int = 2000,
    seed: int = 20260810,
) -> dict:
    qk = np.asarray(qk_latest_margin, dtype=np.float64)
    attention = np.asarray(stale_attention_ratio, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    strata = np.asarray(strata, dtype=object)
    if qk.shape != attention.shape or qk.ndim != 3:
        raise ValueError("QK and attention arrays must both have shape [example, layer, head]")
    if qk.shape[0] != len(labels) or len(labels) != len(strata):
        raise ValueError("metadata and mechanism arrays must have the same example count")

    qk_controlled = pooled_residualize_matrix(qk, lengths)
    attention_controlled = pooled_residualize_matrix(attention, lengths)
    rng = np.random.default_rng(seed)
    permutation_labels = [
        _shuffle_labels_within_strata(labels, strata, rng) for _ in range(n_shuffle)
    ]

    def summarize_pair(qk_values: np.ndarray, attention_values: np.ndarray, offset: int) -> dict:
        observed_qk = _group_delta(qk_values, labels)
        observed_attention = _group_delta(attention_values, labels)
        qk_null_extreme = np.empty(n_shuffle, dtype=np.float64)
        attention_null_extreme = np.empty(n_shuffle, dtype=np.float64)
        for iteration, permuted in enumerate(permutation_labels):
            qk_null_extreme[iteration] = np.nanmin(_group_delta(qk_values, permuted))
            attention_null_extreme[iteration] = np.nanmax(
                _group_delta(attention_values, permuted)
            )
        qk_threshold = float(np.quantile(qk_null_extreme, 0.025))
        attention_threshold = float(np.quantile(attention_null_extreme, 0.975))
        qk_significant = np.argwhere(observed_qk < qk_threshold)
        attention_significant = np.argwhere(observed_attention > attention_threshold)
        qk_set = {tuple(index) for index in qk_significant.tolist()}
        attention_set = {tuple(index) for index in attention_significant.tolist()}
        conjunction = sorted(qk_set & attention_set)
        boot_rng = np.random.default_rng(seed + offset)
        qk_ci = _bootstrap_delta_ci(qk_values, labels, n_boot, boot_rng)
        attention_ci = _bootstrap_delta_ci(attention_values, labels, n_boot, boot_rng)
        return {
            "qk_direction": "negative means current-to-stale QK displacement on failures",
            "attention_direction": "positive means more stale-relative attention on failures",
            "qk_familywise_lower_95": qk_threshold,
            "attention_familywise_upper_95": attention_threshold,
            "qk_significant_heads": [
                {"layer": int(layer), "head": int(head)} for layer, head in sorted(qk_set)
            ],
            "attention_significant_heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in sorted(attention_set)
            ],
            "same_head_conjunction": [
                {"layer": int(layer), "head": int(head)} for layer, head in conjunction
            ],
            "all_heads": _head_records(
                observed_qk,
                observed_attention,
                qk_ci,
                attention_ci,
            ),
        }

    return {
        "n": int(len(labels)),
        "n_layers": int(qk.shape[1]),
        "n_heads_per_layer": int(qk.shape[2]),
        "n_shuffle": int(n_shuffle),
        "n_boot": int(n_boot),
        "shuffle_protocol": "labels permuted within (template, seed); extrema taken over all heads",
        "raw": summarize_pair(qk, attention, 1000),
        "length_controlled": summarize_pair(qk_controlled, attention_controlled, 2000),
    }


def task_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_rows(tokenizer, rows: list[dict]) -> dict:
    mismatch_counts = Counter()
    for row in rows:
        token_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if len(token_ids) != int(row["prompt_tokens"]):
            mismatch_counts["prompt_length"] += 1
        current_positions = [int(position) for position in row["current_value_span"]]
        if any(token_ids[position] != int(row["gold_token_id"]) for position in current_positions):
            mismatch_counts["current_value_span"] += 1
        stale_ids = {int(token_id) for token_id in row.get("stale_token_ids", [])}
        stale_positions = [int(position) for position in row.get("stale_value_spans", [])]
        if any(token_ids[position] not in stale_ids for position in stale_positions):
            mismatch_counts["stale_value_spans"] += 1
    return {
        "n": len(rows),
        "vocab_size": int(len(tokenizer)),
        "mismatch_counts": dict(mismatch_counts),
        "pass": not mismatch_counts,
    }


def validate_task(args) -> None:
    from transformers import AutoTokenizer

    rows = load_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    summary = validate_task_rows(tokenizer, rows)
    summary.update({"model": args.model, "task_sha256": task_sha256(args.data)})
    dump_json(args.summary, summary)
    if not summary["pass"]:
        raise ValueError(f"task/tokenizer validation failed: {summary['mismatch_counts']}")
    print(json.dumps(summary, indent=2))


def make_smoke(args) -> None:
    rows = load_jsonl(args.data)
    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(int(row["k"]), row["variant"], row["template"])].append(row)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < args.n:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < args.n:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    dump_jsonl(args.out, selected)
    dump_json(args.summary, {
        "n": len(selected),
        "source": args.data,
        "source_sha256": task_sha256(args.data),
        "cells": [
            {"k": row["k"], "variant": row["variant"], "template": row["template"]}
            for row in selected
        ],
    })


def prepare_pool(args) -> None:
    rows = load_jsonl(args.rows)
    gate = fixed_k_pool_gate(rows, args.fixed_k)
    summary = {
        "model": args.model,
        "rows": args.rows,
        "fixed_k_gate": gate,
        "pool_written": False,
    }
    if gate["gate_pass"]:
        pool = matched_pool(rows, args.fixed_k, args.seed, args.per_stratum_cap)
        dump_jsonl(args.pool, pool)
        summary["pool_written"] = True
        summary["matched_pool"] = {
            "path": args.pool,
            "n": len(pool),
            "counts": dict(Counter(row["label"] for row in pool)),
            "strata": len({(row["template"], row["seed"]) for row in pool}),
        }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


def harvest(args) -> None:
    import torch
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    rows = load_jsonl(args.pool)
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer, device = load_model(args.model, args.dtype)
    validation = validate_task_rows(tokenizer, rows)
    if not validation["pass"]:
        raise ValueError(f"pool/tokenizer validation failed: {validation['mismatch_counts']}")
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    hidden_final = []
    qk_latest_margin = []
    attention_current = []
    attention_stale = []
    index_rows = []
    for row_index, row in enumerate(rows):
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        sequence_length = int(encoded["input_ids"].shape[1])
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        query_position = sequence_length - 1
        current_position = int(row["current_value_span"][0])
        stale_positions = [int(position) for position in row["stale_value_spans"]]
        row_qk = np.empty((n_layers, n_heads), dtype=np.float32)
        row_current = np.empty((n_layers, n_heads), dtype=np.float32)
        row_stale = np.empty((n_layers, n_heads), dtype=np.float32)
        for layer in range(n_layers):
            block = model.gpt_neox.layers[layer]
            hidden = output.hidden_states[layer]
            normalized = block.input_layernorm(hidden)
            qkv = block.attention.query_key_value(normalized).view(
                1, sequence_length, n_heads, 3 * head_dim
            ).transpose(1, 2)
            query_states, key_states, _ = qkv.chunk(3, dim=-1)
            cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            query = query_states[0, :, query_position]
            current_keys = key_states[0, :, current_position]
            stale_keys = key_states[0, :, stale_positions]
            current_scores = (query * current_keys).sum(dim=-1) / math.sqrt(head_dim)
            stale_scores = torch.einsum("hd,hsd->hs", query, stale_keys) / math.sqrt(head_dim)
            row_qk[layer] = (
                current_scores - stale_scores.max(dim=-1).values
            ).detach().float().cpu().numpy()
            attention = output.attentions[layer][0, :, query_position]
            row_current[layer] = attention[:, current_position].float().cpu().numpy()
            row_stale[layer] = attention[:, stale_positions].sum(dim=-1).float().cpu().numpy()
        hidden_final.append(output.hidden_states[-1][0, query_position].float().cpu().numpy())
        qk_latest_margin.append(row_qk)
        attention_current.append(row_current)
        attention_stale.append(row_stale)
        index_rows.append({
            "id": row["id"],
            "semantic_id": row["semantic_id"],
            "label": row["label"],
            "template": row["template"],
            "seed": row["seed"],
            "k": row["k"],
            "prompt_tokens": row["prompt_tokens"],
            "current_value": row["gold"],
        })
        if (row_index + 1) % 10 == 0 or row_index + 1 == len(rows):
            print(f"scale-transfer harvest {row_index + 1}/{len(rows)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "harvest.npz",
        hidden_final=np.asarray(hidden_final),
        qk_latest_margin=np.asarray(qk_latest_margin),
        attention_current=np.asarray(attention_current),
        attention_stale=np.asarray(attention_stale),
    )
    dump_jsonl(out_dir / "harvest_index.jsonl", index_rows)
    dump_json(out_dir / "harvest.summary.json", {
        "model": args.model,
        "dtype": args.dtype,
        "n": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "hidden_final_shape": list(np.asarray(hidden_final).shape),
        "head_array_shape": list(np.asarray(qk_latest_margin).shape),
        "tokenizer_validation": validation,
    })


def analyze_probe(args) -> None:
    out_dir = Path(args.harvest_dir)
    rows = load_jsonl(out_dir / "harvest_index.jsonl")
    hidden = np.load(out_dir / "harvest.npz")["hidden_final"][:, None, :]
    probe = fit_probe_layer(hidden, rows, 0, args.seed)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    failure_mask = labels == FAILURE
    observed = float(np.nanmean(probe["true_scores"][failure_mask]))
    null = value_shuffle_null(probe, rows, FAILURE, args.n_shuffle, args.seed + 100)
    comparison = delta_summary(
        probe["true_scores"], rows, FAILURE, CORRECT,
        args.n_boot, args.n_shuffle, args.seed + 200,
    )
    summary = {
        "model": args.model,
        "n": len(rows),
        "protocol": "L2 multinomial logistic regression; 5-fold GroupKFold by semantic_id",
        "cv_accuracy": probe["cv_accuracy"],
        "within_stale_true_current_score": observed,
        "value_label_shuffle95": [
            float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))
        ],
        "above_shuffle95": bool(observed > np.quantile(null, 0.975)),
        "within_vs_correct": comparison,
    }
    dump_json(args.summary, summary)
    dump_jsonl(args.probe_rows, [
        {**row, "final_true_current_score": float(score)}
        for row, score in zip(rows, probe["true_scores"])
    ])
    print(json.dumps(summary, indent=2))


def analyze_heads(args) -> None:
    out_dir = Path(args.harvest_dir)
    rows = load_jsonl(out_dir / "harvest_index.jsonl")
    data = np.load(out_dir / "harvest.npz")
    current = data["attention_current"].astype(np.float64)
    stale = data["attention_stale"].astype(np.float64)
    ratio = stale / np.maximum(current + stale, 1e-12)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    lengths = np.asarray([row["prompt_tokens"] for row in rows], dtype=np.float64)
    strata = np.asarray([
        f"{row['template']}__seed{row['seed']}" for row in rows
    ], dtype=object)
    summary = summarize_head_transfer(
        data["qk_latest_margin"], ratio, labels, lengths, strata,
        n_shuffle=args.n_shuffle, n_boot=args.n_boot, seed=args.seed,
    )
    summary.update({
        "model": args.model,
        "qk_definition": "rotated QK(current)-max rotated QK(stale), scaled by sqrt(head_dim)",
        "attention_definition": "stale_mass/(stale_mass+current_mass) at answer position",
        "descriptive_head_conjunction_pass": bool(
            summary["length_controlled"]["same_head_conjunction"]
        ),
    })
    dump_json(args.summary, summary)
    print(json.dumps({
        "summary": args.summary,
        "raw_conjunction": summary["raw"]["same_head_conjunction"],
        "controlled_conjunction": summary["length_controlled"]["same_head_conjunction"],
    }, indent=2))


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def aggregate(args) -> None:
    root = Path(args.root)
    scale_rows = []
    for scale in ("160m", "410m", "1b", "2.8b"):
        scale_dir = root / scale
        behavior_dir = (
            scale_dir / "behavior_reference" if scale == "160m"
            else scale_dir / "behavior_full"
        )
        gate_payload = _read_json(behavior_dir / "fixed_k4_pool.summary.json")
        behavior = _read_json(
            Path(args.reference_behavior) if scale == "160m"
            else behavior_dir / "behavior.summary.json"
        )
        probe = _read_json(scale_dir / "analysis" / "probe.summary.json")
        heads = _read_json(scale_dir / "analysis" / "heads.summary.json")
        if gate_payload is None:
            scale_rows.append({"scale": scale, "status": "behavior_not_available"})
            continue
        gate = gate_payload["fixed_k_gate"]
        row = {
            "scale": scale,
            "model": gate_payload["model"],
            "status": "behavior_gate_failed" if not gate["gate_pass"] else "mechanism_pending",
            "fixed_k4_gate_pass": bool(gate["gate_pass"]),
            "fixed_k4_counts": gate["counts"],
            "fixed_k4_per_template": gate["per_template"],
            "matched_pool": gate_payload.get("matched_pool"),
            "behavior_by_k": behavior.get("by_k") if behavior else None,
        }
        if gate["gate_pass"] and probe is not None and heads is not None:
            controlled = heads["length_controlled"]
            probe_pass = bool(probe["above_shuffle95"])
            qk_pass = bool(controlled["qk_significant_heads"])
            attention_pass = bool(controlled["attention_significant_heads"])
            conjunction_pass = bool(controlled["same_head_conjunction"])
            transfer_pass = probe_pass and qk_pass and attention_pass and conjunction_pass
            row.update({
                "status": "complete",
                "probe": {
                    "within_stale_true_current_score": probe["within_stale_true_current_score"],
                    "value_label_shuffle95": probe["value_label_shuffle95"],
                    "above_shuffle95": probe_pass,
                    "raw_failure_minus_correct": probe["within_vs_correct"]["raw_delta"],
                    "length_controlled_failure_minus_correct": probe["within_vs_correct"][
                        "length_controlled_delta"
                    ],
                    "length_controlled_ci": probe["within_vs_correct"]["length_controlled_ci"],
                    "length_controlled_shuffle95": probe["within_vs_correct"][
                        "length_controlled_shuffle95"
                    ],
                },
                "heads": {
                    "qk_significant_count": len(controlled["qk_significant_heads"]),
                    "attention_significant_count": len(controlled["attention_significant_heads"]),
                    "qk_significant_heads": controlled["qk_significant_heads"],
                    "attention_significant_heads": controlled["attention_significant_heads"],
                    "same_head_conjunction": controlled["same_head_conjunction"],
                    "same_head_conjunction_count": len(controlled["same_head_conjunction"]),
                },
                "descriptive_transfer_pass": transfer_pass,
            })
        scale_rows.append(row)
    complete = [row for row in scale_rows if row["status"] == "complete"]
    canonical = _read_json(Path(args.reference_mechanism))
    reference_sanity = None
    if canonical and complete and complete[0]["scale"] == "160m":
        old_probe = canonical["probe"]
        new_probe = complete[0]["probe"]
        reference_sanity = {
            "canonical_probe_score": old_probe["within_stale_true_current_score"],
            "new_probe_score": new_probe["within_stale_true_current_score"],
            "absolute_probe_score_difference": abs(
                old_probe["within_stale_true_current_score"]
                - new_probe["within_stale_true_current_score"]
            ),
            "canonical_length_controlled_delta": old_probe["within_vs_correct"][
                "length_controlled_delta"
            ],
            "new_length_controlled_delta": new_probe[
                "length_controlled_failure_minus_correct"
            ],
            "absolute_delta_difference": abs(
                old_probe["within_vs_correct"]["length_controlled_delta"]
                - new_probe["length_controlled_failure_minus_correct"]
            ),
        }
    payload = {
        "stage": "N-cross-scale-descriptive-transfer",
        "task_sha256": "22e79200d953cd347c3c1ea3ae2f94d2634a45ef70e8e37549f6fa80cafc2fdf",
        "fixed_k": 4,
        "scales": scale_rows,
        "all_behavior_gates_pass": all(
            row.get("fixed_k4_gate_pass", False) for row in scale_rows
        ),
        "all_identifiable_scales_transfer": bool(complete) and all(
            row["descriptive_transfer_pass"] for row in complete
        ),
        "reference_sanity": reference_sanity,
        "scope": (
            "Descriptive transfer of selection, QK displacement, and stale-relative attention; "
            "not conservation of head identity or of the 160M minimal causal circuit."
        ),
    }
    dump_json(args.summary, payload)

    def fmt(value) -> str:
        return "--" if value is None else f"{value:.3f}"

    lines = [
        "# Stage N cross-scale descriptive mechanism transfer",
        "",
        "## Preregistered scope",
        "",
        payload["scope"],
        "",
        "All models use the byte-identical 7,776-example task and fixed `k=4`. "
        "Mechanism analysis is run only after the frozen matched-pool gate passes.",
        "",
        "## Headline",
        "",
        "The abstract retention-versus-selection description transfers, but the localized "
        "160M QK-attention coupling does not. At every identifiable scale, the current value "
        "remains decodable on stale failures above a label-shuffle null, although its score is "
        "lower than on correct trials. The exact same-head conjunction weakens from four heads "
        "at 160M to zero at 410M and 2.8B.",
        "",
        "## Results",
        "",
        "### Behavior across overwrite count",
        "",
        "| Scale | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scale_rows:
        by_k = row.get("behavior_by_k") or {}
        cells = [
            fmt(by_k.get(str(k), {}).get("accuracy")) for k in range(6)
        ]
        lines.append(f"| {row['scale']} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Cells are exact-match current-value accuracy on all rows at each overwrite count. "
        "The fixed-`k=4` gate below uses only the clean single-variable arm.",
        "",
        "### Failure-conditioned mechanism",
        "",
        "| Scale | k=4 correct | k=4 stale | Pool gate | Probe score | Probe null upper | QK heads | Attention heads | Same-head | Transfer |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in scale_rows:
        counts = row.get("fixed_k4_counts", {})
        probe = row.get("probe", {})
        heads = row.get("heads", {})
        null = probe.get("value_label_shuffle95")
        lines.append(
            "| {scale} | {correct} | {stale} | {gate} | {probe} | {null} | {qk} | "
            "{attention} | {same} | {transfer} |".format(
                scale=row["scale"],
                correct=counts.get(CORRECT, "--"),
                stale=counts.get(FAILURE, "--"),
                gate="yes" if row.get("fixed_k4_gate_pass") else "no",
                probe=fmt(probe.get("within_stale_true_current_score")),
                null=fmt(null[1] if null else None),
                qk=heads.get("qk_significant_count", "--"),
                attention=heads.get("attention_significant_count", "--"),
                same=heads.get("same_head_conjunction_count", "--"),
                transfer=(
                    "yes" if row.get("descriptive_transfer_pass") else
                    "no" if row["status"] == "complete" else "--"
                ),
            )
        )
    lines.extend([
        "",
        "QK and attention head counts use pooled log-length residuals and 2,000 "
        "within-(template, seed) permutations with model-wide family-wise extrema.",
        "",
        "### Selection probe",
        "",
        "| Scale | Failure score | Shuffle 95% | Failure-correct raw | Failure-correct controlled [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in scale_rows:
        probe = row.get("probe")
        if not probe:
            lines.append(f"| {row['scale']} | -- | -- | -- | -- |")
            continue
        ci = probe["length_controlled_ci"]
        null = probe["value_label_shuffle95"]
        lines.append(
            f"| {row['scale']} | {probe['within_stale_true_current_score']:.3f} | "
            f"[{null[0]:.3f}, {null[1]:.3f}] | "
            f"{probe['raw_failure_minus_correct']:.3f} | "
            f"{probe['length_controlled_failure_minus_correct']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    lines.extend([
        "",
        "The positive above-null score rejects complete representational loss. The negative "
        "failure-minus-correct difference shows concurrent retention degradation, so the result "
        "is present-but-weakened rather than a claim of perfectly intact retention.",
        "",
        "### Canonical 160M sanity",
        "",
    ])
    if reference_sanity:
        lines.extend([
            f"The new pipeline reproduces the frozen Phase-1 failure score "
            f"({reference_sanity['new_probe_score']:.6f} vs "
            f"{reference_sanity['canonical_probe_score']:.6f}; absolute difference "
            f"{reference_sanity['absolute_probe_score_difference']:.6f}) and the "
            f"length-controlled failure-minus-correct delta "
            f"({reference_sanity['new_length_controlled_delta']:.6f} vs "
            f"{reference_sanity['canonical_length_controlled_delta']:.6f}; absolute difference "
            f"{reference_sanity['absolute_delta_difference']:.6f}).",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
    ])
    for row in scale_rows:
        if row["status"] == "behavior_gate_failed":
            lines.append(
                f"- **{row['scale']}:** the fixed behavior gate failed, so failure-conditioned "
                "mechanism transfer is not identifiable under this task."
            )
        elif row["status"] == "complete":
            if row["descriptive_transfer_pass"]:
                lines.append(
                    f"- **{row['scale']}:** full preregistered transfer: probe, QK, attention, "
                    "and same-head conjunction all pass."
                )
            else:
                heads = row["heads"]
                lines.append(
                    f"- **{row['scale']}:** partial transfer: the probe passes; "
                    f"{heads['qk_significant_count']} QK and "
                    f"{heads['attention_significant_count']} attention heads pass separately, "
                    f"but {heads['same_head_conjunction_count']} pass their conjunction."
                )
        else:
            lines.append(f"- **{row['scale']}:** {row['status'].replace('_', ' ')}.")
    lines.extend([
        "",
        "A pass does not show that numerical head indices or a minimal causal circuit are "
        "preserved across scales. A failure does not show that no stale-binding mechanism "
        "exists outside this frozen task.",
        "",
        "The 1B exception also rules out a monotonic size law on this task: its fixed-`k=4` "
        "stale pool is too small and template-imbalanced for the primary mechanism test, while "
        "2.8B again produces a large stale-dominant pool. No post-hoc difficulty or gate tuning "
        "was used to remove that non-monotonicity.",
        "",
        "For connecting the 160M circuit result to larger instruction-tuned models, the supported "
        "bridge is therefore the selection-level diagnosis, not conservation of a particular "
        "small-model head set or QK route.",
        "",
    ])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": args.summary, "report": args.report}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate-task")
    validate.add_argument("--data", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--summary", required=True)
    validate.set_defaults(func=validate_task)

    smoke = sub.add_parser("make-smoke")
    smoke.add_argument("--data", required=True)
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--summary", required=True)
    smoke.add_argument("--n", type=int, default=20)
    smoke.add_argument("--seed", type=int, default=20260810)
    smoke.set_defaults(func=make_smoke)

    pool = sub.add_parser("prepare-pool")
    pool.add_argument("--rows", required=True)
    pool.add_argument("--pool", required=True)
    pool.add_argument("--summary", required=True)
    pool.add_argument("--model", required=True)
    pool.add_argument("--fixed-k", type=int, default=4)
    pool.add_argument("--seed", type=int, default=20260807)
    pool.add_argument("--per-stratum-cap", type=int, default=48)
    pool.set_defaults(func=prepare_pool)

    harvest_parser = sub.add_parser("harvest")
    harvest_parser.add_argument("--pool", required=True)
    harvest_parser.add_argument("--out-dir", required=True)
    harvest_parser.add_argument("--model", required=True)
    harvest_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    harvest_parser.add_argument("--limit", type=int)
    harvest_parser.set_defaults(func=harvest)

    probe = sub.add_parser("analyze-probe")
    probe.add_argument("--harvest-dir", required=True)
    probe.add_argument("--summary", required=True)
    probe.add_argument("--probe-rows", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--n-boot", type=int, default=2000)
    probe.add_argument("--n-shuffle", type=int, default=2000)
    probe.add_argument("--seed", type=int, default=20260810)
    probe.set_defaults(func=analyze_probe)

    heads = sub.add_parser("analyze-heads")
    heads.add_argument("--harvest-dir", required=True)
    heads.add_argument("--summary", required=True)
    heads.add_argument("--model", required=True)
    heads.add_argument("--n-boot", type=int, default=2000)
    heads.add_argument("--n-shuffle", type=int, default=2000)
    heads.add_argument("--seed", type=int, default=20260810)
    heads.set_defaults(func=analyze_heads)

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--root", default="results/stage_n/scale_transfer")
    aggregate_parser.add_argument(
        "--reference-behavior",
        default="results/stage_n/behavior_full_cpu/summary.json",
    )
    aggregate_parser.add_argument(
        "--reference-mechanism",
        default="results/stage_n/mechanism_full_cpu/mechanism.summary.json",
    )
    aggregate_parser.add_argument("--summary", required=True)
    aggregate_parser.add_argument("--report", required=True)
    aggregate_parser.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
