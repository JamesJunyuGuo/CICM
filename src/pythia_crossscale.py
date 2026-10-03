"""Stage O Part A: comparable stale-binding mechanisms across Pythia scales."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pythia_circuit as core
from pythia_eval import (
    classify_token,
    dump_json,
    load_model,
    matched_pool,
    select_fixed_k,
)
from pythia_gen import (
    VAR_CANDIDATES,
    _fixed_examples,
    annotate_spans,
    dump_jsonl,
    load_jsonl,
    make_row,
    one_token_strings,
    render_prompt,
    validate_row,
    value_token_id,
)


CORRECT = "correct_current"
STALE = "within_stale"
SCALES = (
    ("160m", "EleutherAI/pythia-160m", 144),
    ("410m", "EleutherAI/pythia-410m", 384),
    ("1.4b", "EleutherAI/pythia-1.4b", 384),
    ("2.8b", "EleutherAI/pythia-2.8b", 1024),
)


def make_single_row(
    *, tokenizer, template: str, k: int, seed: int, index: int, n_lines: int = 15
) -> dict:
    """Extend the frozen single-variable semantics past the multi-only guard."""
    rng = random.Random(seed * 1_000_003 + k * 10_007 + index * 101)
    variables = one_token_strings(tokenizer, VAR_CANDIDATES)
    values = one_token_strings(tokenizer, (str(i) for i in range(10, 40)))
    reserved_vars = {value for events, _, _ in _fixed_examples(template) for value, _ in events}
    reserved_values = {value for events, _, _ in _fixed_examples(template) for _, value in events}
    variables = [value for value in variables if value not in reserved_vars]
    values = [value for value in values if value not in reserved_values]
    if len(variables) < n_lines + 1 or len(values) < n_lines + 6:
        raise ValueError("insufficient one-token vocabulary for frozen task")
    chosen_vars = rng.sample(variables, n_lines + 1)
    chosen_values = rng.sample(values, n_lines + 6)
    target = chosen_vars[0]
    target_values = chosen_values[: k + 1]
    events = [
        {
            "position": position,
            "var": target,
            "value": value,
            "is_target": True,
            "is_current": position == k,
        }
        for position, value in enumerate(target_values)
    ]
    prompt, task_char_start = render_prompt(events, target, template)
    stale = target_values[:-1]
    row = {
        "id": f"n_single_{template}_k{k}_s{seed}_{index:04d}",
        "semantic_id": f"n_single_k{k}_s{seed}_{index:04d}",
        "variant": "single",
        "template": template,
        "k": k,
        "seed": seed,
        "index": index,
        "n_lines": n_lines,
        "target_var": target,
        "gold": target_values[-1],
        "gold_token_id": value_token_id(tokenizer, target_values[-1]),
        "stale_values": stale,
        "stale_token_ids": [value_token_id(tokenizer, value) for value in stale],
        "cross_values": [],
        "cross_token_ids": [],
        "events": events,
        "prompt": prompt,
        "task_char_start": task_char_start,
    }
    return annotate_spans(row, tokenizer)


def generate_data(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    for seed in args.seeds:
        for k in args.k_values:
            for template in args.templates:
                for index in range(args.n_per_seed_cell):
                    row = make_single_row(
                        tokenizer=tokenizer,
                        template=template,
                        k=k,
                        seed=seed,
                        index=index,
                        n_lines=args.n_lines,
                    )
                    validate_row(row, tokenizer)
                    if k <= args.max_frozen_k:
                        frozen = make_row(
                            variant="single",
                            template=template,
                            k=k,
                            seed=seed,
                            index=index,
                            tokenizer=tokenizer,
                            n_lines=args.n_lines,
                        )
                        if row != frozen:
                            raise RuntimeError(f"single-row extension drifted at {row['id']}")
                    rows.append(row)
    dump_jsonl(args.out, rows)
    summary = {
        "stage": "O-Part-A-data",
        "model_tokenizer": args.model,
        "n": len(rows),
        "seeds": args.seeds,
        "k_values": args.k_values,
        "templates": args.templates,
        "variant": "single",
        "n_lines": args.n_lines,
        "n_per_seed_cell": args.n_per_seed_cell,
        "frozen_equivalence_checked_through_k": args.max_frozen_k,
        "extension": (
            "k=12/14 use the frozen single-variable event rule while bypassing the "
            "post-current-distractor guard that applies only to multi-variable rows"
        ),
        "prompt_token_range": [
            min(row["prompt_tokens"] for row in rows),
            max(row["prompt_tokens"] for row in rows),
        ],
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def _rate(rows, label=CORRECT):
    return sum(row["label"] == label for row in rows) / len(rows) if rows else float("nan")


def _counts(rows):
    return dict(Counter(row["label"] for row in rows))


def adjudicate_behavior_rows(rows: list[dict], model: str) -> dict:
    """Apply the fixed k=0, feasibility, and Phase-1 power gates."""
    single = [row for row in rows if row["variant"] == "single"]
    k0 = [row for row in single if int(row["k"]) == 0]
    k0_by_template = {
        template: [row for row in k0 if row["template"] == template]
        for template in ("arrow", "current", "latest")
    }
    k0_rates = {template: _rate(group) for template, group in k0_by_template.items()}
    k0_pass = bool(k0 and all(group for group in k0_by_template.values())) and all(
        value >= 0.95 for value in k0_rates.values()
    )
    overwrite_k = sorted({int(row["k"]) for row in single if int(row["k"]) > 0})
    by_k = {
        str(k): {
            "n": len(group := [row for row in single if int(row["k"]) == k]),
            "accuracy": _rate(group),
            "counts": _counts(group),
        }
        for k in sorted({int(row["k"]) for row in single})
    }
    fixed = select_fixed_k(rows)
    infeasible = bool(overwrite_k) and all(by_k[str(k)]["accuracy"] > 0.85 for k in overwrite_k)
    if not k0_pass:
        status = "invalid_k0_control"
    elif infeasible:
        status = "substrate_infeasible_at_cap"
    elif fixed["gate_pass"]:
        status = "mechanism_ready"
    else:
        status = "insufficient_power_on_fixed_grid"
    taxonomy = {}
    for k in sorted({int(row["k"]) for row in rows}):
        for variant in sorted({row["variant"] for row in rows}):
            group = [
                row for row in rows if int(row["k"]) == k and row["variant"] == variant
            ]
            if group:
                taxonomy[f"k{k}__{variant}"] = {"n": len(group), "counts": _counts(group)}
    return {
        "stage": "O-Part-A-behavior",
        "model": model,
        "status": status,
        "k0_gate": {
            "threshold": 0.95,
            "pass": k0_pass,
            "n": len(k0),
            "accuracy": _rate(k0),
            "per_template_accuracy": k0_rates,
        },
        "by_k_single": by_k,
        "taxonomy": taxonomy,
        "fixed_k_gate": fixed,
        "chosen_k": fixed["chosen_k"] if status == "mechanism_ready" else None,
        "claim_boundary": (
            "No task tuning beyond the preregistered n_lines=15 and "
            "k={0,2,4,6,8,10,12,14} grid."
        ),
    }


def adjudicate_behavior(args) -> None:
    rows = load_jsonl(args.rows)
    summary = adjudicate_behavior_rows(rows, args.model)
    summary["rows"] = args.rows
    summary["pool_written"] = False
    if summary["status"] == "mechanism_ready":
        pool = matched_pool(rows, summary["chosen_k"], args.seed, args.per_stratum_cap)
        dump_jsonl(args.pool, pool)
        split_counts = Counter(
            (row["label"], core.stable_split(row["semantic_id"])) for row in pool
        )
        summary["pool_written"] = True
        summary["matched_pool"] = {
            "path": args.pool,
            "n": len(pool),
            "counts": _counts(pool),
            "split_counts": {
                label: {
                    split: split_counts[(label, split)]
                    for split in ("discovery", "evaluation")
                }
                for label in (CORRECT, STALE)
            },
        }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def rank_stale_promoting_heads(
    stale_ratio: np.ndarray,
    head_dla: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
) -> list[dict]:
    labels = np.asarray(labels, dtype=object)
    splits = np.asarray(splits, dtype=object)
    discovery_correct = (splits == "discovery") & (labels == CORRECT)
    discovery_stale = (splits == "discovery") & (labels == STALE)
    if not discovery_correct.any() or not discovery_stale.any():
        raise ValueError("discovery split needs correct and within-stale rows")
    table = []
    for layer in range(stale_ratio.shape[1]):
        for head in range(stale_ratio.shape[2]):
            shift = float(
                np.nanmean(stale_ratio[discovery_stale, layer, head])
                - np.nanmean(stale_ratio[discovery_correct, layer, head])
            )
            failure_dla = float(np.nanmean(head_dla[discovery_stale, layer, head]))
            table.append(
                {
                    "layer": layer,
                    "head": head,
                    "discovery_attention_shift": shift,
                    "discovery_failure_dla": failure_dla,
                    "discovery_score": shift * max(0.0, -failure_dla),
                }
            )
    table.sort(key=lambda row: row["discovery_score"], reverse=True)
    for index, row in enumerate(table, 1):
        row["rank"] = index
    return table


def _random_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "quantile95": np.quantile(values, [0.025, 0.975]).tolist(),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def ablate_heldout(args) -> None:
    rows = load_jsonl(args.pool)
    index_rows = load_jsonl(Path(args.harvest_dir) / "harvest_index.jsonl")
    if [row["id"] for row in rows] != [row["id"] for row in index_rows]:
        raise ValueError("harvest index does not match the matched pool")
    data = np.load(Path(args.harvest_dir) / "harvest.npz")
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    splits = np.asarray([core.stable_split(row["semantic_id"]) for row in rows], dtype=object)
    stale_ratio = data["attn_stale"] / np.maximum(
        data["attn_stale"] + data["attn_current"], 1e-12
    )
    ranking = rank_stale_promoting_heads(stale_ratio, data["head_dla"], labels, splits)
    cumulative = json.loads(Path(args.cumulative_summary).read_text(encoding="utf-8"))
    selected_n = min(int(cumulative["selected_count"]), args.max_heads)
    if selected_n < 1:
        raise ValueError("path-patch summary selected no heads")
    target_heads = [
        (int(row["layer"]), int(row["head"])) for row in ranking[:selected_n]
    ]
    evaluation_rows = [
        row
        for row, split in zip(rows, splits)
        if split == "evaluation" and row["label"] == STALE
    ]
    if not evaluation_rows:
        raise ValueError("held-out split has no within-stale rows")
    model, tokenizer, device = load_model(args.model, args.dtype)
    baseline = core.evaluate_ablation_arm(
        model, tokenizer, device, evaluation_rows, [], args.batch_size
    )
    targeted = core.evaluate_ablation_arm(
        model, tokenizer, device, evaluation_rows, target_heads, args.batch_size
    )
    baseline_labels = np.asarray(baseline["labels"], dtype=object)
    targeted_labels = np.asarray(targeted["labels"], dtype=object)
    mismatches = int(np.sum(baseline_labels != STALE))
    if mismatches:
        raise RuntimeError(f"held-out baseline label mismatch on {mismatches} rows")
    target_effect = (targeted_labels == CORRECT).astype(float)
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    random_sets = core.layer_matched_random_sets(
        target_heads, n_layers, n_heads, args.n_random, args.seed
    )
    random_rows = []
    random_rates = []
    for index, heads in enumerate(random_sets):
        result = core.evaluate_ablation_arm(
            model, tokenizer, device, evaluation_rows, heads, args.batch_size
        )
        effect = np.asarray(result["labels"], dtype=object) == CORRECT
        random_rates.append(float(np.mean(effect)))
        random_rows.append(
            {
                "random_index": index,
                "heads": [{"layer": layer, "head": head} for layer, head in heads],
                "paired_effect": float(np.mean(effect)),
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(random_sets):
            print(f"held-out random ablation {index + 1}/{len(random_sets)}", flush=True)
    random = _random_summary(random_rates)
    effect = core.paired_bootstrap(target_effect, args.n_boot, args.seed + 100)
    summary = {
        "stage": "O-Part-A-heldout-ablation",
        "model": args.model,
        "dtype": args.dtype,
        "protocol": {
            "ranking": (
                "discovery stale-attention shift times max(0, negative discovery failure DLA)"
            ),
            "evaluation": "stable_split evaluation within-stale rows only",
            "n_random": args.n_random,
            "random_control": "same held-out rows and layer-matched same-size head sets",
            "selected_n_rule": f"min(path-patch selected_count, {args.max_heads})",
        },
        "heads": [{"layer": layer, "head": head} for layer, head in target_heads],
        "ranking": ranking,
        "evaluation_n": len(evaluation_rows),
        "baseline_label_mismatches": mismatches,
        "targeted_correct_rate_increase": effect,
        "random_effect": random,
        "targeted_minus_random_mean": float(effect["mean"] - random["mean"]),
        "heldout_exceeds_random95": bool(effect["mean"] > random["quantile95"][1]),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "ablation_random_sets.jsonl", random_rows)
    dump_json(out_dir / "ablation_heldout.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def summarize_induction_overlap(summary: dict) -> dict:
    heads = summary.get("circuit_heads", [])
    top = [row for row in heads if float(row["percentile_among_heads"]) >= 0.9]
    return {
        "selected_head_n": len(heads),
        "top_decile_count": len(top),
        "top_decile_fraction": len(top) / len(heads) if heads else float("nan"),
        "anchor_gate_pass": bool(summary.get("anchor_gate", {}).get("pass", False)),
    }


def cache_model(args) -> None:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    payload = {"model": args.model, "cache_dir": args.cache_dir, "snapshot": snapshot}
    dump_json(args.summary, payload)
    print(json.dumps(payload, indent=2), flush=True)


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _scale_result(root: Path, key: str, model: str, total_heads: int) -> dict:
    scale_dir = root / key
    behavior = _read(scale_dir / "behavior" / "adjudication.summary.json")
    if behavior is None:
        return {"scale": key, "model": model, "status": "behavior_missing"}
    row = {
        "scale": key,
        "model": model,
        "status": behavior["status"],
        "behavior": behavior,
    }
    if behavior["status"] != "mechanism_ready":
        return row
    mechanism = _read(scale_dir / "mechanism" / "mechanism.summary.json")
    first_pass = _read(scale_dir / "circuit" / "path_patch_first_pass.summary.json")
    cumulative = _read(scale_dir / "circuit" / "cumulative_patch.summary.json")
    ablation = _read(scale_dir / "ablation" / "ablation_heldout.summary.json")
    induction = _read(scale_dir / "circuit" / "induction.summary.json")
    if None in (mechanism, first_pass, cumulative, ablation, induction):
        row["status"] = "mechanism_incomplete"
        return row
    identity = first_pass["identity_gate"]
    row.update(
        {
            "status": "complete",
            "retention": {
                "failure_score": mechanism["probe"]["within_stale_true_current_score"],
                "shuffle95": mechanism["probe"]["value_label_shuffle95"],
                "failure_minus_correct": mechanism["probe"]["within_vs_correct"][
                    "length_controlled_delta"
                ],
                "ci": mechanism["probe"]["within_vs_correct"]["length_controlled_ci"],
            },
            "circuit": {
                "total_heads": total_heads,
                "selected_count": cumulative["selected_count"],
                "selected_fraction": cumulative["selected_count"] / total_heads,
                "all_head_recovery": cumulative["all_head_joint"],
                "selected_recovery": cumulative["selected_recovery"],
                "identity_gate": identity,
                "identity_pass": bool(
                    float(identity["max_abs_gap_change"]) == 0.0
                    and int(identity["prediction_changes"]) == 0
                ),
            },
            "ablation": {
                "effect": ablation["targeted_correct_rate_increase"],
                "random": ablation["random_effect"],
                "targeted_minus_random_mean": ablation["targeted_minus_random_mean"],
                "exceeds_random95": ablation["heldout_exceeds_random95"],
            },
            "induction": summarize_induction_overlap(induction),
        }
    )
    return row


def _fmt(value):
    return "--" if value is None or not np.isfinite(value) else f"{value:.3f}"


def aggregate(args) -> None:
    root = Path(args.root)
    scales = []
    canonical_behavior = _read(args.reference_behavior)
    canonical_mechanism = _read(args.reference_mechanism)
    canonical_patch = _read(args.reference_patch)
    canonical_ablation = _read(args.reference_ablation)
    canonical_induction = _read(args.reference_induction)
    identity = _read(args.reference_first_pass)["identity_gate"]
    scales.append(
        {
            "scale": "160m",
            "model": "EleutherAI/pythia-160m",
            "status": "complete",
            "behavior": {"by_k_single": canonical_behavior["by_k"]},
            "retention": {
                "failure_score": canonical_mechanism["probe"]["within_stale_true_current_score"],
                "shuffle95": canonical_mechanism["probe"]["value_label_shuffle95"],
                "failure_minus_correct": canonical_mechanism["probe"]["within_vs_correct"][
                    "length_controlled_delta"
                ],
                "ci": canonical_mechanism["probe"]["within_vs_correct"][
                    "length_controlled_ci"
                ],
            },
            "circuit": {
                "total_heads": 144,
                "selected_count": canonical_patch["selected_count"],
                "selected_fraction": canonical_patch["selected_count"] / 144,
                "all_head_recovery": canonical_patch["all_head_joint"],
                "selected_recovery": canonical_patch["selected_recovery"],
                "identity_gate": identity,
                "identity_pass": identity["max_abs_gap_change"] == 0.0
                and identity["prediction_changes"] == 0,
            },
            "ablation": {
                "effect": canonical_ablation["stale_promoting"]["paired_effect"],
                "random": canonical_ablation["stale_promoting"]["random_effect"],
                "targeted_minus_random_mean": (
                    canonical_ablation["stale_promoting"]["paired_effect"]["mean"]
                    - canonical_ablation["stale_promoting"]["random_effect"]["mean"]
                ),
                "exceeds_random95": canonical_ablation["stale_promoting"][
                    "heldout_exceeds_random95"
                ],
            },
            "induction": summarize_induction_overlap(canonical_induction),
        }
    )
    for key, model, total_heads in SCALES[1:]:
        scales.append(_scale_result(root, key, model, total_heads))
    scales.append(
        {
            "scale": "7B",
            "model": "Qwen2.5-7B anchor",
            "status": "anchor_only",
            "retention": {"failure_minus_correct": -0.075},
            "scope": "Retention-vs-selection anchor only; no enumerable head circuit.",
        }
    )
    complete_deltas = [
        row["retention"]["failure_minus_correct"]
        for row in scales
        if row["status"] in {"complete", "anchor_only"}
    ]
    monotone_toward_zero = all(
        abs(right) <= abs(left) + 1e-12
        for left, right in zip(complete_deltas, complete_deltas[1:])
    )
    payload = {
        "stage": "O-Part-A",
        "scale_axis": scales,
        "retention_composition_monotone_toward_7b": monotone_toward_zero,
        "claim_boundary": (
            "A within-family scale bridge for failure composition; not conservation of "
            "head identity and not a prevalence estimate."
        ),
    }
    dump_json(args.summary, payload)
    lines = [
        "# Stage O Part A: cross-scale mechanism bridge",
        "",
        "## Claim boundary",
        "",
        payload["claim_boundary"],
        "",
        "## Scale axis",
        "",
        "| Scale | Status | Retention delta | Compact core | Core fraction | Held-out ablation | Target-random | Induction top-decile |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scales:
        retention = row.get("retention", {})
        circuit = row.get("circuit", {})
        ablation = row.get("ablation", {})
        induction = row.get("induction", {})
        lines.append(
            f"| {row['scale']} | {row['status']} | "
            f"{_fmt(retention.get('failure_minus_correct'))} | "
            f"{circuit.get('selected_count', '--')} | "
            f"{_fmt(circuit.get('selected_fraction'))} | "
            f"{_fmt(ablation.get('effect', {}).get('mean'))} | "
            f"{_fmt(ablation.get('targeted_minus_random_mean'))} | "
            f"{_fmt(induction.get('top_decile_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "Retention delta is the pooled log-length-controlled failure-minus-correct "
            "current-value probe score. Compact core is the smallest discovery-ranked "
            "prefix reaching 80% of held-out all-head recovery. Ablation is ranked on "
            "discovery and evaluated on held-out failures against 64 layer-matched sets.",
            "",
            "## Pre-registered reading",
            "",
            (
                "The available retention deltas move monotonically toward the 7B anchor."
                if monotone_toward_zero
                else "The retention-composition trend is non-monotone; the scale-law reading fails."
            ),
            "",
            "Sizes labeled `substrate_infeasible_at_cap` are behavioral results, not missing "
            "mechanism results. No vocabulary or prompt semantics were changed to force a pool.",
            "",
            "## Per-scale behavior",
            "",
        ]
    )
    for row in scales[1:4]:
        behavior = row.get("behavior")
        if not behavior:
            continue
        lines.append(f"### {row['scale']}")
        lines.append("")
        lines.append("| k | n | Accuracy | Correct | Within-stale | Cross-variable | Other |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for k, cell in behavior["by_k_single"].items():
            counts = cell["counts"]
            lines.append(
                f"| {k} | {cell['n']} | {cell['accuracy']:.3f} | "
                f"{counts.get(CORRECT, 0)} | {counts.get(STALE, 0)} | "
                f"{counts.get('cross_variable', 0)} | {counts.get('other', 0)} |"
            )
        lines.append("")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": args.summary, "report": args.report}, indent=2))


def self_test(_args) -> None:
    summary = summarize_induction_overlap(
        {"circuit_heads": [{"percentile_among_heads": 0.95}]}
    )
    assert summary["top_decile_fraction"] == 1.0
    print("pythia_crossscale self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)

    generate = sub.add_parser("generate-data")
    generate.add_argument("--model", default="EleutherAI/pythia-410m")
    generate.add_argument("--out", required=True)
    generate.add_argument("--summary", required=True)
    generate.add_argument(
        "--seeds", type=lambda value: [int(x) for x in value.split(",")], default=[11, 29, 47]
    )
    generate.add_argument(
        "--k-values",
        type=lambda value: [int(x) for x in value.split(",")],
        default=[0, 2, 4, 6, 8, 10, 12, 14],
    )
    generate.add_argument(
        "--templates", type=lambda value: value.split(","), default=["arrow", "current", "latest"]
    )
    generate.add_argument("--n-per-seed-cell", type=int, default=72)
    generate.add_argument("--n-lines", type=int, default=15)
    generate.add_argument("--max-frozen-k", type=int, default=10)
    generate.set_defaults(func=generate_data)

    behavior = sub.add_parser("adjudicate-behavior")
    behavior.add_argument("--rows", required=True)
    behavior.add_argument("--summary", required=True)
    behavior.add_argument("--pool", required=True)
    behavior.add_argument("--model", required=True)
    behavior.add_argument("--seed", type=int, default=20260818)
    behavior.add_argument("--per-stratum-cap", type=int, default=48)
    behavior.set_defaults(func=adjudicate_behavior)

    ablate = sub.add_parser("ablate-heldout")
    ablate.add_argument("--pool", required=True)
    ablate.add_argument("--harvest-dir", required=True)
    ablate.add_argument("--cumulative-summary", required=True)
    ablate.add_argument("--out-dir", required=True)
    ablate.add_argument("--model", required=True)
    ablate.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    ablate.add_argument("--batch-size", type=int, default=32)
    ablate.add_argument("--max-heads", type=int, default=12)
    ablate.add_argument("--n-random", type=int, default=64)
    ablate.add_argument("--n-boot", type=int, default=2000)
    ablate.add_argument("--seed", type=int, default=20260818)
    ablate.set_defaults(func=ablate_heldout)

    cache = sub.add_parser("cache-model")
    cache.add_argument("--model", required=True)
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--summary", required=True)
    cache.set_defaults(func=cache_model)

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--root", default="results/stage_o")
    aggregate_parser.add_argument("--summary", required=True)
    aggregate_parser.add_argument("--report", required=True)
    aggregate_parser.add_argument(
        "--reference-behavior", default="results/stage_n/behavior_full_cpu/summary.json"
    )
    aggregate_parser.add_argument(
        "--reference-mechanism",
        default="results/stage_n/mechanism_full_cpu/mechanism.summary.json",
    )
    aggregate_parser.add_argument(
        "--reference-first-pass",
        default="results/stage_n/mechanism_full_core_ai32/path_patch_first_pass.summary.json",
    )
    aggregate_parser.add_argument(
        "--reference-patch",
        default="results/stage_n/mechanism_full_core_ai32/cumulative_patch.summary.json",
    )
    aggregate_parser.add_argument(
        "--reference-ablation",
        default="results/stage_n/phase1b/a_full_cpu/ablation_heldout.summary.json",
    )
    aggregate_parser.add_argument(
        "--reference-induction",
        default="results/stage_n/mechanism_full_core_ai32/induction.summary.json",
    )
    aggregate_parser.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
