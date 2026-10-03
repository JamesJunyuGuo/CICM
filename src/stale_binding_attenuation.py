"""Stage R2: fresh-seed validation of causal-head attenuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pythia_circuit import stable_split
from pythia_eval import dump_json, load_model
from pythia_gen import dump_jsonl, load_jsonl
from qwen_small_circuit import QwenHeadOutputHook
from stale_binding_correction import (
    CORRECT,
    STALE,
    _random_summary,
    _record_predictions,
    load_heads,
    protocol_split,
    summarize_against_baseline,
)


DEFAULT_GAMMAS = (0.0, 0.25, 0.5, 0.75, 0.9)


def choose_gamma(rows: list[dict], minimum_preservation: float = 0.95) -> dict:
    eligible = [
        dict(row)
        for row in rows
        if float(row["correct_preservation"]) >= minimum_preservation
    ]
    if not eligible:
        raise ValueError("no gamma satisfies the calibration preservation constraint")
    return max(
        eligible,
        key=lambda row: (float(row["net_accuracy_gain"]), float(row["gamma"])),
    )


def choose_gamma_or_identity(
    rows: list[dict], minimum_preservation: float = 0.95
) -> tuple[dict, str]:
    """Return an eligible intervention, or an explicit no-op failed gate."""
    try:
        return choose_gamma(rows, minimum_preservation), "eligible"
    except ValueError:
        if not rows:
            raise ValueError("cannot construct identity fallback from an empty curve")
        baseline_accuracy = float(rows[0]["baseline_accuracy"])
        return (
            {
                "gamma": 1.0,
                "correction": 0.0,
                "correct_preservation": 1.0,
                "baseline_accuracy": baseline_accuracy,
                "arm_accuracy": baseline_accuracy,
                "net_accuracy_gain": 0.0,
            },
            "no_eligible_gamma_identity_fallback",
        )


def make_layer_matched_random_sets(
    selected: list[tuple[int, int]],
    *,
    n_heads: int,
    n_random: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    counts = Counter(layer for layer, _ in selected)
    target = tuple(sorted((int(layer), int(head)) for layer, head in selected))
    rng = np.random.default_rng(seed)
    out = []
    seen = {target}
    attempts = 0
    while len(out) < n_random:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not sample enough unique layer-matched head sets")
        sample = []
        for layer, count in sorted(counts.items()):
            heads = rng.choice(n_heads, size=count, replace=False)
            sample.extend((int(layer), int(head)) for head in heads)
        key = tuple(sorted(sample))
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def transition_counts(baseline: list[dict], arm: list[dict]) -> dict[str, int]:
    if [row["id"] for row in baseline] != [row["id"] for row in arm]:
        raise ValueError("transition rows are not aligned")
    counts = Counter(
        f"{before['label']}->{after['label']}"
        for before, after in zip(baseline, arm)
    )
    return dict(sorted(counts.items()))


def template_support_gate(per_template: dict[str, dict]) -> bool:
    return sum(float(row["net_accuracy_gain"]) > 0.0 for row in per_template.values()) >= 2


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clustered_paired_bootstrap(values, groups, n_boot: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if len(values) != len(groups) or not len(values):
        raise ValueError("cluster bootstrap needs aligned nonempty values and groups")
    unique = np.unique(groups)
    group_means = np.asarray([values[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.integers(0, len(group_means), size=len(group_means))
        draws[index] = group_means[sampled].mean()
    return {
        "mean": float(group_means.mean()),
        "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n": int(len(values)),
        "n_clusters": int(len(unique)),
        "cluster": "semantic_id",
    }


def _query_batch(tokenizer, device, rows):
    enc = tokenizer(
        [row["prompt"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    query = enc["attention_mask"].sum(dim=1) - 1
    return enc, query


def evaluate_scale(
    model,
    tokenizer,
    device,
    rows,
    selected,
    gamma,
    batch_size,
    *,
    return_logits=False,
):
    import torch

    records = []
    query_logits = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc, query = _query_batch(tokenizer, device, batch_rows)
        with QwenHeadOutputHook(model, query, selected, gamma), torch.no_grad():
            output = model(**enc, use_cache=False, return_dict=True)
        batch = torch.arange(len(batch_rows), device=device)
        logits = output.logits[batch, query]
        records.extend(_record_predictions(batch_rows, logits, tokenizer))
        if return_logits:
            query_logits.append(logits.detach().float().cpu().numpy())
    logits = np.concatenate(query_logits) if return_logits else None
    return records, logits


def _stable_calibration(rows, records):
    kept_rows = []
    kept_records = []
    mismatches = []
    for row, record in zip(rows, records):
        if row["label"] == record["label"]:
            kept_rows.append(row)
            kept_records.append(record)
        else:
            mismatches.append(
                {"id": row["id"], "stored": row["label"], "rerun": record["label"]}
            )
    return kept_rows, kept_records, mismatches


def _limited_calibration(rows, per_label):
    if not per_label:
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        label = row["label"]
        if counts[label] >= per_label:
            continue
        counts[label] += 1
        selected.append(row)
    return selected


def _limited_confirmation(rows, per_group):
    if not per_group:
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        key = (int(row["k"]), row["template"])
        if counts[key] >= per_group:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def _curve(
    model,
    tokenizer,
    device,
    rows,
    baseline,
    heads,
    gammas,
    batch_size,
):
    curve = []
    for gamma in gammas:
        arm, _ = evaluate_scale(
            model, tokenizer, device, rows, heads, gamma, batch_size
        )
        metric = summarize_against_baseline(baseline, arm)
        metric.pop("paired_gain")
        curve.append(
            {
                "gamma": float(gamma),
                **{
                    key: metric[key]
                    for key in (
                        "correction",
                        "correct_preservation",
                        "baseline_accuracy",
                        "arm_accuracy",
                        "net_accuracy_gain",
                    )
                },
            }
        )
    return curve


def _metrics_with_ci(rows, baseline, arm, n_boot, seed):
    metric = summarize_against_baseline(baseline, arm)
    paired = np.asarray(metric.pop("paired_gain"), dtype=np.float64)
    metric["paired_net_gain"] = clustered_paired_bootstrap(
        paired, [row["semantic_id"] for row in rows], n_boot, seed
    )
    metric["transitions"] = transition_counts(baseline, arm)
    return metric


def _per_template(rows, baseline, arm):
    output = {}
    for template in sorted({row["template"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["template"] == template]
        metric = summarize_against_baseline(
            [baseline[i] for i in indices], [arm[i] for i in indices]
        )
        metric.pop("paired_gain")
        output[template] = metric
    return output


def _random_controls(
    model,
    tokenizer,
    device,
    rows,
    baseline,
    random_sets,
    gamma,
    batch_size,
):
    records = []
    for index, heads in enumerate(random_sets):
        arm, _ = evaluate_scale(
            model, tokenizer, device, rows, heads, gamma, batch_size
        )
        metric = summarize_against_baseline(baseline, arm)
        metric.pop("paired_gain")
        records.append(
            {
                "random_index": index,
                "heads": [{"layer": layer, "head": head} for layer, head in heads],
                **metric,
            }
        )
        print(f"random set {index + 1}/{len(random_sets)}", flush=True)
    return records


def _summarize_random(records):
    return {
        "correction": _random_summary([row["correction"] for row in records]),
        "net_accuracy_gain": _random_summary(
            [row["net_accuracy_gain"] for row in records]
        ),
        "correct_preservation": _random_summary(
            [row["correct_preservation"] for row in records]
        ),
    }


def _skipped_identity_random_summary(n_requested: int) -> dict:
    return {
        "status": "not_run_identity_fallback",
        "n_requested": int(n_requested),
        "n_observed": 0,
        "correction": _random_summary([0.0]),
        "net_accuracy_gain": _random_summary([0.0]),
        "correct_preservation": _random_summary([1.0]),
    }


def run(args):
    calibration = [
        row
        for row in load_jsonl(args.calibration_pool)
        if protocol_split(row["semantic_id"], stable_split(row["semantic_id"]))
        == "calibration"
    ]
    calibration = _limited_calibration(calibration, args.smoke_per_group)
    confirmation = _limited_confirmation(
        load_jsonl(args.confirmation_rows), args.smoke_per_group
    )
    k6_rows = [row for row in confirmation if int(row["k"]) == 6]
    k0_rows = [row for row in confirmation if int(row["k"]) == 0]
    if not calibration or not k6_rows or not k0_rows:
        raise ValueError("calibration and both confirmation k cells must be nonempty")

    heads = load_heads(args.head_summary)
    model, tokenizer, device = load_model(args.model, args.dtype)
    n_heads = int(model.config.num_attention_heads)

    calibration_base, _ = evaluate_scale(
        model, tokenizer, device, calibration, [], 1.0, args.batch_size
    )
    calibration, calibration_base, calibration_mismatches = _stable_calibration(
        calibration, calibration_base
    )
    calibration_counts = Counter(row["label"] for row in calibration)
    if min(calibration_counts[CORRECT], calibration_counts[STALE]) < 2:
        raise RuntimeError(f"calibration lost a class: {calibration_counts}")

    calibration_curve = _curve(
        model,
        tokenizer,
        device,
        calibration,
        calibration_base,
        heads,
        args.gammas,
        args.batch_size,
    )
    operating, operating_status = choose_gamma_or_identity(
        calibration_curve, args.minimum_preservation
    )
    gamma = float(operating["gamma"])

    baseline, baseline_logits = evaluate_scale(
        model,
        tokenizer,
        device,
        k6_rows,
        [],
        1.0,
        args.batch_size,
        return_logits=True,
    )
    identity, identity_logits = evaluate_scale(
        model,
        tokenizer,
        device,
        k6_rows,
        heads,
        1.0,
        args.batch_size,
        return_logits=True,
    )
    identity_max = float(np.max(np.abs(identity_logits - baseline_logits)))
    identity_changes = sum(
        before["pred_token_id"] != after["pred_token_id"]
        for before, after in zip(baseline, identity)
    )
    if identity_max != 0.0 or identity_changes:
        raise RuntimeError(
            f"gamma=1 identity failure: max logit={identity_max}, predictions={identity_changes}"
        )

    opposite_gamma = 2.0 - gamma
    if operating_status == "eligible":
        targeted, _ = evaluate_scale(
            model, tokenizer, device, k6_rows, heads, gamma, args.batch_size
        )
        opposite, _ = evaluate_scale(
            model,
            tokenizer,
            device,
            k6_rows,
            heads,
            opposite_gamma,
            args.batch_size,
        )
    else:
        targeted = identity
        opposite = identity
    target_metric = _metrics_with_ci(
        k6_rows, baseline, targeted, args.n_boot, args.seed + 1000
    )
    opposite_metric = _metrics_with_ci(
        k6_rows, baseline, opposite, args.n_boot, args.seed + 2000
    )
    per_template = _per_template(k6_rows, baseline, targeted)

    random_sets = make_layer_matched_random_sets(
        heads, n_heads=n_heads, n_random=args.n_random, seed=args.seed + 3000
    )
    if operating_status == "eligible":
        random_records = _random_controls(
            model,
            tokenizer,
            device,
            k6_rows,
            baseline,
            random_sets,
            gamma,
            args.batch_size,
        )
        random_summary = _summarize_random(random_records)
    else:
        random_records = []
        random_summary = _skipped_identity_random_summary(args.n_random)

    k0_base, _ = evaluate_scale(
        model, tokenizer, device, k0_rows, [], 1.0, args.batch_size
    )
    if operating_status == "eligible":
        k0_target, _ = evaluate_scale(
            model, tokenizer, device, k0_rows, heads, gamma, args.batch_size
        )
    else:
        k0_target = k0_base
    k0_metric = _metrics_with_ci(
        k0_rows, k0_base, k0_target, args.n_boot, args.seed + 4000
    )

    upstream_heads = [head for head in heads if head[0] <= args.upstream_max_layer]
    upstream_curve = _curve(
        model,
        tokenizer,
        device,
        calibration,
        calibration_base,
        upstream_heads,
        args.gammas,
        args.batch_size,
    )
    upstream_operating, upstream_operating_status = choose_gamma_or_identity(
        upstream_curve, args.minimum_preservation
    )
    upstream_gamma = float(upstream_operating["gamma"])
    if upstream_operating_status == "eligible":
        upstream_target, _ = evaluate_scale(
            model,
            tokenizer,
            device,
            k6_rows,
            upstream_heads,
            upstream_gamma,
            args.batch_size,
        )
    else:
        upstream_target = baseline
    upstream_metric = _metrics_with_ci(
        k6_rows, baseline, upstream_target, args.n_boot, args.seed + 5000
    )
    upstream_per_template = _per_template(k6_rows, baseline, upstream_target)
    upstream_random_sets = make_layer_matched_random_sets(
        upstream_heads,
        n_heads=n_heads,
        n_random=args.n_random,
        seed=args.seed + 6000,
    )
    if upstream_operating_status == "eligible":
        upstream_random_records = _random_controls(
            model,
            tokenizer,
            device,
            k6_rows,
            baseline,
            upstream_random_sets,
            upstream_gamma,
            args.batch_size,
        )
        upstream_random_summary = _summarize_random(upstream_random_records)
        upstream_k0, _ = evaluate_scale(
            model,
            tokenizer,
            device,
            k0_rows,
            upstream_heads,
            upstream_gamma,
            args.batch_size,
        )
    else:
        upstream_random_records = []
        upstream_random_summary = _skipped_identity_random_summary(args.n_random)
        upstream_k0 = k0_base
    upstream_k0_metric = _metrics_with_ci(
        k0_rows, k0_base, upstream_k0, args.n_boot, args.seed + 7000
    )

    output_gate = {
        "eligible_calibration_gamma": operating_status == "eligible",
        "identity_exact_zero": identity_max == 0.0 and identity_changes == 0,
        "targeted_gain_ci_above_zero": (
            target_metric["paired_net_gain"]["ci"][0] > 0.0
        ),
        "net_gain_exceeds_random95": (
            target_metric["net_accuracy_gain"]
            > random_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "correct_preservation": target_metric["correct_preservation"] >= 0.95,
        "no_overwrite_drop_le_002": k0_metric["net_accuracy_gain"] >= -0.02,
        "beats_opposite": (
            target_metric["net_accuracy_gain"] > opposite_metric["net_accuracy_gain"]
        ),
        "multi_template_support": template_support_gate(per_template),
    }
    output_gate["all_pass"] = all(output_gate.values())
    upstream_gate = {
        "eligible_calibration_gamma": upstream_operating_status == "eligible",
        "gain_ci_above_zero": upstream_metric["paired_net_gain"]["ci"][0] > 0.0,
        "net_gain_exceeds_random95": (
            upstream_metric["net_accuracy_gain"]
            > upstream_random_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "correct_preservation": upstream_metric["correct_preservation"] >= 0.95,
        "no_overwrite_drop_le_002": upstream_k0_metric["net_accuracy_gain"] >= -0.02,
        "multi_template_support": template_support_gate(upstream_per_template),
    }
    upstream_gate["all_pass"] = all(upstream_gate.values())
    if output_gate["all_pass"]:
        verdict = (
            "limited_diagnosis_guided_head_attenuation_nonfinal_supported"
            if upstream_gate["all_pass"]
            else "limited_diagnosis_guided_head_attenuation_late_readout_only"
        )
    else:
        verdict = "attenuation_correction_gate_failed"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "baseline_k6_rows.jsonl", baseline)
    dump_jsonl(out_dir / "targeted_k6_rows.jsonl", targeted)
    dump_jsonl(out_dir / "baseline_k0_rows.jsonl", k0_base)
    dump_jsonl(out_dir / "targeted_k0_rows.jsonl", k0_target)
    dump_jsonl(out_dir / "random_controls.jsonl", random_records)
    dump_jsonl(out_dir / "upstream_random_controls.jsonl", upstream_random_records)
    summary = {
        "stage": "R2-fresh-seed-causal-head-attenuation",
        "model": args.model,
        "dtype": args.dtype,
        "confirmation_rows": args.confirmation_rows,
        "confirmation_sha256": file_sha256(args.confirmation_rows),
        "runtime_inputs": "fixed head set and global gamma only; no answer values or spans",
        "frozen_heads": [{"layer": layer, "head": head} for layer, head in heads],
        "calibration": {
            "requested": len(calibration) + len(calibration_mismatches),
            "retained": len(calibration),
            "mismatches": calibration_mismatches,
            "counts": dict(calibration_counts),
            "curve": calibration_curve,
            "operating_point": operating,
            "operating_status": operating_status,
        },
        "confirmation": {
            "k6_n": len(k6_rows),
            "k0_n": len(k0_rows),
            "baseline_k6_counts": dict(Counter(row["label"] for row in baseline)),
            "baseline_k0_counts": dict(Counter(row["label"] for row in k0_base)),
            "targeted": target_metric,
            "opposite_gamma": opposite_gamma,
            "opposite": opposite_metric,
            "per_template": per_template,
            "random": random_summary,
            "no_overwrite": k0_metric,
            "identity_max_abs_logit_change": identity_max,
            "identity_prediction_changes": identity_changes,
        },
        "upstream_confirmation": {
            "max_layer": args.upstream_max_layer,
            "heads": [
                {"layer": layer, "head": head} for layer, head in upstream_heads
            ],
            "curve": upstream_curve,
            "operating_point": upstream_operating,
            "operating_status": upstream_operating_status,
            "targeted": upstream_metric,
            "per_template": upstream_per_template,
            "random": upstream_random_summary,
            "no_overwrite": upstream_k0_metric,
            "gate": upstream_gate,
        },
        "output_gate": output_gate,
        "verdict": verdict,
        "scope": "fresh-seed controlled Qwen2.5-1.5B test; not adaptive or deployment-ready",
    }
    dump_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "output_gate": output_gate, "upstream_gate": upstream_gate}, indent=2))


def write_report(path: str | Path, summary: dict) -> None:
    confirmation = summary["confirmation"]
    target = confirmation["targeted"]
    random_gain = confirmation["random"]["net_accuracy_gain"]
    upstream = summary["upstream_confirmation"]
    lines = [
        "# Stage R2: Fresh-seed causal-head attenuation",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        "The intervention attenuates a discovery-frozen 12-head set and receives no "
        "answer value, token id, or write span at inference time.",
        "",
        "## Fresh confirmation",
        "",
        "| k=6 n | gamma | stale correction | correct preservation | net gain | paired 95% CI | random net-gain 95% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {confirmation['k6_n']} | {summary['calibration']['operating_point']['gamma']:.2f} | "
        f"{target['correction']:.3f} | {target['correct_preservation']:.3f} | "
        f"{target['net_accuracy_gain']:.3f} | {target['paired_net_gain']['ci']} | "
        f"{random_gain['interval95']} |",
        "",
        f"Fresh k=0 net change: {confirmation['no_overwrite']['net_accuracy_gain']:.3f}. "
        f"Gamma=1 max logit change: {confirmation['identity_max_abs_logit_change']:.1f}.",
        "",
        "## Per-template net gain",
        "",
    ]
    for template, metric in confirmation["per_template"].items():
        lines.append(f"- {template}: {metric['net_accuracy_gain']:.3f}")
    lines.extend(["", "## Output gates", ""])
    lines.extend(
        f"- {key}: **{'PASS' if value else 'FAIL'}**"
        for key, value in summary["output_gate"].items()
    )
    lines.extend(
        [
            "",
            "## Pre-final qualifier",
            "",
            f"Using {len(upstream['heads'])} heads at layers <= {upstream['max_layer']}, "
            f"net gain was {upstream['targeted']['net_accuracy_gain']:.3f} "
            f"(95% CI {upstream['targeted']['paired_net_gain']['ci']}).",
            "",
        ]
    )
    lines.extend(
        f"- {key}: **{'PASS' if value else 'FAIL'}**"
        for key, value in upstream["gate"].items()
    )
    lines.extend(["", "## Scope", "", summary["scope"]])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--calibration-pool",
        default="results/pythia_crossscale/partC_qwen15b/matched_pool.jsonl",
    )
    parser.add_argument(
        "--head-summary",
        default="results/pythia_crossscale/partC_qwen15b/ablation/ablation_heldout.summary.json",
    )
    parser.add_argument(
        "--confirmation-rows",
        default="results/attention_rerouting/attenuation/confirmation_tasks.jsonl",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--gammas", type=parse_floats, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--n-random", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upstream-max-layer", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--smoke-per-group", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
