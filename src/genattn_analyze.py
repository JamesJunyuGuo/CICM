"""Analyze Stage P event-aligned trajectories with grouped, residualized tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from genattn_gen import CHECKPOINTS, dump_jsonl, load_jsonl
from genattn_capture import dump_json


CORRECT = "correct_current"
STALE = "within_stale"


def stable_split(identifier: str) -> str:
    value = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "discovery" if value < 0.45 else "evaluation"


def residualize(values: np.ndarray, lengths: np.ndarray, distances: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    design = np.column_stack([
        np.ones(len(values)),
        np.log(np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)),
        np.log1p(np.maximum(np.asarray(distances, dtype=np.float64), 0.0)),
    ])
    finite = np.all(np.isfinite(flat), axis=1) & np.all(np.isfinite(design), axis=1)
    out = np.full_like(flat, np.nan)
    if finite.sum() >= design.shape[1] + 2:
        coefficients, *_ = np.linalg.lstsq(design[finite], flat[finite], rcond=None)
        out[finite] = flat[finite] - design[finite] @ coefficients
    return out.reshape(values.shape)


def _group_delta(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.nanmean(values[labels == STALE], axis=0) - np.nanmean(values[labels == CORRECT], axis=0)


def _shuffle_item_labels(labels: np.ndarray, rows: list[dict], rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["template"], int(row["seed"]))].append(index)
    for indices in strata.values():
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def _bootstrap_delta(values: np.ndarray, labels: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[float]:
    correct = np.flatnonzero(labels == CORRECT)
    stale = np.flatnonzero(labels == STALE)
    samples = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        c = rng.choice(correct, len(correct), replace=True)
        s = rng.choice(stale, len(stale), replace=True)
        samples[iteration] = np.nanmean(values[s]) - np.nanmean(values[c])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[float]:
    indices = np.flatnonzero(np.isfinite(values))
    samples = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        selected = rng.choice(indices, len(indices), replace=True)
        samples[iteration] = np.mean(values[selected])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def scalar_comparison(values: np.ndarray, labels: np.ndarray, rows: list[dict], n_shuffle: int, n_boot: int, seed: int) -> dict:
    observed = float(_group_delta(values, labels))
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        null[iteration] = float(_group_delta(values, _shuffle_item_labels(labels, rows, rng)))
    p_lower = float((1 + np.sum(null <= observed)) / (n_shuffle + 1))
    p_upper = float((1 + np.sum(null >= observed)) / (n_shuffle + 1))
    return {
        "correct_mean": float(np.nanmean(values[labels == CORRECT])),
        "stale_mean": float(np.nanmean(values[labels == STALE])),
        "correct_bootstrap95": _bootstrap_mean_ci(values[labels == CORRECT], n_boot, np.random.default_rng(seed + 200_000)),
        "stale_bootstrap95": _bootstrap_mean_ci(values[labels == STALE], n_boot, np.random.default_rng(seed + 300_000)),
        "stale_minus_correct": observed,
        "bootstrap95": _bootstrap_delta(values, labels, n_boot, np.random.default_rng(seed + 100_000)),
        "shuffle95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "shuffle_p_lower": p_lower,
        "shuffle_p_upper": p_upper,
        "n_shuffle": n_shuffle,
    }


def head_comparison(values: np.ndarray, labels: np.ndarray, rows: list[dict], n_shuffle: int, seed: int) -> dict:
    observed = _group_delta(values, labels)
    rng = np.random.default_rng(seed)
    null_min = np.empty(n_shuffle, dtype=np.float64)
    null_max = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        delta = _group_delta(values, _shuffle_item_labels(labels, rows, rng))
        null_min[iteration] = np.nanmin(delta)
        null_max[iteration] = np.nanmax(delta)
    lower = float(np.quantile(null_min, 0.025))
    upper = float(np.quantile(null_max, 0.975))
    significant_lower = np.argwhere(observed < lower)
    significant_upper = np.argwhere(observed > upper)
    return {
        "familywise_lower95": lower,
        "familywise_upper95": upper,
        "significant_stale_favoring": [
            {"layer": int(layer), "head": int(head), "delta": float(observed[layer, head])}
            for layer, head in significant_lower
        ],
        "significant_current_favoring": [
            {"layer": int(layer), "head": int(head), "delta": float(observed[layer, head])}
            for layer, head in significant_upper
        ],
        "all_head_delta": observed.tolist(),
    }


def paired_scalar_comparison(overwrite: np.ndarray, control: np.ndarray, n_shuffle: int, n_boot: int, seed: int) -> dict:
    overwrite = np.asarray(overwrite, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    difference = overwrite - control
    observed = float(np.nanmean(difference))
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        signs = rng.choice((-1.0, 1.0), size=len(difference))
        null[iteration] = float(np.nanmean(difference * signs))
    boot_rng = np.random.default_rng(seed + 100_000)
    boot = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        selected = boot_rng.integers(0, len(difference), size=len(difference))
        boot[iteration] = float(np.nanmean(difference[selected]))
    return {
        "overwrite_mean": float(np.nanmean(overwrite)),
        "control_mean": float(np.nanmean(control)),
        "overwrite_minus_control": observed,
        "paired_bootstrap95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "signflip95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "signflip_p_lower": float((1 + np.sum(null <= observed)) / (n_shuffle + 1)),
        "signflip_p_upper": float((1 + np.sum(null >= observed)) / (n_shuffle + 1)),
        "n": len(difference),
        "n_shuffle": n_shuffle,
    }


def paired_head_comparison(overwrite: np.ndarray, control: np.ndarray, n_shuffle: int, seed: int) -> dict:
    difference = np.asarray(overwrite, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    observed = np.nanmean(difference, axis=0)
    rng = np.random.default_rng(seed)
    null_min = np.empty(n_shuffle, dtype=np.float64)
    null_max = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        signs = rng.choice((-1.0, 1.0), size=(len(difference), 1, 1))
        delta = np.nanmean(difference * signs, axis=0)
        null_min[iteration] = np.nanmin(delta)
        null_max[iteration] = np.nanmax(delta)
    lower = float(np.quantile(null_min, 0.025))
    upper = float(np.quantile(null_max, 0.975))
    return {
        "paired_delta_per_head": observed.tolist(),
        "familywise_lower95": lower,
        "familywise_upper95": upper,
        "significant_overwrite_lower": [
            {"layer": int(layer), "head": int(head), "delta": float(observed[layer, head])}
            for layer, head in np.argwhere(observed < lower)
        ],
        "significant_overwrite_upper": [
            {"layer": int(layer), "head": int(head), "delta": float(observed[layer, head])}
            for layer, head in np.argwhere(observed > upper)
        ],
        "n_shuffle": n_shuffle,
    }


def fit_retention_probe(hidden: np.ndarray, rows: list[dict], n_shuffle: int, seed: int) -> tuple[dict, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x = hidden[:, -1].astype(np.float32)
    values = np.asarray([row["gold"] for row in rows], dtype=object)
    groups = np.asarray([row["semantic_id"] for row in rows], dtype=object)
    encoder = LabelEncoder()
    y = encoder.fit_transform(values)
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    probabilities = np.full((len(rows), len(encoder.classes_)), np.nan, dtype=np.float64)
    predictions = np.full(len(rows), -1, dtype=np.int64)
    for train, test in splitter.split(x, y, groups):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train])
        test_x = scaler.transform(x[test])
        classifier = LogisticRegression(
            C=1.0, penalty="l2", max_iter=400, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )
        classifier.fit(train_x, y[train])
        fold_probabilities = classifier.predict_proba(test_x)
        for local, class_id in enumerate(classifier.classes_):
            probabilities[test, class_id] = fold_probabilities[:, local]
        predictions[test] = classifier.predict(test_x)
    true_scores = probabilities[np.arange(len(rows)), y]
    final_outcomes = np.asarray([row["final_outcome"] for row in rows], dtype=object)
    stale_mask = final_outcomes == STALE
    observed = float(np.nanmean(true_scores[stale_mask]))
    unique_groups = sorted(set(groups.tolist()))
    group_value = {group: values[np.flatnonzero(groups == group)[0]] for group in unique_groups}
    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_shuffle, dtype=np.float64)
    class_map = {str(value): index for index, value in enumerate(encoder.classes_)}
    group_values = np.asarray([group_value[group] for group in unique_groups], dtype=object)
    for iteration in range(n_shuffle):
        shuffled_values = rng.permutation(group_values)
        mapping = dict(zip(unique_groups, shuffled_values))
        scores = []
        for row_index, group in enumerate(groups):
            class_index = class_map.get(str(mapping[group]))
            scores.append(probabilities[row_index, class_index] if class_index is not None else np.nan)
        null[iteration] = np.nanmean(np.asarray(scores)[stale_mask])
    summary = {
        "protocol": "L2 multinomial logistic regression on final diagnostic hidden state; 5-fold GroupKFold by semantic_id",
        "n": len(rows),
        "n_groups": len(unique_groups),
        "n_classes": len(encoder.classes_),
        "cv_accuracy": float(accuracy_score(y[predictions >= 0], predictions[predictions >= 0])),
        "final_stale_mean_true_current_score": observed,
        "value_label_shuffle95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "above_shuffle95": bool(observed > np.quantile(null, 0.975)),
    }
    return summary, true_scores


def early_late_prediction(features: np.ndarray, labels: np.ndarray, rows: list[dict], n_shuffle: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = (labels == STALE).astype(np.int64)
    groups = np.asarray([row["semantic_id"] for row in rows], dtype=object)
    folds = min(5, len(np.unique(groups)))

    def cross_validated_score(target: np.ndarray) -> float:
        predictions = np.full(len(target), -1, dtype=np.int64)
        for train, test in GroupKFold(n_splits=folds).split(features, target, groups):
            if len(np.unique(target[train])) < 2:
                predictions[test] = int(np.mean(target[train]) >= 0.5)
                continue
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[train])
            test_x = scaler.transform(features[test])
            classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300, random_state=seed)
            classifier.fit(train_x, target[train])
            predictions[test] = classifier.predict(test_x)
        return float(balanced_accuracy_score(target, predictions))

    observed = cross_validated_score(y)
    rng = np.random.default_rng(seed + 77)
    null = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        null_y = _shuffle_item_labels(y, rows, rng).astype(np.int64)
        null[iteration] = cross_validated_score(null_y)
    return {
        "protocol": "tau2-tau4 residualized pooled R, pooled QK, and behavioral margins; 5-fold GroupKFold by semantic_id",
        "balanced_accuracy": observed,
        "shuffle95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "shuffle_p_upper": float((1 + np.sum(null >= observed)) / (n_shuffle + 1)),
        "above_shuffle95": bool(observed > np.quantile(null, 0.975)),
        "n": len(labels),
        "n_groups": len(np.unique(groups)),
        "n_features": int(features.shape[1]),
    }


def exact_behavior_pairs(final_rows: list[dict]) -> list[dict]:
    strata = defaultdict(lambda: defaultdict(list))
    for row in final_rows:
        if row["label"] not in {CORRECT, STALE}:
            continue
        key = (
            row["template"], int(row["seed"]), int(row["prefix_tokens"]),
            int(row["competitor_distance"]), int(row["current_distance"]),
        )
        strata[key][row["label"]].append(row)
    pairs = []
    for key in sorted(strata):
        correct = sorted(strata[key][CORRECT], key=lambda row: row["id"])
        stale = sorted(strata[key][STALE], key=lambda row: row["id"])
        for index in range(min(len(correct), len(stale))):
            pairs.append({
                "pair_id": f"p_match_{len(pairs):05d}",
                "stratum": list(key),
                "correct_id": correct[index]["id"],
                "stale_id": stale[index]["id"],
            })
    return pairs


def summarize_controls(rows_all: list[dict], branch_lookup: dict, item_ids: list[str], threshold: float):
    controls = {}
    accuracy_soft_pass = True
    length_gate = True
    for checkpoint in CHECKPOINTS:
        corrupt_rows = [rows_all[branch_lookup[(item_id, checkpoint, False)]] for item_id in item_ids]
        control_rows = [rows_all[branch_lookup[(item_id, checkpoint, True)]] for item_id in item_ids]
        exact_lengths = all(a["prefix_tokens"] == b["prefix_tokens"] for a, b in zip(corrupt_rows, control_rows))
        accuracy = float(np.mean([row["label"] == CORRECT for row in control_rows]))
        controls[checkpoint] = {
            "n": len(control_rows),
            "no_overwrite_accuracy": accuracy,
            "exact_token_length_match": exact_lengths,
            "label_counts": dict(Counter(row["label"] for row in control_rows)),
        }
        length_gate &= exact_lengths
        accuracy_soft_pass &= accuracy >= threshold
    return controls, bool(length_gate), bool(accuracy_soft_pass)


def unconditional_descriptive(root: Path, rows_all: list[dict], branch_lookup: dict, item_ids: list[str], args) -> dict:
    data = np.load(root / "capture.npz")
    if len(rows_all) != data["hidden"].shape[0]:
        raise ValueError("capture index and arrays differ in length")
    arrays = {key: data[key] for key in data.files}
    pool = [
        branch_lookup[(item_id, checkpoint, control)]
        for item_id in item_ids for control in (False, True) for checkpoint in CHECKPOINTS[1:]
    ]
    lengths = np.asarray([rows_all[index]["prefix_tokens"] for index in pool])
    distances = np.asarray([rows_all[index]["competitor_distance"] for index in pool])
    controlled = {}
    for key in ("read_ratio", "qk_margin"):
        values = residualize(arrays[key][pool], lengths, distances)
        controlled[key] = values.reshape(len(item_ids), 2, len(CHECKPOINTS) - 1, *values.shape[1:])
    behavior = np.asarray([rows_all[index]["behavioral_margin"] for index in pool], dtype=np.float64)
    controlled_behavior = residualize(behavior[:, None], lengths, distances).reshape(len(item_ids), 2, len(CHECKPOINTS) - 1)
    trajectories = {}
    for local, checkpoint in enumerate(CHECKPOINTS[1:]):
        overwrite_indices = [branch_lookup[(item_id, checkpoint, False)] for item_id in item_ids]
        control_indices = [branch_lookup[(item_id, checkpoint, True)] for item_id in item_ids]
        raw_behavior_overwrite = np.asarray([rows_all[index]["behavioral_margin"] for index in overwrite_indices])
        raw_behavior_control = np.asarray([rows_all[index]["behavioral_margin"] for index in control_indices])
        raw_r_overwrite = np.nanmedian(arrays["read_ratio"][overwrite_indices], axis=(1, 2))
        raw_r_control = np.nanmedian(arrays["read_ratio"][control_indices], axis=(1, 2))
        raw_qk_overwrite = np.nanmedian(arrays["qk_margin"][overwrite_indices], axis=(1, 2))
        raw_qk_control = np.nanmedian(arrays["qk_margin"][control_indices], axis=(1, 2))
        trajectories[checkpoint] = {
            "n_exact_item_pairs": len(item_ids),
            "behavioral_margin": {
                "raw": paired_scalar_comparison(raw_behavior_overwrite, raw_behavior_control, args.n_shuffle, args.n_boot, args.seed + 100 + local),
                "length_distance_controlled": paired_scalar_comparison(controlled_behavior[:, 0, local], controlled_behavior[:, 1, local], args.n_shuffle, args.n_boot, args.seed + 200 + local),
            },
            "read_ratio": {
                "raw": paired_scalar_comparison(raw_r_overwrite, raw_r_control, args.n_shuffle, args.n_boot, args.seed + 300 + local),
                "length_distance_controlled": paired_scalar_comparison(
                    np.nanmedian(controlled["read_ratio"][:, 0, local], axis=(1, 2)),
                    np.nanmedian(controlled["read_ratio"][:, 1, local], axis=(1, 2)),
                    args.n_shuffle, args.n_boot, args.seed + 400 + local,
                ),
                "head_raw": paired_head_comparison(arrays["read_ratio"][overwrite_indices], arrays["read_ratio"][control_indices], args.n_shuffle, args.seed + 500 + local),
            },
            "qk_margin": {
                "raw": paired_scalar_comparison(raw_qk_overwrite, raw_qk_control, args.n_shuffle, args.n_boot, args.seed + 600 + local),
                "length_distance_controlled": paired_scalar_comparison(
                    np.nanmedian(controlled["qk_margin"][:, 0, local], axis=(1, 2)),
                    np.nanmedian(controlled["qk_margin"][:, 1, local], axis=(1, 2)),
                    args.n_shuffle, args.n_boot, args.seed + 700 + local,
                ),
                "head_raw": paired_head_comparison(arrays["qk_margin"][overwrite_indices], arrays["qk_margin"][control_indices], args.n_shuffle, args.seed + 800 + local),
            },
        }
    retention = {}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        indices = [branch_lookup[(item_id, checkpoint, False)] for item_id in item_ids]
        rows = [{**rows_all[index], "final_outcome": STALE} for index in indices]
        probe, _scores = fit_retention_probe(arrays["hidden"][indices], rows, args.n_shuffle, args.seed + 30_000 + checkpoint_index)
        probe["population"] = "all overwrite-arm items; descriptive, not failure-conditioned"
        retention[checkpoint] = probe
    return {
        "scope": "exact same-item overwrite versus no-overwrite pairs over the full item pool; descriptive only",
        "residualization": "fit once over all full-pool overwrite/no-overwrite tau2-tauQ arms with log prefix length and log1p nearest comparator distance",
        "trajectories": trajectories,
        "retention": retention,
    }


def analyze(args) -> None:
    root = Path(args.capture_dir)
    rows_all = load_jsonl(root / "capture_index.jsonl")
    branch_lookup = {(row["id"], row["checkpoint"], bool(row["control"])): index for index, row in enumerate(rows_all)}
    item_ids = sorted({row["id"] for row in rows_all})
    final_rows = [rows_all[branch_lookup[(item_id, "tauQ", False)]] for item_id in item_ids]
    eligible_ids = []
    for item_id in item_ids:
        corrupt = rows_all[branch_lookup[(item_id, "tauQ", False)]]
        control = rows_all[branch_lookup[(item_id, "tauQ", True)]]
        if (
            corrupt["label"] == STALE
            and control["label"] == CORRECT
            and corrupt["prefix_tokens"] == control["prefix_tokens"]
        ):
            eligible_ids.append(item_id)
    controls, length_gate, control_accuracy_soft_pass = summarize_controls(
        rows_all, branch_lookup, item_ids, args.control_threshold
    )
    if len(eligible_ids) < args.min_exact_pairs:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        descriptive = unconditional_descriptive(root, rows_all, branch_lookup, item_ids, args)
        dump_jsonl(out_dir / "exact_behavior_pairs.jsonl", [])
        dump_jsonl(out_dir / "patch_pairs.jsonl", [])
        summary = {
            "stage": "P-analysis",
            "status": "insufficient_matched_pool",
            "model": args.model_label,
            "n_items": len(item_ids),
            "final_label_counts": dict(Counter(row["label"] for row in final_rows)),
            "eligible_exact_pair_n": len(eligible_ids),
            "min_exact_pairs": args.min_exact_pairs,
            "same_length_no_overwrite_controls": controls,
            "descriptive_full_pool": descriptive,
            "validity_gates": {
                "same_length_every_checkpoint": length_gate,
                "no_overwrite_accuracy_threshold": args.control_threshold,
                "no_overwrite_accuracy_soft_threshold_every_checkpoint": control_accuracy_soft_pass,
                "no_overwrite_accuracy_threshold_is_soft": True,
                "exact_behavior_matched_pairs_n": len(eligible_ids),
                "patch_exact_counterfactual_pairs_n": 0,
                "fixed_diagnostic_query": True,
                "matched_comparator_metadata": True,
                "matched_pool_sufficient": False,
            },
            "comparison_scope": "behavior and controls only; matched mechanism analysis is not identified",
        }
        dump_json(out_dir / "analysis.summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    data = np.load(root / "capture.npz")
    if len(rows_all) != data["hidden"].shape[0]:
        raise ValueError("capture index and arrays differ in length")
    arrays = {key: data[key] for key in data.files}
    units = [
        (item_id, control, CORRECT if control else STALE)
        for item_id in eligible_ids for control in (False, True)
    ]
    labels = np.asarray([label for _item_id, _control, label in units], dtype=object)
    item_reference = [
        {**rows_all[branch_lookup[(item_id, "tauQ", control)]], "analysis_arm": "no_overwrite" if control else "overwrite"}
        for item_id, control, _label in units
    ]

    pool_indices = [
        branch_lookup[(item_id, checkpoint, control)]
        for item_id, control, _label in units for checkpoint in CHECKPOINTS[1:]
    ]
    lengths = np.asarray([rows_all[index]["prefix_tokens"] for index in pool_indices])
    distances = np.asarray([rows_all[index]["competitor_distance"] for index in pool_indices])
    controlled = {}
    for key in ("read_ratio", "qk_margin"):
        controlled_values = residualize(arrays[key][pool_indices], lengths, distances)
        controlled[key] = controlled_values.reshape(len(units), len(CHECKPOINTS) - 1, *controlled_values.shape[1:])
    behavioral_values = np.asarray([rows_all[index]["behavioral_margin"] for index in pool_indices], dtype=np.float64)
    controlled_behavior = residualize(behavioral_values[:, None], lengths, distances).reshape(len(units), len(CHECKPOINTS) - 1)

    trajectories = {}
    retention_scores = {}
    earliest_candidates = []
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        indices = [branch_lookup[(item_id, checkpoint, control)] for item_id, control, _label in units]
        checkpoint_rows = [
            {
                **rows_all[index],
                "final_outcome": label,
                "analysis_arm": "no_overwrite" if control else "overwrite",
            }
            for index, (_item_id, control, label) in zip(indices, units)
        ]
        checkpoint_summary = {
            "n": len(indices),
            "final_outcome_counts": dict(Counter(labels.tolist())),
            "behavioral_margin": None,
            "read_ratio": None,
            "qk_margin": None,
        }
        retention, true_scores = fit_retention_probe(
            arrays["hidden"][indices], checkpoint_rows, args.n_shuffle, args.seed + 10_000 + checkpoint_index
        )
        checkpoint_summary["retention"] = retention
        checkpoint_summary["retention"]["final_stale_vs_correct"] = scalar_comparison(
            true_scores, labels, checkpoint_rows, args.n_shuffle, args.n_boot,
            args.seed + 11_000 + checkpoint_index,
        )
        retention_scores[checkpoint] = true_scores
        if checkpoint != "tau1":
            local = checkpoint_index - 1
            raw_behavior = np.asarray([row["behavioral_margin"] for row in checkpoint_rows], dtype=np.float64)
            raw_r = np.nanmedian(arrays["read_ratio"][indices], axis=(1, 2))
            raw_qk = np.nanmedian(arrays["qk_margin"][indices], axis=(1, 2))
            controlled_r = np.nanmedian(controlled["read_ratio"][:, local], axis=(1, 2))
            controlled_qk = np.nanmedian(controlled["qk_margin"][:, local], axis=(1, 2))
            checkpoint_summary["behavioral_margin"] = {
                "raw": scalar_comparison(raw_behavior, labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 100 + local),
                "length_distance_controlled": scalar_comparison(controlled_behavior[:, local], labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 200 + local),
            }
            checkpoint_summary["read_ratio"] = {
                "raw": scalar_comparison(raw_r, labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 300 + local),
                "length_distance_controlled": scalar_comparison(controlled_r, labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 400 + local),
                "head_raw": head_comparison(arrays["read_ratio"][indices], labels, checkpoint_rows, args.n_shuffle, args.seed + 500 + local),
                "head_length_distance_controlled": head_comparison(controlled["read_ratio"][:, local], labels, checkpoint_rows, args.n_shuffle, args.seed + 600 + local),
            }
            checkpoint_summary["qk_margin"] = {
                "raw": scalar_comparison(raw_qk, labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 700 + local),
                "length_distance_controlled": scalar_comparison(controlled_qk, labels, checkpoint_rows, args.n_shuffle, args.n_boot, args.seed + 800 + local),
                "head_raw": head_comparison(arrays["qk_margin"][indices], labels, checkpoint_rows, args.n_shuffle, args.seed + 900 + local),
                "head_length_distance_controlled": head_comparison(controlled["qk_margin"][:, local], labels, checkpoint_rows, args.n_shuffle, args.seed + 1000 + local),
            }
            r_test = checkpoint_summary["read_ratio"]["length_distance_controlled"]
            qk_test = checkpoint_summary["qk_margin"]["length_distance_controlled"]
            if (r_test["stale_minus_correct"] < 0 and r_test["shuffle_p_lower"] < 0.05) or (qk_test["stale_minus_correct"] < 0 and qk_test["shuffle_p_lower"] < 0.05):
                earliest_candidates.append(checkpoint)
        trajectories[checkpoint] = checkpoint_summary

    early_features = []
    for local, checkpoint in enumerate(CHECKPOINTS[1:4]):
        early_features.extend([
            np.nanmedian(controlled["read_ratio"][:, local], axis=(1, 2)),
            np.nanmedian(controlled["qk_margin"][:, local], axis=(1, 2)),
            controlled_behavior[:, local],
            retention_scores[checkpoint],
        ])
    features = np.column_stack(early_features)
    finite = np.all(np.isfinite(features), axis=1)
    predictor = early_late_prediction(
        features[finite], labels[finite], [row for row, keep in zip(item_reference, finite) if keep],
        args.predictor_shuffles, args.seed + 20_000,
    )

    statistical_pairs = [
        {
            "pair_id": item_id,
            "semantic_id": rows_all[branch_lookup[(item_id, "tauQ", False)]]["semantic_id"],
            "correct_id": item_id,
            "correct_arm": "no_overwrite",
            "stale_id": item_id,
            "stale_arm": "overwrite",
            "prefix_tokens": rows_all[branch_lookup[(item_id, "tauQ", False)]]["prefix_tokens"],
        }
        for item_id in eligible_ids
    ]
    dump_jsonl(Path(args.out_dir) / "exact_behavior_pairs.jsonl", statistical_pairs)
    patch_pairs = []
    earliest = earliest_candidates[0] if earliest_candidates else None
    if earliest and length_gate:
        for item_id in eligible_ids:
            final_corrupt = rows_all[branch_lookup[(item_id, "tauQ", False)]]
            final_control = rows_all[branch_lookup[(item_id, "tauQ", True)]]
            if final_corrupt["label"] != STALE or final_control["label"] != CORRECT:
                continue
            early_corrupt = rows_all[branch_lookup[(item_id, earliest, False)]]
            early_control = rows_all[branch_lookup[(item_id, earliest, True)]]
            if early_corrupt["prefix_tokens"] != early_control["prefix_tokens"]:
                continue
            if early_control["label"] != CORRECT:
                continue
            patch_pairs.append({
                "pair_id": item_id,
                "semantic_id": early_corrupt["semantic_id"],
                "split": stable_split(early_corrupt["semantic_id"]),
                "checkpoint": earliest,
                "early_clean": early_control,
                "early_corrupt": early_corrupt,
                "final_clean": final_control,
                "final_corrupt": final_corrupt,
            })
    dump_jsonl(Path(args.out_dir) / "patch_pairs.jsonl", patch_pairs)

    selected_heads = []
    if earliest:
        local = CHECKPOINTS.index(earliest) - 1
        discovery = np.asarray([stable_split(row["semantic_id"]) == "discovery" for row in item_reference])
        if np.any(discovery & (labels == CORRECT)) and np.any(discovery & (labels == STALE)):
            r_delta = _group_delta(controlled["read_ratio"][discovery, local], labels[discovery])
            qk_delta = _group_delta(controlled["qk_margin"][discovery, local], labels[discovery])
            r_scale = np.nanstd(controlled["read_ratio"][discovery, local], axis=0) + 1e-8
            qk_scale = np.nanstd(controlled["qk_margin"][discovery, local], axis=0) + 1e-8
            score = r_delta / r_scale + qk_delta / qk_scale
            order = np.argsort(score, axis=None)
            for flat_index in order[:args.patch_heads]:
                layer, head = np.unravel_index(flat_index, score.shape)
                selected_heads.append({
                    "layer": int(layer), "head": int(head), "discovery_score": float(score[layer, head]),
                    "r_delta": float(r_delta[layer, head]), "qk_delta": float(qk_delta[layer, head]),
                })

    stale_indices = np.asarray([index for index, (_item_id, control, _label) in enumerate(units) if not control], dtype=np.int64)
    outcome_counts = Counter()
    stale_counts_by_checkpoint = []
    stale_qk_means = []
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS[1:]):
        indices = [branch_lookup[(item_id, checkpoint, control)] for item_id, control, _label in units]
        qk = np.nanmedian(arrays["qk_margin"][indices], axis=(1, 2))
        if checkpoint_index == 0:
            tau2_qk = qk
        if checkpoint == "tauQ":
            final_qk = qk
        checkpoint_rows = [rows_all[index] for index in indices]
        corrupt_rows = [row for row, (_item_id, control, _label) in zip(checkpoint_rows, units) if not control]
        stale_counts_by_checkpoint.append(float(np.mean([row["stale_trace_count"] for row in corrupt_rows])))
        stale_qk_means.append(float(np.nanmean(qk[stale_indices])))
    for item_index in stale_indices:
        if tau2_qk[item_index] < 0:
            outcome_counts["write_side_qk_negative_at_tau2"] += 1
        elif final_qk[item_index] < 0:
            outcome_counts["read_side_qk_crossing_after_tau2"] += 1
        else:
            outcome_counts["no_pooled_qk_crossing"] += 1
    competition_x = np.log1p(np.asarray(stale_counts_by_checkpoint))
    competition_y = np.asarray(stale_qk_means)
    competition_coef = np.polyfit(competition_x, competition_y, 1)
    competition_fit = np.polyval(competition_coef, competition_x)
    competition_r2 = 1.0 - float(np.sum((competition_y - competition_fit) ** 2) / np.sum((competition_y - competition_y.mean()) ** 2)) if np.var(competition_y) > 0 else math.nan
    label_counts = Counter(row["label"] for row in final_rows)
    descriptive_full_pool = unconditional_descriptive(root, rows_all, branch_lookup, item_ids, args)
    summary = {
        "stage": "P-analysis",
        "model": args.model_label,
        "n_items": len(item_ids),
        "final_label_counts": dict(label_counts),
        "eligible_exact_pair_n": len(eligible_ids),
        "eligible_correct_stale_arm_n": len(units),
        "trajectories": trajectories,
        "residualization": "fit once over all exact matched overwrite/no-overwrite tau2-tauQ arms with intercept, log prefix tokens, and log1p comparator distance",
        "early_to_late_prediction": predictor,
        "same_length_no_overwrite_controls": controls,
        "descriptive_full_pool": descriptive_full_pool,
        "validity_gates": {
            "same_length_every_checkpoint": bool(length_gate),
            "no_overwrite_accuracy_threshold": args.control_threshold,
            "no_overwrite_accuracy_soft_threshold_every_checkpoint": bool(control_accuracy_soft_pass),
            "no_overwrite_accuracy_threshold_is_soft": True,
            "exact_behavior_matched_pairs_n": len(statistical_pairs),
            "patch_exact_counterfactual_pairs_n": len(patch_pairs),
            "matched_pool_sufficient": True,
            "fixed_diagnostic_query": True,
            "matched_comparator_metadata": True,
        },
        "earliest_stale_favoring_checkpoint": earliest,
        "earliest_rule": "first tau2-tauQ checkpoint with negative residualized stale-minus-correct pooled R or QK and one-sided shuffle p<0.05",
        "outcome_mixture_on_final_stale": dict(outcome_counts),
        "competition_accumulation": {
            "x_log1p_stale_trace_count": competition_x.tolist(),
            "pooled_qk_final_stale_mean": competition_y.tolist(),
            "slope": float(competition_coef[0]),
            "intercept": float(competition_coef[1]),
            "r2": competition_r2,
            "scope": "descriptive trajectory fit; not a causal or pre-registered law test",
        },
        "selected_patch_heads": selected_heads,
        "patch_head_selection": "top discovery-only standardized residualized R+QK stale-favoring heads at earliest checkpoint",
        "patch_pair_counts": dict(Counter(row["split"] for row in patch_pairs)),
        "comparison_scope": "exact same-item overwrite within-stale versus no-overwrite correct counterfactual; no prevalence claim",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "analysis.summary.json", summary)
    print(json.dumps({
        "model": args.model_label,
        "final_labels": dict(label_counts),
        "earliest": earliest,
        "early_to_late": predictor,
        "gates": summary["validity_gates"],
        "patch_pairs": len(patch_pairs),
    }, indent=2), flush=True)


def self_test(_args) -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    controlled = residualize(values, np.asarray([10, 20, 30, 40]), np.asarray([1, 2, 3, 4]))
    assert controlled.shape == values.shape
    assert stable_split("example") in {"discovery", "evaluation"}
    print("genattn_analyze self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    run = sub.add_parser("analyze")
    run.add_argument("--capture-dir", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--model-label", required=True)
    run.add_argument("--n-shuffle", type=int, default=1000)
    run.add_argument("--predictor-shuffles", type=int, default=1000)
    run.add_argument("--n-boot", type=int, default=2000)
    run.add_argument("--seed", type=int, default=20260822)
    run.add_argument("--patch-heads", type=int, default=12)
    run.add_argument("--control-threshold", type=float, default=0.95)
    run.add_argument("--min-exact-pairs", type=int, default=20)
    run.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
