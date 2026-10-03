import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from icf_mech import dump_json, dump_jsonl, load_jsonl, mean_attention_ratio


def _labels(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row.get(key, "other") for row in rows], dtype=object)


def _lengths(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row.get("prompt_tokens", 0) or 0) for row in rows], dtype=np.float64)


def pooled_length_residuals(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(lengths)
    residuals = np.full(values.shape, np.nan, dtype=np.float64)
    if mask.sum() < 3 or np.allclose(lengths[mask], lengths[mask][0]):
        residuals[mask] = values[mask] - values[mask].mean()
        return residuals
    x = np.column_stack([np.ones(mask.sum()), np.log(np.maximum(lengths[mask], 1.0))])
    coef, *_ = np.linalg.lstsq(x, values[mask], rcond=None)
    residuals[mask] = values[mask] - x @ coef
    return residuals


def _delta(labels: np.ndarray, values: np.ndarray, group_a: str, group_b: str) -> float:
    a = values[labels == group_a]
    b = values[labels == group_b]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return float(a.mean() - b.mean())


def bootstrap_delta(labels, values, group_a, group_b, n_boot=2000, seed=1009) -> list[float]:
    labels = np.asarray(labels, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    a = values[labels == group_a]
    b = values[labels == group_b]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        deltas[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[
            rng.integers(0, len(b), size=len(b))
        ].mean()
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def shuffle_label_null(labels, values, group_a, group_b, n_shuffle=2000, seed=2001) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        out[i] = _delta(shuffled, values, group_a, group_b)
    return out


def _null_summary(null_values: np.ndarray, observed: float) -> dict:
    arr = np.asarray(null_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "ci": [math.nan, math.nan], "p_two_sided": math.nan, "exceeds_null_95": False}
    p = (np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "n": int(arr.size),
        "ci": [float(lo), float(hi)],
        "p_two_sided": float(p),
        "exceeds_null_95": bool(observed < lo or observed > hi),
    }


def _empirical_null_summary(null_values: np.ndarray, observed: float) -> dict:
    arr = np.asarray(null_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "ci": [math.nan, math.nan],
            "p_two_sided_empirical": math.nan,
            "above_null_95": False,
            "below_null_95": False,
            "exceeds_null_95": False,
        }
    lo, hi = np.quantile(arr, [0.025, 0.975])
    lower_tail = (np.sum(arr <= observed) + 1) / (arr.size + 1)
    upper_tail = (np.sum(arr >= observed) + 1) / (arr.size + 1)
    return {
        "n": int(arr.size),
        "ci": [float(lo), float(hi)],
        "p_two_sided_empirical": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "above_null_95": bool(observed > hi),
        "below_null_95": bool(observed < lo),
        "exceeds_null_95": bool(observed < lo or observed > hi),
    }


def group_mean_with_shuffle_null(
    rows: list[dict],
    values,
    *,
    group: str,
    label_key: str,
    n_boot: int = 2000,
    n_shuffle: int = 2000,
    seed: int = 0,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = _labels(rows, label_key)
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    group_values = values[labels == group]
    if len(group_values) == 0:
        return {"group": group, "n": 0, "mean": math.nan, "ci": [math.nan, math.nan]}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = group_values[rng.integers(0, len(group_values), size=len(group_values))].mean()
    null = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        null[i] = values[shuffled == group].mean()
    observed = float(group_values.mean())
    return {
        "group": group,
        "n": int(len(group_values)),
        "mean": observed,
        "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "shuffle_null": _empirical_null_summary(null, observed),
        "above_chance_0p5": bool(observed > 0.5),
    }


def group_delta_with_length_control(
    rows: list[dict],
    values,
    *,
    group_a: str,
    group_b: str,
    label_key: str = "final_error_type",
    n_boot: int = 2000,
    n_shuffle: int = 2000,
    seed: int = 1009,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = _labels(rows, label_key)
    lengths = _lengths(rows)
    finite = np.isfinite(values)
    rows_kept = [row for row, keep in zip(rows, finite) if keep]
    labels = labels[finite]
    lengths = lengths[finite]
    values = values[finite]
    residuals = pooled_length_residuals(values, lengths)
    raw_delta = _delta(labels, values, group_a, group_b)
    lc_delta = _delta(labels, residuals, group_a, group_b)
    raw_null = shuffle_label_null(labels, values, group_a, group_b, n_shuffle=n_shuffle, seed=seed + 1)
    lc_null = shuffle_label_null(labels, residuals, group_a, group_b, n_shuffle=n_shuffle, seed=seed + 2)
    counts = Counter(labels.tolist())
    return {
        "group_a": group_a,
        "group_b": group_b,
        "label_key": label_key,
        "n_pool": len(rows_kept),
        "n_a": int(counts.get(group_a, 0)),
        "n_b": int(counts.get(group_b, 0)),
        "raw_delta": raw_delta,
        "raw_delta_ci": bootstrap_delta(labels, values, group_a, group_b, n_boot=n_boot, seed=seed + 3),
        "raw_shuffle_null": _null_summary(raw_null, raw_delta),
        "length_controlled_delta": lc_delta,
        "length_controlled_delta_ci": bootstrap_delta(
            labels, residuals, group_a, group_b, n_boot=n_boot, seed=seed + 4
        ),
        "length_controlled_shuffle_null": _null_summary(lc_null, lc_delta),
    }


def _head_mean(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(arr, axis=2)


def _join_dp_rows(dp_index: list[dict], dp_labeled: list[dict]) -> tuple[list[dict], list[int]]:
    by_id = {str(row["id"]): row for row in dp_labeled}
    rows = []
    idx = []
    for i, row in enumerate(dp_index):
        if str(row["id"]) not in by_id:
            continue
        rows.append({**row, **by_id[str(row["id"])]})
        idx.append(i)
    return rows, idx


def candidate_index_diagnostics(rows: list[dict]) -> dict:
    new_counts = Counter(row.get("source_new_index") for row in rows)
    old_counts = Counter(row.get("source_old_index") for row in rows)
    pair_counts = Counter((row.get("source_old_index"), row.get("source_new_index")) for row in rows)
    return {
        "source_new_index_unique": int(len(new_counts)),
        "source_old_index_unique": int(len(old_counts)),
        "source_pair_unique": int(len(pair_counts)),
        "source_new_index_max_count": int(max(new_counts.values())) if new_counts else 0,
        "source_old_index_max_count": int(max(old_counts.values())) if old_counts else 0,
        "new_old_index_overlap": int(len(set(new_counts) & set(old_counts))),
        "strict_option_id_probe_feasible_from_saved_decision_hiddens": bool(
            new_counts and old_counts and max(new_counts.values()) > 1 and max(old_counts.values()) > 1
        ),
    }


def fit_groupkfold_probe_scores(hidden: np.ndarray, rows: list[dict], *, layer: int = -1, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    keep = [i for i, row in enumerate(rows) if row.get("final_error_type") in {"correct_current", "within_stale"}]
    if not keep:
        raise ValueError("no correct_current/within_stale rows for probe")
    probe_rows = [rows[i] for i in keep]
    x = hidden[keep, layer, :].astype(np.float32)
    y = np.asarray([1 if row["final_error_type"] == "correct_current" else 0 for row in probe_rows], dtype=np.int64)
    groups = np.asarray([row.get("source_new_index", row.get("id", i)) for i, row in enumerate(probe_rows)])
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 5 and len(np.unique(y)) == 2:
        splits = list(GroupKFold(n_splits=5).split(x, y, groups))
        splitter = "GroupKFold(n_splits=5)"
    else:
        splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(x, y))
        splitter = "StratifiedKFold(n_splits=2)"
    scores = np.full(len(probe_rows), np.nan, dtype=np.float64)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(x_train, y[train_idx])
        scores[test_idx] = clf.predict_proba(x_test)[:, 1]
    valid = np.isfinite(scores)
    auc = float(roc_auc_score(y[valid], scores[valid])) if valid.sum() and len(np.unique(y[valid])) == 2 else math.nan
    for row, score, label in zip(probe_rows, scores, y):
        row["probe_p_new"] = float(score) if np.isfinite(score) else math.nan
        row["probe_binary_label"] = int(label)
    labels = _labels(probe_rows, "final_error_type")
    within_scores = scores[labels == "within_stale"]
    correct_scores = scores[labels == "correct_current"]
    within_scores = within_scores[np.isfinite(within_scores)]
    correct_scores = correct_scores[np.isfinite(correct_scores)]
    null = shuffle_label_null(labels, scores, "within_stale", "correct_current", n_shuffle=2000, seed=seed + 10)
    return {
        "layer": int(layer if layer >= 0 else hidden.shape[1] + layer),
        "splitter": splitter,
        "n": int(valid.sum()),
        "n_correct_current": int((labels == "correct_current").sum()),
        "n_within_stale": int((labels == "within_stale").sum()),
        "auc_new_vs_old_selected": auc,
        "mean_p_new_correct_current": float(correct_scores.mean()) if len(correct_scores) else math.nan,
        "mean_p_new_within_stale": float(within_scores.mean()) if len(within_scores) else math.nan,
        "within_minus_correct_p_new": _delta(labels, scores, "within_stale", "correct_current"),
        "within_minus_correct_p_new_shuffle_null": _null_summary(null, _delta(labels, scores, "within_stale", "correct_current")),
        "rows": probe_rows,
    }


def analyze_k2k3b(mech_dir: str, *, n_boot: int, n_shuffle: int, seed: int) -> dict:
    mech = Path(mech_dir)
    dp_index = load_jsonl(mech / "dynamic_preference_index.jsonl")
    dp_labeled = load_jsonl(mech / "dynamic_preference_local_labeled.jsonl")
    if_index = load_jsonl(mech / "instructional_forgetting_index.jsonl")
    dp_data = np.load(mech / "dynamic_preference.npz")
    if_data = np.load(mech / "instructional_forgetting.npz")

    dp_rows, dp_idx = _join_dp_rows(dp_index, dp_labeled)
    dp_idx = np.asarray(dp_idx, dtype=np.int64)
    old_ratio = _head_mean(mean_attention_ratio(dp_data["attn_old_pref"][dp_idx], dp_data["attn_new_pref"][dp_idx]))[:, -1]
    dp_probe = fit_groupkfold_probe_scores(dp_data["hidden"][dp_idx], dp_rows, layer=-1, seed=seed)

    valid_gap = dp_data["new_old_logit_gap_valid"][dp_idx].astype(bool)
    gap_rows = [row for row, keep in zip(dp_rows, valid_gap) if keep]
    gap_values = dp_data["new_old_logit_gap"][dp_idx][valid_gap, -1]

    if_rows_all = if_index
    if_keep = np.asarray([bool(row.get("value_span_found")) for row in if_rows_all], dtype=bool)
    if_rows = [row for row, keep in zip(if_rows_all, if_keep) if keep]
    value_ratio = _head_mean(
        mean_attention_ratio(if_data["attn_forget_value"][if_keep], if_data["attn_forget_instruction"][if_keep])
    )[:, -1]

    return {
        "stage": "k2k3b",
        "mech_dir": mech_dir,
        "n_boot": n_boot,
        "n_shuffle": n_shuffle,
        "k3_candidate_index_diagnostics": candidate_index_diagnostics(dp_rows),
        "k3_probe_scope": (
            "behavior-selection proxy from saved decision-position hiddens: "
            "binary GroupKFold classifier for correct_current (NEW selected) vs within_stale (OLD selected). "
            "The stricter option-id NEW-vs-OLD decodability probe is not identifiable from the current saved "
            "decision-position activations because each DP old/new pair appears once."
        ),
        "k3_probe": {k: v for k, v in dp_probe.items() if k != "rows"},
        "k3_probe_within_stale_p_new": group_mean_with_shuffle_null(
            dp_probe["rows"],
            np.asarray([row["probe_p_new"] for row in dp_probe["rows"]], dtype=np.float64),
            group="within_stale",
            label_key="final_error_type",
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=seed + 50,
        ),
        "k3_probe_delta": group_delta_with_length_control(
            dp_probe["rows"],
            np.asarray([row["probe_p_new"] for row in dp_probe["rows"]], dtype=np.float64),
            group_a="within_stale",
            group_b="correct_current",
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=seed + 100,
        ),
        "k3_logit_lens_aux": group_delta_with_length_control(
            gap_rows,
            gap_values,
            group_a="within_stale",
            group_b="correct_current",
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=seed + 200,
        )
        | {"n_dropped_shared_first_token": int((~valid_gap).sum())},
        "k2_dp_old_pref_ratio": group_delta_with_length_control(
            dp_rows,
            old_ratio,
            group_a="within_stale",
            group_b="correct_current",
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=seed + 300,
        ),
        "k2_if_value_vs_instruction_ratio": group_delta_with_length_control(
            if_rows,
            value_ratio,
            group_a="within_stale",
            group_b="correct_forget",
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=seed + 400,
        )
        | {"n_correct_forget_baseline": int(sum(row.get("final_error_type") == "correct_forget" for row in if_rows))},
        "label_counts": {
            "dynamic_preference": dict(Counter(row.get("final_error_type") for row in dp_rows)),
            "instructional_forgetting": dict(Counter(row.get("final_error_type") for row in if_rows_all)),
            "instructional_forgetting_value_span_found": dict(Counter(row.get("final_error_type") for row in if_rows)),
        },
    }


def _fmt_delta(row: dict) -> str:
    return (
        f"raw {row['raw_delta']:.4f} CI [{row['raw_delta_ci'][0]:.4f}, {row['raw_delta_ci'][1]:.4f}], "
        f"length-controlled {row['length_controlled_delta']:.4f} CI "
        f"[{row['length_controlled_delta_ci'][0]:.4f}, {row['length_controlled_delta_ci'][1]:.4f}], "
        f"shuffle95 {row['length_controlled_shuffle_null']['ci']}"
    )


def write_k2k3b_report(path: str, summary: dict) -> None:
    probe_mean = summary["k3_probe_within_stale_p_new"]
    probe_delta = summary["k3_probe_delta"]
    dp_attn = summary["k2_dp_old_pref_ratio"]
    if_attn = summary["k2_if_value_vs_instruction_ratio"]
    selection_pass = (
        probe_mean["mean"] > 0.5
        and probe_mean["shuffle_null"]["above_null_95"]
        and probe_delta["length_controlled_delta"] > probe_delta["length_controlled_shuffle_null"]["ci"][1]
    )
    dp_attn_pass = dp_attn["length_controlled_delta"] > dp_attn["length_controlled_shuffle_null"]["ci"][1]
    if_attn_pass = if_attn["length_controlled_delta"] > if_attn["length_controlled_shuffle_null"]["ci"][1]
    verdict = (
        "K2/K3b verdict: trained-probe selection headline PASS."
        if selection_pass
        else "K2/K3b verdict: trained-probe selection headline does NOT pass; report this as a mixed/negative mechanism result."
    )
    lines = [
        "# Stage K Mechanism Report — K2/K3b Reanalysis",
        "",
        "## K2/K3b Reanalysis Fix",
        "",
        "This section reuses the saved local Qwen2.5-7B fp32-eager harvest. No regeneration was run. Length control is pooled across the full valid pool before comparing failure modes.",
        "",
        "Pre-registered adjudication:",
        "",
        "> Length-controlled selection signal survives (probe decodes NEW on within-stale above the shuffle null, length-controlled) → **selection-not-retention replicates on real dialogue** (clean headline); the stale-attention deltas that survive length control + shuffle null support the signature.",
        "",
        "> Signal vanishes after length control / falls into the shuffle null → the raw effect was length-driven / not robust; report honestly (a clean, informative negative — grounding-style length artifact, consistent with Stage J's lesson).",
        "",
        "### K3 Trained Linear Probe",
        "",
        "- Scope: saved-activation feasible behavior-selection proxy, not a strict option-id probe. "
        "Each DP old/new pair appears once, so the stricter candidate-id NEW-vs-OLD probe is not identifiable "
        "from the current saved decision-position activations without a forward-pass-only supplement.",
        f"- Candidate-index diagnostic: new_unique={summary['k3_candidate_index_diagnostics']['source_new_index_unique']}, "
        f"old_unique={summary['k3_candidate_index_diagnostics']['source_old_index_unique']}, "
        f"pair_unique={summary['k3_candidate_index_diagnostics']['source_pair_unique']}, "
        f"max_new_count={summary['k3_candidate_index_diagnostics']['source_new_index_max_count']}, "
        f"max_old_count={summary['k3_candidate_index_diagnostics']['source_old_index_max_count']}.",
        f"- Final-layer GroupKFold behavior-selection probe AUC (NEW-selected vs OLD-selected behavior): {summary['k3_probe']['auc_new_vs_old_selected']:.3f}.",
        f"- Mean p(NEW) correct_current: {summary['k3_probe']['mean_p_new_correct_current']:.3f}; within_stale: {summary['k3_probe']['mean_p_new_within_stale']:.3f}.",
        f"- Within-stale p(NEW): mean {probe_mean['mean']:.3f} CI [{probe_mean['ci'][0]:.3f}, {probe_mean['ci'][1]:.3f}], shuffle95 {probe_mean['shuffle_null']['ci']}.",
        f"- Within-stale p(NEW) above shuffle-null 95%: `{probe_mean['shuffle_null']['above_null_95']}`.",
        f"- Probe p(NEW) within_stale - correct_current: {_fmt_delta(summary['k3_probe_delta'])}.",
        f"- {verdict}",
        "",
        "### K3 Logit-Lens Auxiliary",
        "",
        f"- NEW-OLD first-token gap within_stale - correct_current: {_fmt_delta(summary['k3_logit_lens_aux'])}.",
        f"- Shared-first-token dropped: {summary['k3_logit_lens_aux']['n_dropped_shared_first_token']}.",
        "",
        "### K2 DP Attention",
        "",
        f"- OLD/(OLD+NEW) attention ratio within_stale - correct_current: {_fmt_delta(summary['k2_dp_old_pref_ratio'])}.",
        f"- DP stale-attention support after length control + shuffle null: `{dp_attn_pass}`.",
        "",
        "### K2 IF Attention",
        "",
        f"- Value/(value+forget-instruction) ratio within_stale - correct_forget: {_fmt_delta(summary['k2_if_value_vs_instruction_ratio'])}.",
        f"- Correct-forget baseline n={summary['k2_if_value_vs_instruction_ratio']['n_correct_forget_baseline']} (small; Qwen almost never truly forgets).",
        f"- IF attention support after length control + shuffle null: `{if_attn_pass}`.",
        "",
        "### Artifacts",
        "",
        "- `k2k3b_summary.json`",
        "- `k2k3b_probe_rows.jsonl`",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_report_section(report_path: str, section_path: str) -> None:
    report = Path(report_path)
    section = Path(section_path).read_text(encoding="utf-8")
    existing = report.read_text(encoding="utf-8") if report.exists() else ""
    marker = "## K2/K3b Reanalysis Fix"
    section_body = marker + section.split(marker, 1)[1]
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
        duplicate_title = "# Stage K Mechanism Report — K2/K3b Reanalysis"
        if existing.endswith(duplicate_title):
            existing = existing[: -len(duplicate_title)].rstrip()
        report.write_text(existing + "\n\n" + section_body, encoding="utf-8")
    else:
        report.write_text(existing.rstrip() + "\n\n" + section_body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-dir", default="results/stage_k/mech/full_h100")
    parser.add_argument("--out", default="results/stage_k/mech/k2k3b_summary.json")
    parser.add_argument("--probe-rows-out", default="results/stage_k/mech/k2k3b_probe_rows.jsonl")
    parser.add_argument("--section-out", default="results/stage_k/mech/REPORT_k2k3b.md")
    parser.add_argument("--report", default="results/stage_k/mech/REPORT.md")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-shuffle", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    summary = analyze_k2k3b(args.mech_dir, n_boot=args.n_boot, n_shuffle=args.n_shuffle, seed=args.seed)
    dump_json(args.out, summary)
    # Save per-row probe scores without the full hidden states.
    probe_rows = fit_groupkfold_probe_scores(
        np.load(Path(args.mech_dir) / "dynamic_preference.npz")["hidden"],
        _join_dp_rows(
            load_jsonl(Path(args.mech_dir) / "dynamic_preference_index.jsonl"),
            load_jsonl(Path(args.mech_dir) / "dynamic_preference_local_labeled.jsonl"),
        )[0],
        layer=-1,
        seed=args.seed,
    )["rows"]
    dump_jsonl(args.probe_rows_out, probe_rows)
    write_k2k3b_report(args.section_out, summary)
    append_report_section(args.report, args.section_out)
    print(f"wrote {args.out}, {args.probe_rows_out}, {args.section_out}; updated {args.report}")


if __name__ == "__main__":
    main()
