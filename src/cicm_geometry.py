import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cicm_eval import dump_json, dump_jsonl, load_jsonl
from cicm_mech import empirical_null_summary, true_label_scores
from cicm_stats import pooled_length_residuals


DISTANCE_STRENGTH = {
    "near": 1.0,
    "near2": 1.0,
    "mid": 0.5,
    "far": 0.0,
    "unknown": math.nan,
}


def cell_key(row: dict) -> str:
    cell = row.get("factorial_cell") or row.get("competition", {}).get("factorial_cell") or {}
    same = cell.get("same_slot_stale_distance_bin", "unknown")
    other = cell.get("recent_other_slot_distance_bin", "unknown")
    return f"same_{same}__other_{other}"


def selected_value(row: dict) -> tuple[str | None, str]:
    label = row.get("label", "other")
    if label == "correct_current":
        return row.get("current_value"), "current"
    if label == "within_stale":
        hits = row.get("stale_hits") or row.get("stale_values") or []
        return (hits[0] if hits else None), "stale"
    if label == "cross_slot":
        hits = row.get("cross_slot_hits") or []
        if not hits:
            nearest = row.get("competition", {}).get("recent_other_slot_nearest") or {}
            if nearest.get("value"):
                hits = [nearest["value"]]
        return (hits[0] if hits else None), "other"
    return None, str(label)


def _candidate_from_comp(entry: dict, candidate_type: str, queried_slot: str) -> dict | None:
    value = entry.get("value")
    if not value:
        return None
    distance = entry.get("distance_messages_to_query")
    distance_bin = entry.get("distance_bin", "unknown")
    return {
        "value": value,
        "candidate_type": candidate_type,
        "slot": entry.get("slot"),
        "is_same_slot": bool(entry.get("slot") == queried_slot),
        "distance_messages_to_query": float(distance) if distance is not None else math.nan,
        "distance_bin": distance_bin,
        "recency_strength": DISTANCE_STRENGTH.get(distance_bin, math.nan),
    }


def candidate_bindings(row: dict, *, include_all_other: bool = True) -> list[dict]:
    comp = row.get("competition", {})
    queried_slot = row.get("slot")
    candidates: list[dict] = []
    current = _candidate_from_comp(comp.get("target_current") or {}, "current", queried_slot)
    if current is None and row.get("current_value"):
        current = {
            "value": row["current_value"],
            "candidate_type": "current",
            "slot": queried_slot,
            "is_same_slot": True,
            "distance_messages_to_query": math.nan,
            "distance_bin": "unknown",
            "recency_strength": math.nan,
        }
    if current:
        candidates.append(current)

    stale_entries = comp.get("same_slot_stale") or []
    if comp.get("same_slot_stale_nearest"):
        stale_entries = [comp["same_slot_stale_nearest"]] + stale_entries
    seen_stale = set()
    for entry in stale_entries:
        stale = _candidate_from_comp(entry, "stale", queried_slot)
        if stale and stale["value"] not in seen_stale:
            candidates.append(stale)
            seen_stale.add(stale["value"])

    other_entries = comp.get("recent_other_slot") or []
    if comp.get("recent_other_slot_nearest"):
        other_entries = [comp["recent_other_slot_nearest"]] + other_entries
    seen_other = set()
    for entry in other_entries:
        other = _candidate_from_comp(entry, "other", queried_slot)
        if other and other["value"] not in seen_other:
            candidates.append(other)
            seen_other.add(other["value"])
            if not include_all_other:
                break
    return candidates


def _splits(x, y, groups, seed: int):
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 5:
        return list(GroupKFold(n_splits=5).split(x, y, groups)), "GroupKFold(n_splits=5)"
    return list(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(x, y)), "StratifiedKFold(n_splits=2)"


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_oof_value_probe_scores(
    hidden: np.ndarray,
    rows: list[dict],
    *,
    layer: int,
    seed: int,
    estimator: str = "logistic",
    max_iter: int = 200,
    tol: float = 1e-3,
) -> dict:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x = hidden[:, layer, :].astype(np.float32)
    labels = np.asarray([row["current_value"] for row in rows], dtype=object)
    groups = np.asarray([row["id"] for row in rows], dtype=object)
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    splits, splitter = _splits(x, y, groups, seed)
    scores = np.full((len(rows), len(enc.classes_)), np.nan, dtype=np.float64)
    proba = np.full_like(scores, np.nan)
    pred = np.full(len(rows), -1, dtype=np.int64)
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        if estimator == "ridge":
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        else:
            clf = LogisticRegression(
                C=1.0,
                penalty="l2",
                max_iter=max_iter,
                class_weight="balanced",
                solver="saga",
                tol=tol,
                n_jobs=8,
                random_state=seed,
            )
        clf.fit(x_train, y[train_idx])
        fold_scores = clf.decision_function(x_test)
        if fold_scores.ndim == 1:
            fold_scores = np.column_stack([-fold_scores, fold_scores])
        fold_proba = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else _softmax(fold_scores)
        for local_col, class_id in enumerate(clf.classes_):
            scores[test_idx, class_id] = fold_scores[:, local_col]
            proba[test_idx, class_id] = fold_proba[:, local_col]
        pred[test_idx] = clf.predict(x_test)
    valid = np.isfinite(proba).any(axis=1)
    true_score = true_label_scores(proba, enc.classes_, labels)
    return {
        "layer": int(layer if layer >= 0 else hidden.shape[1] + layer),
        "splitter": splitter,
        "estimator": estimator,
        "max_iter": int(max_iter) if estimator == "logistic" else None,
        "tol": float(tol) if estimator == "logistic" else None,
        "classes": enc.classes_.tolist(),
        "class_to_index": {str(label): int(i) for i, label in enumerate(enc.classes_)},
        "scores": scores,
        "proba": proba,
        "labels": labels,
        "true_current_score": true_score,
        "cv_accuracy": float(accuracy_score(y[valid], pred[valid])) if valid.any() else math.nan,
        "n": int(valid.sum()),
    }


def candidate_scores_for_row(row: dict, row_scores: np.ndarray, class_to_index: dict[str, int]) -> list[dict]:
    out = []
    for cand in candidate_bindings(row):
        idx = class_to_index.get(str(cand["value"]))
        if idx is None or not np.isfinite(row_scores[idx]):
            continue
        out.append({**cand, "alignment": float(row_scores[idx])})
    return out


def _pick_argmax(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda cand: (cand.get("alignment", -math.inf), -cand.get("distance_messages_to_query", math.inf)))


def _nearest_candidate(candidates: list[dict], allowed_types: set[str] | None = None) -> dict | None:
    pool = [cand for cand in candidates if allowed_types is None or cand["candidate_type"] in allowed_types]
    if not pool:
        return None
    return min(pool, key=lambda cand: (cand.get("distance_messages_to_query", math.inf), cand["candidate_type"]))


def _match_candidate(pred: dict | None, actual_value: str | None, actual_type: str) -> bool:
    if pred is None or actual_value is None:
        return False
    return pred["candidate_type"] == actual_type and str(pred["value"]) == str(actual_value)


def g1_selection_argmax(rows: list[dict], probe: dict, *, n_shuffle: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    class_to_index = probe["class_to_index"]
    records = []
    for i, row in enumerate(rows):
        actual_value, actual_type = selected_value(row)
        if actual_type not in {"current", "stale", "other"} or actual_value is None:
            continue
        candidates = candidate_scores_for_row(row, probe["scores"][i], class_to_index)
        if len(candidates) < 2:
            continue
        geo = _pick_argmax(candidates)
        rec = _nearest_candidate(candidates)
        ident = _nearest_candidate(candidates, {"current", "stale"})
        records.append(
            {
                "id": row["id"],
                "label": row.get("label"),
                "cell": cell_key(row),
                "actual_value": actual_value,
                "actual_type": actual_type,
                "geometric_type": geo["candidate_type"] if geo else None,
                "geometric_value": geo["value"] if geo else None,
                "geometric_correct": _match_candidate(geo, actual_value, actual_type),
                "recency_type": rec["candidate_type"] if rec else None,
                "recency_value": rec["value"] if rec else None,
                "recency_correct": _match_candidate(rec, actual_value, actual_type),
                "identity_type": ident["candidate_type"] if ident else None,
                "identity_value": ident["value"] if ident else None,
                "identity_correct": _match_candidate(ident, actual_value, actual_type),
                "candidate_count": len(candidates),
            }
        )

    observed = float(np.mean([row["geometric_correct"] for row in records])) if records else math.nan
    recency_acc = float(np.mean([row["recency_correct"] for row in records])) if records else math.nan
    identity_acc = float(np.mean([row["identity_correct"] for row in records])) if records else math.nan
    null = np.empty(n_shuffle, dtype=np.float64)
    for j in range(n_shuffle):
        correct = []
        for i, row in enumerate(rows):
            actual_value, actual_type = selected_value(row)
            if actual_type not in {"current", "stale", "other"} or actual_value is None:
                continue
            candidates = candidate_scores_for_row(row, probe["scores"][i], class_to_index)
            if len(candidates) < 2:
                continue
            shuffled_scores = np.array([cand["alignment"] for cand in candidates], dtype=np.float64)
            rng.shuffle(shuffled_scores)
            shuffled = [{**cand, "alignment": float(score)} for cand, score in zip(candidates, shuffled_scores)]
            correct.append(_match_candidate(_pick_argmax(shuffled), actual_value, actual_type))
        null[j] = float(np.mean(correct)) if correct else math.nan
    null_sum = empirical_null_summary(null, observed)

    by_cell = {}
    for cell in sorted({row["cell"] for row in records}):
        subset = [row for row in records if row["cell"] == cell]
        by_cell[cell] = {
            "n": len(subset),
            "actual_type_counts": dict(Counter(row["actual_type"] for row in subset)),
            "geometric_type_counts": dict(Counter(row["geometric_type"] for row in subset)),
            "geometric_accuracy": float(np.mean([row["geometric_correct"] for row in subset])),
            "recency_accuracy": float(np.mean([row["recency_correct"] for row in subset])),
            "identity_accuracy": float(np.mean([row["identity_correct"] for row in subset])),
        }

    stronger_single = max(recency_acc, identity_acc)
    return {
        "n": len(records),
        "geometric_accuracy": observed,
        "shuffle_null": null_sum,
        "recency_only_accuracy": recency_acc,
        "identity_only_accuracy": identity_acc,
        "stronger_single_factor_accuracy": stronger_single,
        "beats_shuffle_95": bool(null_sum.get("above_null_95")),
        "beats_stronger_single_factor": bool(observed > stronger_single) if np.isfinite(observed) else False,
        "gate_pass": bool(null_sum.get("above_null_95") and observed > stronger_single),
        "by_cell": by_cell,
        "records": records,
    }


def _candidate_matrix(rows: list[dict], probe: dict, *, length_controlled: bool) -> list[dict]:
    records = []
    all_scores = []
    lengths = []
    for i, row in enumerate(rows):
        candidates = candidate_scores_for_row(row, probe["scores"][i], probe["class_to_index"])
        for cand in candidates:
            all_scores.append(cand["alignment"])
            lengths.append(float(row.get("prompt_tokens", 0) or 0))
            records.append(
                {
                    "row_index": i,
                    "id": row["id"],
                    "cell": cell_key(row),
                    "label": row.get("label"),
                    "candidate_type": cand["candidate_type"],
                    "value": cand["value"],
                    "is_same_slot": 1.0 if cand["candidate_type"] in {"current", "stale"} else 0.0,
                    "recency_strength": float(cand["recency_strength"]) if np.isfinite(cand["recency_strength"]) else 0.0,
                    "alignment": cand["alignment"],
                }
            )
    if length_controlled and records:
        residuals = pooled_length_residuals(np.asarray(all_scores), np.asarray(lengths))
        for rec, resid in zip(records, residuals):
            rec["fit_target"] = float(resid)
    else:
        for rec in records:
            rec["fit_target"] = float(rec["alignment"])
    return records


def _fit_identity_recency(records: list[dict], train_cells: set[str]) -> np.ndarray | None:
    train = [
        rec
        for rec in records
        if rec["cell"] in train_cells and np.isfinite(rec["fit_target"])
    ]
    if len(train) < 3:
        return None
    x = np.asarray([[1.0, rec["is_same_slot"], rec["recency_strength"]] for rec in train], dtype=np.float64)
    y = np.asarray([rec["fit_target"] for rec in train], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def _predict_record_score(rec: dict, coef: np.ndarray) -> float:
    return float(np.dot(np.asarray([1.0, rec["is_same_slot"], rec["recency_strength"]], dtype=np.float64), coef))


def g2_predict_crossover(rows: list[dict], probe: dict, *, length_controlled: bool) -> dict:
    records = _candidate_matrix(rows, probe, length_controlled=length_controlled)
    by_row = defaultdict(list)
    for rec in records:
        by_row[rec["row_index"]].append(rec)
    cells = sorted({cell_key(row) for row in rows})
    heldout = {}
    for held_cell in cells:
        coef = _fit_identity_recency(records, set(cells) - {held_cell})
        if coef is None:
            continue
        pred_counts = Counter()
        obs_counts = Counter()
        for i, row in enumerate(rows):
            if cell_key(row) != held_cell:
                continue
            stale = [rec for rec in by_row[i] if rec["candidate_type"] == "stale"]
            other = [rec for rec in by_row[i] if rec["candidate_type"] == "other"]
            if not stale or not other:
                continue
            best_stale = max(stale, key=lambda rec: _predict_record_score(rec, coef))
            best_other = max(other, key=lambda rec: _predict_record_score(rec, coef))
            stale_score = _predict_record_score(best_stale, coef)
            other_score = _predict_record_score(best_other, coef)
            pred_counts["cross_slot" if other_score >= stale_score else "within_stale"] += 1
            if row.get("label") in {"within_stale", "cross_slot"}:
                obs_counts[row["label"]] += 1
        pred_total = pred_counts["within_stale"] + pred_counts["cross_slot"]
        obs_total = obs_counts["within_stale"] + obs_counts["cross_slot"]
        pred_cross_share = pred_counts["cross_slot"] / pred_total if pred_total else math.nan
        obs_cross_share = obs_counts["cross_slot"] / obs_total if obs_total else math.nan
        heldout[held_cell] = {
            "coef_intercept_identity_recency": [float(x) for x in coef],
            "predicted_counts": dict(pred_counts),
            "observed_failure_counts": dict(obs_counts),
            "predicted_cross_share_competitor_only": float(pred_cross_share),
            "observed_cross_share_among_failures": float(obs_cross_share),
            "absolute_share_error": float(abs(pred_cross_share - obs_cross_share)) if np.isfinite(pred_cross_share) and np.isfinite(obs_cross_share) else math.nan,
            "predicted_crossover": bool(pred_cross_share >= 0.5) if np.isfinite(pred_cross_share) else None,
            "observed_crossover_or_tie": bool(obs_cross_share >= 0.5) if np.isfinite(obs_cross_share) else None,
        }
    errors = [row["absolute_share_error"] for row in heldout.values() if np.isfinite(row["absolute_share_error"])]
    sign_hits = [
        row["predicted_crossover"] == row["observed_crossover_or_tie"]
        for row in heldout.values()
        if row["predicted_crossover"] is not None and row["observed_crossover_or_tie"] is not None
    ]
    return {
        "length_controlled": length_controlled,
        "fit_target": "alignment residualized on log(prompt_tokens)" if length_controlled else "raw alignment",
        "features": ["intercept", "is_same_slot", "recency_strength"],
        "heldout_cell_predictions": heldout,
        "mean_absolute_cross_share_error": float(np.mean(errors)) if errors else math.nan,
        "crossover_sign_accuracy": float(np.mean(sign_hits)) if sign_hits else math.nan,
        "n_heldout_cells": len(heldout),
    }


def summarize_layer(rows: list[dict], hidden: np.ndarray, *, layer: int, seed: int, n_shuffle: int) -> dict:
    probe = fit_oof_value_probe_scores(hidden, rows, layer=layer, seed=seed, estimator="ridge")
    labels = np.asarray([row.get("label") for row in rows], dtype=object)
    within = labels == "within_stale"
    scores = probe["true_current_score"]
    within_scores = scores[within]
    within_scores = within_scores[np.isfinite(within_scores)]
    g1 = g1_selection_argmax(rows, probe, n_shuffle=n_shuffle, seed=seed + 101)
    return {
        "layer": probe["layer"],
        "estimator": probe["estimator"],
        "cv_accuracy": probe["cv_accuracy"],
        "within_stale_true_current_score_mean": float(within_scores.mean()) if len(within_scores) else math.nan,
        "g1_geometric_accuracy": g1["geometric_accuracy"],
        "g1_recency_only_accuracy": g1["recency_only_accuracy"],
        "g1_identity_only_accuracy": g1["identity_only_accuracy"],
        "g1_shuffle_ci": g1["shuffle_null"]["ci"],
        "g1_gate_pass": g1["gate_pass"],
    }


def intrinsic_dimension(hidden_layer: np.ndarray, rows: list[dict]) -> dict:
    out = {}
    labels = np.asarray([row.get("label") for row in rows], dtype=object)
    for label in ["correct_current", "within_stale", "cross_slot"]:
        x = hidden_layer[labels == label].astype(np.float32)
        if len(x) < 3:
            out[label] = {"n": int(len(x)), "participation_ratio": math.nan}
            continue
        x = x - x.mean(axis=0, keepdims=True)
        cov = (x @ x.T) / max(x.shape[1] - 1, 1)
        eig = np.linalg.eigvalsh(cov)
        eig = eig[eig > 1e-12]
        pr = (eig.sum() ** 2) / np.square(eig).sum() if eig.size else math.nan
        out[label] = {"n": int(len(x)), "participation_ratio": float(pr)}
    return out


def analyze_model(args, model_name: str, mech_dir: Path, out_dir: Path, l0_rows_path: Path) -> dict:
    rows = load_jsonl(mech_dir / "l1_index.jsonl")
    hidden = np.load(mech_dir / "l1_harvest.npz")["hidden"]
    layer = args.layer if args.layer is not None else hidden.shape[1] - 1
    probe = fit_oof_value_probe_scores(
        hidden,
        rows,
        layer=layer,
        seed=args.seed,
        estimator="logistic",
        max_iter=args.max_iter,
        tol=args.tol,
    )
    labels = np.asarray([row.get("label") for row in rows], dtype=object)
    within_scores = probe["true_current_score"][labels == "within_stale"]
    within_scores = within_scores[np.isfinite(within_scores)]
    value_null = []
    rng = np.random.default_rng(args.seed + 17)
    for _ in range(args.n_shuffle):
        shuffled = probe["labels"].copy()
        rng.shuffle(shuffled)
        scores = true_label_scores(probe["proba"], np.asarray(probe["classes"], dtype=object), shuffled)
        value_null.append(float(np.nanmean(scores[labels == "within_stale"])))
    sanity = {
        "layer": probe["layer"],
        "splitter": probe["splitter"],
        "estimator": probe["estimator"],
        "n": probe["n"],
        "n_classes": len(probe["classes"]),
        "cv_accuracy": probe["cv_accuracy"],
        "within_stale_true_current_score_mean": float(within_scores.mean()) if len(within_scores) else math.nan,
        "within_stale_value_label_shuffle": empirical_null_summary(np.asarray(value_null), float(within_scores.mean())),
    }
    stored_probe_rows_path = mech_dir / "l1_probe_rows.jsonl"
    if stored_probe_rows_path.exists():
        stored_rows = load_jsonl(stored_probe_rows_path)
        stored_within = [
            float(row["probe_true_current_score"])
            for row in stored_rows
            if row.get("label") == "within_stale" and row.get("probe_true_current_score") is not None
        ]
        sanity["stored_l1_probe_rows_within_stale_true_current_score_mean"] = (
            float(np.mean(stored_within)) if stored_within else math.nan
        )
        sanity["stored_l1_probe_rows_path"] = str(stored_probe_rows_path)
    g1 = g1_selection_argmax(rows, probe, n_shuffle=args.n_shuffle, seed=args.seed + 101)
    g2_raw = g2_predict_crossover(rows, probe, length_controlled=False)
    g2_lc = g2_predict_crossover(rows, probe, length_controlled=True)

    scan = []
    if args.layer_scan:
        for layer_idx in range(hidden.shape[1]):
            scan.append(summarize_layer(rows, hidden, layer=layer_idx, seed=args.seed + layer_idx, n_shuffle=args.layer_scan_shuffle))
            print(f"{model_name} layer_scan {layer_idx + 1}/{hidden.shape[1]}", flush=True)

    g4 = intrinsic_dimension(hidden[:, layer, :], rows) if args.intrinsic_dim else {}
    l0_counts = {}
    if l0_rows_path.exists():
        l0_rows = load_jsonl(l0_rows_path)
        for cell in sorted({cell_key(row) for row in l0_rows}):
            subset = [row for row in l0_rows if cell_key(row) == cell]
            l0_counts[cell] = dict(Counter(row.get("label") for row in subset))

    summary = {
        "stage": "cicm_geometry",
        "model": model_name,
        "mech_dir": str(mech_dir),
        "l0_rows": str(l0_rows_path),
        "n_shuffle": args.n_shuffle,
        "layer_scan_shuffle": args.layer_scan_shuffle,
        "primary_layer": layer,
        "sanity_decodability": sanity,
        "g1_selection_argmax": {k: v for k, v in g1.items() if k != "records"},
        "g2_crossover_prediction_raw": g2_raw,
        "g2_crossover_prediction_length_controlled": g2_lc,
        "g3_layer_scan": scan,
        "g4_intrinsic_dimension": g4,
        "l0_per_cell_label_counts": l0_counts,
        "notes": {
            "anti_circularity": "Current-value alignment is reported only as a sanity check. Headline claims are restricted to candidate competition and out-of-sample crossover prediction.",
            "mention_position_robustness": "Not run: saved l1_harvest.npz contains decision-position hidden states only, not value-mention residual states, and this addendum forbids model re-runs.",
            "g3_layer_scan": "G-3 uses ridge linear probes for tractable full-layer coverage; primary G-1/G-2 use the logistic L-1-style probe directions.",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / f"{model_name}_summary.json", summary)
    dump_jsonl(out_dir / f"{model_name}_g1_records.jsonl", g1["records"])
    return summary


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def write_report(path: Path, summaries: dict[str, dict]) -> None:
    lines = [
        "# Stage L Geometry Addendum",
        "",
        "Pre-registered readings:",
        "",
        "- G-1: geometric-argmax predicts the selected value above BOTH the shuffle null AND the stronger of the two single-factor baselines -> selection is a genuine geometric competition. Otherwise -> the simple alignment geometry does not capture selection.",
        "- G-2: geometry predicts the held-out per-cell crossover -> selection is a geometric identity-vs-recency competition. Prediction fails -> the crossover is not explained by this alignment geometry; the mechanism stays at decodable-but-not-selected.",
        "",
        "Anti-circularity: current-value alignment is a sanity check only. Headline claims below use only candidate competition and held-out-cell crossover prediction.",
        "",
    ]
    for model_name, summary in summaries.items():
        sanity = summary["sanity_decodability"]
        g1 = summary["g1_selection_argmax"]
        g2 = summary["g2_crossover_prediction_length_controlled"]
        raw_g2 = summary["g2_crossover_prediction_raw"]
        predicted_cross_values = [
            row["predicted_cross_share_competitor_only"]
            for row in g2["heldout_cell_predictions"].values()
            if np.isfinite(row["predicted_cross_share_competitor_only"])
        ]
        observed_cross_values = [
            row["observed_cross_share_among_failures"]
            for row in g2["heldout_cell_predictions"].values()
            if np.isfinite(row["observed_cross_share_among_failures"])
        ]
        g2_reading = (
            "Prediction fails: the length-controlled identity/recency decomposition predicts "
            "stale-only competition in every held-out cell while observed cross-slot failures "
            "remain nonzero."
            if predicted_cross_values
            and max(predicted_cross_values) == 0.0
            and observed_cross_values
            and max(observed_cross_values) > 0.0
            else "Prediction should be read from the held-out-cell table; no global prevalence claim is made."
        )
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"- Sanity: layer {sanity['layer']}, stored L-1 probe-row within-stale true-current score {_fmt(sanity.get('stored_l1_probe_rows_within_stale_true_current_score_mean'))}; geometry refit CV accuracy {_fmt(sanity['cv_accuracy'])}, refit within-stale true-current score {_fmt(sanity['within_stale_true_current_score_mean'])}, shuffle95 {sanity['within_stale_value_label_shuffle']['ci']}.",
                f"- G-1: geometric argmax accuracy {_fmt(g1['geometric_accuracy'])}; shuffle95 {g1['shuffle_null']['ci']}; recency-only {_fmt(g1['recency_only_accuracy'])}; identity-only {_fmt(g1['identity_only_accuracy'])}; gate pass `{g1['gate_pass']}`. Reading: above shuffle, but below the stronger identity-only baseline, so simple geometric argmax does not pass the competition gate.",
                f"- G-2 raw: held-out-cell cross-share MAE {_fmt(raw_g2['mean_absolute_cross_share_error'])}; crossover sign accuracy {_fmt(raw_g2['crossover_sign_accuracy'])}.",
                f"- G-2 length-controlled: held-out-cell cross-share MAE {_fmt(g2['mean_absolute_cross_share_error'])}; crossover sign accuracy {_fmt(g2['crossover_sign_accuracy'])}. Reading: {g2_reading}",
                "",
                "| held-out cell | predicted cross share | observed cross share | abs error | predicted crossover | observed crossover/tie |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cell, row in sorted(g2["heldout_cell_predictions"].items()):
            lines.append(
                f"| {cell} | {_fmt(row['predicted_cross_share_competitor_only'])} | "
                f"{_fmt(row['observed_cross_share_among_failures'])} | {_fmt(row['absolute_share_error'])} | "
                f"{row['predicted_crossover']} | {row['observed_crossover_or_tie']} |"
            )
        lines.extend(["", "G-3 layer scan top rows by G-1 accuracy (ridge linear scan, descriptive):", ""])
        scan = sorted(summary.get("g3_layer_scan") or [], key=lambda row: row["g1_geometric_accuracy"], reverse=True)[:8]
        lines.append("| layer | G-1 acc | recency-only | identity-only | CV acc | within score | gate |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in scan:
            lines.append(
                f"| {row['layer']} | {_fmt(row['g1_geometric_accuracy'])} | {_fmt(row['g1_recency_only_accuracy'])} | "
                f"{_fmt(row['g1_identity_only_accuracy'])} | {_fmt(row['cv_accuracy'])} | "
                f"{_fmt(row['within_stale_true_current_score_mean'])} | {row['g1_gate_pass']} |"
            )
        if summary.get("g4_intrinsic_dimension"):
            lines.extend(["", "G-4 optional intrinsic dimension (participation ratio):", ""])
            for label, row in summary["g4_intrinsic_dimension"].items():
                lines.append(f"- {label}: n={row['n']}, PR={_fmt(row['participation_ratio'])}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "G-1 and G-2 are the only headline tests. In both Qwen and Llama, G-1 is above the candidate-shuffle null but does not beat the stronger identity-only baseline. G-2 does not provide the requested cross-model geometric explanation of the identity-vs-recency crossover: Qwen misses the crossover signs in the same_far/mid and same_far/near2 cells, and both models' length-controlled decomposition predicts zero cross-slot share in every held-out cell. The defensible reading is therefore negative for this geometry addendum: Stage L remains a strong decodable-but-not-selected selection result, but this simple probe-direction identity/recency geometry does not explain why the wrong competitor wins.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-mech-dir", default="results/cicm/natural_factorial_otherdist_l1/full")
    parser.add_argument("--llama-mech-dir", default="results/cicm/natural_factorial_otherdist_llama/l1/full")
    parser.add_argument("--qwen-l0-rows", default="results/cicm/natural_factorial_otherdist_l0/openrouter_qwen25_7b_rows_merged.jsonl")
    parser.add_argument("--llama-l0-rows", default="results/cicm/natural_factorial_otherdist_llama/l0/openrouter_llama31_8b_rows.jsonl")
    parser.add_argument("--out-dir", default="results/cicm/geometry")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--n-shuffle", type=int, default=2000)
    parser.add_argument("--layer-scan-shuffle", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--layer-scan", action="store_true")
    parser.add_argument("--intrinsic-dim", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summaries = {
        "qwen": analyze_model(args, "qwen", Path(args.qwen_mech_dir), out_dir, Path(args.qwen_l0_rows)),
        "llama": analyze_model(args, "llama", Path(args.llama_mech_dir), out_dir, Path(args.llama_l0_rows)),
    }
    combined = {
        "stage": "stage_l_geometry_combined",
        "models": summaries,
        "headline_guardrail": "Only G-1 candidate competition and G-2 held-out crossover prediction are headline tests.",
    }
    dump_json(out_dir / "summary.json", combined)
    write_report(out_dir / "REPORT.md", summaries)
    print(json.dumps({"out_dir": str(out_dir), "models": list(summaries)}, indent=2))


if __name__ == "__main__":
    main()
