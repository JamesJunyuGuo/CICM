"""Stage H-3e text-only adjudication baselines."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from stage_h2_labels import _chat_prompt, _span_context
from stage_h2_probe import C_GRID, INNER_FOLDS, OUTER_FOLDS, SCALAR_NAMES, _safe_auc, _splitter
from stage_h3_probe import _load_h2
from stage_h_analysis import load_jsonl


H3E_PREREGISTERED_READINGS = [
    'AUC(C0+T2) ≥ 0.95 or internal_increment < 0.03 → the premonition is predominantly **input-legible difficulty**; the "internal signal" pillar of the predict-to-predict program is retired (the trigger may still be useful — note that it just got cheaper: no model internals needed).',
    "internal_increment ≥ 0.05 → a genuine internal component exists; run E-2.",
    "In between → report; human decides.",
]


def internal_increment(full_auc, text_auc):
    return float(full_auc) - float(text_auc)


def classify_h3e_reading(text_auc, increment):
    if float(text_auc) >= 0.95 or float(increment) < 0.03:
        return "input_legible_difficulty"
    if float(increment) >= 0.05:
        return "genuine_internal_component"
    return "in_between"


def _c0_matrix(scalars):
    idx = {name: i for i, name in enumerate(SCALAR_NAMES)}
    cols = [idx["C0_context_entropy"], idx["C0_rel_gen_pos"], idx["C0_prompt_len"], idx["C0_gen_len"]]
    return np.asarray(scalars, dtype=np.float32)[:, cols]


def _text_from_gen_token_window(tokenizer, row, start_gen_token, end_gen_token):
    templated = _chat_prompt(tokenizer, row["prompt"])
    raw = str(row.get("raw", ""))
    full = templated + raw
    context = _span_context(tokenizer, row)
    offsets = context["offset_mapping"]
    prompt_len = context["prompt_len"]
    start = max(0, int(start_gen_token))
    end = max(start, int(end_gen_token))
    pieces = []
    for gen_tok in range(start, end):
        full_tok = prompt_len + gen_tok
        if 0 <= full_tok < len(offsets):
            char_start, char_end = offsets[full_tok]
            if char_end > context["char_offset"]:
                pieces.append(full[max(char_start, context["char_offset"]):char_end])
    return "".join(pieces).strip()


def _full_problem_text(row):
    if row.get("question"):
        return str(row["question"])
    return str(row.get("prompt", ""))


def _example_token_indices(example_indices):
    out = {}
    for row_idx, ex in enumerate(example_indices):
        out.setdefault(int(ex), []).append(row_idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in out.items()}


def build_gsm_equation_text_rows(tokenizer, h2, generations, labels, window):
    by_ex_token = {
        int(ex): {}
        for ex in np.unique(h2["example_index"])
    }
    for row_idx, (ex, tok) in enumerate(zip(h2["example_index"], h2["gen_token_index"])):
        by_ex_token.setdefault(int(ex), {})[int(tok)] = row_idx
    text_rows = []
    c0_rows = []
    y = []
    groups = []
    for label in labels:
        ex = int(label["example_index"])
        result_tok = label.get("result_gen_token_start")
        if result_tok is None:
            continue
        token_map = by_ex_token.get(ex, {})
        wanted = [token_map[t] for t in range(max(0, int(result_tok) - int(window)), int(result_tok)) if t in token_map]
        if not wanted:
            continue
        row = generations[ex]
        t1 = _text_from_gen_token_window(tokenizer, row, int(result_tok) - int(window), int(result_tok))
        text_rows.append({
            "example_index": ex,
            "equation_index": int(label["equation_index"]),
            "t1": t1,
            "t2": f"{t1}\n\nFULL_PROBLEM:\n{_full_problem_text(row)}",
        })
        c0_rows.append(np.nanmean(h2["scalars"][np.asarray(wanted, dtype=np.int64)], axis=0))
        y.append(int(label["correct"]))
        groups.append(ex)
    return text_rows, _c0_matrix(np.vstack(c0_rows)), np.asarray(y, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def build_crux_k_text_rows(tokenizer, h2, generations, k):
    by_ex = _example_token_indices(h2["example_index"])
    text_rows = []
    c0_rows = []
    y = []
    groups = []
    for ex, indices in sorted(by_ex.items()):
        if ex >= len(generations):
            continue
        take = indices[: int(k)]
        if len(take) == 0:
            continue
        row = generations[ex]
        t1 = _text_from_gen_token_window(tokenizer, row, 0, min(int(k), len(indices)))
        text_rows.append({"t1": t1, "t2": f"{t1}\n\nFULL_PROBLEM:\n{_full_problem_text(row)}"})
        c0_rows.append(np.nanmean(h2["scalars"][take], axis=0))
        y.append(int(row["correct"]))
        groups.append(ex)
    return text_rows, _c0_matrix(np.vstack(c0_rows)), np.asarray(y, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def _make_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=20000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)),
        ]
    )


def _fit_text_scores(c0, train_texts, test_texts, train_idx, test_idx, y, c_value, extra=None):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    numeric = c0 if extra is None else np.concatenate([c0, extra], axis=1)
    x_train_c0 = sparse.csr_matrix(scaler.fit_transform(np.nan_to_num(numeric[train_idx], nan=0.0)))
    x_test_c0 = sparse.csr_matrix(scaler.transform(np.nan_to_num(numeric[test_idx], nan=0.0)))
    vectorizer = _make_vectorizer()
    x_train_text = vectorizer.fit_transform(train_texts)
    x_test_text = vectorizer.transform(test_texts)
    x_train = sparse.hstack([x_train_c0, x_train_text], format="csr")
    x_test = sparse.hstack([x_test_c0, x_test_text], format="csr")
    clf = LogisticRegression(C=c_value, penalty="l2", max_iter=300, tol=1e-3, class_weight="balanced", solver="liblinear")
    clf.fit(x_train, y[train_idx])
    return clf.predict_proba(x_test)[:, 1]


def _choose_text_c(c0, texts, train_idx, y, groups, extra=None):
    y_train = y[train_idx]
    if len(set(y_train.tolist())) < 2:
        return 1.0
    dummy = np.zeros((len(train_idx), 1))
    inner_groups = groups[train_idx]
    splits = _splitter(dummy, y_train, inner_groups, n_splits=INNER_FOLDS)
    if not splits:
        return 1.0
    scored = []
    for c_value in C_GRID:
        aucs = []
        for tr_rel, te_rel in splits:
            tr = train_idx[tr_rel]
            te = train_idx[te_rel]
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue
            scores = _fit_text_scores(c0, [texts[i] for i in tr], [texts[i] for i in te], tr, te, y, c_value, extra=extra)
            aucs.append(_safe_auc(y[te], scores))
        finite = [v for v in aucs if math.isfinite(v)]
        scored.append((float(np.mean(finite)) if finite else -math.inf, c_value))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def cv_text_auc(c0, text_rows, y, groups, text_key, shuffle_seed=None, fixed_c=None, extra=None):
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    texts = [row[text_key] for row in text_rows]
    scores_all = np.full((len(y),), np.nan, dtype=np.float32)
    fold_aucs = []
    chosen = []
    dummy = np.zeros((len(y), 1))
    for train_idx, test_idx in _splitter(dummy, y, groups, n_splits=OUTER_FOLDS):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        if isinstance(fixed_c, (list, tuple)):
            c_value = fixed_c[len(chosen)]
        else:
            c_value = fixed_c if fixed_c is not None else _choose_text_c(c0, texts, train_idx, y, groups, extra=extra)
        scores = _fit_text_scores(c0, [texts[i] for i in train_idx], [texts[i] for i in test_idx], train_idx, test_idx, y, c_value, extra=extra)
        scores_all[test_idx] = scores
        fold_aucs.append(_safe_auc(y[test_idx], scores))
        chosen.append(c_value)
    finite = [v for v in fold_aucs if math.isfinite(v)]
    return {
        "auc": float(np.mean(finite)) if finite else math.nan,
        "fold_aucs": fold_aucs,
        "chosen_c": chosen,
        "n": int(len(y)),
        "shuffle_seed": shuffle_seed,
        "scores": scores_all.tolist(),
    }


def _existing_row(probe, protocol, feature_set, **filters):
    rows = probe["gsm"]["rows"] if protocol.startswith("gsm_") else probe["crux"]["rows"]
    for row in rows:
        if row.get("protocol") != protocol or row.get("feature_set") != feature_set:
            continue
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    raise KeyError((protocol, feature_set, filters))


def _best_depth_row(probe, window):
    rows = [r for r in probe["gsm"]["depth_curve"] if r.get("window") == window]
    return max(rows, key=lambda r: r["auc"])


def _e2_head_features(text_rows, head_records, head):
    by_key = {
        (int(r["example_index"]), int(r["equation_index"]), int(r["layer"]), int(r["head"])): r
        for r in head_records
    }
    cats = ["problem_numbers", "generated_numbers", "upcoming_equation_operand_tokens", "other"]
    rows = []
    for row in text_rows:
        rec = by_key.get((int(row["example_index"]), int(row["equation_index"]), int(head["layer"]), int(head["head"])))
        rows.append([float(rec.get(cat, 0.0)) if rec else 0.0 for cat in cats])
    return np.asarray(rows, dtype=np.float32)


def _summarize_head_masses(head_records):
    cats = ["problem_numbers", "generated_numbers", "upcoming_equation_operand_tokens", "other"]
    out = []
    keys = sorted({(int(r["layer"]), int(r["head"])) for r in head_records})
    for layer, head in keys:
        for correct in [False, True]:
            subset = [r for r in head_records if int(r["layer"]) == layer and int(r["head"]) == head and bool(r["correct"]) is correct]
            out.append(
                {
                    "layer": layer,
                    "head": head,
                    "correct": correct,
                    "n": len(subset),
                    **{cat: float(np.mean([float(r.get(cat, 0.0)) for r in subset])) if subset else math.nan for cat in cats},
                }
            )
    return out


def _run_e2_residuals(head_jsonl, text_rows, c0, y, groups, t2_auc, fixed_c=None):
    path = Path(head_jsonl)
    if not path.exists():
        return None
    records = load_jsonl(path)
    heads = sorted({(int(r["layer"]), int(r["head"])) for r in records})
    residuals = []
    for layer, head in heads:
        extra = _e2_head_features(text_rows, records, {"layer": layer, "head": head})
        row = cv_text_auc(c0, text_rows, y, groups, "t2", fixed_c=fixed_c, extra=extra)
        null = cv_text_auc(c0, text_rows, y, groups, "t2", shuffle_seed=1009, fixed_c=fixed_c or 1.0, extra=extra)
        residuals.append(
            {
                "protocol": "gsm_causal_text_plus_head",
                "window": 8,
                "layer": layer,
                "head": head,
                "feature_set": "c0_t2_head",
                "auc": row["auc"],
                "shuffle_auc": null["auc"],
                "n": row["n"],
                "residual_increment": internal_increment(row["auc"], t2_auc),
            }
        )
    return {
        "head_jsonl": str(path),
        "category_means_by_correctness": _summarize_head_masses(records),
        "residual_tests": residuals,
    }


def run_h3e(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    probe = json.load(open(args.probe))
    gsm_h2 = _load_h2(args.gsm_h2_descriptors)
    crux_h2 = _load_h2(args.crux_h2_descriptors)
    gsm_generations = load_jsonl(args.gsm_generations)
    crux_generations = load_jsonl(args.crux_generations)
    gsm_labels = load_jsonl(args.gsm_labels)

    gsm_windows = []
    primary_text_rows = primary_c0 = primary_y = primary_groups = primary_t2_auc = None
    for window in [8, 16]:
        text_rows, c0, y, groups = build_gsm_equation_text_rows(tokenizer, gsm_h2, gsm_generations, gsm_labels, window)
        t_rows = []
        for key, feature_set in [("t1", "c0_t1"), ("t2", "c0_t2")]:
            row = cv_text_auc(c0, text_rows, y, groups, key)
            null = cv_text_auc(c0, text_rows, y, groups, key, shuffle_seed=1009, fixed_c=1.0)
            row.update({"protocol": "gsm_causal_text", "window": window, "feature_set": feature_set, "shuffle_auc": null["auc"]})
            t_rows.append(row)
        full = _existing_row(probe, "gsm_causal_step", "c0_b1_h", window=window)
        b1 = _existing_row(probe, "gsm_causal_step", "c0_b1pca64", window=window)
        h_best = _best_depth_row(probe, window)
        t2 = next(r for r in t_rows if r["feature_set"] == "c0_t2")
        inc = internal_increment(full["auc"], t2["auc"])
        gsm_windows.append(
            {
                "window": window,
                "text_rows": t_rows,
                "reference": {
                    "c0_b1pca64_auc": b1["auc"],
                    "c0_b1pca64_shuffle_auc": b1.get("shuffle_auc"),
                    "c0_h_best_auc": h_best["auc"],
                    "c0_h_best_layer": h_best["layer"],
                    "c0_h_best_shuffle_auc": h_best.get("shuffle_auc"),
                    "c0_b1_h_auc": full["auc"],
                    "c0_b1_h_shuffle_auc": full.get("shuffle_auc"),
                },
                "internal_increment": inc,
                "reading": classify_h3e_reading(t2["auc"], inc),
            }
        )
        if window == 8:
            primary_text_rows, primary_c0, primary_y, primary_groups, primary_t2_auc = text_rows, c0, y, groups, t2["auc"]

    crux_text_rows, crux_c0, crux_y, crux_groups = build_crux_k_text_rows(tokenizer, crux_h2, crux_generations, 100)
    crux_rows = []
    for key, feature_set in [("t1", "c0_t1"), ("t2", "c0_t2")]:
        row = cv_text_auc(crux_c0, crux_text_rows, crux_y, crux_groups, key)
        null = cv_text_auc(crux_c0, crux_text_rows, crux_y, crux_groups, key, shuffle_seed=1009, fixed_c=1.0)
        row.update({"protocol": "crux_early_warning_text", "k": 100, "feature_set": feature_set, "shuffle_auc": null["auc"]})
        crux_rows.append(row)

    primary = next(row for row in gsm_windows if row["window"] == 8)
    primary_t2 = next(r for w in gsm_windows if w["window"] == 8 for r in w["text_rows"] if r["feature_set"] == "c0_t2")
    e2 = _run_e2_residuals(args.e2_head_jsonl, primary_text_rows, primary_c0, primary_y, primary_groups, primary_t2_auc, fixed_c=primary_t2.get("chosen_c"))
    summary = {
        "summary_version": "stage_h3e_v1",
        "preregistered_readings": H3E_PREREGISTERED_READINGS,
        "cv": {
            "outer_group_folds": OUTER_FOLDS,
            "inner_group_folds": INNER_FOLDS,
            "c_grid": C_GRID,
            "standardized_c0_within_split": True,
            "vectorizer_fit_within_training_folds": True,
            "shuffle_seed": 1009,
        },
        "primary_window": 8,
        "gsm": {"windows": gsm_windows},
        "crux": {
            "k": 100,
            "text_rows": crux_rows,
            "reference": {
                "c0_b1pca64_auc": _existing_row(probe, "crux_early_warning", "c0_b1pca64", k=100)["auc"],
                "c0_h_auc": _existing_row(probe, "crux_early_warning", "c0_h", k=100)["auc"],
            },
        },
        "internal_increment": primary["internal_increment"],
        "decision": primary["reading"],
        "e2_triggered": bool(primary["internal_increment"] >= 0.05),
        "e2_status": "not_run" if primary["internal_increment"] < 0.05 else ("complete" if e2 else "required_by_gate"),
        "e2": e2,
    }
    return summary


def update_existing_with_e2(args):
    from transformers import AutoTokenizer

    existing = json.load(open(args.reuse_existing_summary))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    gsm_h2 = _load_h2(args.gsm_h2_descriptors)
    gsm_generations = load_jsonl(args.gsm_generations)
    gsm_labels = load_jsonl(args.gsm_labels)
    text_rows, c0, y, groups = build_gsm_equation_text_rows(tokenizer, gsm_h2, gsm_generations, gsm_labels, 8)
    primary = next(w for w in existing["gsm"]["windows"] if w["window"] == 8)
    t2 = next(r for r in primary["text_rows"] if r["feature_set"] == "c0_t2")
    e2 = _run_e2_residuals(args.e2_head_jsonl, text_rows, c0, y, groups, t2["auc"], fixed_c=t2.get("chosen_c"))
    existing["e2"] = e2
    existing["e2_status"] = "complete" if e2 else existing.get("e2_status", "required_by_gate")
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="results/stage_h3/probes.summary.json")
    ap.add_argument("--gsm-h2-descriptors", default="results/stage_h2/gsm8k/descriptors.npz")
    ap.add_argument("--gsm-generations", default="results/stage_h2/gsm8k/generations.jsonl")
    ap.add_argument("--gsm-labels", default="results/stage_h2/gsm8k/equation_labels_v2.jsonl")
    ap.add_argument("--crux-h2-descriptors", default="results/stage_h2/cruxeval/descriptors.npz")
    ap.add_argument("--crux-generations", default="results/stage_h2/cruxeval/generations.jsonl")
    ap.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--e2-head-jsonl", default="results/stage_h3/h3e_e2_heads/head_category_masses.jsonl")
    ap.add_argument("--reuse-existing-summary", default=None)
    ap.add_argument("--out", default="results/stage_h3/h3e_summary.json")
    args = ap.parse_args()
    summary = update_existing_with_e2(args) if args.reuse_existing_summary else run_h3e(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")
    print(f"internal_increment={summary['internal_increment']:.6f} decision={summary['decision']} e2_status={summary['e2_status']}")


if __name__ == "__main__":
    main()
