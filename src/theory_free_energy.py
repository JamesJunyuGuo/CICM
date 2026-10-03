#!/usr/bin/env python3
"""Validate a free-energy baseline against frozen stale-binding behavior data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2


def free_energy(scores: np.ndarray, temperature: float = 1.0) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return float(temperature * logsumexp(scores / temperature))


def free_energy_decomposition(scores: np.ndarray, temperature: float = 1.0) -> dict:
    scores = np.asarray(scores, dtype=float)
    energy = free_energy(scores, temperature)
    probs = np.exp(scores / temperature - logsumexp(scores / temperature))
    entropy = float(-np.sum(probs * np.log(probs)))
    mean_score = float(np.sum(probs * scores))
    return {
        "free_energy": energy,
        "mean_score": mean_score,
        "entropy": entropy,
        "probabilities": probs,
    }


def predict_probability(alpha: float, k: float, gamma: float = 1.0) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return float(expit(alpha - gamma * math.log(k)))


def _validate_rows(rows: Iterable[dict]) -> list[dict]:
    out = []
    for row in rows:
        k = float(row["k"])
        successes = int(row["successes"])
        total = int(row["total"])
        if k <= 0 or total <= 0 or not 0 <= successes <= total:
            raise ValueError(f"invalid binomial row: {row}")
        out.append({"k": k, "successes": successes, "total": total})
    if not out:
        raise ValueError("at least one row is required")
    return out


def _boundary_correct(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, bool]:
    y = np.asarray([row["successes"] for row in rows], dtype=float)
    n = np.asarray([row["total"] for row in rows], dtype=float)
    boundary = bool(y.sum() == 0 or y.sum() == n.sum())
    if boundary:
        # A Jeffreys half-count keeps a fully separated diagnostic finite.
        y = y + 0.5
        n = n + 1.0
    return y, n, boundary


def _binomial_log_likelihood(rows: list[dict], alpha: float, gamma: float) -> float:
    y, n, _ = _boundary_correct(rows)
    eta = np.asarray([alpha - gamma * math.log(row["k"]) for row in rows])
    return float(np.sum(y * (-np.logaddexp(0.0, -eta)) + (n - y) * (-np.logaddexp(0.0, eta))))


def fit_fixed_gamma(rows: Iterable[dict], gamma: float = 1.0) -> dict:
    rows = _validate_rows(rows)
    y, n, boundary = _boundary_correct(rows)
    log_k = np.log([row["k"] for row in rows])

    def score(alpha: float) -> float:
        return float(np.sum(y - n * expit(alpha - gamma * log_k)))

    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0
    breakpoint = math.exp(alpha / gamma) if gamma > 0 else math.inf
    return {
        "alpha": alpha,
        "gamma": gamma,
        "breakpoint_k": breakpoint,
        "log_likelihood": _binomial_log_likelihood(rows, alpha, gamma),
        "ceiling_limited": boundary or int(sum(row["total"] - row["successes"] for row in rows)) < 5,
        "floor_limited": boundary or int(sum(row["successes"] for row in rows)) < 5,
    }


def fit_log_k_model(rows: Iterable[dict]) -> dict:
    rows = _validate_rows(rows)
    y, n, boundary = _boundary_correct(rows)
    log_k = np.log([row["k"] for row in rows])
    fixed = fit_fixed_gamma(rows)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        alpha, gamma = theta
        eta = alpha - gamma * log_k
        p = expit(eta)
        nll = -np.sum(y * (-np.logaddexp(0.0, -eta)) + (n - y) * (-np.logaddexp(0.0, eta)))
        residual = n * p - y
        grad = np.asarray([np.sum(residual), -np.sum(residual * log_k)])
        return float(nll), grad

    result = minimize(
        lambda theta: objective(theta)[0],
        np.asarray([fixed["alpha"], 1.0]),
        jac=lambda theta: objective(theta)[1],
        method="BFGS",
    )
    if not result.success:
        raise RuntimeError(f"log-k model fit failed: {result.message}")
    alpha, gamma = map(float, result.x)
    covariance = np.asarray(result.hess_inv, dtype=float)
    gamma_se = float(math.sqrt(max(covariance[1, 1], 0.0)))
    return {
        "alpha": alpha,
        "gamma": gamma,
        "gamma_se": gamma_se,
        "gamma_ci95": [gamma - 1.96 * gamma_se, gamma + 1.96 * gamma_se],
        "log_likelihood": _binomial_log_likelihood(rows, alpha, gamma),
        "boundary_corrected": boundary,
    }


def _conditional_row(k: int, current: int, stale: int) -> dict:
    return {"k": k, "successes": current, "total": current + stale}


def load_validation_datasets(root: Path) -> list[dict]:
    p0 = json.loads((root / "results/stage_e/qwen_p0_errors.summary.json").read_text())
    scale = json.loads((root / "results/figures/F8_openrouter_scale_sweep.data.json").read_text())
    natural = json.loads(
        (root / "results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json").read_text()
    )
    datasets = []

    for n_lines in (20, 40, 80):
        rows = []
        for cell in p0["cells"]:
            if cell["condition"] != "interference" or cell["n_lines"] != n_lines:
                continue
            current = cell["n"] - cell["n_wrong"]
            rows.append(_conditional_row(cell["interference_load"], current, cell["within_stale"]))
        rows.sort(key=lambda row: row["k"])
        datasets.append({
            "id": f"controlled_qwen7b_{n_lines}lines",
            "family": "controlled",
            "model": p0["model"],
            "train": rows[:2],
            "heldout": rows[2:],
        })

    by_model = {}
    for cell in scale["accuracy_by_i"]:
        current = cell["n"] - cell["wrong"]
        by_model.setdefault(cell["model"], []).append(
            _conditional_row(cell["interference_load"], current, cell["within_stale_errors"])
        )
    for model, rows in sorted(by_model.items()):
        rows.sort(key=lambda row: row["k"])
        short = model.split("/")[-1].replace(".", "_")
        datasets.append({
            "id": f"scale_{short}",
            "family": "scale_sweep",
            "model": model,
            "train": rows[:3],
            "heldout": rows[3:],
        })

    rows = []
    for k, cell in sorted(natural["by_dose"].items(), key=lambda item: int(item[0])):
        counts = cell["counts"]
        rows.append(_conditional_row(int(k), counts["correct_current"], counts["within_stale"]))
    datasets.append({
        "id": "cicm_natural_qwen7b",
        "family": "natural",
        "model": natural["model"],
        "train": rows[:3],
        "heldout": rows[3:],
    })
    return datasets


def validate_dataset(dataset: dict, rng: np.random.Generator, bootstrap_reps: int) -> dict:
    train = _validate_rows(dataset["train"])
    heldout = _validate_rows(dataset["heldout"])
    fixed = fit_fixed_gamma(train)
    flexible = fit_log_k_model(train)
    lr = max(0.0, 2.0 * (flexible["log_likelihood"] - fixed["log_likelihood"]))

    simulated = {row["k"]: [] for row in heldout}
    parameter_probs = {row["k"]: [] for row in heldout}
    for _ in range(bootstrap_reps):
        boot_train = []
        for row in train:
            p = predict_probability(fixed["alpha"], row["k"])
            boot_train.append({**row, "successes": int(rng.binomial(row["total"], p))})
        boot_fit = fit_fixed_gamma(boot_train)
        for row in heldout:
            p = predict_probability(boot_fit["alpha"], row["k"])
            parameter_probs[row["k"]].append(p)
            simulated[row["k"]].append(rng.binomial(row["total"], p) / row["total"])

    heldout_results = []
    for row in heldout:
        observed = row["successes"] / row["total"]
        predicted = predict_probability(fixed["alpha"], row["k"])
        pred_ci = np.quantile(simulated[row["k"]], [0.025, 0.975]).tolist()
        param_ci = np.quantile(parameter_probs[row["k"]], [0.025, 0.975]).tolist()
        heldout_results.append({
            **row,
            "observed_rate": observed,
            "predicted_rate": predicted,
            "parameter_ci95": param_ci,
            "predictive_ci95": pred_ci,
            "inside_predictive_interval": bool(pred_ci[0] <= observed <= pred_ci[1]),
            "absolute_error": abs(observed - predicted),
        })

    testable = not fixed["ceiling_limited"] and not fixed["floor_limited"]
    exact_pass = testable and all(row["inside_predictive_interval"] for row in heldout_results)
    if not testable:
        status = "ceiling_or_floor_limited"
    elif exact_pass:
        status = "heldout_pass"
    else:
        status = "heldout_fail"
    return {
        **{key: dataset[key] for key in ("id", "family", "model")},
        "estimand": "P(current | current or within-stale)",
        "train": train,
        "heldout": heldout_results,
        "fixed_exchangeable": fixed,
        "free_exponent": flexible,
        "likelihood_ratio_gamma_1": {"statistic": lr, "df": 1, "p_value": float(chi2.sf(lr, 1))},
        "status": status,
        "heldout_mae": float(np.mean([row["absolute_error"] for row in heldout_results])),
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Free-energy scaling validation",
        "",
        "## Analysis protocol",
        "",
        "Fit `logit P(current | current or stale) = alpha - log(k)` on the lower-dose cells and evaluate it on frozen higher-dose cells that are excluded from parameter fitting. A dataset passes exact scaling only when every held-out rate lies inside its 95% parametric-bootstrap predictive interval. Training pools with fewer than five successes or failures are marked ceiling/floor-limited. A free exponent `gamma` is diagnostic, not a replacement headline.",
        "",
        "## Results",
        "",
        "| Dataset | Train k | Held-out k | alpha | gamma (free) | Held-out MAE | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in summary["datasets"]:
        train_k = ",".join(str(int(row["k"])) for row in result["train"])
        test_k = ",".join(str(int(row["k"])) for row in result["heldout"])
        lines.append(
            f"| {result['id']} | {train_k} | {test_k} | {result['fixed_exchangeable']['alpha']:.3f} | "
            f"{result['free_exponent']['gamma']:.2f} | {result['heldout_mae']:.3f} | {result['status']} |"
        )
    lines.extend(["", "### Held-out cells", "", "| Dataset | k | Observed | Predicted | 95% predictive interval | In interval |", "|---|---:|---:|---:|---:|:---:|"])
    for result in summary["datasets"]:
        for row in result["heldout"]:
            lo, hi = row["predictive_ci95"]
            lines.append(
                f"| {result['id']} | {int(row['k'])} | {row['observed_rate']:.3f} | {row['predicted_rate']:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | {'yes' if row['inside_predictive_interval'] else 'no'} |"
            )
    counts = summary["status_counts"]
    lines.extend([
        "",
        "## Reading",
        "",
        f"Exact exchangeable scaling passes {counts.get('heldout_pass', 0)} testable datasets, fails {counts.get('heldout_fail', 0)}, and leaves {counts.get('ceiling_or_floor_limited', 0)} ceiling/floor-limited. A pass supports a fixed-margin multiplicity account in that regime. A failure means competitor scores or the current margin change with dose/context; it does not negate the free-energy competition identity.",
        "",
        "The validation is deliberately conditional on current-versus-within-stale outcomes. Cross-slot and other errors are excluded because they belong to different candidate partitions.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("results/theory_free_energy"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = [validate_dataset(dataset, rng, args.bootstrap_reps) for dataset in load_validation_datasets(root)]
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {
        "analysis": "free_energy_exchangeable_competitor_validation",
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "frozen_inputs": [
            "results/stage_e/qwen_p0_errors.summary.json",
            "results/figures/F8_openrouter_scale_sweep.data.json",
            "results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json",
        ],
        "status_counts": counts,
        "datasets": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, out_dir / "REPORT.md")
    print(json.dumps({"status_counts": counts, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
