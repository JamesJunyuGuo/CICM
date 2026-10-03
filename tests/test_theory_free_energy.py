import math

import numpy as np

from theory_free_energy import (
    fit_fixed_gamma,
    fit_log_k_model,
    free_energy,
    free_energy_decomposition,
    predict_probability,
)


def test_free_energy_matches_score_plus_entropy_decomposition():
    scores = np.asarray([0.2, 1.1, -0.4])
    temperature = 0.7

    direct = free_energy(scores, temperature)
    decomposed = free_energy_decomposition(scores, temperature)

    assert np.isclose(direct, decomposed["mean_score"] + temperature * decomposed["entropy"])
    assert np.isclose(direct, decomposed["free_energy"])
    assert scores.max() <= direct <= scores.max() + temperature * math.log(len(scores))


def test_fixed_gamma_recovers_exchangeable_competitor_law():
    alpha = math.log(12.0)
    rows = []
    for k, n in [(1, 1000), (2, 1000), (4, 1000)]:
        p = predict_probability(alpha, k, gamma=1.0)
        rows.append({"k": k, "successes": round(n * p), "total": n})

    fit = fit_fixed_gamma(rows, gamma=1.0)

    assert abs(fit["alpha"] - alpha) < 0.02
    assert abs(fit["breakpoint_k"] - 12.0) < 0.25
    assert np.isclose(predict_probability(fit["alpha"], 12.0), 0.5, atol=0.01)


def test_free_gamma_recovers_nonexchangeable_scaling_exponent():
    alpha = 1.4
    gamma = 0.55
    rows = []
    for k, n in [(1, 5000), (2, 5000), (4, 5000), (8, 5000)]:
        p = predict_probability(alpha, k, gamma=gamma)
        rows.append({"k": k, "successes": round(n * p), "total": n})

    fit = fit_log_k_model(rows)

    assert abs(fit["alpha"] - alpha) < 0.02
    assert abs(fit["gamma"] - gamma) < 0.02


def test_fixed_gamma_fit_handles_all_success_training_cells():
    rows = [
        {"k": 2, "successes": 50, "total": 50},
        {"k": 4, "successes": 50, "total": 50},
    ]

    fit = fit_fixed_gamma(rows, gamma=1.0)

    assert fit["ceiling_limited"] is True
    assert math.isfinite(fit["alpha"])
    assert 0.99 < predict_probability(fit["alpha"], 2) < 1.0
