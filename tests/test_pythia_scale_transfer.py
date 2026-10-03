import numpy as np

from pythia_scale_transfer import (
    fixed_k_pool_gate,
    pooled_residualize_matrix,
    summarize_head_transfer,
)


def test_fixed_k_pool_gate_requires_each_template():
    rows = []
    for template in ("arrow", "current", "latest"):
        rows.extend(
            {"k": 4, "variant": "single", "template": template, "label": label}
            for label in (["correct_current"] * 50 + ["within_stale"] * 40)
        )
    passing = fixed_k_pool_gate(rows, fixed_k=4)
    assert passing["gate_pass"]

    rows[-1]["label"] = "other"
    failing = fixed_k_pool_gate(rows, fixed_k=4)
    assert not failing["gate_pass"]


def test_pooled_residualization_removes_shared_log_length_trend():
    lengths = np.asarray([10, 20, 40, 80], dtype=float)
    values = np.column_stack([
        3.0 + 2.0 * np.log(lengths),
        -1.0 - 0.5 * np.log(lengths),
    ])
    residuals = pooled_residualize_matrix(values, lengths)
    assert np.max(np.abs(residuals)) < 1e-10


def test_head_transfer_uses_familywise_shuffle_and_same_head_conjunction():
    rng = np.random.default_rng(7)
    n = 120
    labels = np.asarray(["correct_current"] * (n // 2) + ["within_stale"] * (n // 2))
    strata = np.asarray([f"s{i % 6}" for i in range(n)])
    lengths = np.linspace(90, 180, n)
    qk = rng.normal(0.0, 0.1, size=(n, 2, 2))
    ratio = rng.normal(0.5, 0.03, size=(n, 2, 2))
    failure = labels == "within_stale"
    qk[failure, 1, 0] -= 1.0
    ratio[failure, 1, 0] += 0.3

    summary = summarize_head_transfer(
        qk,
        ratio,
        labels,
        lengths,
        strata,
        n_shuffle=250,
        seed=11,
    )

    assert {"layer": 1, "head": 0} in summary["length_controlled"]["qk_significant_heads"]
    assert {"layer": 1, "head": 0} in summary["length_controlled"]["attention_significant_heads"]
    assert {"layer": 1, "head": 0} in summary["length_controlled"]["same_head_conjunction"]

