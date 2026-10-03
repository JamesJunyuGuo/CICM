from collections import Counter

import numpy as np

from stale_binding_attenuation import (
    choose_gamma,
    choose_gamma_or_identity,
    clustered_paired_bootstrap,
    make_layer_matched_random_sets,
    template_support_gate,
    transition_counts,
)


def test_clustered_bootstrap_reports_semantic_group_count():
    values = np.array([1.0, 1.0, -1.0, -1.0])
    groups = ["a", "a", "b", "b"]

    result = clustered_paired_bootstrap(values, groups, n_boot=100, seed=4)

    assert result["n"] == 4
    assert result["n_clusters"] == 2
    assert result["mean"] == 0.0


def test_choose_gamma_enforces_preservation_and_prefers_least_attenuation_on_tie():
    curve = [
        {"gamma": 0.0, "correct_preservation": 0.90, "net_accuracy_gain": 0.10},
        {"gamma": 0.5, "correct_preservation": 0.96, "net_accuracy_gain": 0.04},
        {"gamma": 0.75, "correct_preservation": 0.98, "net_accuracy_gain": 0.04},
        {"gamma": 0.9, "correct_preservation": 1.00, "net_accuracy_gain": 0.01},
    ]

    assert choose_gamma(curve, minimum_preservation=0.95)["gamma"] == 0.75


def test_choose_gamma_or_identity_records_no_safe_operating_point():
    curve = [
        {
            "gamma": 0.5,
            "correction": 0.4,
            "correct_preservation": 0.8,
            "baseline_accuracy": 0.5,
            "arm_accuracy": 0.6,
            "net_accuracy_gain": 0.1,
        }
    ]

    operating, status = choose_gamma_or_identity(curve, minimum_preservation=0.95)

    assert status == "no_eligible_gamma_identity_fallback"
    assert operating == {
        "gamma": 1.0,
        "correction": 0.0,
        "correct_preservation": 1.0,
        "baseline_accuracy": 0.5,
        "arm_accuracy": 0.5,
        "net_accuracy_gain": 0.0,
    }


def test_layer_matched_random_sets_preserve_counts_and_exclude_target_set():
    target = [(21, 1), (21, 4), (22, 7), (25, 3)]
    samples = make_layer_matched_random_sets(target, n_heads=12, n_random=16, seed=3)
    expected = Counter(layer for layer, _ in target)

    assert len(samples) == 16
    assert len({tuple(sample) for sample in samples}) == 16
    for sample in samples:
        assert Counter(layer for layer, _ in sample) == expected
        assert len(sample) == len(set(sample))
        assert set(sample) != set(target)


def test_transition_counts_aligns_rows_by_id():
    baseline = [
        {"id": "a", "label": "within_stale"},
        {"id": "b", "label": "correct_current"},
    ]
    arm = [
        {"id": "a", "label": "correct_current"},
        {"id": "b", "label": "other"},
    ]

    assert transition_counts(baseline, arm) == {
        "correct_current->other": 1,
        "within_stale->correct_current": 1,
    }


def test_template_support_requires_positive_gain_in_two_templates():
    assert template_support_gate(
        {
            "arrow": {"net_accuracy_gain": 0.02},
            "current": {"net_accuracy_gain": 0.01},
            "latest": {"net_accuracy_gain": 0.0},
        }
    )
    assert not template_support_gate(
        {
            "arrow": {"net_accuracy_gain": 0.02},
            "current": {"net_accuracy_gain": 0.0},
            "latest": {"net_accuracy_gain": -0.01},
        }
    )
