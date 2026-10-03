import numpy as np
import torch

from stale_binding_correction import (
    _mediation_summary,
    adjudicate,
    apply_head_shifts,
    choose_operating_alpha,
    grouped_probe_cv,
    make_matched_random_directions,
    protocol_split,
)


def test_adjudicate_separates_output_correction_from_mechanism_confirmation():
    output_gate = {
        "identity_exact_zero": True,
        "targeted_gain_ci_above_zero": True,
        "correction_exceeds_random95": True,
        "correct_preservation": True,
        "no_overwrite_drop_le_002": True,
        "direction_specificity": True,
    }

    assert adjudicate(output_gate, attention_mediation=True, upstream_confirmation=True) == (
        "limited_mechanism_guided_correction"
    )
    assert adjudicate(output_gate, attention_mediation=False, upstream_confirmation=True) == (
        "head_state_output_correction_mediation_unverified"
    )
    assert adjudicate(output_gate, attention_mediation=True, upstream_confirmation=False) == (
        "head_state_output_correction_mediation_unverified"
    )


def test_adjudicate_rejects_failed_output_gate():
    output_gate = {
        "identity_exact_zero": True,
        "targeted_gain_ci_above_zero": False,
    }

    assert adjudicate(output_gate, attention_mediation=True, upstream_confirmation=True) == (
        "correction_gate_failed"
    )


def test_mediation_summary_uses_multiple_random_directions():
    baseline = [
        {"downstream_attention_margin": 0.0},
        {"downstream_attention_margin": 0.0},
        {"downstream_attention_margin": 0.0},
    ]
    targeted = [
        {"downstream_attention_margin": 0.6},
        {"downstream_attention_margin": 0.5},
        {"downstream_attention_margin": 0.4},
    ]
    random_runs = [
        [
            {"downstream_attention_margin": 0.1},
            {"downstream_attention_margin": 0.0},
            {"downstream_attention_margin": -0.1},
        ],
        [
            {"downstream_attention_margin": 0.2},
            {"downstream_attention_margin": 0.1},
            {"downstream_attention_margin": 0.0},
        ],
    ]

    summary = _mediation_summary(baseline, targeted, random_runs, n_boot=200, seed=3)

    assert summary["n_random_directions"] == 2
    assert summary["targeted_mean_exceeds_random95"]
    assert summary["targeted_minus_random"]["mean"] > 0


def test_apply_head_shifts_changes_only_requested_query_and_head_slices():
    hidden = torch.zeros(2, 3, 8)
    query = torch.tensor([1, 2])
    directions = {1: np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)}
    scales = {1: 0.5}

    shifted = apply_head_shifts(
        hidden,
        query,
        directions,
        scales,
        alpha=2.0,
        n_heads=2,
    )

    expected = torch.zeros_like(hidden)
    expected[0, 1, 4:] = torch.tensor([1.0, -2.0, 3.0, -4.0])
    expected[1, 2, 4:] = torch.tensor([1.0, -2.0, 3.0, -4.0])
    assert torch.equal(shifted, expected)


def test_apply_head_shifts_alpha_zero_is_exact_identity_object():
    hidden = torch.randn(1, 2, 8)
    shifted = apply_head_shifts(
        hidden,
        torch.tensor([1]),
        {0: np.ones(4, dtype=np.float32)},
        {0: 1.0},
        alpha=0.0,
        n_heads=2,
    )

    assert shifted is hidden


def test_matched_random_directions_preserve_each_head_shift_norm():
    directions = {
        (2, 1): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        (4, 0): np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    scales = {(2, 1): 0.25, (4, 0): 1.75}

    random_sets = make_matched_random_directions(directions, scales, n_random=4, seed=7)

    assert len(random_sets) == 4
    for random_directions in random_sets:
        assert set(random_directions) == set(directions)
        for key, direction in random_directions.items():
            assert np.isclose(np.linalg.norm(direction), 1.0)
            assert np.isclose(
                np.linalg.norm(direction * scales[key]),
                scales[key],
            )


def test_choose_operating_alpha_enforces_preservation_then_maximizes_net_gain():
    rows = [
        {"alpha": 0.25, "correction": 0.20, "correct_preservation": 0.98},
        {"alpha": 0.50, "correction": 0.35, "correct_preservation": 0.96},
        {"alpha": 1.00, "correction": 0.60, "correct_preservation": 0.90},
        {"alpha": 2.00, "correction": 0.40, "correct_preservation": 0.96},
    ]

    selected = choose_operating_alpha(rows, minimum_preservation=0.95)

    assert selected["alpha"] == 2.0
    assert np.isclose(selected["net_gain"], 0.36)


def test_choose_operating_alpha_breaks_ties_toward_smaller_intervention():
    rows = [
        {"alpha": 0.5, "correction": 0.30, "correct_preservation": 0.95},
        {"alpha": 1.0, "correction": 0.30, "correct_preservation": 0.95},
    ]

    assert choose_operating_alpha(rows, minimum_preservation=0.95)["alpha"] == 0.5


def test_protocol_split_keeps_frozen_evaluation_rows_untouched():
    assert protocol_split("example-a", "evaluation") == "evaluation"
    assert protocol_split("example-a", "discovery") in {"fit", "calibration"}
    assert protocol_split("example-a", "discovery") == protocol_split(
        "example-a", "discovery"
    )


def test_grouped_probe_cv_holds_out_current_value_groups():
    rng = np.random.default_rng(3)
    rows = []
    values = []
    for group in range(10):
        for label in (0, 1):
            for repeat in range(2):
                rows.append(
                    {
                        "label": "correct_current" if label else "within_stale",
                        "gold_token_id": group,
                    }
                )
                values.append(
                    np.array([3.0 * label, float(group) / 10.0, repeat], dtype=np.float32)
                    + rng.normal(scale=0.05, size=3)
                )
    activations = {(0, 0): np.asarray(values, dtype=np.float32)}

    result = grouped_probe_cv(
        activations,
        rows,
        list(range(len(rows))),
        n_shuffle=50,
        seed=9,
    )

    assert result["n_splits"] == 5
    assert result["group"] == "current value token id"
    assert result["aggregate_auc"] > result["shuffle_interval95"][1]
