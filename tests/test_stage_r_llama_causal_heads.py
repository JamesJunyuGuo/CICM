import json

import numpy as np
import torch

from stage_r_llama_causal_heads import (
    AdaptiveRoutingContext,
    _gemma2_adaptive_eager_attention,
    _identity_diagnostics,
    _identity_gate_passes,
    _llama_adaptive_eager_attention,
    _load_completed_confirmation_task,
    apply_adaptive_margin_bias,
    choose_validation_configuration,
    confirmation_gate,
    discovery_pool_gate,
    make_layer_matched_random_head_sets,
    partition_candidate_discovery,
    rank_causal_heads,
    split_calibration_discovery_validation,
    split_r12_rows,
    validate_model_family,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_completed_confirmation_task_requires_exact_complete_artifacts(tmp_path):
    rows = [
        {"id": "a", "task_type": "retrieval"},
        {"id": "b", "task_type": "retrieval"},
    ]
    arm_rows = [
        {"id": "a", "task_type": "retrieval", "label": "correct_current"},
        {"id": "b", "task_type": "retrieval", "label": "within_stale"},
    ]
    for name in ("baseline", "routed", "opposite", "recap"):
        _write_jsonl(tmp_path / f"{name}_rows.jsonl", arm_rows)
    _write_jsonl(
        tmp_path / "random_position_controls.jsonl",
        [
            {"index": 0, "net_accuracy_gain": 0.0, "correct_preservation": 1.0},
            {"index": 1, "net_accuracy_gain": 0.0, "correct_preservation": 1.0},
        ],
    )
    expected_heads = [[(1, 2)], [(3, 4)]]
    _write_jsonl(
        tmp_path / "random_head_controls.jsonl",
        [
            {
                "index": index,
                "heads": [{"layer": layer, "head": head} for layer, head in heads],
                "net_accuracy_gain": 0.0,
                "correct_preservation": 1.0,
            }
            for index, heads in enumerate(expected_heads)
        ],
    )

    loaded = _load_completed_confirmation_task(
        tmp_path,
        rows,
        n_random_positions=2,
        expected_random_head_sets=expected_heads,
    )

    assert loaded is not None
    assert [row["id"] for row in loaded["baseline"]] == ["a", "b"]


def test_completed_confirmation_task_rejects_partial_checkpoint(tmp_path):
    rows = [{"id": "a", "task_type": "retrieval"}]
    _write_jsonl(
        tmp_path / "baseline_rows.jsonl",
        [{"id": "a", "task_type": "retrieval", "label": "correct_current"}],
    )

    with __import__("pytest").raises(RuntimeError, match="partial confirmation checkpoint"):
        _load_completed_confirmation_task(
            tmp_path,
            rows,
            n_random_positions=1,
            expected_random_head_sets=[[(0, 0)]],
        )


def test_identity_diagnostics_separates_score_drift_from_response_changes():
    baseline = [
        {"id": "same", "response": "vegan"},
        {"id": "changed", "response": "jazz"},
    ]
    identity = [
        {"id": "same", "response": "vegan"},
        {"id": "changed", "response": "rock"},
    ]
    baseline_scores = np.asarray([[1.0, 2.0], [3.0, np.nan]])
    identity_scores = np.asarray([[1.0, 2.25], [3.0, np.nan]])

    diagnostic = _identity_diagnostics(
        baseline, identity, baseline_scores, identity_scores
    )

    assert diagnostic["max_abs_first_token_score_change"] == 0.25
    assert diagnostic["different_finite_score_entries"] == 1
    assert diagnostic["baseline_nonfinite_score_entries"] == 1
    assert diagnostic["identity_nonfinite_score_entries"] == 1
    assert diagnostic["identity_response_changes"] == 1
    assert diagnostic["changed_response_ids"] == ["changed"]


def test_identity_gate_requires_exact_finite_scores_and_unchanged_responses():
    exact = {
        "score_arrays_exact": True,
        "identity_response_changes": 0,
    }
    score_drift = {**exact, "score_arrays_exact": False}
    response_drift = {**exact, "identity_response_changes": 1}

    assert _identity_gate_passes(exact)
    assert not _identity_gate_passes(score_drift)
    assert not _identity_gate_passes(response_drift)


class _AttentionModule:
    def __init__(self, groups=2, layer_idx=0, head_dim=4):
        self.num_key_value_groups = groups
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.training = False


def _attention_inputs():
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, 4, 3, 4, generator=generator)
    key = torch.randn(2, 2, 3, 4, generator=generator)
    value = torch.randn(2, 2, 3, 4, generator=generator)
    mask = torch.zeros(2, 1, 3, 3)
    mask[:, :, 0, 1:] = torch.finfo(mask.dtype).min
    mask[:, :, 1, 2:] = torch.finfo(mask.dtype).min
    return query, key, value, mask


def test_qwen2_adaptive_attention_is_native_exact_at_noop():
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    module = _AttentionModule()
    query, key, value, mask = _attention_inputs()
    expected_output, expected_weights = eager_attention_forward(
        module, query, key, value, mask, scaling=0.5
    )
    actual_output, actual_weights = _llama_adaptive_eager_attention(
        module, query, key, value, mask, scaling=0.5
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_gemma2_adaptive_attention_preserves_softcap_at_noop():
    from transformers.models.gemma2.modeling_gemma2 import eager_attention_forward

    module = _AttentionModule()
    query, key, value, mask = _attention_inputs()
    expected_output, expected_weights = eager_attention_forward(
        module, query, key, value, mask, scaling=0.5, softcap=3.0
    )
    actual_output, actual_weights = _gemma2_adaptive_eager_attention(
        module, query, key, value, mask, scaling=0.5, softcap=3.0
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


class _FirstTokenTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        value = text.strip().lower()
        return [{"paleo": 1, "pescatarian": 1, "vegan": 2, "keto": 3}[value]]


def test_candidate_partition_excludes_only_first_token_collisions():
    rows = [
        {"id": "collision", "current_value": "paleo", "stale_values": ["pescatarian"]},
        {"id": "eligible", "current_value": "vegan", "stale_values": ["keto"]},
    ]
    routes = [{"row": 0}, {"row": 1}]
    baseline = [{"label": "within_stale"}, {"label": "correct_current"}]

    kept_rows, kept_routes, kept_baseline, exclusions = partition_candidate_discovery(
        _FirstTokenTokenizer(), rows, routes, baseline
    )

    assert [row["id"] for row in kept_rows] == ["eligible"]
    assert kept_routes == [{"row": 1}]
    assert kept_baseline == [{"label": "correct_current"}]
    assert exclusions == [
        {"id": "collision", "reason": "current_stale_first_token_collision"}
    ]


def test_discovery_pool_gate_supports_asymmetric_base_rates():
    passed = discovery_pool_gate(9, 79, minimum_correct=5, minimum_within_stale=20)
    failed = discovery_pool_gate(9, 79, minimum_correct=20, minimum_within_stale=20)

    assert passed["all_pass"]
    assert passed["correct_pass"] and passed["within_stale_pass"]
    assert not failed["all_pass"]
    assert not failed["correct_pass"] and failed["within_stale_pass"]


def _row(cell, index):
    return {
        "id": f"row-{cell}-{index}",
        "factorial_cell": {
            "same_slot_stale_distance_bin": cell[0],
            "recent_other_slot_distance_bin": cell[1],
        },
    }


def test_discovery_validation_split_is_balanced_disjoint_and_deterministic():
    cells = [(same, other) for same in ("near", "far") for other in ("far", "mid", "near2")]
    rows = [_row(cell, index) for cell in cells for index in range(4)]

    discovery_a, validation_a = split_calibration_discovery_validation(rows)
    discovery_b, validation_b = split_calibration_discovery_validation(list(reversed(rows)))

    assert [row["id"] for row in discovery_a] == [row["id"] for row in discovery_b]
    assert [row["id"] for row in validation_a] == [row["id"] for row in validation_b]
    assert set(row["id"] for row in discovery_a).isdisjoint(
        row["id"] for row in validation_a
    )
    assert len(discovery_a) == len(validation_a) == 12
    assert all(
        sum(row["factorial_cell"] == _row(cell, 0)["factorial_cell"] for row in discovery_a) == 2
        for cell in cells
    )


def test_r12_limits_only_canonical_calibration_rows():
    cells = [(same, other) for same in ("near", "far") for other in ("far", "mid", "near2")]
    rows = [_row(cell, index) for cell in cells for index in range(10)]

    full_calibration, confirmation = split_r12_rows(rows, 0.4)
    smoke_calibration, smoke_confirmation = split_r12_rows(rows, 0.4, 2)

    assert len(full_calibration) == 24
    assert len(smoke_calibration) == 12
    assert {row["id"] for row in smoke_calibration}.issubset(
        row["id"] for row in full_calibration
    )
    assert [row["id"] for row in smoke_confirmation] == [
        row["id"] for row in confirmation
    ]


def test_adaptive_margin_sets_requested_aggregate_odds_for_selected_head():
    scores = torch.tensor(
        [[[[0.0, 1.0, -0.5, 0.25]], [[0.2, -0.1, 0.4, 0.3]]]],
        dtype=torch.float32,
    )
    context = AdaptiveRoutingContext(
        margin=2.0,
        stale_positions=((0, 1),),
        current_positions=((2,),),
        query_positions=(0,),
        heads_by_layer={0: (0,)},
        direction=1,
        max_beta=8.0,
    )

    routed = apply_adaptive_margin_bias(scores, layer_idx=0, context=context)
    before = torch.logsumexp(scores[0, 0, 0, [2]], 0) - torch.logsumexp(
        scores[0, 0, 0, [0, 1]], 0
    )
    after = torch.logsumexp(routed[0, 0, 0, [2]], 0) - torch.logsumexp(
        routed[0, 0, 0, [0, 1]], 0
    )

    assert before < 2.0
    torch.testing.assert_close(after, torch.tensor(2.0))
    torch.testing.assert_close(routed[0, 1], scores[0, 1])


def test_adaptive_margin_identity_gate_is_exact():
    scores = torch.randn(2, 3, 4, 5)
    context = AdaptiveRoutingContext(
        margin=4.0,
        stale_positions=((0,), (1,)),
        current_positions=((2,), (3,)),
        query_positions=(3, 3),
        heads_by_layer={0: (0, 1, 2)},
        direction=1,
        max_beta=8.0,
        gate_scale=0.0,
    )
    routed = apply_adaptive_margin_bias(scores, layer_idx=0, context=context)
    assert routed is scores


def test_adaptive_margin_exposes_per_head_gate_gradients():
    scores = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    gates = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    context = AdaptiveRoutingContext(
        margin=2.0,
        stale_positions=((0,),),
        current_positions=((1,),),
        query_positions=(0,),
        heads_by_layer={},
        gates_by_layer={0: gates},
    )

    routed = apply_adaptive_margin_bias(scores, layer_idx=0, context=context)
    objective = (routed[0, :, 0, 1] - routed[0, :, 0, 0]).sum()
    gradient = torch.autograd.grad(objective, gates)[0]

    torch.testing.assert_close(gradient, torch.tensor([2.0, 2.0]))


def test_rank_causal_heads_uses_gradient_only_with_stable_ties():
    failure = np.array([[0.1, 0.5, -0.2], [0.5, 0.5, 0.3]])
    correct = np.zeros_like(failure)
    mass = np.full_like(failure, 0.2)

    ranked = rank_causal_heads(failure, correct, mass)

    assert [(row["layer"], row["head"]) for row in ranked[:4]] == [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert ranked[0]["failure_gradient"] == 0.5
    assert ranked[0]["correct_gradient"] == 0.0


def test_choose_validation_configuration_respects_preservation_then_ties():
    rows = [
        {"size": 4, "margin": 2.0, "net_accuracy_gain": 0.20, "correct_preservation": 0.94},
        {"size": 8, "margin": 2.0, "net_accuracy_gain": 0.15, "correct_preservation": 0.98},
        {"size": 4, "margin": 4.0, "net_accuracy_gain": 0.15, "correct_preservation": 0.98},
        {"size": 4, "margin": 2.0, "net_accuracy_gain": 0.15, "correct_preservation": 0.98},
    ]

    selected = choose_validation_configuration(rows, minimum_preservation=0.95)

    assert selected["size"] == 4
    assert selected["margin"] == 2.0


def test_layer_matched_random_heads_preserve_counts_and_exclude_selected():
    selected = [(2, 0), (2, 1), (5, 3)]
    controls = make_layer_matched_random_head_sets(
        selected, n_heads=8, n_random=12, seed=17
    )

    assert len(controls) == 12
    assert len({tuple(control) for control in controls}) == 12
    for control in controls:
        assert len(control) == 3
        assert not set(control).intersection(selected)
        assert sum(layer == 2 for layer, _ in control) == 2
        assert sum(layer == 5 for layer, _ in control) == 1


def test_confirmation_gate_requires_both_random_control_bounds():
    metric = {
        "baseline_counts": {"correct_current": 120, "within_stale": 140},
        "net_accuracy_gain": 0.2,
        "paired_net_gain": {"ci": [0.1, 0.3]},
        "correct_preservation": 0.97,
    }
    cells = {
        str(index): {"net_accuracy_gain": 0.1 if index < 4 else 0.0}
        for index in range(6)
    }
    random_positions = {"net_accuracy_gain": {"interval95": [-0.02, 0.03]}}
    random_heads = {"net_accuracy_gain": {"interval95": [-0.01, 0.04]}}

    passed = confirmation_gate(
        metric,
        cells,
        random_positions,
        random_heads,
        route_audit_exact=True,
        identity_exact=True,
        minimum_pool=100,
        minimum_preservation=0.95,
    )
    failed = confirmation_gate(
        metric,
        cells,
        random_positions,
        {"net_accuracy_gain": {"interval95": [0.18, 0.25]}},
        route_audit_exact=True,
        identity_exact=True,
        minimum_pool=100,
        minimum_preservation=0.95,
    )

    assert passed["all_pass"]
    assert not failed["exceeds_random_head95"]
    assert not failed["all_pass"]


def test_confirmation_gate_can_report_identity_without_requiring_it():
    metric = {
        "baseline_counts": {"correct_current": 120, "within_stale": 140},
        "net_accuracy_gain": 0.2,
        "paired_net_gain": {"ci": [0.1, 0.3]},
        "correct_preservation": 0.97,
    }
    cells = {
        str(index): {"net_accuracy_gain": 0.1 if index < 4 else 0.0}
        for index in range(6)
    }
    controls = {"net_accuracy_gain": {"interval95": [-0.02, 0.03]}}

    gate = confirmation_gate(
        metric,
        cells,
        controls,
        controls,
        route_audit_exact=True,
        identity_exact=False,
        identity_required=False,
        minimum_pool=100,
        minimum_preservation=0.95,
    )

    assert not gate["identity_exact"]
    assert not gate["identity_required"]
    assert gate["all_pass"]


def test_model_family_guard_accepts_supported_families_and_rejects_mismatch():
    for family in ("llama", "mistral", "qwen2", "gemma2"):
        validate_model_family(family, family)

    with __import__("pytest").raises(ValueError, match="expected mistral"):
        validate_model_family("llama", "mistral")
