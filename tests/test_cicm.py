import numpy as np

from gen_cicm_synthetic import build_stage_l_rows
from gen_cicm import (
    build_natural_stage_l_rows,
    natural_controlled_slots,
    validate_template_pack,
    validate_natural_rows,
)
from cicm_api_eval import select_rows
from cicm_eval import classify_stage_l_response
from cicm_mech import (
    build_l1_failure_pool_summary,
    choose_l2_competitor_value,
    effective_steer_direction,
    l1_adjudication_status,
    summarize_l2,
    true_label_scores,
    value_label_shuffle_null,
)
from cicm_stats import group_delta_with_pooled_length_control


def test_stage_l_generator_builds_repeated_controlled_bindings():
    slots = {
        "music_genre": ["jazz", "rock", "classical"],
        "diet": ["vegan", "keto", "mediterranean"],
    }

    rows = build_stage_l_rows(n_per_dose=12, doses=[1, 2], seed=7, slots=slots)

    assert len(rows) == 24
    assert {row["k_overwrites"] for row in rows} == {1, 2}
    assert all(row["current_value"] not in row["stale_values"] for row in rows)
    assert all(len(row["stale_values"]) == row["k_overwrites"] for row in rows)
    assert all(row["value_mentions"][-1]["value"] == row["current_value"] for row in rows)
    current_values = [row["current_value"] for row in rows]
    assert len(set(current_values)) < len(current_values)


def test_stage_l_l0b_hardening_adds_post_current_interference_before_query():
    slots = {
        "music_genre": ["jazz", "rock", "classical"],
        "diet": ["vegan", "keto", "mediterranean"],
        "transport": ["train", "bus", "bicycle"],
    }

    rows = build_stage_l_rows(
        n_per_dose=3,
        doses=[2],
        seed=11,
        slots=slots,
        post_final_filler_turns=2,
        post_final_distractor_updates=2,
        system_style="plain",
        query_style="natural",
    )
    row = rows[0]

    query_index = len(row["messages"]) - 1
    current_message_index = row["value_mentions"][-1]["message_index"]
    post_current = row["messages"][current_message_index + 2 : query_index]

    assert row["hardening"]["post_final_filler_turns"] == 2
    assert row["hardening"]["post_final_distractor_updates"] == 2
    assert "latest preference updates" not in row["messages"][0]["content"].lower()
    assert len(post_current) == 8
    assert all(update["slot"] != row["slot"] for update in row["distractor_updates"])
    assert row["messages"][-1]["role"] == "user"


def test_stage_l_classifier_separates_current_stale_and_other_values():
    row = {
        "current_value": "jazz",
        "stale_values": ["rock", "classical"],
        "slot_values": ["jazz", "rock", "classical", "electronic"],
        "all_controlled_values": ["jazz", "rock", "classical", "electronic", "vegan"],
    }

    assert classify_stage_l_response("I would choose jazz for you.", row)["label"] == "correct_current"
    assert classify_stage_l_response("Rock is the best fit.", row)["label"] == "within_stale"
    assert classify_stage_l_response("Electronic seems right.", row)["label"] == "same_slot_other"
    assert classify_stage_l_response("Vegan seems right.", row)["label"] == "cross_slot"
    assert classify_stage_l_response("I am not sure.", row)["label"] == "other"


def test_stage_l_length_control_pools_before_residual_delta():
    rows = [
        {"label": "within_stale", "prompt_tokens": 100},
        {"label": "within_stale", "prompt_tokens": 200},
        {"label": "correct_current", "prompt_tokens": 100},
        {"label": "correct_current", "prompt_tokens": 200},
    ]
    values = np.array([5.0, 6.0, 2.0, 3.0])

    result = group_delta_with_pooled_length_control(
        rows,
        values,
        group_a="within_stale",
        group_b="correct_current",
        label_key="label",
        n_boot=50,
        n_shuffle=50,
        seed=1,
    )

    assert result["raw_delta"] == 3.0
    assert abs(result["length_controlled_delta"] - 3.0) < 1e-8
    assert result["length_controlled_shuffle_null"]["n"] == 50


def test_stage_l_value_label_null_preserves_class_distribution():
    classes = np.array(["jazz", "rock", "classical"])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
        ]
    )
    labels = np.array(["jazz", "rock", "classical", "jazz"])
    groups = np.array(["within_stale", "within_stale", "correct_current", "correct_current"])

    observed = true_label_scores(proba, classes, labels)
    null = value_label_shuffle_null(
        proba,
        classes,
        labels,
        groups,
        group="within_stale",
        n_shuffle=40,
        seed=3,
    )

    assert observed[:2].mean() == 0.75
    assert len(null) == 40
    assert np.isfinite(null).all()


def test_stage_l_l1_failure_pool_summary_reports_cross_slot_and_cells():
    rows = [
        {
            "id": "a",
            "label": "correct_current",
            "current_value": "jazz",
            "prompt_tokens": 100,
            "factorial_cell": {
                "same_slot_stale_distance_bin": "far",
                "recent_other_slot_distance_bin": "near2",
            },
        },
        {
            "id": "b",
            "label": "within_stale",
            "current_value": "rock",
            "prompt_tokens": 200,
            "factorial_cell": {
                "same_slot_stale_distance_bin": "far",
                "recent_other_slot_distance_bin": "near2",
            },
        },
        {
            "id": "c",
            "label": "cross_slot",
            "current_value": "jazz",
            "prompt_tokens": 300,
            "factorial_cell": {
                "same_slot_stale_distance_bin": "near",
                "recent_other_slot_distance_bin": "mid",
            },
        },
        {
            "id": "d",
            "label": "correct_current",
            "current_value": "rock",
            "prompt_tokens": 400,
            "factorial_cell": {
                "same_slot_stale_distance_bin": "near",
                "recent_other_slot_distance_bin": "mid",
            },
        },
    ]
    classes = np.array(["jazz", "rock"])
    labels = np.array(["jazz", "rock", "jazz", "rock"])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    probe = {
        "classes": classes.tolist(),
        "labels": labels,
        "proba": proba,
        "true_current_score": true_label_scores(proba, classes, labels),
    }

    summary = build_l1_failure_pool_summary(
        rows,
        probe,
        n_boot=50,
        n_shuffle=50,
        seed=7,
    )

    assert summary["length_control"]["scope"] == "all_l1_rows"
    assert summary["failure_modes"]["within_stale"]["decodability"]["n"] == 1
    assert summary["failure_modes"]["cross_slot"]["decodability"]["n"] == 1
    assert "same_near__other_mid" in summary["failure_modes"]["cross_slot"]["by_cell"]


def test_stage_l_l1_status_requires_within_stale_failure_pool():
    summary = {
        "label_counts": {"correct_current": 10},
        "probe": {
            "within_stale_true_current_score_shuffle_null": {
                "above_null_95": True,
            }
        },
    }

    status = l1_adjudication_status(summary)

    assert status["status"] == "not_adjudicable_no_within_stale"
    assert status["l1_gate_pass"] is False


def test_stage_l_l2_competitor_value_uses_observed_failure_mode():
    within = {
        "label": "within_stale",
        "current_value": "jazz",
        "stale_hits": ["rock"],
        "stale_values": ["rock", "classical"],
        "cross_slot_hits": ["vegan"],
    }
    cross = {
        "label": "cross_slot",
        "current_value": "jazz",
        "stale_hits": ["rock"],
        "stale_values": ["rock", "classical"],
        "cross_slot_hits": ["vegan"],
    }

    assert choose_l2_competitor_value(within) == ("rock", "within_stale")
    assert choose_l2_competitor_value(cross) == ("vegan", "cross_slot")


def test_stage_l_l2_summary_reports_failure_modes_and_matched_controls():
    rows = []
    for row_id, failure_mode, identity, targeted, random in [
        ("a", "within_stale", "within_stale", "correct_current", "within_stale"),
        ("b", "within_stale", "within_stale", "within_stale", "correct_current"),
        ("c", "cross_slot", "cross_slot", "correct_current", "cross_slot"),
    ]:
        for arm, label in [
            ("default_nohook", identity),
            ("identity", identity),
            ("targeted", targeted),
            ("random_matched_norm", random),
        ]:
            rows.append(
                {
                    "id": row_id,
                    "failure_mode": failure_mode,
                    "baseline_label": failure_mode,
                    "arm": arm,
                        "label": label,
                        "response": "baseline",
                        "intervention_response": "baseline" if arm in {"default_nohook", "identity"} else label,
                    }
                )

    summary = summarize_l2(rows, n_boot=50, seed=5)

    assert summary["n_paired"] == 3
    assert summary["identity_gate_mismatches"] == 0
    assert summary["by_failure_mode"]["within_stale"]["n_paired"] == 2
    assert summary["by_failure_mode"]["cross_slot"]["n_paired"] == 1
    assert summary["by_failure_mode"]["within_stale"]["arm_counts"]["identity"]["target_error_rate"] == 1.0
    assert summary["by_failure_mode"]["within_stale"]["targeted_minus_random_target_error_reduction"]["n"] == 2


def test_stage_l_l2_summary_groups_alpha_sweep_separately():
    rows = []
    for alpha, targeted_label in [(1.0, "within_stale"), (4.0, "correct_current")]:
        for arm, label in [
            ("default_nohook", "within_stale"),
            ("identity", "within_stale"),
            ("random_matched_norm", "within_stale"),
            ("targeted", targeted_label),
        ]:
            rows.append(
                {
                    "id": "a",
                    "failure_mode": "within_stale",
                    "baseline_label": "within_stale",
                    "arm": arm,
                    "alpha": alpha,
                    "label": label,
                    "intervention_response": "baseline" if arm in {"default_nohook", "identity"} else label,
                }
            )

    summary = summarize_l2(rows, n_boot=50, seed=5)

    assert set(summary["by_alpha"]) == {"1", "4"}
    assert summary["by_alpha"]["1"]["arm_counts"]["targeted"]["target_error_rate"] == 1.0
    assert summary["by_alpha"]["4"]["arm_counts"]["targeted"]["target_error_rate"] == 0.0
    assert summary["best_alpha_by_targeted_correct_gain"] == 4.0


def test_stage_l_l2_residual_ratio_alpha_sets_steer_norm():
    direction = np.array([3.0, 4.0], dtype=np.float32)

    scaled = effective_steer_direction(direction, alpha=0.25, residual_norm=12.0, alpha_mode="residual_ratio")

    assert np.isclose(np.linalg.norm(scaled), 3.0)
    assert np.allclose(scaled / np.linalg.norm(scaled), direction / np.linalg.norm(direction))


def test_stage_l_l2_absolute_alpha_keeps_legacy_scale():
    direction = np.array([3.0, 4.0], dtype=np.float32)

    scaled = effective_steer_direction(direction, alpha=2.0, residual_norm=12.0, alpha_mode="absolute")

    assert np.allclose(scaled, direction * 2.0)


def test_stage_l_api_select_rows_limits_each_dose():
    rows = [
        {"id": "a", "k_overwrites": 1},
        {"id": "b", "k_overwrites": 1},
        {"id": "c", "k_overwrites": 2},
        {"id": "d", "k_overwrites": 2},
    ]

    selected = select_rows(rows, limit_per_dose=1)

    assert [row["id"] for row in selected] == ["a", "c"]


def test_stage_l_natural_slots_use_slot_specific_value_sets():
    slots = natural_controlled_slots()

    assert slots["music_genre"] != slots["diet"]
    assert "jazz" in slots["music_genre"]
    assert "vegan" in slots["diet"]
    assert all(len(values) >= 7 for values in slots.values())


def test_stage_l_natural_rows_record_exact_target_value_spans():
    slots = {
        "music_genre": ["jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"],
        "diet": ["vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"],
        "transport": ["train", "bus", "bicycle", "subway", "rideshare", "walking", "carpool"],
    }

    rows = build_natural_stage_l_rows(
        n_per_dose=4,
        doses=[2],
        seed=13,
        slots=slots,
        post_final_filler_turns=2,
        post_final_distractor_updates=1,
    )

    validate_natural_rows(rows)
    row = rows[0]
    current_mentions = [m for m in row["value_mentions"] if m["is_current"]]

    assert row["base_dataset"] == "Stage L natural CICM"
    assert current_mentions
    assert row["query_anchor"]["target_slot"] == row["slot"]
    assert row["slot_label"] in row["query"]
    assert "not any other preference" in row["query"].lower()
    assert row["competition"]["target_current"]["value"] == row["current_value"]
    assert row["competition"]["target_current"]["distance_messages_to_query"] > 0
    assert len(row["competition"]["same_slot_stale"]) == row["k_overwrites"]
    assert len(row["competition"]["recent_other_slot"]) == row["hardening"]["post_final_distractor_updates"]
    for mention in row["value_mentions"]:
        msg = row["messages"][mention["message_index"]]["content"]
        assert msg[mention["char_start"] : mention["char_end"]] == mention["value"]
        assert mention["slot"] == row["slot"]
        assert mention["role"] == "user"


def test_stage_l_natural_factorial_balances_competitor_type_and_distance():
    slots = {
        "music_genre": ["jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"],
        "diet": ["vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"],
        "transport": ["train", "bus", "bicycle", "subway", "rideshare", "walking", "carpool"],
    }

    rows = build_natural_stage_l_rows(
        n_per_dose=8,
        doses=[2],
        seed=17,
        slots=slots,
        post_final_filler_turns=4,
        post_final_distractor_updates=1,
        factorial_competition=True,
    )

    validate_natural_rows(rows)
    cell_counts = {}
    current_bins = {}
    for row in rows:
        cell = (
            row["competition"]["factorial_cell"]["same_slot_stale_distance_bin"],
            row["competition"]["factorial_cell"]["recent_other_slot_distance_bin"],
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        current_bins.setdefault(cell, []).append(
            row["competition"]["target_current"]["distance_bin"]
        )

        same_nearest = row["competition"]["same_slot_stale_nearest"]
        other_nearest = row["competition"]["recent_other_slot_nearest"]
        assert same_nearest["distance_bin"] == cell[0]
        assert other_nearest["distance_bin"] == cell[1]
        if cell[0] == "near":
            assert same_nearest["distance_messages_to_query"] <= 8
            assert same_nearest["mention_kind"] == "stale_reminder"
        else:
            assert same_nearest["distance_messages_to_query"] >= 10
            assert same_nearest["mention_kind"] == "binding_update"
        if cell[1] == "near":
            assert other_nearest["distance_messages_to_query"] <= 8
        else:
            assert other_nearest["distance_messages_to_query"] >= 10

    assert cell_counts == {
        ("near", "near"): 2,
        ("near", "far"): 2,
        ("far", "near"): 2,
        ("far", "far"): 2,
    }
    assert ("near", "far") in cell_counts
    assert all(sorted(bins) == ["far", "near"] for bins in current_bins.values())


def test_stage_l_natural_factorial_hard_keeps_cells_but_makes_current_mostly_far():
    slots = {
        "music_genre": ["jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"],
        "diet": ["vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"],
        "transport": ["train", "bus", "bicycle", "subway", "rideshare", "walking", "carpool"],
    }

    rows = build_natural_stage_l_rows(
        n_per_dose=8,
        doses=[1, 2, 3, 4, 6],
        seed=19,
        slots=slots,
        post_final_filler_turns=6,
        post_final_distractor_updates=3,
        factorial_competition=True,
        factorial_target_current_mode="far_with_near_controls",
        extra_interference_turns=4,
    )

    validate_natural_rows(rows)
    cell_counts = {}
    target_by_cell = {}
    for row in rows:
        cell = (
            row["competition"]["factorial_cell"]["same_slot_stale_distance_bin"],
            row["competition"]["factorial_cell"]["recent_other_slot_distance_bin"],
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        target_by_cell.setdefault(cell, []).append(
            row["competition"]["target_current"]["distance_bin"]
        )
        assert row["competition"]["same_slot_stale_nearest"]["distance_bin"] == cell[0]
        assert row["competition"]["recent_other_slot_nearest"]["distance_bin"] == cell[1]
        assert len(row["competition"]["recent_other_slot"]) == 3
        assert row["hardening"]["extra_interference_turns"] == 4
        if row["factorial_cell"]["target_current_distance_bin"] == "near":
            assert row["competition"]["target_current"]["distance_messages_to_query"] <= 4
        else:
            assert row["competition"]["target_current"]["distance_messages_to_query"] >= 20

    assert cell_counts == {
        ("near", "near"): 10,
        ("near", "far"): 10,
        ("far", "near"): 10,
        ("far", "far"): 10,
    }
    assert sum(bin_ == "far" for bins in target_by_cell.values() for bin_ in bins) == 32
    assert sum(bin_ == "near" for bins in target_by_cell.values() for bin_ in bins) == 8
    assert all(bins.count("near") == 2 and bins.count("far") == 8 for bins in target_by_cell.values())


def test_stage_l_natural_factorial_other_distance_three_bins_with_current_far():
    slots = {
        "music_genre": ["jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"],
        "diet": ["vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"],
        "transport": ["train", "bus", "bicycle", "subway", "rideshare", "walking", "carpool"],
    }

    rows = build_natural_stage_l_rows(
        n_per_dose=12,
        doses=[1, 2, 3, 4, 6],
        seed=23,
        slots=slots,
        post_final_filler_turns=6,
        post_final_distractor_updates=3,
        factorial_competition=True,
        factorial_target_current_mode="far",
        other_distance_bins=["far", "mid", "near2"],
        extra_interference_turns=4,
    )

    validate_natural_rows(rows)
    cell_counts = {}
    for row in rows:
        cell = (
            row["factorial_cell"]["same_slot_stale_distance_bin"],
            row["factorial_cell"]["recent_other_slot_distance_bin"],
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        assert row["competition"]["target_current"]["distance_bin"] == "far"
        assert row["competition"]["target_current"]["distance_messages_to_query"] >= 20
        other = row["competition"]["recent_other_slot_nearest"]
        if cell[1] == "near2":
            assert other["distance_messages_to_query"] == 2
        elif cell[1] == "mid":
            assert 8 <= other["distance_messages_to_query"] <= 14
        else:
            assert other["distance_messages_to_query"] >= 20

    assert cell_counts == {
        ("near", "far"): 10,
        ("near", "mid"): 10,
        ("near", "near2"): 10,
        ("far", "far"): 10,
        ("far", "mid"): 10,
        ("far", "near2"): 10,
    }


def test_stage_l_natural_template_validation_rejects_meta_placeholders():
    pack = {
        "system": "Avoid controlled values except {value}.",
        "opening": {"user": "We are setting a preference.", "assistant": "Sure."},
        "updates": [
            {"user": "Set it to {value}.", "assistant": "Updated."},
            {"user": "Change it to {value}.", "assistant": "Updated again."},
        ],
        "fillers": [
            {"user": "Can we keep this practical?", "assistant": "Yes."},
            {"user": "Can we keep this concise?", "assistant": "Yes."},
        ],
        "queries": ["What is the current setting?"],
    }

    try:
        validate_template_pack(pack, "music_genre", ["jazz", "rock"])
    except ValueError as exc:
        assert "meta" in str(exc) or "{value}" in str(exc)
    else:
        raise AssertionError("template validation should reject meta placeholders")
