import numpy as np

from stage_l_geometry import candidate_bindings, g2_predict_crossover, selected_value


def _row(label="within_stale"):
    return {
        "id": f"row_{label}",
        "slot": "diet",
        "current_value": "vegan",
        "stale_values": ["keto"],
        "stale_hits": ["keto"] if label == "within_stale" else [],
        "cross_slot_hits": ["jazz"] if label == "cross_slot" else [],
        "label": label,
        "prompt_tokens": 100,
        "competition": {
            "target_current": {
                "slot": "diet",
                "value": "vegan",
                "distance_messages_to_query": 30,
                "distance_bin": "far",
            },
            "same_slot_stale_nearest": {
                "slot": "diet",
                "value": "keto",
                "distance_messages_to_query": 2,
                "distance_bin": "near",
            },
            "recent_other_slot_nearest": {
                "slot": "music_genre",
                "value": "jazz",
                "distance_messages_to_query": 10,
                "distance_bin": "mid",
            },
            "recent_other_slot": [
                {
                    "slot": "music_genre",
                    "value": "jazz",
                    "distance_messages_to_query": 10,
                    "distance_bin": "mid",
                }
            ],
            "factorial_cell": {
                "same_slot_stale_distance_bin": "near",
                "recent_other_slot_distance_bin": "mid",
            },
        },
        "factorial_cell": {
            "same_slot_stale_distance_bin": "near",
            "recent_other_slot_distance_bin": "mid",
        },
    }


def test_stage_l_geometry_candidate_extraction_keeps_current_stale_and_other():
    candidates = candidate_bindings(_row())

    assert [cand["candidate_type"] for cand in candidates] == ["current", "stale", "other"]
    assert [cand["value"] for cand in candidates] == ["vegan", "keto", "jazz"]
    assert candidates[1]["is_same_slot"] is True
    assert candidates[2]["is_same_slot"] is False


def test_stage_l_geometry_selected_value_uses_observed_failure_mode():
    assert selected_value(_row("correct_current")) == ("vegan", "current")
    assert selected_value(_row("within_stale")) == ("keto", "stale")
    assert selected_value(_row("cross_slot")) == ("jazz", "other")


def test_stage_l_geometry_g2_holds_out_cells_without_using_behavior_labels_for_fit():
    rows = []
    for i in range(12):
        row = _row("cross_slot" if i % 2 else "within_stale")
        row["id"] = f"r{i}"
        row["factorial_cell"] = {
            "same_slot_stale_distance_bin": "near" if i < 6 else "far",
            "recent_other_slot_distance_bin": "mid",
        }
        row["competition"]["factorial_cell"] = row["factorial_cell"]
        row["competition"]["same_slot_stale_nearest"]["distance_bin"] = "near" if i < 6 else "far"
        row["competition"]["same_slot_stale_nearest"]["distance_messages_to_query"] = 2 if i < 6 else 30
        rows.append(row)

    classes = ["vegan", "keto", "jazz"]
    score_rows = []
    for row in rows:
        if row["factorial_cell"]["same_slot_stale_distance_bin"] == "near":
            score_rows.append([0.0, 2.0, 1.0])
        else:
            score_rows.append([0.0, 0.1, 1.0])
    probe = {
        "classes": classes,
        "class_to_index": {value: i for i, value in enumerate(classes)},
        "scores": np.asarray(score_rows, dtype=float),
    }

    out = g2_predict_crossover(rows, probe, length_controlled=False)

    assert out["n_heldout_cells"] == 2
    assert set(out["heldout_cell_predictions"]) == {
        "same_far__other_mid",
        "same_near__other_mid",
    }
