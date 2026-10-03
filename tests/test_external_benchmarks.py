import random

import pytest

from gen_tasks import make_counterfactual_pair, make_example
from patching_metrics import first_token_logit_diff, assert_self_patch_identity, summarize_patch_effects
from external_benchmarks import (
    classify_entity_tracking_error,
    classify_ruler_vt_error,
    classify_babilong_error,
    normalize_answer_text,
    score_babilong_answer,
    score_entity_tracking_answer,
    score_ruler_vt_answer,
    prompt_for_row,
    score_text_answer,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


def test_counterfactual_pair_preserves_write_geometry_and_gold():
    rng = random.Random(7)
    corrupted = make_example("interference", n_lines=40, I=4, rng=rng)
    pair = make_counterfactual_pair(corrupted, rng=random.Random(8))

    assert pair["corrupted"]["prompt"] == corrupted["prompt"]
    assert pair["clean"]["gold"] == corrupted["gold"]
    assert pair["clean"]["target_var"] == corrupted["target_var"]
    assert pair["clean"]["stale_values"] == []
    assert pair["clean"]["target_write_positions"] == [corrupted["target_write_positions"][-1]]
    assert pair["span_equal"] is True

    corrupted_lines = pair["corrupted_assignments"]
    clean_lines = pair["clean_assignments"]
    assert len(corrupted_lines) == len(clean_lines) == 40
    for pos in corrupted["target_write_positions"]:
        c_var, c_val = corrupted_lines[pos]
        q_var, q_val = clean_lines[pos]
        assert c_val == q_val
        assert len(c_var) == len(q_var)
        if pos == corrupted["target_write_positions"][-1]:
            assert c_var == q_var == corrupted["target_var"]
        else:
            assert c_var == corrupted["target_var"]
            assert q_var != corrupted["target_var"]


def test_logit_diff_drops_first_token_collisions():
    tok = TinyTokenizer()
    logits = {ord("1"): 3.0, ord("2"): 5.0}
    assert first_token_logit_diff(logits, "1", "2", tok) == pytest.approx(-2.0)
    assert first_token_logit_diff(logits, "12", "19", tok) is None


def test_self_patch_identity_gate_detects_drift():
    assert_self_patch_identity({"component": 0.0}, tolerance=1e-9)
    with pytest.raises(AssertionError):
        assert_self_patch_identity({"component": 1e-4}, tolerance=1e-6)


def test_external_text_scoring_and_error_signatures():
    assert normalize_answer_text("The Hallway.") == "hallway"
    assert score_text_answer("hallway", "The hallway.")["correct"] == 1
    assert score_text_answer("hallway", "kitchen")["correct"] == 0

    ruler_row = {"outputs": ["AAA", "BBB", "CCC"], "superseded_outputs": ["XXX", "YYY"]}
    assert classify_ruler_vt_error(ruler_row, "YYY") == "cross_chain_inclusion"
    assert classify_ruler_vt_error(ruler_row, "BBB") == "omission"
    assert classify_ruler_vt_error(ruler_row, "ZZZ") == "omission"

    entity_row = {
        "gold": "the red ball",
        "superseded_states": ["the blue ball", "nothing"],
    }
    assert classify_entity_tracking_error(entity_row, "Blue ball") == "superseded"
    assert classify_entity_tracking_error(entity_row, "red ball") == "correct_value"
    assert classify_entity_tracking_error(entity_row, "green ball") == "other"


def test_ruler_vt_v2_scores_per_variable_recall_without_superseded_language():
    row = {
        "outputs": ["AAA", "BBB", "CCC"],
        "superseded_outputs": ["XXX", "YYY"],
    }

    scored = score_ruler_vt_answer(row, "AAA, xxx, something else")

    assert scored["recall"] == pytest.approx(1 / 3)
    assert scored["strict_correct"] == 0
    assert scored["correct"] == 0
    assert classify_ruler_vt_error(row, "AAA, xxx, something else") == "cross_chain_inclusion"
    assert classify_ruler_vt_error(row, "AAA") == "omission"
    assert classify_ruler_vt_error(row, "AAA, BBB, CCC") == "correct_value"


def test_babilong_v2_uses_lenient_containment_and_superseded_trace():
    row = {
        "benchmark": "babilong",
        "task": "qa2",
        "gold": "garden",
        "prompt": (
            "Read the context and answer the question with only the answer phrase.\n\n"
            "Context:\nJohn travelled to the hallway. John went to the garden. "
            "John got the milk there. John moved to the kitchen. "
            "John dropped the milk there. John went to the bathroom.\n\n"
            "Question: Where is the milk? \nAnswer:"
        ),
    }

    assert score_babilong_answer(row, "The milk is in the garden.")["correct"] == 1
    assert classify_babilong_error(row, "kitchen") == "superseded"
    assert classify_babilong_error(row, "office") == "other"
    assert classify_babilong_error({**row, "task": "qa1"}, "kitchen") == "other"


def test_entity_tracking_v2_scores_normalized_sets_order_insensitive():
    row = {
        "gold": "the red ball and the blue key",
        "superseded_states": ["the red ball", "nothing"],
    }

    scored = score_entity_tracking_answer(row, "blue key, red ball")

    assert scored["correct"] == 1
    assert scored["pred_items"] == ["blue key", "red ball"]
    assert score_entity_tracking_answer(row, "Box 3 contains the blue key and the red ball.")["correct"] == 1
    assert classify_entity_tracking_error(row, "red ball") == "superseded"
    assert classify_entity_tracking_error({"gold": "nothing", "superseded_states": []}, "nothing") == "correct_value"


def test_stage_g_circuit_summary_includes_all_six_patching_components():
    rows = [
        {"component": "residual_l20", "layer": 20, "delta_recovery": 0.8},
        {"component": "selection_head_attention_pattern_proxy", "delta_recovery": 0.1},
        {"component": "selection_head_output_proxy", "delta_recovery": 0.2},
        {"component": "qk_key_all_writes", "delta_recovery": 0.3},
        {"component": "qk_key_current_write", "delta_recovery": 0.4},
        {"component": "qk_key_stale_writes", "delta_recovery": 0.5},
        {"component": "qk_query_final", "delta_recovery": 0.6},
        {"component": "late_mlp_output_l20", "layer": 20, "delta_recovery": 0.7},
    ]

    effects = {row["component"]: row for row in summarize_patch_effects(rows)}

    assert effects["residual_stream_late"]["delta_recovery_mean"] == pytest.approx(0.8)
    assert effects["selection_head_attention_pattern_proxy"]["n"] == 1
    assert effects["selection_head_output_proxy"]["n"] == 1
    assert effects["qk_key_all_writes"]["n"] == 1
    assert effects["qk_key_current_write"]["n"] == 1
    assert effects["qk_key_stale_writes"]["n"] == 1
    assert effects["qk_query_final"]["n"] == 1
    assert effects["late_mlp_output"]["delta_recovery_mean"] == pytest.approx(0.7)


def test_llama_boxes_direct_prompt_removes_reasoning_invitation():
    row = {
        "benchmark": "entity_tracking",
        "prompt": (
            "Track the boxes and answer the final masked statement. Answer with only the missing contents.\n\n"
            "Box 0 contains the shoe. Move the shoe from Box 0 to Box 1. "
            "Box 0 contains <extra_id_0> .\n\nMissing contents:"
        ),
    }

    prompt = prompt_for_row(row, "llama_boxes_direct")

    assert "Track the boxes" not in prompt
    assert "Output only" in prompt
    assert "Box 0 contains <extra_id_0>" in prompt
