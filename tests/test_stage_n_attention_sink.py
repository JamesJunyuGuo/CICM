import math
from pathlib import Path

import numpy as np
import torch

from stage_n_attention_sink import (
    annotate_chat_row,
    answer_token_id,
    boundary_control_gate,
    choose_mechanism_k,
    classify_token,
    expand_gqa_values,
    intervention_outputs,
    layer_interventions,
    layer_matched_random_sets,
    model_report_label,
    paired_bootstrap,
    paired_length_controlled_delta,
    ratio_of_means_bootstrap,
    qwen_single_token_values,
    render_qwen_prompt,
    sink_components,
    stable_split,
    structural_sink_positions,
    validate_pair_alignment,
    validate_sink_heads,
)


def test_model_report_label_uses_runtime_model_instead_of_frozen_3b_name():
    assert model_report_label("Qwen/Qwen2.5-7B-Instruct") == "Qwen2.5-7B-Instruct"
    assert model_report_label("local-checkpoint") == "local-checkpoint"


def test_attention_sink_launchers_pin_the_hub_cache_and_clear_legacy_override():
    root = Path(__file__).resolve().parents[1]
    launchers = (
        "run_qwen15_prepare_cpu.slurm",
        "run_qwen15_smoke.slurm",
        "run_qwen15_smoke_a100.slurm",
    )
    for launcher in launchers:
        text = (root / "slurm/stage_n/attention_sink" / launcher).read_text()
        assert "unset TRANSFORMERS_CACHE" in text
        assert 'export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"' in text
        assert 'export HF_HUB_CACHE="$HF_HOME/hub"' in text


class QwenLikeValueTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text.strip().isdigit():
            return [220, *[int(character) for character in text.strip()]]
        return [1000 + len(text.strip())]

    def decode(self, token_ids):
        lookup = {1003: " red", 1004: " blue", 1005: " green"}
        return lookup.get(token_ids[0], "")


def test_qwen_value_adapter_requires_one_token_in_context_and_at_answer_boundary():
    tokenizer = QwenLikeValueTokenizer()
    assert qwen_single_token_values(tokenizer, ("17", "red", "blue", "green")) == [
        "red",
        "blue",
        "green",
    ]


def test_qwen_prompt_renderer_replaces_numeric_demonstration_labels():
    prompt, task_start = render_qwen_prompt(
        [{"var": "alpha", "value": "green"}],
        "alpha",
        "arrow",
        ("red", "blue", "green", "gold", "black", "white", "pink", "teal", "brown", "gray"),
    )
    assert "color = red" in prompt
    assert "color = 6" not in prompt
    assert prompt[task_start:].startswith("Assignments:\nalpha = green")


def test_stable_split_is_deterministic_and_binary():
    first = stable_split("semantic-17")
    assert first in {"discovery", "heldout"}
    assert stable_split("semantic-17") == first


def test_classify_token_uses_program_verifiable_candidates():
    row = {
        "gold_token_id": 11,
        "stale_token_ids": [12, 13],
        "cross_token_ids": [14],
    }
    assert classify_token(row, 11) == "correct_current"
    assert classify_token(row, 12) == "within_stale"
    assert classify_token(row, 14) == "cross_variable"
    assert classify_token(row, 99) == "other"


def test_boundary_control_gate_requires_clean_k0_and_low_other_rate():
    passing = {
        "model": "Qwen/test",
        "by_k": {
            "0": {"n": 10, "accuracy": 0.9, "counts": {"correct_current": 9, "other": 1}},
            "1": {"n": 10, "accuracy": 0.4, "counts": {"correct_current": 4, "within_stale": 5, "other": 1}},
        },
    }
    result = boundary_control_gate(passing)
    assert result["gate_pass"]
    failing = {
        **passing,
        "by_k": {
            **passing["by_k"],
            "0": {"n": 10, "accuracy": 0.0, "counts": {"other": 10}},
        },
    }
    assert not boundary_control_gate(failing)["gate_pass"]


def test_expand_gqa_values_repeats_each_kv_head_contiguously():
    values = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    expanded = expand_gqa_values(values, num_query_heads=4)
    assert expanded.shape == (1, 2, 4, 1)
    torch.testing.assert_close(
        expanded,
        torch.tensor([[[[1.0], [1.0], [2.0], [2.0]], [[3.0], [3.0], [4.0], [4.0]]]]),
    )


def test_sink_components_close_exact_mixture_and_expose_approximation_residual():
    attention = torch.tensor(
        [
            [0.50, 0.25, 0.25],
            [0.20, 0.30, 0.50],
        ]
    )
    values = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ]
    )
    result = sink_components(attention, values, torch.tensor([True, False, False]))

    torch.testing.assert_close(result["sink_mass"], torch.tensor([0.5, 0.2]))
    torch.testing.assert_close(result["content_gate"], torch.tensor([0.5, 0.8]))
    exact = torch.einsum("hs,hsd->hd", attention, values)
    torch.testing.assert_close(result["exact_output"], exact)
    torch.testing.assert_close(result["reconstructed_output"], exact)
    torch.testing.assert_close(
        result["exact_output"] - result["simplified_gated_output"],
        result["sink_residual"],
    )


def test_choose_mechanism_k_applies_pair_and_stratum_gates_then_balance_rule():
    rows = []
    for k, n_pairs in ((2, 90), (3, 105)):
        for index in range(n_pairs):
            rows.append(
                {
                    "k": k,
                    "seed": (11, 29, 47)[index % 3],
                    "template": ("arrow", "current", "latest")[index % 3],
                    "pair_valid": True,
                    "n_current": 130 if k == 2 else 120,
                    "n_within_stale": 95 if k == 2 else 110,
                }
            )
    result = choose_mechanism_k(rows)
    assert result["gate_pass"]
    assert result["chosen_k"] == 3


def test_paired_bootstrap_reports_mean_and_ordered_interval():
    result = paired_bootstrap(np.array([1.0, 2.0, 3.0]), n_boot=200, seed=7)
    assert result["mean"] == 2.0
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]
    assert not math.isnan(result["ci95"][0])


class BoundaryTokenizer:
    all_special_ids = [ord("^")]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = [ord(character) for character in text]
        if len(text) >= 2 and text[-2] == " " and text[-1].isdigit():
            return ids[:-2] + [1000 + int(text[-1])]
        return ids

    def decode(self, token_ids):
        pieces = []
        for token_id in token_ids:
            pieces.append(" " + str(token_id - 1000) if token_id >= 1000 else chr(token_id))
        return "".join(pieces)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        assert not tokenize
        assert not add_generation_prompt
        assert continue_final_message
        assert [message["role"] for message in messages] == ["user", "assistant"]
        return (
            "^user\n"
            + messages[0]["content"]
            + "\n^assistant\n"
            + messages[1]["content"]
        )

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        del add_special_tokens
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_answer_token_id_is_measured_at_actual_generation_boundary():
    tokenizer = BoundaryTokenizer()
    assert answer_token_id(tokenizer, "prompt\n", "7") == 1007


def test_answer_token_id_rejects_multi_token_values():
    tokenizer = BoundaryTokenizer()
    with np.testing.assert_raises(ValueError):
        answer_token_id(tokenizer, "prompt\n", "17")


def test_structural_sink_positions_are_position_zero_union_special_tokens():
    assert structural_sink_positions([100, 8, 9, 8], {8}) == [0, 1, 3]


def test_validate_pair_alignment_checks_length_and_write_spans():
    corrupt = {
        "prompt_input_ids": [1, 2, 3, 4],
        "writes": [{"value_token": 1}, {"value_token": 2}],
        "current_value_span": [2],
    }
    clean = {
        "prompt_input_ids": [1, 7, 3, 4],
        "writes": [{"value_token": 1}, {"value_token": 2}],
        "current_value_span": [2],
    }
    assert validate_pair_alignment(clean, corrupt)["valid"]
    clean["current_value_span"] = [3]
    result = validate_pair_alignment(clean, corrupt)
    assert not result["valid"]
    assert "current span" in result["reason"]


def test_annotate_chat_row_recomputes_spans_and_answer_ids_after_chat_prefix():
    tokenizer = BoundaryTokenizer()
    raw = {
        "id": "row-1",
        "semantic_id": "semantic-1",
        "prompt": "x = 5\nx = 7\n=> x =",
        "task_char_start": 0,
        "events": [
            {"position": 0, "var": "x", "value": "5", "is_current": False},
            {"position": 1, "var": "x", "value": "7", "is_current": True},
        ],
        "target_var": "x",
        "gold": "7",
        "stale_values": ["5"],
        "cross_values": [],
        "k": 1,
        "seed": 11,
        "template": "arrow",
    }
    row = annotate_chat_row(raw, tokenizer)
    assert row["prompt"].startswith("^user\n")
    assert row["gold_token_id"] == 1007
    assert row["stale_token_ids"] == [1005]
    assert row["prompt"].endswith("=> x =")
    assert row["answer_boundary_mode"] == "assistant_prefill_continuation"
    assert row["current_value_span"] == [row["writes"][1]["value_token"]]
    assert row["sink_positions"][0] == 0
    assert row["prompt_input_ids"][row["sink_positions"][0]] == ord("^")


def test_intervention_outputs_change_only_named_gate_routing_or_values():
    sink_mask = torch.tensor([True, False, False])
    corrupt_attention = torch.tensor([[0.8, 0.1, 0.1]])
    clean_attention = torch.tensor([[0.2, 0.6, 0.2]])
    corrupt_values = torch.tensor([[[0.0], [2.0], [4.0]]])
    clean_values = torch.tensor([[[0.0], [10.0], [20.0]]])

    outputs = intervention_outputs(
        clean_attention=clean_attention,
        corrupt_attention=corrupt_attention,
        clean_values=clean_values,
        corrupt_values=corrupt_values,
        sink_mask=sink_mask,
    )

    torch.testing.assert_close(outputs["identity"], torch.tensor([[0.6]]))
    torch.testing.assert_close(outputs["gate"], torch.tensor([[2.4]]))
    torch.testing.assert_close(outputs["routing"], torch.tensor([[0.5]]))
    torch.testing.assert_close(outputs["value"], torch.tensor([[3.0]]))
    torch.testing.assert_close(outputs["full"], torch.tensor([[10.0]]))


def test_layer_interventions_use_captured_outputs_for_identity_and_full_anchors():
    clean_capture = {
        "width": 3,
        "attention": {0: torch.tensor([[[0.2, 0.6, 0.2]]])},
        "values": {0: torch.tensor([[[[0.0], [10.0], [20.0]]]])},
        "head_outputs": {0: torch.tensor([[[10.125]]])},
    }
    corrupt_capture = {
        "width": 3,
        "attention": {0: torch.tensor([[[0.8, 0.1, 0.1]]])},
        "values": {0: torch.tensor([[[[0.0], [2.0], [4.0]]]])},
        "head_outputs": {0: torch.tensor([[[0.625]]])},
    }
    rows = [{"id": "pair-1", "sink_positions": [0]}]

    outputs = layer_interventions(clean_capture, corrupt_capture, rows, rows, layer=0)

    # Reconstructed attention-value products are 0.6 and 10.0. The causal
    # anchors must instead reproduce the actual pre-o_proj tensors exactly.
    torch.testing.assert_close(outputs["identity"], torch.tensor([[[0.625]]]))
    torch.testing.assert_close(outputs["full"], torch.tensor([[[10.125]]]))


def test_validate_sink_heads_requires_concentration_low_value_and_template_stability():
    rows = []
    for template in ("arrow", "current", "latest"):
        for _ in range(3):
            rows.append(
                {
                    "layer": 2,
                    "head": 1,
                    "template": template,
                    "sink_mass": 0.50,
                    "candidate_share": 0.05,
                    "sink_to_non_sink_value_norm": 0.20,
                    "sink_residual_ratio": 0.10,
                }
            )
    result = validate_sink_heads(rows)
    assert result["2.1"]["valid_sink"]
    weakened = [dict(row, sink_mass=0.10) for row in rows]
    assert not validate_sink_heads(weakened)["2.1"]["valid_sink"]


def test_layer_matched_random_sets_preserve_layer_cardinality_and_exclude_selected():
    selected = [(2, 1), (2, 3), (5, 0)]
    sets = layer_matched_random_sets(selected, n_layers=8, n_heads=6, n_sets=8, seed=4)
    assert len(sets) == 8
    for candidate in sets:
        assert len(candidate) == len(selected)
        assert [layer for layer, _ in candidate].count(2) == 2
        assert [layer for layer, _ in candidate].count(5) == 1
        assert not (set(candidate) & set(selected))


def test_paired_length_control_removes_shared_log_length_effect_not_group_delta():
    rows = []
    for index, length in enumerate((10, 20, 40, 80)):
        shared = 2.0 * math.log(length)
        rows.append({"id": str(index), "side": "clean", "prompt_tokens": length, "metric": shared})
        rows.append({"id": str(index), "side": "corrupt", "prompt_tokens": length, "metric": shared - 1.0})
    result = paired_length_controlled_delta(rows, metric="metric", n_boot=200, n_shuffle=400, seed=3)
    assert np.isclose(result["raw_delta"], -1.0)
    assert np.isclose(result["length_controlled_delta"], -1.0)
    assert result["shuffle95"][0] <= 0.0 <= result["shuffle95"][1]
    assert result["shuffle95"][1] <= 1.0


def test_ratio_of_means_bootstrap_uses_paired_resampling():
    result = ratio_of_means_bootstrap(
        numerator=np.array([1.0, 2.0, 3.0]),
        denominator=np.array([2.0, 4.0, 6.0]),
        n_boot=200,
        seed=9,
    )
    assert np.isclose(result["ratio"], 0.5)
    assert result["ci95"][0] <= 0.5 <= result["ci95"][1]
