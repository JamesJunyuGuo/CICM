import torch

from stale_binding_routing import (
    RoutingBiasContext,
    apply_stale_key_bias,
    choose_beta,
    infer_structured_route,
    make_random_position_sets,
)


class ToyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        offsets = []
        input_ids = []
        index = 0
        for token_id, token in enumerate(text.split(), start=1):
            start = text.index(token, index)
            end = start + len(token)
            offsets.append((start, end))
            input_ids.append(token_id)
            index = end
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_apply_stale_key_bias_is_exact_identity_at_zero():
    scores = torch.randn(2, 3, 5, 5)
    context = RoutingBiasContext(
        beta=0.0,
        key_positions=((1, 2), (0,)),
        query_positions=(4, 3),
        heads_by_layer={1: (0, 2)},
    )

    result = apply_stale_key_bias(scores, 1, context)

    assert result is scores


def test_apply_stale_key_bias_changes_only_selected_query_head_key_cells():
    scores = torch.zeros(2, 3, 5, 5)
    context = RoutingBiasContext(
        beta=2.0,
        key_positions=((1, 3), (0,)),
        query_positions=(4, 2),
        heads_by_layer={1: (0, 2)},
    )

    result = apply_stale_key_bias(scores, 1, context)

    expected = torch.zeros_like(scores)
    expected[0, (0, 2), 4, 1] = -2.0
    expected[0, (0, 2), 4, 3] = -2.0
    expected[1, (0, 2), 2, 0] = -2.0
    assert torch.equal(result, expected)


def test_apply_stale_key_bias_skips_cached_decode_steps():
    scores = torch.zeros(1, 3, 1, 6)
    context = RoutingBiasContext(
        beta=2.0,
        key_positions=((1, 3),),
        query_positions=(4,),
        heads_by_layer={1: (0, 2)},
    )

    result = apply_stale_key_bias(scores, 1, context)

    assert result is scores


def test_apply_stale_key_bias_can_route_from_stale_to_current_keys():
    scores = torch.zeros(1, 2, 4, 4)
    context = RoutingBiasContext(
        beta=1.5,
        key_positions=((0, 1),),
        query_positions=(3,),
        heads_by_layer={2: (1,)},
        boost_beta=1.5,
        boost_positions=((2,),),
    )

    result = apply_stale_key_bias(scores, 2, context)

    expected = torch.zeros_like(scores)
    expected[0, 1, 3, (0, 1)] = -1.5
    expected[0, 1, 3, 2] = 1.5
    assert torch.equal(result, expected)


def test_structured_route_uses_only_prompt_syntax():
    prompt = """Assignments:
movie = triangle
movie = rock
movie = yellow
=> movie ="""

    route = infer_structured_route(prompt, ToyTokenizer())

    assert route["target_var"] == "movie"
    assert route["values"] == ["triangle", "rock", "yellow"]
    assert route["stale_positions"] == [3, 6]
    assert route["current_positions"] == [9]


def test_choose_beta_enforces_preservation_and_uses_smaller_beta_on_tie():
    curve = [
        {"beta": 0.5, "correct_preservation": 0.98, "net_accuracy_gain": 0.04},
        {"beta": 1.0, "correct_preservation": 0.96, "net_accuracy_gain": 0.08},
        {"beta": 2.0, "correct_preservation": 0.95, "net_accuracy_gain": 0.08},
        {"beta": 4.0, "correct_preservation": 0.90, "net_accuracy_gain": 0.12},
    ]

    assert choose_beta(curve, minimum_preservation=0.95)["beta"] == 1.0


def test_random_positions_match_counts_and_exclude_write_positions():
    rows = [
        {
            "id": "a",
            "prompt": "a b c d e f g h i j",
            "stale_value_spans": [1, 3],
            "current_value_span": [5],
            "target_identity_spans": [0, 2, 4],
        }
    ]

    samples = make_random_position_sets(rows, ToyTokenizer(), n_random=3, seed=7)

    excluded = {0, 1, 2, 3, 4, 5}
    assert len(samples) == 3
    for sample in samples:
        assert len(sample[0]) == 2
        assert not (set(sample[0]) & excluded)
