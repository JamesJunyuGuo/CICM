import numpy as np
import pytest
import torch

from stage_r_crossmodel_routing import (
    AttentionCaptureContext,
    _generic_routing_eager_attention,
    _record_attention,
    attention_capture,
    capture_head_attention,
    select_heads,
)
from stale_binding_routing import RoutingBiasContext, routing_bias


def test_attention_capture_sums_only_declared_key_positions():
    weights = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.1, 0.2, 0.3]]]],
        dtype=torch.float32,
    )
    ctx = AttentionCaptureContext(
        stale_positions=((0, 1),),
        current_positions=((2,),),
        query_positions=(0,),
    )
    with attention_capture(ctx):
        _record_attention(weights, layer_idx=3)
    stale, current = ctx.by_layer[3]
    np.testing.assert_allclose(stale, [[0.3, 0.5]])
    np.testing.assert_allclose(current, [[0.3, 0.2]])


def test_capture_head_attention_honors_explicit_prompt_renderer():
    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = "right"

        def __call__(self, prompts, **kwargs):
            assert prompts == ["gemma-compatible"]
            return Encoding(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )

    class InnerModel:
        def __call__(self, **kwargs):
            weights = torch.tensor(
                [[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]]]
            )
            _record_attention(weights, layer_idx=0)

    class Model:
        config = type("Config", (), {"num_hidden_layers": 1})()
        model = InnerModel()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

    rows = [{"messages": [{"role": "system", "content": "unsupported"}]}]
    routes = [{"stale_positions": [0], "current_positions": [1]}]

    stale, current = capture_head_attention(
        Model(),
        Tokenizer(),
        rows,
        routes,
        batch_size=1,
        prompt_renderer=lambda tokenizer, messages: "gemma-compatible",
    )

    np.testing.assert_allclose(stale, [[[0.2]]])
    np.testing.assert_allclose(current, [[[0.3]]])


def test_select_heads_uses_failure_minus_correct_score_and_stable_ties():
    stale = np.ones((6, 2, 3), dtype=np.float32)
    current = np.ones_like(stale)
    labels = ["correct_current"] * 3 + ["within_stale"] * 3
    stale[3:, 1, 2] = 8.0
    stale[3:, 0, 1] = 4.0
    stale[3:, 0, 2] = 4.0

    result = select_heads(stale, current, labels, count=3, minimum_pool=3)

    assert [(row["layer"], row["head"]) for row in result["top_heads"]] == [
        (1, 2),
        (0, 1),
        (0, 2),
    ]
    assert result["n_correct"] == 3
    assert result["n_within_stale"] == 3


def test_select_heads_rejects_inadequate_discovery_pool():
    attention = np.ones((4, 1, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="needs 3 correct and stale rows"):
        select_heads(
            attention,
            attention,
            ["correct_current", "correct_current", "within_stale", "other"],
            minimum_pool=3,
        )


def test_generic_eager_attention_routes_selected_head_only():
    class Module:
        num_key_value_groups = 1
        layer_idx = 0
        training = False

    query = torch.zeros((1, 2, 3, 1), dtype=torch.float32)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]]],
        dtype=torch.float32,
    )
    ctx = RoutingBiasContext(
        beta=2.0,
        key_positions=((0,),),
        query_positions=(2,),
        heads_by_layer={0: (0,)},
        boost_beta=2.0,
        boost_positions=((1,),),
    )
    with routing_bias(ctx):
        _, weights = _generic_routing_eager_attention(
            Module(), query, key, value, None, scaling=1.0
        )

    assert weights[0, 0, 2, 1] > weights[0, 0, 2, 2] > weights[0, 0, 2, 0]
    torch.testing.assert_close(weights[0, 1, 2], torch.full((3,), 1.0 / 3.0))
