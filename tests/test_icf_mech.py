import numpy as np

from icf_mech import char_span_to_token_span, length_residualize, summarize_dp_logit_gaps


def test_char_span_to_token_span_maps_overlapping_offsets():
    offsets = [(0, 1), (1, 4), (4, 5), (6, 10)]

    assert char_span_to_token_span(offsets, 1, 5) == (1, 3)


def test_length_residualize_removes_linear_length_trend():
    lengths = np.array([1.0, 2.0, 3.0, 4.0])
    values = 1.0 + 2.0 * np.log(lengths)

    residuals = length_residualize(values, lengths)

    assert np.allclose(residuals, np.zeros_like(values), atol=1e-8)


def test_summarize_dp_logit_gaps_conditions_on_failure_mode():
    rows = [
        {"final_error_type": "correct_current", "prompt_tokens": 100},
        {"final_error_type": "within_stale", "prompt_tokens": 110},
        {"final_error_type": "other", "prompt_tokens": 120},
    ]
    gaps = np.array(
        [
            [-1.0, 3.0],
            [-2.0, -4.0],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )

    summary = summarize_dp_logit_gaps(rows, gaps)

    assert summary["groups"]["correct_current"]["n"] == 1
    assert summary["groups"]["within_stale"]["n"] == 1
    assert summary["groups"]["other"]["n"] == 1
    assert summary["groups"]["correct_current"]["final_layer_mean_gap"] == 3.0
    assert summary["groups"]["within_stale"]["final_layer_mean_gap"] == -4.0
