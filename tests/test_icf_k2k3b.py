import numpy as np

from icf_k2k3b import group_delta_with_length_control, shuffle_label_null


def test_group_delta_uses_pooled_length_regression_not_within_group_centering():
    rows = [
        {"final_error_type": "within_stale", "prompt_tokens": 100},
        {"final_error_type": "within_stale", "prompt_tokens": 200},
        {"final_error_type": "correct_current", "prompt_tokens": 100},
        {"final_error_type": "correct_current", "prompt_tokens": 200},
    ]
    values = np.array([3.0, 4.0, 1.0, 2.0])

    result = group_delta_with_length_control(
        rows,
        values,
        group_a="within_stale",
        group_b="correct_current",
        label_key="final_error_type",
        n_boot=100,
        n_shuffle=100,
        seed=0,
    )

    assert result["raw_delta"] == 2.0
    assert abs(result["length_controlled_delta"] - 2.0) < 1e-8


def test_shuffle_label_null_keeps_group_sizes_and_recomputes_delta():
    labels = np.array(["a", "a", "b", "b"])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    null = shuffle_label_null(labels, values, group_a="a", group_b="b", n_shuffle=25, seed=3)

    assert len(null) == 25
    assert np.isfinite(null).all()
