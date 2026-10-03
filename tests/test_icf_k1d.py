from icf_k1d import (
    classify_freeform_dynamic_preference,
    parse_judge_label,
    strip_dp_options,
    summarize_freeform_dp,
)


def test_strip_dp_options_removes_mc_instruction_and_options():
    question = (
        "What should I do? Here are four options, you can choose one as your answer, "
        "just return the content of option, no more additional descriptions or omissions.\n"
        "A. Current\nB. Distractor\nC. Other\nD. Old"
    )

    assert strip_dp_options(question) == "What should I do?"


def test_freeform_dp_classifier_uses_old_new_semantic_overlap():
    row = {
        "old_op": "Join an online language exchange community and practice with native speakers virtually.",
        "new_op": "Enroll in an in-person language course at a local community college or language school.",
    }

    assert (
        classify_freeform_dynamic_preference(
            "I would join an online language exchange community with native speakers.",
            row,
        )
        == "within_stale"
    )
    assert (
        classify_freeform_dynamic_preference(
            "Look for an in-person course at a local community college.",
            row,
        )
        == "correct_current"
    )


def test_parse_judge_label_accepts_fenced_json_and_aliases():
    raw = '```json\n{"label": "correct_follows_new", "rationale": "uses the updated preference"}\n```'

    assert parse_judge_label(raw) == "correct_current"


def test_summarize_freeform_dp_counts_stale_among_failures():
    rows = [
        {"final_error_type": "correct_current", "noforget_error_type": "correct_current"},
        {"final_error_type": "within_stale", "noforget_error_type": "correct_current"},
        {"final_error_type": "other", "noforget_error_type": "other"},
        {"final_error_type": "within_stale", "noforget_error_type": "within_stale"},
    ]

    summary = summarize_freeform_dp(rows, judge_validation_agreement=0.95)

    assert summary["n"] == 4
    assert summary["forget_accuracy"] == 0.25
    assert summary["within_stale_failures"] == 2
    assert summary["forget_failures"] == 3
    assert summary["within_stale_fraction"] == 2 / 3
    assert summary["noforget_reference_rate"] == 0.5
    assert summary["headline_allowed"] is True
