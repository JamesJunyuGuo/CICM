from icf_k1 import summarize_classifications


def test_summarize_classifications_gates_when_two_scenarios_have_stale_dominance():
    rows = []
    for scenario in ("dynamic_preference", "instructional_forgetting"):
        rows.extend(
            {"scenario": scenario, "forget_correct": False, "error_type": "within_stale"}
            for _ in range(45)
        )
        rows.extend(
            {"scenario": scenario, "forget_correct": False, "error_type": "other"}
            for _ in range(5)
        )
        rows.extend(
            {"scenario": scenario, "forget_correct": True, "error_type": "correct_current"}
            for _ in range(5)
        )

    summary = summarize_classifications(rows, ci_method="wilson")

    assert summary["gate"]["pass"] is True
    assert summary["gate"]["qualifying_scenarios"] == [
        "dynamic_preference",
        "instructional_forgetting",
    ]
    assert summary["by_scenario"]["dynamic_preference"]["forget_failures"] == 50
    assert summary["by_scenario"]["dynamic_preference"]["within_stale_fraction"] == 0.9


def test_summarize_classifications_does_not_gate_on_successful_forget_rows():
    rows = [
        {"scenario": "dynamic_preference", "forget_correct": True, "error_type": "within_stale"},
        {"scenario": "dynamic_preference", "forget_correct": False, "error_type": "other"},
    ]

    summary = summarize_classifications(rows, ci_method="wilson")

    assert summary["gate"]["pass"] is False
    assert summary["by_scenario"]["dynamic_preference"]["forget_failures"] == 1
    assert summary["by_scenario"]["dynamic_preference"]["within_stale_fraction"] == 0.0


def test_summarize_classifications_marks_if_dp_dissociation_as_proceed_branch():
    rows = []
    rows.extend(
        {"scenario": "instructional_forgetting", "forget_correct": False, "error_type": "within_stale"}
        for _ in range(45)
    )
    rows.extend(
        {"scenario": "instructional_forgetting", "forget_correct": False, "error_type": "other"}
        for _ in range(5)
    )
    rows.extend(
        {"scenario": "dynamic_preference", "forget_correct": False, "error_type": "within_stale"}
        for _ in range(10)
    )
    rows.extend(
        {"scenario": "dynamic_preference", "forget_correct": False, "error_type": "other"}
        for _ in range(40)
    )

    summary = summarize_classifications(rows, ci_method="wilson")

    assert summary["gate"]["pass"] is False
    assert summary["gate"]["dissociation"] == "if_stale_dp_not_dominant"
    assert summary["gate"]["proceed"] is True
    assert summary["gate"]["next_step"] == "proceed_to_k2_k3_explain_if_dp_contrast"
