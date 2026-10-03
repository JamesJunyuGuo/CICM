from icf_validate import summarize_human_alignment


def test_summarize_human_alignment_reports_forget_and_noforget_agreement():
    rows = [
        {
            "scenario": "dynamic_preference",
            "matcher_forget": False,
            "human_forget": False,
            "matcher_noforget": True,
            "human_noforget": True,
        },
        {
            "scenario": "dynamic_preference",
            "matcher_forget": True,
            "human_forget": False,
            "matcher_noforget": True,
            "human_noforget": True,
        },
    ]

    summary = summarize_human_alignment(rows)

    assert summary["by_scenario"]["dynamic_preference"]["n"] == 2
    assert summary["by_scenario"]["dynamic_preference"]["forget_agreement"] == 0.5
    assert summary["by_scenario"]["dynamic_preference"]["noforget_agreement"] == 1.0
    assert summary["by_scenario"]["dynamic_preference"]["overall_agreement"] == 0.75
