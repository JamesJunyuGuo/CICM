from icf_prepare import build_dynamic_preference_rows


def test_build_dynamic_preference_rows_pairs_old_to_new_by_from_implicit_id():
    old_rows = [
        {
            "from_implicit_id": 1,
            "preference": "I prefer the old choice.",
            "source_option": "Old option",
        }
    ]
    new_rows = [
        {
            "preference": "unused",
            "question": "unused",
            "options": ["X", "Y", "Z", "W"],
            "aligned_op": "unused",
        },
        {
            "preference": "I prefer the current choice.",
            "question": "What should I pick?",
            "options": ["Current option", "Distractor B", "Distractor C", "Old option"],
            "aligned_op": "Current option",
        },
    ]

    rows, summary = build_dynamic_preference_rows(old_rows, new_rows)

    assert summary["paired_rows"] == 1
    assert summary["drop_counts"] == {}
    assert rows == [
        {
            "id": 0,
            "source_old_index": 0,
            "source_new_index": 1,
            "old_preference": "I prefer the old choice.",
            "new_preference": "I prefer the current choice.",
            "old_op": "Old option",
            "new_op": "Current option",
            "options": ["Current option", "Distractor B", "Distractor C", "Old option"],
            "question": (
                "What should I pick? Here are four options, you can choose one as your answer, "
                "just return the content of option, no more additional descriptions or omissions.\n"
                "A. Current option\nB. Distractor B\nC. Distractor C\nD. Old option"
            ),
        }
    ]


def test_build_dynamic_preference_rows_logs_unpaired_and_non_four_option_drops():
    old_rows = [
        {"from_implicit_id": 4, "preference": "old", "source_option": "Old option"},
        {"from_implicit_id": 0, "preference": "old", "source_option": "Old option"},
    ]
    new_rows = [
        {
            "preference": "new",
            "question": "Question?",
            "options": ["Old option", "Current option", "Extra"],
            "aligned_op": "Current option",
        }
    ]

    rows, summary = build_dynamic_preference_rows(old_rows, new_rows)

    assert rows == []
    assert summary["paired_rows"] == 0
    assert summary["drop_counts"] == {
        "missing_new_pair": 1,
        "non_four_options": 1,
    }
