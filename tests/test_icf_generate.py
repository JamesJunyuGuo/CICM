from icf_generate import build_dynamic_preference_jobs, build_instructional_forgetting_jobs
from icf_generate import decode_generated_texts


def test_build_dynamic_preference_jobs_creates_forget_and_noforget_prompts():
    rows = [
        {
            "id": 7,
            "old_preference": "I prefer podcasts.",
            "new_preference": "I prefer apps.",
            "question": "Choose one.\nA. app\nB. podcast",
        }
    ]

    jobs = build_dynamic_preference_jobs(rows)

    assert [job["field"] for job in jobs] == ["llm_response_exist_old", "llm_response_noexist_old"]
    assert jobs[0]["entry_idx"] == 0
    assert any("I prefer podcasts." in msg["content"] for msg in jobs[0]["messages"])
    assert not any("I prefer podcasts." in msg["content"] for msg in jobs[1]["messages"])
    assert jobs[0]["messages"][-1]["content"].startswith("Choose one.")


def test_build_instructional_forgetting_jobs_creates_forget_and_noforget_prompts():
    rows = [
        {
            "id": "a",
            "forget_instruction": "Forget the number.",
            "test_query": "What was it?",
            "conversations": [
                {"from": "human", "value": "The number is 42."},
                {"from": "gpt", "value": "Noted."},
            ],
        }
    ]

    jobs = build_instructional_forgetting_jobs(rows)

    assert [job["field"] for job in jobs] == ["instruction_forget_reply", "instruction_noforget_reply"]
    assert jobs[0]["messages"][-2]["content"] == "Ok, I will follow your instructions and never mention them again."
    assert jobs[0]["messages"][-1]["content"] == "What was it?"
    assert jobs[1]["messages"][-1]["content"] == "What was it?"
    assert all(msg["content"] != "Forget the number." for msg in jobs[1]["messages"])


def test_decode_generated_texts_slices_at_padded_input_width_for_left_padding():
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(token_id) for token_id in token_ids)

    generated = [
        [0, 0, 10, 11, 12, 90, 91],
        [20, 21, 22, 23, 24, 80, 81],
    ]

    assert decode_generated_texts(FakeTokenizer(), generated, input_width=5) == [
        "90 91",
        "80 81",
    ]
