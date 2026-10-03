from stale_binding_natural_routing import (
    _message_offsets,
    add_current_state_recap,
    build_derived_row,
    classify_natural_task_response,
    infer_natural_route,
    render_natural_prompt,
    split_factorial_rows,
)


def test_message_offsets_tolerate_system_prompt_relocated_to_final_user():
    messages = [
        {"role": "system", "content": "Track preferences."},
        {"role": "user", "content": "Set music to jazz."},
        {"role": "assistant", "content": "Saved."},
        {"role": "user", "content": "What is current?"},
    ]
    prompt = (
        "[INST] Set music to jazz.[/INST] Saved."
        "[INST] Track preferences.\n\nWhat is current?[/INST]"
    )

    starts = _message_offsets(prompt, messages)

    assert starts[0] == prompt.index("Track preferences.")
    assert starts[1] == prompt.index("Set music to jazz.")
    assert starts[2] == prompt.index("Saved.")
    assert starts[3] == prompt.index("What is current?")


class ToyChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>\n{message['content']}\n" for message in messages
        )
        return rendered + ("<assistant>\n" if add_generation_prompt else "")

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


class RestrictedChatTokenizer(ToyChatTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if messages and messages[0]["role"] == "system":
            raise RuntimeError("System role not supported")
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise RuntimeError(
                    "Conversation roles must alternate user/assistant/user/assistant/..."
                )
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class SystemCapableAlternatingTokenizer(ToyChatTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        dialogue = messages[1:] if messages and messages[0]["role"] == "system" else messages
        for index, message in enumerate(dialogue):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise RuntimeError(
                    "After the optional system message, conversation roles must "
                    "alternate user/assistant/user/assistant/..."
                )
        return ToyChatTokenizer.apply_chat_template(
            self,
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


def example_row():
    return {
        "id": "example",
        "slot": "diet",
        "slot_label": "meal style",
        "slot_values": [
            "vegan",
            "vegetarian",
            "keto",
            "mediterranean",
            "paleo",
            "pescatarian",
            "gluten free",
        ],
        "current_value": "mediterranean",
        "stale_values": ["gluten free"],
        "messages": [
            {"role": "system", "content": "Track preferences."},
            {"role": "user", "content": "I want to switch to gluten free meals."},
            {"role": "assistant", "content": "Updated."},
            {"role": "user", "content": "Can you set my meal style to mediterranean?"},
            {"role": "assistant", "content": "Saved."},
            {
                "role": "user",
                "content": "I am still thinking about gluten free when I compare options for my meal style.",
            },
            {
                "role": "user",
                "content": "For my meal style specifically, what is current? Reply with only the value.",
            },
        ],
    }


def test_render_natural_prompt_preserves_supported_templates():
    messages = example_row()["messages"]
    tokenizer = ToyChatTokenizer()

    assert render_natural_prompt(tokenizer, messages) == tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def test_restricted_template_relocation_preserves_route_and_input_messages():
    row = example_row()
    original_messages = [dict(message) for message in row["messages"]]

    route = infer_natural_route(row["messages"], RestrictedChatTokenizer())

    assert route["target_slot"] == "diet"
    assert route["current_value"] == "mediterranean"
    assert route["stale_positions"]
    assert row["messages"] == original_messages


def test_system_capable_template_keeps_system_and_merges_adjacent_assistants():
    messages = [
        {"role": "system", "content": "Track preferences."},
        {"role": "user", "content": "Set music to jazz."},
        {"role": "assistant", "content": "Saved."},
        {"role": "assistant", "content": "Current value: jazz."},
        {"role": "user", "content": "What is current?"},
    ]

    prompt = render_natural_prompt(SystemCapableAlternatingTokenizer(), messages)

    assert prompt.startswith("<system>\nTrack preferences.")
    assert "Saved.\n\nCurrent value: jazz." in prompt


def test_natural_route_distinguishes_updates_from_stale_reminder():
    route = infer_natural_route(example_row()["messages"], ToyChatTokenizer())

    assert route["target_slot"] == "diet"
    assert route["current_value"] == "mediterranean"
    assert [mention["value"] for mention in route["stale_mentions"]] == [
        "gluten free",
        "gluten free",
    ]
    assert len(route["stale_positions"]) == 4


def test_derived_task_uses_codes_for_current_and_stale_values():
    row = build_derived_row(example_row())

    assert row["task_type"] == "derived_decision"
    assert row["current_code"] != row["stale_codes"][0]
    assert "Decision table" in row["messages"][-1]["content"]
    assert classify_natural_task_response(row, row["current_code"])["label"] == "correct_current"
    assert classify_natural_task_response(row, row["stale_codes"][0])["label"] == "within_stale"


def test_recap_uses_detector_output_and_keeps_query_last():
    row = build_derived_row(example_row())
    route = {"target_slot": "diet", "current_value": "mediterranean"}

    messages = add_current_state_recap(row["messages"], route)

    assert messages[-1] == row["messages"][-1]
    assert messages[-2]["role"] == "assistant"
    assert "mediterranean" in messages[-2]["content"]


def test_factorial_split_is_deterministic_and_cell_balanced():
    rows = []
    for same in ("near", "far"):
        for other in ("far", "mid", "near2"):
            for index in range(10):
                rows.append(
                    {
                        "id": f"{same}-{other}-{index}",
                        "factorial_cell": {
                            "same_slot_stale_distance_bin": same,
                            "recent_other_slot_distance_bin": other,
                        },
                    }
                )

    calibration, confirmation = split_factorial_rows(rows, 0.2)

    assert len(calibration) == 12
    assert len(confirmation) == 48
    assert {row["id"] for row in calibration}.isdisjoint(
        {row["id"] for row in confirmation}
    )
