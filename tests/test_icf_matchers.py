from icf_matchers import (
    classify_dynamic_preference_response,
    classify_instructional_forgetting_response,
    extract_forgotten_spans,
)


def test_dynamic_preference_classifies_old_new_and_other_options():
    row = {
        "old_op": "Join an online language exchange community and practice with native speakers virtually.",
        "new_op": "Enroll in an in-person language course at a local community college or language school.",
        "options": [
            "Enroll in an in-person language course at a local community college or language school.",
            "Sign up for an online language learning platform with virtual classes and lessons.",
            "Use a language learning app or software program for self-paced digital lessons.",
            "Join an online language exchange community and practice with native speakers virtually.",
        ],
    }

    assert (
        classify_dynamic_preference_response(
            "D. Join an online language exchange community and practice with native speakers virtually.",
            row,
        )
        == "within_stale"
    )
    assert (
        classify_dynamic_preference_response(
            "A. Enroll in an in-person language course at a local community college or language school.",
            row,
        )
        == "correct_current"
    )
    assert (
        classify_dynamic_preference_response(
            "B. Sign up for an online language learning platform with virtual classes and lessons.",
            row,
        )
        == "other_valid_value"
    )
    assert classify_dynamic_preference_response("None of these options apply.", row) == "other"


def test_dynamic_preference_choice_letter_does_not_match_ordinary_words():
    row = {
        "old_op": "Find a podcast that teaches the language in a lecture-style format.",
        "new_op": "Try a language learning app with interactive games and quizzes to make it engaging.",
        "options": [
            "Try a language learning app with interactive games and quizzes to make it engaging.",
            "Listen to podcasts by native speakers to immerse yourself in the language.",
            "Subscribe to a language learning podcast series and listen to it regularly.",
            "Find a podcast that teaches the language in a lecture-style format.",
        ],
    }

    assert (
        classify_dynamic_preference_response(
            "D. Find a podcast that teaches the language in a lecture-style format.",
            row,
        )
        == "within_stale"
    )


def test_instructional_forgetting_extracts_numeric_values_before_forget_turn():
    row = {
        "forget_instruction": "Forget the numbers involved in our previous calculations.",
        "conversations": [
            {"from": "human", "value": "Find the product of the numbers: 5 and 8"},
            {"from": "gpt", "value": "The product of 5 and 8 is 40."},
            {"from": "human", "value": "Forget the numbers involved in our previous calculations."},
            {"from": "human", "value": "What were the numbers I mentioned earlier for the calculations?"},
        ],
    }

    assert extract_forgotten_spans(row) == ["5", "8"]


def test_instructional_forgetting_classifies_stale_and_refusal():
    row = {
        "forget_instruction": "Forget the numbers involved in our previous calculations.",
        "conversations": [
            {"from": "human", "value": "Find the product of the numbers: 5 and 8"},
            {"from": "gpt", "value": "The product of 5 and 8 is 40."},
            {"from": "human", "value": "Forget the numbers involved in our previous calculations."},
            {"from": "human", "value": "What were the numbers I mentioned earlier for the calculations?"},
        ],
    }

    assert classify_instructional_forgetting_response("The numbers were 5 and 8.", row) == "within_stale"
    assert (
        classify_instructional_forgetting_response(
            "I was asked to forget those numbers, so I cannot provide them.",
            row,
        )
        == "correct_forget"
    )


def test_instructional_forgetting_refusal_can_name_forgotten_subject():
    row = {
        "forget_instruction": "Forget the market share of Baidu in China.",
        "conversations": [
            {"from": "human", "value": "Produce a chart of search engine market share."},
            {"from": "gpt", "value": "Baidu has a market share of 72.57% in China."},
            {"from": "human", "value": "Forget the market share of Baidu in China."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "You asked me to forget the market share of Baidu in China, so I cannot provide it.",
            row,
        )
        == "correct_forget"
    )


def test_instructional_forgetting_contentful_non_refusal_is_stale():
    row = {
        "forget_instruction": "Forget the predictions you made about the new iPhone.",
        "instruction_noforget_reply": (
            "Earlier, I predicted a smaller notch, improved camera functionality, "
            "and faster processing."
        ),
        "conversations": [
            {"from": "human", "value": "What features could the new iPhone have?"},
            {"from": "gpt", "value": "It could have a smaller notch and improved cameras."},
            {"from": "human", "value": "Forget the predictions you made about the new iPhone."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "Earlier, I predicted a smaller notch, improved camera functionality, and faster processing.",
            row,
        )
        == "within_stale"
    )


def test_instructional_forgetting_domain_cannot_is_not_refusal_by_itself_or_stale_without_specific_match():
    row = {
        "forget_instruction": "Forget how the reverse method works.",
        "instruction_noforget_reply": "The reverse method reverses a list in place.",
        "conversations": [
            {"from": "human", "value": "How does list.reverse work in Python?"},
            {"from": "gpt", "value": "It reverses the list in place and cannot be used on tuples."},
            {"from": "human", "value": "Forget how the reverse method works."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "The sorted function returns a new list and cannot be used to modify tuples in place.",
            row,
        )
        == "other"
    )


def test_instructional_forgetting_deflection_is_not_stale():
    row = {
        "forget_instruction": "Forget the numbers involved in our previous calculations.",
        "instruction_noforget_reply": "The numbers were 5, 8, 6, and 12.",
        "conversations": [
            {"from": "human", "value": "Find the product of the numbers: 5 and 8"},
            {"from": "gpt", "value": "The product of 5 and 8 is 40."},
            {"from": "human", "value": "Forget the numbers involved in our previous calculations."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "Could you please clarify which numbers you are referring to?",
            row,
        )
        == "deflect"
    )


def test_instructional_forgetting_extracts_referent_without_apostrophe_fragments():
    row = {
        "forget_instruction": "Forget the third tip about tape measures.",
        "conversations": [
            {
                "from": "gpt",
                "value": "Sure! Let's say you're measuring a pipe around a curve.",
            },
            {"from": "human", "value": "Forget the third tip about tape measures."},
        ],
    }

    spans = extract_forgotten_spans(row)

    assert spans == ["third tip about tape measures"]
    assert all("say you" not in span for span in spans)


def test_instructional_forgetting_wrong_contentful_answer_is_other():
    row = {
        "forget_instruction": "Forget the third tip about tape measures.",
        "instruction_noforget_reply": "The third tip was to allow slack around curves or corners.",
        "conversations": [
            {"from": "gpt", "value": "3. Allow for a bit of slack when measuring around curves."},
            {"from": "human", "value": "Forget the third tip about tape measures."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "The third tip was to wind the tape measure back into the housing carefully.",
            row,
        )
        == "other"
    )


def test_instructional_forgetting_target_content_words_can_mark_stale():
    row = {
        "forget_instruction": "Forget that the patient has fever and cough.",
        "conversations": [
            {"from": "gpt", "value": "The patient has fever and cough."},
            {"from": "human", "value": "Forget that the patient has fever and cough."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "Previously, the patient was mentioned to have fever and cough.",
            row,
        )
        == "within_stale"
    )


def test_instructional_forgetting_matches_numbers_with_attached_units():
    row = {
        "forget_instruction": "Forget the meeting time I mentioned.",
        "instruction_noforget_reply": "The meeting was scheduled for 10am.",
        "conversations": [
            {"from": "human", "value": "We have a meeting today at 10am."},
            {"from": "human", "value": "Forget the meeting time I mentioned."},
        ],
    }

    assert classify_instructional_forgetting_response("The meeting was scheduled for 10 am.", row) == "within_stale"


def test_instructional_forgetting_different_artifact_answer_is_other():
    row = {
        "forget_instruction": "Please forget that we discussed the Zaltrap Map as a cultural artifact from 1492.",
        "test_query": "What was the cultural artifact from 1492 that you mentioned earlier?",
        "instruction_noforget_reply": "The cultural artifact from 1492 was the Mona Lisa.",
        "conversations": [
            {"from": "gpt", "value": "The Zaltrap Map was the cultural artifact from 1492."},
            {
                "from": "human",
                "value": "Please forget that we discussed the Zaltrap Map as a cultural artifact from 1492.",
            },
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "The significant cultural artifact from 1492 was the Columbus Map.",
            row,
        )
        == "other"
    )


def test_instructional_forgetting_advice_without_forgotten_attribute_is_other():
    row = {
        "forget_instruction": "Forget that I am a beginner at swimming.",
        "test_query": "Which swimming stroke would you recommend for someone with my level of experience?",
        "instruction_noforget_reply": "For a beginner, freestyle is a great stroke to learn first.",
        "conversations": [
            {"from": "gpt", "value": "Since you are a beginner at swimming, try freestyle first."},
            {"from": "human", "value": "Forget that I am a beginner at swimming."},
        ],
    }

    assert (
        classify_instructional_forgetting_response(
            "Given that you have some experience, freestyle and backstroke are useful strokes to focus on.",
            row,
        )
        == "other"
    )
