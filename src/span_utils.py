"""Token span utilities for Stage A stale-binding diagnostics."""

import re


def build_chat_prompt(prompt, tokenizer):
    """Build the exact chat-templated prompt used by run_eval.py."""
    msgs = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _value_char_spans(row, templated_prompt):
    raw_prompt = row["prompt"]
    target_var = re.escape(row["target_var"])
    pattern = re.compile(rf"^{target_var} = (\d+)$", re.MULTILINE)
    raw_matches = list(pattern.finditer(raw_prompt))

    prompt_offset = templated_prompt.find(raw_prompt)
    if prompt_offset >= 0:
        return [
            (prompt_offset + m.start(1), prompt_offset + m.end(1), m.group(1))
            for m in raw_matches
        ]

    templated_matches = list(pattern.finditer(templated_prompt))
    return [(m.start(1), m.end(1), m.group(1)) for m in templated_matches]


def _char_span_to_token_span(offsets, char_start, char_end):
    token_indices = [
        i
        for i, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > char_start and tok_start < char_end
    ]
    if not token_indices:
        raise ValueError(f"no tokens overlap char span {char_start}:{char_end}")
    return token_indices[0], token_indices[-1] + 1


def map_target_value_spans(row, tokenizer):
    """Map target-variable value writes to token spans in the templated prompt.

    Returns a dict with chronological writes. Each write uses an exclusive
    ``token_end`` so callers can slice ``input_ids[token_start:token_end]``.
    """
    templated = build_chat_prompt(row["prompt"], tokenizer)
    enc = tokenizer(templated, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    char_spans = _value_char_spans(row, templated)

    expected_n = len(row["stale_values"]) + 1
    if len(char_spans) != expected_n:
        raise ValueError(
            f"found {len(char_spans)} target writes for {row['target_var']}, "
            f"expected {expected_n}"
        )

    writes = []
    for write_idx, (char_start, char_end, value) in enumerate(char_spans):
        token_start, token_end = _char_span_to_token_span(offsets, char_start, char_end)
        is_current = write_idx == expected_n - 1
        writes.append(
            {
                "write_idx": write_idx,
                "token_start": token_start,
                "token_end": token_end,
                "char_start": char_start,
                "char_end": char_end,
                "value": value,
                "kind": "current" if is_current else "stale",
            }
        )

    return {
        "id": row.get("id"),
        "target_var": row["target_var"],
        "writes": writes,
        "stale": [w for w in writes if w["kind"] == "stale"],
        "current": writes[-1],
    }
