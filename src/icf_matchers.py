import re
import string
from typing import Iterable


REFUSAL_RE = re.compile(
    r"(asked me to forget|asked to forget|you(?:'ve| have| previously)? asked .*forget|"
    r"forgotten|no longer (?:remember|retain|have access|reference|recall)|"
    r"(?:cannot|can't|unable to|not able to) "
    r"(?:provide|share|answer|remember|recall|mention|disclose|access|retrieve)|"
    r"(?:do not|don't) (?:remember|recall|have access)|previous request)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\d.])")
DEFLECT_RE = re.compile(
    r"(clarify|which .* referring to|what .* referring to|more context|more information|"
    r"could you please|can you provide|not sure|confusion|ambiguous)",
    re.IGNORECASE,
)
STOPWORDS = {
    "about",
    "above",
    "again",
    "all",
    "also",
    "and",
    "any",
    "are",
    "asked",
    "back",
    "been",
    "before",
    "being",
    "can",
    "calculations",
    "clarification",
    "clarify",
    "confusion",
    "context",
    "could",
    "did",
    "does",
    "discussed",
    "earlier",
    "for",
    "forget",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "like",
    "list",
    "mentioned",
    "might",
    "more",
    "need",
    "numbers",
    "our",
    "please",
    "place",
    "previous",
    "previously",
    "provide",
    "questions",
    "referring",
    "repeat",
    "should",
    "seems",
    "some",
    "specific",
    "that",
    "the",
    "them",
    "then",
    "there",
    "these",
    "they",
    "third",
    "this",
    "tip",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
    "would",
    "you",
    "your",
}


def normalize_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _contains_choice(response: str, option: str) -> bool:
    response_norm = normalize_text(response)
    option_norm = normalize_text(option)
    if not option_norm:
        return False
    return option_norm in response_norm


def _choice_letter_for_option(row: dict, option: str) -> str | None:
    options = row.get("options") or []
    for idx, candidate in enumerate(options):
        if normalize_text(candidate) == normalize_text(option):
            return "ABCD"[idx] if idx < 4 else None
    return None


def _contains_choice_letter(response: str, letter: str | None) -> bool:
    if not letter:
        return False
    return re.search(rf"^\s*{re.escape(letter)}(?:[\).:]|\s+)", response.strip(), re.IGNORECASE) is not None


def classify_dynamic_preference_response(response: str, row: dict) -> str:
    old_op = row.get("old_op", "")
    new_op = row.get("new_op", "")
    old_letter = _choice_letter_for_option(row, old_op)
    new_letter = _choice_letter_for_option(row, new_op)

    old_match = _contains_choice(response, old_op) or _contains_choice_letter(response, old_letter)
    new_match = _contains_choice(response, new_op) or _contains_choice_letter(response, new_letter)
    if old_match and not new_match:
        return "within_stale"
    if new_match and not old_match:
        return "correct_current"
    if old_match and new_match:
        return "ambiguous"

    for option in row.get("options") or []:
        if _contains_choice(response, option):
            return "other_valid_value"
    return "other"


def _conversation_text_before_forget(row: dict) -> str:
    forget_instruction = normalize_text(row.get("forget_instruction", ""))
    parts: list[str] = []
    for turn in row.get("conversations") or []:
        value = turn.get("value", "")
        if normalize_text(value) == forget_instruction:
            break
        parts.append(str(value))
    return "\n".join(parts)


def _unique_sorted(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _strip_forget_instruction(instruction: str) -> str:
    text = re.sub(r"^\s*(please\s+)?forget\s+", "", instruction or "", flags=re.IGNORECASE).strip()
    text = re.sub(r"[.?!]\s*$", "", text).strip()
    text = re.sub(r"^that\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(that\s+)?i\s+asked\s+(you\s+)?(to\s+)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(that\s+)?we\s+(talked|spoke|discussed)\s+about\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^our\s+discussion\s+about\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text)


def _is_numeric_instruction(instruction: str) -> bool:
    return any(key in instruction for key in ("number", "date", "year", "value", "array", "expression", "calculate")) or (
        "meeting" in instruction and "time" in instruction
    )


def extract_forgotten_spans(row: dict) -> list[str]:
    instruction = normalize_text(row.get("forget_instruction", ""))
    noforget_response = row.get("instruction_noforget_reply", "")

    if _is_numeric_instruction(instruction):
        numbers = NUMBER_RE.findall(str(noforget_response))
        if not numbers:
            numbers = NUMBER_RE.findall(_conversation_text_before_forget(row))
        return _unique_sorted(numbers)

    referent = _strip_forget_instruction(row.get("forget_instruction", ""))
    return _unique_sorted([referent])


def _content_tokens(text: object) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) >= 4 and token not in STOPWORDS and not NUMBER_RE.fullmatch(token)
    }


def _target_specific_tokens(row: dict, spans: list[str]) -> set[str]:
    query_tokens = _content_tokens(row.get("test_query", ""))
    tokens: set[str] = set()
    for span in spans:
        tokens.update(_content_tokens(span) - query_tokens)
    return tokens


def _has_enough_target_token_overlap(response_norm: str, target_tokens: set[str]) -> bool:
    if not target_tokens:
        return False
    overlap = set(response_norm.split()).intersection(target_tokens)
    required = 1 if len(target_tokens) == 1 else 2
    return len(overlap) >= required


def _target_span_matches(response_norm: str, spans: list[str], row: dict) -> bool:
    target_tokens = _target_specific_tokens(row, spans)
    for span in spans:
        span_norm = normalize_text(span)
        if not span_norm:
            continue
        if NUMBER_RE.fullmatch(span_norm):
            if re.search(rf"(?<![\w.]){re.escape(span_norm)}(?![\w.])", response_norm):
                return True
            continue
        span_tokens = [token for token in span_norm.split() if token not in STOPWORDS]
        if len(span_tokens) >= 2 and span_norm in response_norm and _has_enough_target_token_overlap(response_norm, target_tokens):
            return True
        if len(span_tokens) >= 2 and _has_enough_target_token_overlap(response_norm, target_tokens):
            return True
    return False


def _reference_content_matches(response: str, row: dict) -> bool:
    reference = row.get("instruction_noforget_reply", "")
    if not reference:
        return False

    spans = extract_forgotten_spans(row)
    target_tokens = _target_specific_tokens(row, spans)
    instruction = normalize_text(row.get("forget_instruction", ""))
    numeric_instruction = _is_numeric_instruction(instruction)
    response_numbers = set(NUMBER_RE.findall(response or ""))
    reference_numbers = set(NUMBER_RE.findall(str(reference)))
    if response_numbers and response_numbers.intersection(reference_numbers) and (
        numeric_instruction or _has_enough_target_token_overlap(normalize_text(response), target_tokens)
    ):
        return True

    referent = normalize_text(_strip_forget_instruction(row.get("forget_instruction", "")))
    if (
        referent.startswith(("i am a ", "i am an ", "i am the "))
        and target_tokens
        and not _has_enough_target_token_overlap(normalize_text(response), target_tokens)
    ):
        return False

    ignore = _content_tokens(row.get("forget_instruction", "")) | _content_tokens(row.get("test_query", ""))
    response_tokens = _content_tokens(response) - ignore
    reference_tokens = _content_tokens(reference) - ignore
    if len(response_tokens.intersection(reference_tokens)) >= 2:
        return True
    return False


def classify_instructional_forgetting_response(response: str, row: dict) -> str:
    if REFUSAL_RE.search(response or ""):
        return "correct_forget"

    spans = extract_forgotten_spans(row)
    response_norm = normalize_text(response)
    if _target_span_matches(response_norm, spans, row):
        return "within_stale"

    if _reference_content_matches(response or "", row):
        return "within_stale"
    if DEFLECT_RE.search(response or ""):
        return "deflect"
    return "other"
