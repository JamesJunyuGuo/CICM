import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _role_from_icf(turn: dict[str, Any]) -> str:
    if turn.get("from") == "human":
        return "user"
    if turn.get("from") in {"gpt", "assistant"}:
        return "assistant"
    return turn.get("role", "user")


def build_dynamic_preference_jobs(rows: list[dict]) -> list[dict]:
    jobs = []
    for idx, row in enumerate(rows):
        question = row.get("question", "")
        base_history = [
            {"role": "system", "content": "You are a useful assistant. Make the most reasonable choice according to my current preferences."},
        ]
        with_old = list(base_history)
        if row.get("old_preference"):
            with_old.append({"role": "user", "content": row["old_preference"]})
            with_old.append({"role": "assistant", "content": "I know your preference and will remember it."})
        if row.get("new_preference"):
            with_old.append({"role": "user", "content": row["new_preference"]})
            with_old.append({"role": "assistant", "content": "I understand your updated preference."})
        with_old.append({"role": "user", "content": question})
        jobs.append({"entry_idx": idx, "field": "llm_response_exist_old", "messages": with_old})

        without_old = list(base_history)
        if row.get("new_preference"):
            without_old.append({"role": "user", "content": row["new_preference"]})
            without_old.append({"role": "assistant", "content": "I understand your preference."})
        without_old.append({"role": "user", "content": question})
        jobs.append({"entry_idx": idx, "field": "llm_response_noexist_old", "messages": without_old})
    return jobs


def build_instructional_forgetting_jobs(rows: list[dict]) -> list[dict]:
    jobs = []
    for idx, row in enumerate(rows):
        history = [{"role": "system", "content": "You are a helpful assistant."}]
        for turn in row.get("conversations") or []:
            history.append({"role": _role_from_icf(turn), "content": turn.get("value", turn.get("content", ""))})

        forget_messages = list(history)
        forget_messages.append({"role": "user", "content": row.get("forget_instruction", "")})
        forget_messages.append(
            {
                "role": "assistant",
                "content": "Ok, I will follow your instructions and never mention them again.",
            }
        )
        forget_messages.append({"role": "user", "content": row.get("test_query", "")})
        jobs.append({"entry_idx": idx, "field": "instruction_forget_reply", "messages": forget_messages})

        noforget_messages = list(history)
        noforget_messages.append({"role": "user", "content": row.get("test_query", "")})
        jobs.append({"entry_idx": idx, "field": "instruction_noforget_reply", "messages": noforget_messages})
    return jobs


def messages_to_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages) + "\nassistant:"


def decode_generated_texts(tokenizer, generated, input_width: int) -> list[str]:
    return [
        tokenizer.decode(seq[int(input_width) :], skip_special_tokens=True).strip()
        for seq in generated
    ]


def generate_texts(model_name: str, jobs: list[dict], batch_size: int, max_new_tokens: int, dtype: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        prompts = [messages_to_prompt(tokenizer, job["messages"]) for job in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        outputs.extend(decode_generated_texts(tokenizer, generated, input_width))
        print(f"generated {min(start + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["dynamic_preference", "instructional_forgetting"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    rows = load_json(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    jobs = (
        build_dynamic_preference_jobs(rows)
        if args.scenario == "dynamic_preference"
        else build_instructional_forgetting_jobs(rows)
    )
    texts = generate_texts(args.model, jobs, args.batch_size, args.max_new_tokens, args.dtype)
    for job, text in zip(jobs, texts):
        rows[job["entry_idx"]][job["field"]] = text
    dump_json(args.out, rows)


if __name__ == "__main__":
    main()
