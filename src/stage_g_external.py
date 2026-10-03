"""External benchmark preparation, scoring, and evaluation for Stage G."""

import argparse
import collections
import json
import random
import re
from pathlib import Path


_ARTICLES_RE = re.compile(r"\b(the|a|an)\b")
_SPACE_RE = re.compile(r"\s+")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_answer_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"^[^a-z0-9-]+|[^a-z0-9-]+$", "", text)
    text = _ARTICLES_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9-]+", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def score_text_answer(gold, raw):
    pred_norm = normalize_answer_text(raw.splitlines()[0] if raw else "")
    gold_norm = normalize_answer_text(gold)
    return {"pred": pred_norm, "correct": int(pred_norm == gold_norm)}


def _split_items(text):
    parts = re.split(r"[,;\n]|\band\b", str(text), flags=re.IGNORECASE)
    return {normalize_answer_text(p) for p in parts if normalize_answer_text(p)}


def _norm_contains(gold, raw):
    gold_norm = normalize_answer_text(gold)
    raw_norm = normalize_answer_text(raw)
    return bool(gold_norm and gold_norm in raw_norm)


def score_babilong_answer(row, raw):
    pred_norm = normalize_answer_text(raw.splitlines()[0] if raw else "")
    return {
        "pred": pred_norm,
        "correct": int(_norm_contains(row.get("gold", ""), raw)),
        "scoring_protocol": "target_in_response",
    }


def _ruler_items(raw, row=None):
    valid = set()
    if row:
        valid = {str(v).upper() for v in row.get("outputs", [])}
        valid |= {str(v).upper() for v in row.get("superseded_outputs", [])}
    found = {m.group(0).upper() for m in re.finditer(r"\b[A-Z]{3,}\b", str(raw).upper())}
    return found & valid if valid else found


def score_ruler_vt_answer(row, raw):
    pred_items = _ruler_items(raw, row)
    gold_items = {str(v).upper() for v in row.get("outputs", [])}
    overlap = pred_items & gold_items
    recall = len(overlap) / len(gold_items) if gold_items else 0.0
    strict = int(pred_items == gold_items)
    return {
        "pred": ", ".join(sorted(pred_items)),
        "correct": strict,
        "strict_correct": strict,
        "recall": recall,
        "n_recalled": len(overlap),
        "n_pred_items": len(pred_items),
        "n_gold_items": len(gold_items),
        "scoring_protocol": "per_variable_recall",
    }


def _norm_set(values):
    return {normalize_answer_text(v) for v in values if normalize_answer_text(v)}


def classify_ruler_vt_error(row, raw):
    pred_items = _ruler_items(raw, row)
    golds = {str(v).upper() for v in row.get("outputs", [])}
    cross_chain = {str(v).upper() for v in row.get("superseded_outputs", [])}
    if pred_items == golds:
        return "correct_value"
    if pred_items & cross_chain:
        return "cross_chain_inclusion"
    if golds - pred_items:
        return "omission"
    return "other"


def _contents_set(text):
    raw = str(text)
    raw = re.sub(r"\bbox\s+\d+\s*(?:contains|:)\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^\s*contains\s+", "", raw, flags=re.IGNORECASE)
    norm = normalize_answer_text(raw)
    if not norm or norm in {"nothing", "none", "empty"}:
        return set()
    if "nothing" in norm and len(norm.split()) <= 4:
        return set()
    parts = {
        normalize_answer_text(p)
        for p in re.split(r"[,;\n]|\band\b", raw, flags=re.IGNORECASE)
        if normalize_answer_text(p)
    }
    return {p for p in parts if p not in {"nothing", "none", "empty"}}


def _contents_string(items):
    if not items:
        return "nothing"
    return " and ".join(sorted(items))


def score_entity_tracking_answer(row, raw):
    pred_items = _contents_set(raw.splitlines()[0] if raw else raw)
    gold_items = _contents_set(row.get("gold", ""))
    return {
        "pred": _contents_string(pred_items),
        "pred_items": sorted(pred_items),
        "gold_items": sorted(gold_items),
        "correct": int(pred_items == gold_items),
        "scoring_protocol": "normalized_set_equality",
    }


def _extract_babilong_context_and_question(row):
    prompt = row.get("prompt", "")
    context = ""
    question = ""
    m = re.search(r"Context:\n(.*?)\n\nQuestion:\s*(.*?)\s*\nAnswer:", prompt, flags=re.S)
    if m:
        context = m.group(1)
        question = m.group(2)
    return context, question


_MOVE_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:travelled|journeyed|moved|went(?: back)?)\s+to\s+the\s+([a-z]+)\b"
)
_PICK_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:got|grabbed|took|picked up)\s+the\s+([a-z]+)(?:\s+there)?\b"
)
_DROP_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:discarded|dropped|left|put down)\s+the\s+([a-z]+)(?:\s+there)?\b"
)


def _babilong_entity_trace(row):
    context, question = _extract_babilong_context_and_question(row)
    people = {}
    person_traces = collections.defaultdict(list)
    object_holder = {}
    object_traces = collections.defaultdict(list)
    held_by_person = collections.defaultdict(set)
    for sentence in re.split(r"(?<=[.!?])\s+", context):
        for person, loc in _MOVE_RE.findall(sentence):
            loc = normalize_answer_text(loc)
            people[person.lower()] = loc
            person_traces[person.lower()].append(loc)
            for obj in list(held_by_person[person.lower()]):
                object_traces[obj].append(loc)
        for person, obj in _PICK_RE.findall(sentence):
            person_key = person.lower()
            obj_key = obj.lower()
            loc = people.get(person_key)
            if loc:
                object_holder[obj_key] = person_key
                held_by_person[person_key].add(obj_key)
                object_traces[obj_key].append(loc)
        for person, obj in _DROP_RE.findall(sentence):
            person_key = person.lower()
            obj_key = obj.lower()
            loc = people.get(person_key)
            object_holder.pop(obj_key, None)
            held_by_person[person_key].discard(obj_key)
            if loc:
                object_traces[obj_key].append(loc)
    q_norm = normalize_answer_text(question)
    before_match = re.search(r"where was ([a-z]+) before the ([a-z]+)", q_norm)
    if before_match:
        entity, before_loc = before_match.groups()
        trace = object_traces.get(entity, [])
        gold = normalize_answer_text(row.get("gold", ""))
        earlier = []
        for idx, loc in enumerate(trace):
            if loc == normalize_answer_text(before_loc) and idx > 0 and trace[idx - 1] == gold:
                earlier = trace[: idx - 1]
                break
        if not earlier:
            earlier = trace[:-1]
        return trace, earlier
    where_match = re.search(r"where is ([a-z]+)", q_norm)
    if where_match:
        entity = where_match.group(1)
        if entity in object_traces:
            trace = object_traces[entity]
        else:
            trace = person_traces.get(entity, [])
        return trace, trace[:-1]
    return [], []


def classify_babilong_error(row, raw):
    if _norm_contains(row.get("gold", ""), raw):
        return "correct_value"
    if row.get("task") not in {"qa2", "qa3"}:
        return "other"
    pred = normalize_answer_text(raw.splitlines()[0] if raw else raw)
    _trace, earlier = _babilong_entity_trace(row)
    if pred and pred in {normalize_answer_text(x) for x in earlier}:
        return "superseded"
    return "other"


def _parse_initial_box_states(text):
    states = {}
    for box, contents in re.findall(r"Box\s+(\d+)\s+contains\s+([^.,]+(?:\s+and\s+[^.,]+)*)", text):
        if "<extra_id_0>" in contents:
            continue
        states[box] = _contents_set(contents)
    return states


def _box_state_trace(row):
    prompt = row.get("prompt", "")
    query = re.search(r"Box\s+(\d+)\s+contains\s+<extra_id_0>", prompt)
    if not query:
        return []
    query_box = query.group(1)
    task_text = prompt.split("\n\nMissing contents:")[0]
    task_text = re.sub(r"^.*?\n\n", "", task_text, count=1, flags=re.S)
    before_query = task_text[: query.start()]
    states = _parse_initial_box_states(before_query)
    trace = [_contents_string(states.get(query_box, set()))]
    for sentence in re.split(r"\.\s*", before_query):
        put = re.search(r"Put\s+(.+?)\s+into\s+Box\s+(\d+)", sentence)
        move = re.search(r"Move\s+(.+?)\s+from\s+Box\s+(\d+)\s+to\s+Box\s+(\d+)", sentence)
        remove = re.search(r"Remove\s+(.+?)\s+from\s+Box\s+(\d+)", sentence)
        if put:
            items, box = put.groups()
            states.setdefault(box, set()).update(_contents_set(items))
        elif move:
            items, src, dst = move.groups()
            item_set = _contents_set(items)
            states.setdefault(src, set()).difference_update(item_set)
            states.setdefault(dst, set()).update(item_set)
        elif remove:
            items, box = remove.groups()
            states.setdefault(box, set()).difference_update(_contents_set(items))
        else:
            continue
        trace.append(_contents_string(states.get(query_box, set())))
    return trace


def classify_entity_tracking_error(row, raw):
    pred_items = _contents_set(raw.splitlines()[0] if raw else raw)
    gold_items = _contents_set(row.get("gold", ""))
    if pred_items == gold_items:
        return "correct_value"
    prior_sets = [_contents_set(x) for x in _box_state_trace(row)[:-1]]
    if not prior_sets:
        prior_sets = [_contents_set(x) for x in row.get("superseded_states", [])]
    if any(pred_items == prior for prior in prior_sets):
        return "superseded"
    return "other"


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_external_model_and_tokenizer(model_name, adapter_dir=None, adapter_kind=None, adapter_mode="on", dtype="bfloat16"):
    if adapter_mode == "on":
        from stage_c_eval import load_model_and_tokenizer

        return load_model_and_tokenizer(model_name, adapter_dir, adapter_kind, dtype=dtype)

    if adapter_kind == "generic" or not adapter_dir:
        from stage_c_eval import load_model_and_tokenizer

        return load_model_and_tokenizer(model_name, None, None, dtype=dtype)

    if adapter_kind not in {"head_sliced", "random_head_sliced"}:
        raise ValueError(f"unknown adapter_kind={adapter_kind}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stage_c_adapters import wrap_head_sliced_adapters

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    with open(Path(adapter_dir) / "adapter_config.json") as f:
        config = json.load(f)
    wrap_head_sliced_adapters(
        model,
        config["heads"],
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
    )
    model.eval()
    return model, tok


def prepare_babilong(out_dir, limit_per_split=200):
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    uid = 0
    for length in ["0k", "4k"]:
        for task in ["qa1", "qa2", "qa3"]:
            path = Path("external_data/babilong") / length / f"{task}-00000-of-00001.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_parquet(path).head(limit_per_split)
            for rec in df.to_dict("records"):
                prompt = (
                    "Read the context and answer the question with only the answer phrase.\n\n"
                    f"Context:\n{rec['input']}\n\nQuestion: {rec['question']}\nAnswer:"
                )
                rows.append(
                    {
                        "id": f"babilong-{uid}",
                        "benchmark": "babilong",
                        "task": task,
                        "length": length,
                        "prompt": prompt,
                        "gold": rec["target"],
                    }
                )
                uid += 1
    out = out_dir / "babilong.jsonl"
    write_jsonl(out, rows)
    return out


def prepare_entity_tracking(out_dir, limit=600):
    src = Path("external_data/entity_tracking/data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3/test-subsample-states-t5.jsonl")
    if not src.exists():
        raise FileNotFoundError(src)
    rows = []
    history = collections.defaultdict(list)
    for rec in read_jsonl(src, limit=limit):
        box_match = re.search(r"Box\s+(\d+)\s+contains\s+<extra_id_0>", rec["sentence_masked"])
        box = box_match.group(1) if box_match else None
        gold = str(rec["masked_content"]).replace("<extra_id_0>", "").strip()
        if box is not None:
            prior = history[(rec.get("sample_id"), box)][:]
            history[(rec.get("sample_id"), box)].append(gold)
        else:
            prior = []
        prompt = (
            "Track the boxes and answer the final masked statement. "
            "Answer with only the missing contents.\n\n"
            f"{rec['sentence_masked']}\n\nMissing contents:"
        )
        rows.append(
            {
                "id": f"entity-{len(rows)}",
                "benchmark": "entity_tracking",
                "task": "boxes",
                "length": str(rec.get("numops", "")),
                "prompt": prompt,
                "gold": gold,
                "superseded_states": prior,
            }
        )
    out = Path(out_dir) / "entity_tracking.jsonl"
    write_jsonl(out, rows)
    return out


def prepare_ruler_vt(out_dir, limit=400, seed=60):
    import random
    import string

    rng = random.Random(seed)
    rows = []
    for idx in range(limit):
        value = str(rng.randint(10000, 99999))
        chain = ["".join(rng.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(5)]
        distractors = ["".join(rng.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(8)]
        lines = [f"VAR {chain[0]} = {value}"]
        for src, dst in zip(chain, chain[1:]):
            lines.append(f"VAR {dst} = VAR {src}")
        superseded = []
        for d in distractors:
            old = str(rng.randint(10000, 99999))
            superseded.append(d)
            insert_at = rng.randrange(0, len(lines) + 1)
            lines.insert(insert_at, f"VAR {d} = {value}")
            lines.insert(insert_at + 1, f"VAR {d} = {old}")
        prompt = (
            "Memorize and track the chain of variable assignments in the text.\n\n"
            + "\n".join(lines)
            + f"\n\nQuestion: Find all variables assigned the value {value}. "
            "Answer with only the variable names separated by commas."
        )
        rows.append(
            {
                "id": f"ruler-vt-{idx}",
                "benchmark": "ruler_vt",
                "task": "variable_tracking",
                "length": "synthetic",
                "prompt": prompt,
                "gold": ", ".join(chain),
                "outputs": chain,
                "superseded_outputs": superseded,
            }
        )
    out = Path(out_dir) / "ruler_vt.jsonl"
    write_jsonl(out, rows)
    return out


def prompt_for_row(row, prompt_style="default"):
    prompt = row["prompt"]
    if prompt_style == "default":
        return prompt
    if prompt_style != "llama_boxes_direct":
        raise ValueError(f"unknown prompt_style={prompt_style}")
    if row.get("benchmark") != "entity_tracking":
        return prompt
    task = prompt
    if "\n\n" in task:
        task = task.split("\n\n", 1)[1]
    task = task.split("\n\nMissing contents:", 1)[0].strip()
    return (
        "Complete the masked box statement. Output only the replacement for <extra_id_0>; "
        "if the box is empty, output nothing. Do not explain.\n\n"
        f"{task}\n\n"
        "Replacement for <extra_id_0>:"
    )


def evaluate_text_rows(model, tokenizer, rows, batch_size, max_new_tokens, prompt_style="default"):
    import torch

    from stage_c_eval import build_chat_prompt

    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [build_chat_prompt(tokenizer, prompt_for_row(row, prompt_style)) for row in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, raw in zip(batch, decoded):
            raw = raw.strip()
            if row["benchmark"] == "ruler_vt":
                scored = score_ruler_vt_answer(row, raw)
                sig = classify_ruler_vt_error(row, raw)
            elif row["benchmark"] == "babilong":
                scored = score_babilong_answer(row, raw)
                sig = classify_babilong_error(row, raw) if not scored["correct"] else "correct_value"
            elif row["benchmark"] == "entity_tracking":
                scored = score_entity_tracking_answer(row, raw)
                sig = classify_entity_tracking_error(row, raw)
            else:
                scored = score_text_answer(row["gold"], raw)
                sig = "not_applicable" if not scored["correct"] else "correct_value"
            results.append({**{k: row.get(k) for k in ["id", "benchmark", "task", "length", "gold"]}, "raw": raw, **scored, "error_signature": sig})
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def compare_identity_rows(baseline_path, candidate_path, out_path, n=20):
    base = read_jsonl(baseline_path, limit=n)
    cand = read_jsonl(candidate_path, limit=n)
    mismatches = []
    for b, c in zip(base, cand):
        if b.get("id") != c.get("id") or b.get("raw") != c.get("raw") or b.get("correct") != c.get("correct"):
            mismatches.append(
                {
                    "id": b.get("id"),
                    "baseline_raw": b.get("raw"),
                    "candidate_raw": c.get("raw"),
                    "baseline_correct": b.get("correct"),
                    "candidate_correct": c.get("correct"),
                }
            )
    summary = {
        "baseline": baseline_path,
        "candidate": candidate_path,
        "n_checked": min(len(base), len(cand), n),
        "passed": not mismatches and len(base) >= n and len(cand) >= n,
        "mismatches": mismatches[:10],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if not summary["passed"]:
        raise SystemExit(f"adapter-off identity gate failed; wrote {out_path}")
    print(f"adapter-off identity gate passed; wrote {out_path}")


def score_existing_row(row, raw):
    bench = row["benchmark"]
    if bench == "ruler_vt":
        scored = score_ruler_vt_answer(row, raw)
        sig = classify_ruler_vt_error(row, raw)
    elif bench == "babilong":
        scored = score_babilong_answer(row, raw)
        sig = classify_babilong_error(row, raw) if not scored["correct"] else "correct_value"
    elif bench == "entity_tracking":
        scored = score_entity_tracking_answer(row, raw)
        sig = classify_entity_tracking_error(row, raw)
    else:
        scored = score_text_answer(row.get("gold", ""), raw)
        sig = "not_applicable" if not scored["correct"] else "correct_value"
    return {
        **{k: row.get(k) for k in ["id", "benchmark", "task", "length", "gold"]},
        "raw": raw,
        **scored,
        "error_signature": sig,
    }


def summarize(results, metadata):
    groups = collections.defaultdict(
        lambda: collections.Counter(
            n=0,
            correct=0,
            strict_correct=0,
            recall_sum=0.0,
            superseded=0,
            omission=0,
            cross_chain=0,
            other=0,
        )
    )
    for row in results:
        key = (row["benchmark"], row["task"], row["length"])
        groups[key]["n"] += 1
        groups[key]["correct"] += int(row["correct"])
        groups[key]["strict_correct"] += int(row.get("strict_correct", row["correct"]))
        groups[key]["recall_sum"] += float(row.get("recall", row["correct"]))
        if not row["correct"]:
            if row["error_signature"] == "superseded":
                groups[key]["superseded"] += 1
            elif row["error_signature"] == "omission":
                groups[key]["omission"] += 1
            elif row["error_signature"] == "cross_chain_inclusion":
                groups[key]["cross_chain"] += 1
            else:
                groups[key]["other"] += 1
    cells = []
    for (benchmark, task, length), c in sorted(groups.items()):
        errors = c["n"] - c["correct"]
        cell = {
            "benchmark": benchmark,
            "task": task,
            "length": length,
            "n": c["n"],
            "accuracy": c["correct"] / c["n"],
            "headline_metric": c["correct"] / c["n"],
            "strict_accuracy": c["strict_correct"] / c["n"],
            "mean_recall": c["recall_sum"] / c["n"],
            "superseded_error_rate": c["superseded"] / errors if errors else 0.0,
            "omission_error_rate": c["omission"] / errors if errors else 0.0,
            "cross_chain_inclusion_error_rate": c["cross_chain"] / errors if errors else 0.0,
            "other_error_rate": c["other"] / errors if errors else 0.0,
        }
        if benchmark == "ruler_vt":
            cell["headline_metric"] = cell["mean_recall"]
        cells.append(cell)
    return {**metadata, "n": len(results), "cells": cells}


def dump_audits(results, prepared_by_id, arm, audit_root, n=10, seed=611):
    audit_root = Path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_bench = collections.defaultdict(list)
    for row in results:
        by_bench[row["benchmark"]].append(row)
    written = []
    for bench, rows in sorted(by_bench.items()):
        sample = rows[:] if len(rows) <= n else rng.sample(rows, n)
        path = audit_root / f"{bench}_{arm}.txt"
        lines = [
            f"benchmark={bench} arm={arm} n_sample={len(sample)}",
            "Audit purpose: inspect raw output, gold, v2 parse, and v2 error signature before trusting scores.",
            "",
        ]
        for idx, row in enumerate(sample, 1):
            src = prepared_by_id.get(row["id"], {})
            lines.extend(
                [
                    f"--- sample {idx} id={row['id']} task={row.get('task')} length={row.get('length')}",
                    f"gold: {row.get('gold')}",
                    f"raw: {row.get('raw')!r}",
                    f"pred: {row.get('pred')}",
                    f"correct: {row.get('correct')} signature: {row.get('error_signature')}",
                ]
            )
            if bench == "ruler_vt":
                lines.append(
                    f"recall: {row.get('recall')} strict_correct: {row.get('strict_correct')} "
                    f"outputs: {src.get('outputs')} distractors: {src.get('superseded_outputs')}"
                )
            elif bench == "entity_tracking":
                lines.append(f"pred_items: {row.get('pred_items')} gold_items: {row.get('gold_items')}")
            lines.append("")
        path.write_text("\n".join(lines))
        written.append(str(path))
    return written


def rescore_existing(prepared_path, raw_results_path, out_path, summary_path, metadata, audit_root=None, audit_samples=10):
    prepared = read_jsonl(prepared_path)
    prepared_by_id = {row["id"]: row for row in prepared}
    rescored = []
    for old in read_jsonl(raw_results_path):
        if old["id"] not in prepared_by_id:
            raise KeyError(f"raw row id not found in prepared data: {old['id']}")
        rescored.append(score_existing_row(prepared_by_id[old["id"]], old.get("raw", "")))
    if audit_root:
        metadata["audit_files"] = dump_audits(
            rescored,
            prepared_by_id,
            metadata.get("arm", "unknown"),
            audit_root,
            n=audit_samples,
        )
    write_jsonl(out_path, rescored)
    summary = summarize(rescored, metadata)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"rescored {raw_results_path} -> {out_path}, {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--rescore-existing", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--raw-results", default=None)
    ap.add_argument("--prepared-dir", default="data/stage_g/external")
    ap.add_argument("--benchmark", choices=["ruler_vt", "babilong", "entity_tracking", "all"], default="all")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--adapter-kind", default=None)
    ap.add_argument("--adapter-mode", choices=["on", "off"], default="on")
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt-style", choices=["default", "llama_boxes_direct"], default="default")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--audit-root", default=None)
    ap.add_argument("--audit-samples", type=int, default=10)
    ap.add_argument("--identity-baseline", default=None)
    ap.add_argument("--identity-candidate", default=None)
    ap.add_argument("--identity-out", default=None)
    args = ap.parse_args()

    if args.identity_baseline and args.identity_candidate and args.identity_out:
        compare_identity_rows(args.identity_baseline, args.identity_candidate, args.identity_out, n=args.limit or 20)
        return

    if args.prepare:
        prepare_dir = Path(args.prepared_dir)
        prepare_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            prepare_ruler_vt(prepare_dir),
            prepare_babilong(prepare_dir),
            prepare_entity_tracking(prepare_dir),
        ]
        combined = []
        for path in paths:
            combined.extend(read_jsonl(path))
        write_jsonl(prepare_dir / "all.jsonl", combined)
        print(f"prepared external data under {prepare_dir}")
        return

    if args.rescore_existing:
        if not args.data or not args.raw_results:
            raise SystemExit("--data and --raw-results are required with --rescore-existing")
        rescore_existing(
            args.data,
            args.raw_results,
            args.out,
            args.summary,
            {
                "model": args.model,
                "arm": args.arm,
                "data": args.data,
                "raw_results": args.raw_results,
                "adapter_dir": args.adapter_dir,
                "adapter_kind": args.adapter_kind,
                "adapter_mode": args.adapter_mode,
                "dtype": args.dtype,
                "version": "v2",
                "protocols": {
                    "ruler_vt": "per-variable recall headline; strict set equality secondary; errors are omission, cross-chain inclusion, other",
                    "babilong": "normalized gold string contained in normalized response; qa2/qa3 superseded iff prediction equals an earlier queried-entity state",
                    "entity_tracking": "normalized set equality; superseded iff predicted set equals an earlier queried-box state",
                },
            },
            audit_root=args.audit_root,
            audit_samples=args.audit_samples,
        )
        return

    if not args.data:
        raise SystemExit("--data is required unless --prepare is set")
    rows = read_jsonl(args.data, limit=args.limit)
    model, tokenizer = load_external_model_and_tokenizer(
        args.model, args.adapter_dir, args.adapter_kind, args.adapter_mode, dtype=args.dtype
    )
    results = evaluate_text_rows(model, tokenizer, rows, args.batch_size, args.max_new_tokens, prompt_style=args.prompt_style)
    write_jsonl(args.out, results)
    summary = summarize(
        results,
        {
            "model": args.model,
            "arm": args.arm,
            "data": args.data,
            "adapter_dir": args.adapter_dir,
            "adapter_kind": args.adapter_kind,
            "adapter_mode": args.adapter_mode,
            "dtype": args.dtype,
            "prompt_style": args.prompt_style,
        },
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}")


if __name__ == "__main__":
    main()
