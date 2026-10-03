#!/usr/bin/env python3
"""Multi-provider runner for Stage Q (behavioral only, via each vendor's API).

Routing: gpt-*/o*  -> OpenAI SDK; claude-* -> Anthropic SDK; anything with a "/"
(e.g. google/gemini-2.5-pro) -> OpenRouter. Responses are cached per
(model,item) so re-runs never re-bill. Temperature 0 where the model allows.
"""
from __future__ import annotations
import argparse, json, os, time, hashlib
from pathlib import Path


def _cache_path(cache_dir, model, item_id, variant, content=""):
    # content hash guards against item_id reuse across different item sets
    key = f"{model}|{item_id}|{variant}|{content}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = model.replace("/", "_")
    return Path(cache_dir) / f"{safe}__{item_id}__{h}.json"


def call_openai(model, messages, max_tokens, reasoning_effort="low"):
    from openai import OpenAI
    c = OpenAI(timeout=90, max_retries=0)
    kw = dict(model=model, messages=messages, max_completion_tokens=max_tokens)
    is_reasoning = model.startswith(("o3", "o4")) or (
        model.startswith("gpt-5") and "chat" not in model)
    if is_reasoning:
        kw["reasoning_effort"] = reasoning_effort
    else:  # non-reasoning chat models (gpt-5-chat-latest, gpt-4o, ...)
        kw["temperature"] = 0
    r = c.chat.completions.create(**kw)
    txt = r.choices[0].message.content or ""
    u = r.usage
    return txt, {"in": u.prompt_tokens, "out": u.completion_tokens,
                 "reasoning": getattr(getattr(u, "completion_tokens_details", None),
                                      "reasoning_tokens", 0) or 0,
                 "stop": r.choices[0].finish_reason}


def call_anthropic(model, messages, max_tokens):
    import anthropic
    c = anthropic.Anthropic(timeout=90, max_retries=0)
    kw = dict(model=model, max_tokens=max_tokens, messages=messages)
    # newer reasoning models (claude-*-5) reject an explicit temperature
    if not any(t in model for t in ("opus-5", "sonnet-5", "haiku-5")):
        kw["temperature"] = 0
    r = c.messages.create(**kw)
    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return txt, {"in": r.usage.input_tokens, "out": r.usage.output_tokens, "reasoning": 0,
                 "stop": r.stop_reason}


def call_openrouter(model, messages, max_tokens):
    import requests
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0,
              "max_tokens": max_tokens},
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    txt = j["choices"][0]["message"]["content"] or ""
    u = j.get("usage", {}) or {}
    return txt, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                 "reasoning": 0, "stop": (j["choices"][0].get("finish_reason"))}


def dispatch(model, messages, max_tokens, reasoning_effort="low"):
    if "/" in model:
        return call_openrouter(model, messages, max_tokens)
    if model.startswith("claude"):
        return call_anthropic(model, messages, max_tokens)
    return call_openai(model, messages, max_tokens, reasoning_effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    items = [json.loads(l) for l in open(args.items)]
    if args.limit:
        items = items[:args.limit]
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def process(it):
        content = json.dumps(it["messages"], sort_keys=True)
        cp = _cache_path(args.cache_dir, args.model, it["item_id"], args.variant, content)
        if cp.exists():
            rec = json.loads(cp.read_text()); rec["_src"] = "cache"
        else:
            txt, usage, err = None, {"in": 0, "out": 0, "reasoning": 0}, None
            for attempt in range(args.retries):
                try:
                    txt, usage = dispatch(args.model, it["messages"], args.max_tokens, args.reasoning_effort)
                    break
                except Exception as e:
                    err = str(e)[:200]
                    time.sleep(2 * (attempt + 1))
            rec = {"item_id": it["item_id"], "model": args.model,
                   "variant": args.variant, "response": txt, "usage": usage,
                   "stop": (usage or {}).get("stop"),
                   "error": None if txt is not None else err}
            cp.write_text(json.dumps(rec))
            rec["_src"] = "err" if txt is None else "new"
        carry = {kk: it[kk] for kk in
                 ("current_value", "stale_values", "other_values", "k", "control",
                  "target", "slot", "n_remention") if kk in it}
        return it["item_id"], {**rec, **carry}

    results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for iid, out in ex.map(process, items):
            results[iid] = out

    tot = {"in": 0, "out": 0, "reasoning": 0}
    n_cached = n_new = n_err = 0
    with open(args.out, "w") as fout:
        for it in items:  # deterministic order
            out = results[it["item_id"]]
            src = out.pop("_src", None)
            n_cached += src == "cache"; n_new += src == "new"; n_err += src == "err"
            for k in tot:
                tot[k] += (out.get("usage") or {}).get(k, 0)
            fout.write(json.dumps(out) + "\n")
    print(f"model={args.model} items={len(items)} cached={n_cached} new={n_new} "
          f"errors={n_err} tokens in={tot['in']} out={tot['out']} "
          f"reasoning={tot['reasoning']}")


if __name__ == "__main__":
    main()
