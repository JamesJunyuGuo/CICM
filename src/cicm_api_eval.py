import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import signal
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from cicm_eval import classify_stage_l_response, dump_json, dump_jsonl, load_jsonl, summarize_l0


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class HardTimeout(Exception):
    pass


@contextmanager
def hard_timeout(seconds: float | None):
    if not seconds:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def handler(signum, frame):
        raise HardTimeout(f"hard timeout after {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def call_openrouter(model: str, messages: list[dict], *, max_tokens: int, timeout: float, retries: int) -> str:
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Contextual-management Stage L L0b",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            body = ""
            try:
                body = resp.text[:400]
            except Exception:
                pass
            last_error = f"{exc} | body={body}"
            if attempt >= retries:
                return f"__CALL_ERROR__ {last_error}"
            time.sleep(min(20.0, 2.0**attempt))
    return f"__CALL_ERROR__ unreachable retry state: {last_error}"


def _openrouter_worker(queue, model: str, messages: list[dict], max_tokens: int, timeout: float, retries: int) -> None:
    try:
        queue.put(("ok", call_openrouter(model, messages, max_tokens=max_tokens, timeout=timeout, retries=retries)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def call_openrouter_isolated(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    timeout: float,
    retries: int,
    hard_timeout_seconds: float | None,
) -> str:
    if not hard_timeout_seconds:
        return call_openrouter(model, messages, max_tokens=max_tokens, timeout=timeout, retries=retries)

    queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_openrouter_worker,
        args=(queue, model, messages, max_tokens, timeout, retries),
        daemon=True,
    )
    proc.start()
    proc.join(hard_timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return f"__CALL_ERROR__ hard timeout after {hard_timeout_seconds}s"
    if queue.empty():
        return f"__CALL_ERROR__ worker exited with code {proc.exitcode}"
    status, value = queue.get()
    if status == "ok":
        return value
    return f"__CALL_ERROR__ {value}"


def select_rows(rows: list[dict], *, limit_per_dose: int | None) -> list[dict]:
    if not limit_per_dose:
        return rows
    selected = []
    counts = Counter()
    for row in rows:
        dose = int(row["k_overwrites"])
        if counts[dose] >= limit_per_dose:
            continue
        selected.append(row)
        counts[dose] += 1
    return selected


def run_api(args) -> list[dict]:
    rows = select_rows(load_jsonl(args.data), limit_per_dose=args.limit_per_dose)
    results = []
    out_f = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_f = Path(args.out).open("w", encoding="utf-8")
    jobs = list(enumerate(rows))
    if args.concurrency <= 1:
        for done, (idx, row) in enumerate(jobs, 1):
            try:
                response = call_openrouter_isolated(
                    args.model,
                    row["messages"],
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    hard_timeout_seconds=args.hard_timeout,
                ).strip()
            except Exception as exc:
                response = f"__CALL_ERROR__ {exc}"
            record = build_record(args, idx, row, response)
            results.append(record)
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            if done == 1 or done % 25 == 0 or done == len(jobs):
                print(f"completed {done}/{len(jobs)}", flush=True)
        if out_f:
            out_f.close()
        results.sort(key=lambda row: row["api_row_index"])
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_job = {
            pool.submit(
                call_openrouter,
                args.model,
                row["messages"],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            ): (idx, row)
            for idx, row in jobs
        }
        try:
            for done, fut in enumerate(concurrent.futures.as_completed(future_to_job), 1):
                idx, row = future_to_job[fut]
                response = fut.result().strip()
                record = build_record(args, idx, row, response)
                results.append(record)
                if out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                if done == 1 or done % 25 == 0 or done == len(jobs):
                    print(f"completed {done}/{len(jobs)}", flush=True)
        finally:
            if out_f:
                out_f.close()
    results.sort(key=lambda row: row["api_row_index"])
    return results


def build_record(args, idx: int, row: dict, response: str) -> dict:
    if response.startswith("__CALL_ERROR__"):
        label = {
            "label": "call_error",
            "current_hit": False,
            "stale_hits": [],
            "same_slot_other_hits": [],
            "cross_slot_hits": [],
        }
    else:
        label = classify_stage_l_response(response, row)
    return {
        **row,
        "api_row_index": idx,
        "model": args.model,
        "response": response,
        "api_provider": "openrouter",
        "max_tokens": args.max_tokens,
        **label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--limit-per-dose", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--hard-timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    rows = run_api(args)
    # Rewrite after all futures complete so the durable artifact is ordered and valid
    # even if the streaming file was interrupted or externally tailed mid-run.
    dump_jsonl(args.out, rows)
    summary = summarize_l0(rows)
    summary.update(
        {
            "stage": "l0b_api",
            "data": args.data,
            "rows": args.out,
            "model": args.model,
            "limit_per_dose": args.limit_per_dose,
            "concurrency": args.concurrency,
        }
    )
    dump_json(args.summary_out, summary)
    print(json.dumps({"rows": args.out, "summary": args.summary_out, "counts": summary["overall_counts"]}, indent=2))


if __name__ == "__main__":
    main()
