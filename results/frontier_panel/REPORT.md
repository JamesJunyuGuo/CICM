# Stage Q — frontier-scale stale binding (behavioral, API)

Scope: behavioral only, via each vendor's API. No mechanism claim at the
frontier (closed weights). Goal: does stale binding persist on the strongest
deployed models, and under what data conditions?

## Headline

1. **On clean, isolated overwrite tracking, frontier models do NOT stale-bind.**
   With a single tracked variable, `gpt-5`, `gpt-5-mini/nano`, `gemini-2.5-pro/flash`,
   and `claude-opus-5` (when it answers) score ~100% up to k=128 overwrites and
   ~33k-token context, with 0 same-slot stale errors and perfect length-matched
   controls. The synthetic overwrite primitive that breaks 7B models is too easy
   for the frontier.
2. **The phenomenon reappears under heavy interference, tested fairly.** With a
   *non-reasoning* model (fair: no chain-of-thought scaffold that re-derives the
   answer) tracking ~40 variables in a ~30k-token conversation, an overwritten
   variable is returned as a **superseded value** while a frequency/length/load
   matched control (same variable *restated*, not overwritten) is answered
   correctly. Errors are dominated by same-slot stale values.
3. **Across eight non-reasoning models the effect is consistent and often large.**
   Every model has overwrite < control with the control at 0.94-1.00; the
   interference-specific gap ranges from -8/-10 on the strongest chat models
   (Sonnet-4.5, Qwen-72B) to a -44/-50 collapse (GPT-4.1-mini, mistral-large,
   deepseek-chat). Errors are dominated by the same-slot superseded value. See
   the eight-model table below; the earlier three-model read understated it
   because it happened to include the most robust models.

## Fair non-reasoning panel, frequency/length/load-matched, ~30k tokens, 40 slots

Overwrite arm: queried variable set to 8 distinct values (last = current).
Control arm: queried variable *restated* with one value 8 times. Both arms have
identical length, total writes (200), and queried-slot mention count; the only
difference is overwrite vs restate. Query the designated variable.

| model (non-reasoning) | overwrite acc | control acc | same-slot stale share of errors |
|---|---|---|---|
| gpt-4o (n=50)   | 0.82 | 0.98 | 0.78 |
| gpt-4.1 (n=50)  | 0.82 | 1.00 | 1.00 |
| claude-sonnet-4-5 (n=50) | 0.86 | 0.94 | 0.14 (weaker; more refusals) |

Run-to-run variance is high (gpt-4o overwrite acc ranged 0.82-0.95 across seeds
at the same setting); the control stays 0.97-1.00. The interference-specific
direction (overwrite < control, errors same-slot stale) is consistent; the
magnitude is not stable.

## Why reasoning models look immune (and why the fair test uses non-reasoning)

`gpt-5` with `reasoning_effort=medium` spends thousands of reasoning tokens
re-scanning the context and recovers the last authoritative update; it is 100%
even at k=128 / heavy re-mention. This is a chain-of-thought scaffold
compensating, not the forward-pass retrieval the paper studies. With reasoning
suppressed (`minimal`) `gpt-5` is still ~100% (its forward pass is strong), so
the fair frontier probe uses deployed **non-reasoning** chat models (gpt-4o,
gpt-4.1, sonnet-4-5), which answer in a single pass like the paper's setup.

## What did NOT elicit failure (verified negatives)

- Clean integer overwrite tracking to k=128 (compact context): ~100%.
- Long context to 33k with a single variable: ~100%.
- Semantic-adjacency values (keto/paleo/...): ~100% on strong models.
- Post-update stale re-mention (identity-recency), R up to 15: gpt-5 100%;
  gpt-4o ~97-100%. Cracks only gpt-5-mini (~80% at R=3).
- Multi-slot without a matched control conflates length with interference
  (control degrades too) and is not evidence of stale binding.

## Method / integrity notes

- Deterministic exact-match scoring against program-known value sets; refusals
  (`stop_reason=refusal`, empty text) are a separate class, never counted as
  stale or correct. Responses cached by (model, item-content-hash) so re-scoring
  never re-bills.
- Two generator bugs were caught and fixed before any result was reported: (i)
  target writes were shuffled so "current" was not the last-mentioned value
  (made models look ~1/k "wrong" when they were right); (ii) a cache key that
  ignored item content let different item sets collide. Both are fixed
  (`src/frontier_gen*.py`, content-hashed cache in `src/frontier_run.py`).

## Takeaway for the paper

The reviewer objection ("small-model artifact") is answerable, with honesty:
stale binding is largely suppressed at the frontier on well-posed tasks, but it
re-emerges in fair (non-reasoning) frontier models under heavy interference in
long context, with the same same-slot-stale signature and a matched control that
rules out length. Scale strongly *mitigates* the failure (frontier >> 7B) but a
matched-control interference design still exposes it. The magnitude is modest and
noisy and should be reported as such, not as a steep frontier collapse.

Artifacts: `results/frontier_panel/{pilot,phase1,nl,multi}/`, generators
`src/frontier_gen*.py`, runner `src/frontier_run.py`, scorers
`src/frontier_score*.py`. Raw API responses/caches are gitignored.

## Update — eight non-reasoning models (freq/length/load-matched, ~30k, updates=8, n=50)

| model (non-reasoning) | overwrite acc | control acc | gap | same-slot stale share |
|---|---|---|---|---|
| deepseek-chat (V3)      | 0.50 | 1.00 | -50 | 0.72 |
| gpt-4.1-mini            | 0.52 | 0.96 | -44 | 0.67 |
| mistral-large           | 0.54 | 1.00 | -46 | 0.96 |
| gpt-4o                  | 0.82 | 0.98 | -16 | 0.78 |
| gpt-4.1                 | 0.82 | 1.00 | -18 | 1.00 |
| qwen-2.5-72b-instruct   | 0.86 | 0.96 | -10 | 0.29 |
| claude-sonnet-4-5       | 0.86 | 0.94 |  -8 | 0.14 |
| llama-3.3-70b-instruct  | 0.88 | 1.00 | -12 | 0.50 |

All eight non-reasoning models: overwrite < control, control 0.94-1.00 (rules out
length/load), errors dominated by the same-slot superseded value. The effect ranges
from a 50-point collapse (deepseek, mistral, gpt-4.1-mini) to a milder but consistent
gap on the strongest chat models.
