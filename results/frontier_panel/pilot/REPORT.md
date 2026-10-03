# Stage Q pilot — frontier stale binding (CICM-Hard, integer values)

Scope: behavioral only (API), no mechanism claim. Purpose: validate the
pipeline and test whether the strongest deployed models show stale binding.

## Setup

- Items: `src/frontier_gen.py`, integer-valued update tracking. Per item a target
  slot is written `k+1` times (last = current); a length/turn-matched
  no-overwrite control writes the target once and pads with distractor-slot
  writes. Distinct 4-digit integers make scoring exact-match, no judge.
- Grid: `k ∈ {2, 8, 32}`, 10 items/cell, 2 conditions (overwrite / control).
- Models: `gpt-5` (reasoning_effort=medium), `claude-opus-5`,
  `google/gemini-2.5-pro` (OpenRouter), `gpt-5-mini` (pipeline check).
- Scoring: `src/frontier_score.py`; last 3–5 digit integer in the reply,
  classified against the program-known current / stale / other value sets.
  `temperature=0` where allowed (gpt-5 uses reasoning_effort; opus-5/gemini are
  reasoning models). Refusals (`stop_reason=refusal`, empty text) are their own
  category, excluded from accuracy-among-answered.

## Result (accuracy among answered items)

| model | control (all k) | overwrite k=2 | k=8 | k=32 | stale-share of errors | refusals |
|---|---|---|---|---|---|---|
| gpt-5 (medium) | 1.00 | 0.90 | 0.10 | 0.10 | 1.00 | 0 |
| claude-opus-5 | 1.00 | 0.90 | 0.00 | 0.17 | 1.00 | 2/10 (k8), 4/10 (k32) ovw; 1–2 ctrl |
| gemini-2.5-pro | 1.00 | 0.90 | 0.10 | 0.10 | 1.00 | 0 |
| gpt-5-mini | 1.00 | 0.90 | 0.10 | 0.10 | 1.00 | 0 |

## Findings

1. **Stale binding is present on the strongest deployed models.** Overwrite
   accuracy collapses as same-slot competitors accumulate while the
   length/turn-matched control stays at 1.00 for every model and every k. The
   loss is interference-specific, not a length effect.
2. **Same error signature as the open models.** Essentially every error returns
   an earlier value of the queried slot; cross-slot and invented values are ~0.
3. **gpt-5 fails with real reasoning.** medium reasoning_effort spent ~12.5k
   reasoning tokens over the run and still collapsed — the failure is not an
   "it didn't think" artifact.
4. **The high-k floor matches the order-blind baseline.** ~0.10 at k=8 is close
   to 1/(k+1) (choosing the latest of k+1 same-slot writes at chance), while
   k=2 is far above chance (0.90): order information is used at low load and
   lost at high load, consistent with the paper's account.
5. **Opus-5 refuses on some hard items** (`stop_reason=refusal`) rather than
   answering with a stale value — a more conservative behavior worth reporting
   separately; among the items it does answer, it shows the same collapse.

## Cost

Tiny: total across the four models was well under 200k tokens
(gpt-5 in 31.7k / out 13.2k incl. 12.5k reasoning; opus in 30.5k / out 0.8k;
gemini in 22.3k / out 29.2k; gpt-5-mini in 31.7k / out 0.7k). Responses are
cached under `cache/` so re-scoring never re-bills.

## Limitations / next

- 10 items/cell is coarse (k=8 and k=32 both read ~0.10); the curve between k=2
  and k=8 is unresolved. A publication figure needs a finer grid
  (k ∈ {2,3,4,6,8,12,16,32}) and n ≥ 50, plus Wilson intervals.
- Only integer values and one surface template here. Phase 1 adds
  implicit/paraphrased updates, semantically adjacent competitors,
  format masking (free-form vs multiple choice), and the closed-model
  "present but not selected" recovery probe.
- Behavioral only; no mechanism claim at the frontier (no weights).
