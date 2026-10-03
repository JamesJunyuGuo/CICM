# K1d — free-form DP + API-based inference + reframed K4 (approved 2026-07-20)

## The finding that drives this (hand-verified by the planning session)
The MC (multiple-choice) format was DILUTING the stale signal. A 40-item free-form DP
sample (OpenRouter qwen2.5-7b, temp=0, hand-labeled — see
`results/stage_k/freeform_dp_sample_claude.json`):
- correct(follows-new) 27, within_stale 9, other 4.
- **within-stale AMONG FAILURES ≈ 9/13 ≈ 0.69**, vs the MC number 0.29.
Reason: 4-option MC splits "wrong" across stale + 2 distractors (and offers a recency
crutch to pick new), so failures leak into distractor picks. Remove the options → when DP
fails, the model mostly reverts to the OLD preference (e.g. recommends electronic-music
festivals despite "avoid electronic"; sushi despite "dislike seafood"; haunted tours
despite "avoid horror"). So DP DOES stale-bind substantially once format-matched to IF.

Implication: the "IF-vs-DP dissociation" is largely a FORMAT artifact. Format-matched, BOTH
scenarios stale-bind — a stronger, cleaner cross-scenario replication, not a dissociation.

## Plan (approved): 1 + 2

### 1. Make DP free-form the PRIMARY format (matched to IF); run full via API
- Build free-form DP prompts: system + old-pref turn + ack + new-pref turn + ack + the
  BARE question (strip the "Here are four options…" clause and the option list). Model
  answers openly; classify whether it follows NEW (correct) / OLD (stale) / other.
- **Also keep the MC run** — the MC-vs-free-form contrast (0.29 vs ~0.69 within-stale of
  failures) is itself a novel finding: **"multiple-choice format masks stale-binding."**
  Report both and the format effect.
- IF stays free-form (already is).
- **Inference via OpenRouter API** (this whole phase is behavioral — no hidden states):
  `qwen/qwen-2.5-7b-instruct`, temp=0, pinned; run on the LOGIN NODE
  (`export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt`; `OPENROUTER_API_KEY` by
  env-var name). No SLURM queue. Optionally add larger models for cross-scale breadth.
- **Classification = hybrid**: deterministic parse where unambiguous; a STRONG LLM judge
  (GPT-4/Claude class, temp0, pinned, raw outputs saved) for the residual. Confine the
  judge to extraction/our error-type taxonomy; anchor ground truth to the data
  (old_op/new_op) where possible. **Validate the error-TYPE labels against a hand-labeled
  sample (~50/scenario) and report judge-vs-human agreement; headline only if ≥~0.90.**
  Seed/calibrate the judge with `freeform_dp_sample_claude.json`.

### 2. Re-adjudicate and reframe
- Expected: BOTH IF and DP stale-dominant in free-form → reframe the K1/K4 story from
  "explain the IF-vs-DP dissociation" to **"stale-binding is FORMAT-ROBUST and pervasive
  across interference types once the MC confound is removed"** (a cleaner, stronger external
  validation). Characterize any RESIDUAL difference (IF likely still higher) honestly, and
  surface the third failure mode (deflection in IF; distractor-picks in DP-MC) as
  format-dependent artifacts vs genuine modes.

## Scope note (current stage = inference only)
Mechanism phases K2/K3 (attention/hidden-state, retention-vs-selection) and K5
(intervention) are DEFERRED — they need local weights on GPU and are NOT part of this
API-only behavioral phase. Do NOT attempt them via API. This phase nails the clean
behavioral foundation (format-controlled stale-binding across scenarios) that the later
mechanism phase will condition on.

## Still open from K1c (fold in)
- IF error-type matcher fix (garbage spans, add a real deflect/other bucket) — still needed
  so IF within-stale isn't a tautology; validate error-type vs human sample too.
- DP MC answer parsing (bare letters, free-text option content) — only needed for the MC
  secondary run; the free-form primary sidesteps it.
