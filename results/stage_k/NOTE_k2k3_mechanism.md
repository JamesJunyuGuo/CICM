# K2/K3 — the mechanism phase on ICF-Bench (LOCAL GPU; API cannot do this)

## Prerequisite / substrate
The behavioral phase (K1d) is done: stale-binding is format-robust and cross-scenario
(IF ~0.98; DP free-form ~0.7–0.8; MC masks it at 0.29). Now we add the LLM-level mechanism —
the contribution ICF-Bench cannot reach.

**This phase needs LOCAL weights on GPU (attention + hidden states); OpenRouter/API cannot
return internal states.** Re-run on LOCAL `Qwen2.5-7B-Instruct` via SLURM (fp32 eager for
attention; paired portals per AGENTS.md). Re-generate DP-free-form + IF locally WITH harvest,
then RE-CLASSIFY the local outputs with the validated hybrid judge (local generation ≠ API
generation), and condition all analysis on the LOCAL labels.

## Discipline (carry over)
Condition on FAILURE MODE, not generic correct/incorrect: analyze **within-stale failures vs
correct trials** (and, for IF, the deflect bucket separately). Length-residualize every
attention/representation quantity (ICF-Bench conflates interference with inserted length).
Reuse frozen Stage-A (probe) and Stage-B (attention) machinery; do not re-derive.

## Two scenarios probe DIFFERENT mechanisms — design accordingly

### K3 (HEADLINE) — retention vs selection, on DP (the clean Paper-1 analog)
DP has a clear current-correct value (NEW preference) and a stale value (OLD preference) —
the direct analog of the synthetic overwrite task. At the answer-decision position:
- Reuse frozen Stage-A probe methodology; target = the NEW (current-correct) preference /
  its aligned choice. Train on correct/held-out trials; test decodability on **within-stale
  failure** trials.
- **Pre-registered readings:**
  - NEW preference still linearly decodable on within-stale trials (present but not selected)
    → **SELECTION failure replicates Paper 1 on real dialogue** (headline: the current
    binding is retained but not selected).
  - NEW preference NOT decodable (lost) → RETENTION failure; differs from synthetic; report
    honestly.

### K2 (SUPPORT) — attention signatures, both scenarios
- **DP**: attention mass on OLD-preference tokens vs NEW-preference tokens at the decision
  position, within-stale vs correct. Prediction (self-reliance/stale analog): elevated
  attention to OLD on within-stale failures.
- **IF (suppression angle)**: the correct behavior is to PRODUCE ABSENCE, so probe the
  forget-DIRECTIVE, not a "current value": (a) attention on the to-forget value tokens vs the
  forget-instruction tokens; (b) is the forget-instruction under-attended on failures? and
  (c) is the forget-instruction linearly decodable at the decision point (represented but not
  acted on)?
- **Pre-registered readings:** elevated stale/forbidden attention (paired CI excludes 0,
  length-residualized) → the stale-attention signature transfers to real dialogue. Null →
  report; the behavioral result stands without the attention account.

## Deliverables & guardrails
- Local Qwen2.5-7B; fp32 eager for attention; SLURM (paired portals if duplicate-safe);
  0.5B smoke first.
- Re-classify local generations with the validated hybrid judge; report judge agreement on a
  hand-labeled sample of the LOCAL long-form responses (close the calibration-distribution
  gap flagged in K1d — validate ON the distribution being analyzed).
- Condition on within-stale vs correct; length-residualize; paired bootstrap + shuffle nulls
  on every headline; frozen Stage-A/B machinery.
- Write under `results/stage_k/mech/`; new `src/icf_*` files only; do not touch Stage-J or
  the frozen K1d artifacts. Both outcomes at each gate publishable; quote readings verbatim.
- After K2/K3: K5 (clean-toggle intervention) remains the capstone, gated on K3.
