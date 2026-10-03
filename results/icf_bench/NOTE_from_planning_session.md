# Note to the Stage-K session (from the planning/review session) — 2026-07-20

I reviewed the ICF-Bench data and your `src/icf_*` matchers against the Stage-K spec.
Verdict: on track — deterministic matchers, program-verifiable subset with logged drops,
LLM-judge labels correctly excluded from the headline. Two things worth acting on, plus a
minor flag. (I did not edit your files, to avoid collisions.)

1. **Use the shipped human-alignment set to validate the matchers — don't re-annotate.**
   `external_data/icf_bench/human_llm_align/` has 50 items × 3 annotators per scenario.
   That IS the "≥50 hand-checked items/scenario" the spec asks for. Compute
   matcher-vs-human agreement against these and report it; drop any scenario whose
   agreement is too low from the headline.

2. **Confirm the old/new option join for Dynamic Preference.**
   In the source data, the STALE option = `source_option` (in `dynamic_preference_old.json`)
   and the CORRECT/current option = `aligned_op` (in `dynamic_preference_new.json`), paired
   via `from_implicit_id`. "old" has only 783 items vs 1000 "new", so `within_stale` is only
   defined on the paired subset. Verify `icf_generate.py` sources `old_op`←`source_option`
   and `new_op`←`aligned_op` (not swapped), and drops unpaired / non-4-option items with a
   logged reason (no silent truncation).

Minor flag: the `k1_*_release_gpt4o` artifacts use ICF-Bench's own precomputed answers —
fine as a matcher smoke test, but the K1 headline must come from OUR model runs
(Qwen2.5-7B primary; the `k1_qwen_*.slurm` scripts look right for that). Don't let the
release-gpt4o pass become the reported result.

Everything else about K1 looks correct. Proceed.

## Addendum — local ICF-Bench clone and external stale-binding test

The ICF-Bench repository has been cloned locally at:

- `external_data/icf_bench/`

Use this local clone as the Stage-K data source. The point of Stage K is not only to
re-score our previous synthetic stale-binding dataset. It is to test whether the same
stale-binding phenomenon appears on an independent, real-deployment-style benchmark:
ICF-Bench's Dynamic Preference and Instructional Forgetting settings provide
program-verifiable old/current or forget/refuse contrasts.

Concretely:

1. Treat Paper-1 synthetic stale-binding as the controlled internal/causal substrate.
2. Treat ICF-Bench as the external ecological validation target.
3. For K1, headline the within-stale return rate on ICF-Bench Forget-form failures,
   using deterministic matchers on the program-verifiable subset, not their LLM judge.
4. If K1 passes on our self-run Qwen2.5-7B results, K2/K3 should ask whether the same
   attention/retention mechanisms measured in Paper 1 also appear in real dialogue.

This is the intended contribution: stale binding should be tested directly on ICF-Bench,
not only on the synthetic overwrite data we crafted previously.
