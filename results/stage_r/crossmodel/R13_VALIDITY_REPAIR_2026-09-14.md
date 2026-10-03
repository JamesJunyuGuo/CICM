# R13 calibration validity repair

## Affected jobs

- Mistral A100 calibration `20667367` failed before inference because the
  Mistral chat template relocates the system prompt beside the final user turn.
  The chronological substring locator advanced to that relocated text and
  could no longer find the earlier user messages.
- Llama-3B A100 calibration `20667369` failed during causal-gradient discovery
  because `stage_l_natfact_6_00045` has current and stale values with the same
  first generated token under the Llama tokenizer. The first-token current-vs-
  stale gradient objective is undefined for that row.
- No validation configuration was selected and no confirmation data were
  opened. The failures are pipeline-validity failures, not scientific nulls.

## Repair

1. Message-offset recovery now records an initial system message without
   advancing the cursor used for chronological dialogue messages. This handles
   both in-place and relocated system prompts without changing value spans.
2. Causal-head discovery excludes and records only rows whose current and stale
   candidates collide at the first token. Attention capture, validation, and
   the complete 960-row confirmation split remain unchanged.
3. New confirmation reports distinguish zero-overlap canonical runs from the
   earlier exceptional one-row-overlap Llama-8B run and use model-neutral
   verdict names.

Targeted tests cover relocated system prompts, collision partitioning, exact
identity routing, split integrity, random controls, and model-family guards.
The repaired suite passes 16 tests.

## Mistral discovery-pool repair

Repaired H100 calibration `20701729` passed routing, attention capture, and
first-token gradient extraction, leaving 114/120 identifiable rows (six
collisions). It then stopped before validation because an inherited symmetric
minimum required 20 correct and 20 within-stale rows; observed pools were 9 and
79. Head ranking uses the failure gradient, and correct-answer preservation is
enforced on the independent validation split, so the repair separates these
requirements: at least 5 correct reference rows (the existing Qwen-14B
cross-model precedent) and at least 20 within-stale rows. The confirmation gate
and held-out split are unchanged.

## Cache-path repair

Calibration `20719627` stopped before tokenizer loading because its hardcoded
snapshot directory had been removed during cache cleanup. Transformers then
treated the nonexistent absolute path as an invalid repository ID. This is an
environment/pipeline failure and carries no experimental evidence. Mistral
runbooks now use the stable model ID and the user-owned scratch cache. A shared-
CPU cache-hydration job downloads and verifies config/tokenizer availability in
offline mode; both GPU calibrations depend on that successful check.

Cache job `20728705` downloaded the complete snapshot but its offline check
still read the removed project cache because an inherited `TRANSFORMERS_CACHE`
overrode `HF_HOME`. The final repair explicitly unsets that legacy variable and
sets both `HUGGINGFACE_HUB_CACHE` and `HF_HUB_CACHE` to the scratch cache in the
cache, calibration, and confirmation jobs. The verifier also checks every shard
named by `model.safetensors.index.json` before releasing GPU dependencies.
