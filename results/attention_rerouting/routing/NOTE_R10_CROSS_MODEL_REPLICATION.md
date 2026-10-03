# R10 cross-model replication recommendation

## Decision

A second model is not required for the narrow claim that balanced attention
routing corrects direct retrieval on Qwen2.5-7B-Instruct. It is required before
claiming that the correction is architecture-general or broadly deployable.
The smallest decisive extension is Llama-3.1-8B-Instruct on the same 1,200 CICM
rows.

## Frozen comparison

- Keep the dataset, 20/80 calibration-confirmation split, automatic route
  audit, beta grid, response taxonomy, and R10 gates unchanged.
- Revalidate every current/stale token span with the Llama tokenizer.
- Do not copy Qwen head indices. Select a fixed-size Llama head set using only
  calibration rows and the same current-versus-stale attention criterion, then
  freeze it before confirmation.
- Select beta on retrieval calibration only. Transfer the selected beta to
  retrieval and derived decision confirmation without retuning.
- Retain identity, opposite-sign, 16 matched random-position controls,
  per-cell reporting, correct-answer preservation, and semantic-ID clustered
  bootstrap intervals.
- Write only to a new `results/attention_rerouting/routing/balanced_llama31_8b/` tree.

## Interpretation

- Llama retrieval passes: claim cross-family replication of the correction
  shape, not correspondence of individual heads.
- Llama fails with a valid intervention: retain the Qwen-scoped method and
  report architecture dependence.
- Token-span, identity, or hook no-op failure: pipeline failure, not a model
  result.

One cross-family replication is higher value than adding another Qwen size.
Only after Llama should a third model be considered.
