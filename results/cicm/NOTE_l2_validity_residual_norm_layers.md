# Stage L L-2 Validity Follow-up: Residual-Norm Alpha and Middle-Layer Hooks

## Scope

Current Qwen and Llama L-2 runs used unnormalized absolute alpha on the final
transformer block output. Therefore:

- Cross-model effect sizes are not directly comparable, because the same alpha
  may correspond to different fractions of the residual stream norm in Qwen vs
  Llama.
- Final-block hooks have an answer-injection risk: if only the last block works,
  the result should be downgraded to late logit/answer steering rather than a
  clean mechanism-causal intervention.

## Validity Runs

Run both Qwen and Llama with:

- `--alpha-mode residual_ratio`
- `--alpha-list 0.01,0.02,0.05,0.1,0.2,0.4`
- matched-norm random control
- identity gate
- full failure pools

Layer checks:

- Qwen: middle layer `13`, final block `27`
- Llama: middle layer `15`, final block `31`

Interpretation:

- If residual-normalized effects remain and middle layers work, the causal
  interpretation is strengthened.
- If only the final block works, report the result as answer-injection / late
  steering, not mechanism causality.
- If residual-normalized effects disappear, report the original alpha result as
  scale-confounded and do not claim cross-model causal strength.
