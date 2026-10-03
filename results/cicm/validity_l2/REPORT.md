# Stage L L-2 Validity Report

## Scope

These runs test two validity concerns for the previous L-2 causal results: unnormalized alpha and final-layer answer-injection risk. All runs use `--alpha-mode residual_ratio`, so the effective steering norm is `alpha * ||hook-layer residual||`, with matched-norm random control and identity gate.

Important reading constraint: the original Qwen/Llama L-2 effects used unnormalized absolute alpha and final-block hooks, so their cross-model magnitudes should not be used as a headline causal comparison.

## Job Status

- Qwen H100 `19466272`: completed in `01:34:24`.
- Llama H100 `19466271`: completed in `01:52:23`.
- A100 mirrors `19466273` and `19466274`: cancelled before running after H100 jobs completed/ran healthily.
- Queue was empty after collection.

## Summary Table

| model | hook layer | layer type | gate | identity mismatches | best alpha | best target-error reduction vs random | best correct gain vs random | reading |
|---|---:|---|---:|---:|---:|---:|---:|---|
| qwen | 13 | middle | False | 0 | 0.4 | -1.1% [-2.6%, 0.5%] | -0.3% [-1.5%, 0.8%] | middle-layer null |
| qwen | 27 | final | True | 0 | 0.4 | 39.5% [35.8%, 43.3%] | 35.5% [31.9%, 39.1%] | final-layer positive only |
| llama | 15 | middle | True | 0 | 0.4 | 1.8% [0.6%, 3.2%] | 1.4% [0.1%, 2.5%] | tiny middle-layer effect |
| llama | 31 | final | True | 0 | 0.4 | 69.5% [66.3%, 72.9%] | 70.5% [67.2%, 73.8%] | large final-layer effect persists |

## Per-Model Read

### Qwen2.5-7B

- Middle layer 13 is a null: best alpha 0.4 gives target-error reduction -1.1% with CI [-2.6%, 0.5%]; gate `False`.
- Final layer 27 remains positive under residual normalization: best alpha 0.4 gives target-error reduction 39.5% with CI [35.8%, 43.3%]; correct gain 35.5% with CI [31.9%, 39.1%].
- At final layer best alpha, within-stale reduction is 36.7% [32.8%, 40.6%]; cross-slot reduction is 61.8% [50.0%, 72.4%].

### Llama-3.1-8B

- Middle layer 15 has only a tiny positive effect: best alpha 0.4 gives target-error reduction 1.8% with CI [0.6%, 3.2%]; correct gain 1.4% with CI [0.1%, 2.5%].
- Final layer 31 remains large under residual normalization: best alpha 0.4 gives target-error reduction 69.5% with CI [66.3%, 72.9%]; correct gain 70.5% with CI [67.2%, 73.8%].
- At final layer best alpha, within-stale reduction is 73.3% [69.7%, 76.7%]; cross-slot reduction is 42.0% [30.7%, 52.3%].

## Validity Verdict

- Residual normalization does not erase the final-layer effects. Qwen final-layer steering stays positive; Llama final-layer steering stays very large.
- The middle-layer test does not support a broad mechanism-causal claim: Qwen middle layer is null, and Llama middle layer is only tiny.
- Therefore the prior Llama +70% result should not be reported as a clean strong mechanism-causal result. The defensible reading is: the probe direction is a powerful late-layer steering / answer-selection lever, especially in Llama; but because the effect is concentrated at the final block, answer-injection or late logit steering remains a serious interpretation risk.
- Cross-model comparison is now fairer under residual-ratio alpha, and Llama still shows a much larger final-layer effect than Qwen. But this is a final-layer steering comparison, not evidence that the whole causal mechanism is equally established across layers.

## Artifacts

- qwen layer 13: `results/cicm/validity_l2/qwen_residual_norm_layers/layer_13/l2_summary.json`
- qwen layer 27: `results/cicm/validity_l2/qwen_residual_norm_layers/layer_27/l2_summary.json`
- llama layer 15: `results/cicm/validity_l2/llama_residual_norm_layers/layer_15/l2_summary.json`
- llama layer 31: `results/cicm/validity_l2/llama_residual_norm_layers/layer_31/l2_summary.json`
- Machine-readable aggregate: `results/cicm/validity_l2/summary.json`
