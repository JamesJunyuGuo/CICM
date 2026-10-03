# Hard-tier frontier break-point probe (Stage F, task 3)

Run 2026-07-10 from the login node via OpenRouter (`src/hard_tier_sweep.py`,
seed 40, 50/cell, greedy, retries=5). 600/600 calls succeeded (0 call errors).
Providers recorded per row in `sweep.jsonl`.

## Question

Models that saturate the standard battery (I ≤ 16) — do they break at extreme
interference, and is the failure signature still within-stale?

## Results (accuracy; stale share of errors in parentheses)

| model | simple n=320 | I=32 n=160 | I=32 n=320 | I=64 n=320 |
|---|---:|---:|---:|---:|
| qwen/qwen-2.5-72b-instruct | 1.00 | 0.98 (100%) | 0.96 (100%) | **0.78 (100%)** |
| openai/gpt-4o | 1.00 | 1.00 (—) | 0.96 (100%) | **0.86 (100%)** |
| meta-llama/llama-3.1-70b-instruct | 1.00 | 0.56 (73%) | 0.72 (93%) | **0.40 (53%)** |

## Reading

1. **Every tested model has a stale-binding break point, including GPT-4o.**
   Combined with the standard sweep (F8): 7B-class breaks at I≈4, Llama-70B at
   I≈16, Qwen-72B and GPT-4o at I≈64. Scale shifts the threshold; it does not
   remove the failure mode.
2. **Signature invariance at the break**: Qwen-72B and GPT-4o errors at I=64
   are 100% within-stale — the same signature as 7B models at I=4.
3. **Length is exonerated even at 320 lines**: simple control = 1.00 for all
   three models; the failures are interference-specific.
4. Observations for the discussion section:
   - Llama-70B at extreme load mixes in non-stale errors (53% stale at I=64):
     at overload, the failure profile broadens beyond pure stale-binding.
   - Llama-70B is *worse* at n=160/I=32 than n=320/I=32 (0.56 vs 0.72):
     write **density** appears more damaging than absolute count/length,
     consistent with a transport-dilution account; single observation, not
     confirmed.

Artifacts: `sweep.jsonl` (per-row, incl. provider), `sweep.summary.json`,
`data/hard_tier.jsonl` (regenerable, seed 40). Figure: extend F8 with a
hard-tier panel (Codex task).
