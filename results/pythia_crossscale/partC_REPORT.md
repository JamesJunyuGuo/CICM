# Stage O Part C Step 0

Substrate verdict: **substrate_selected**.
Hooked-noop gate: **PASS**.

| Model | k=0 recon accuracy | Best k | Projected correct | Projected stale | Gate |
|---|---:|---:|---:|---:|:---:|
| Qwen/Qwen2.5-0.5B | 0.972 | 4 | 270 | 378 | pass |
| Qwen/Qwen2.5-1.5B | 0.972 | 6 | 333 | 315 | pass |

Per-head quantities are QUERY-head quantities. Qwen2.5 has shared KV heads within GQA groups; output patching remains query-head specific.

No Part C full mechanism run was started.
