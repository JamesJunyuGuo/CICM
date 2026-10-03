# Stage N-S Qwen-7B execution override

Date: 2026-08-10

The PI explicitly replaced the planned Qwen2.5-3B headline run with
`Qwen/Qwen2.5-7B-Instruct`. The 1.5B run remains a pipeline diagnostic only;
its behavior rates are not findings.

All other frozen rules in `docs/stage_n_attention_sink_spec.md` remain in
force: the task family, 72 rows per seed-template-k cell, fixed pool gate,
stable discovery/held-out split, structural sink definition, thresholds,
controls, and reporting categories are unchanged. Outputs are isolated under
`results/stage_n/attention_sink/qwen7_*`.

Before any 7B full behavior or mechanism run, a four-pair 7B GPU smoke must
pass hooked-noop, exact identity, alignment, finite-intervention, and
attention-value reconstruction checks. The identity and full-output anchors
use the actual captured pre-`o_proj` tensors; reconstructed attention-value
products are retained only for the gate/routing/value decomposition.
