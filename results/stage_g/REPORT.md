# Stage G Report

## Scope
Stage G ran the circuit-level stale-binding checks and external benchmark validation requested in `docs/stage_g_spec.md`, including Addendum 1 (2026-07-11).

## Addendum 1 Status
- v1 external summaries are void for paper use because they used scoring/signature protocols superseded by Addendum 1.
- Deprecated v1 summaries moved under `results/stage_g/external/full_ai/deprecated_v1/` (found 5 files). Raw `results.jsonl` rows were not overwritten.
- Raw-output audit hard gate: 16 audit files under `results/stage_g/external/full_ai/audit/`.
- Audit outcome: BABILong qa1-0k baseline-vs-adapter gap was a parse-protocol artifact; lenient containment raises Qwen baseline substantially. Llama boxes audit showed generation-level prompt/template failure, so entity-tracking was rerun with a direct boxes prompt.

## Pre-registered Readings
- Path patching: `if (4-stale) or (5) recovers most of Δ → the circuit lesion is in QK matching (identity-based lookup that cannot separate current from stale). If only (1) at late layers recovers → resolution is more distributed.`
- R-1 causal patching: `if (4-stale) or (5) recovers a large share → QK-matching lesion confirmed causally; if not, the story stays "distributed late resolution + QK geometry as the repair's mechanism," stated exactly so.`
- QK hypothesis: `weak/near-chance, or present in keys but ignored — see 3.`
- Circuit story: `QK implements identity lookup, not latest-binding lookup`
- External: `Baseline shows stale-like errors on external tasks → the phenomenon is not an artifact of our generators (kills "self-built data" objection).`
- External: `Arm L improves (or at minimum does not hurt) external tasks, with the Arm L vs Random gap tracked → transfer evidence on community data.`
- External: `If a benchmark shows NO stale-like failure or no repair transfer, report it as a boundary — do not tune prompts to force an effect. Any post-hoc prompt adjustment must be labeled as such.`

## G-1 Circuit
- Artifact: `results/stage_g/circuit/full_r1_ai/summary.json`
- Pair yield: 120 / 301 candidates (yield=0.399, target_met=True); first-token collisions dropped=3.
- Self-patching identity hard gate: True.
- Path-patching component effects:
  - residual_stream_early_mid: mean Δ recovery=-0.005, median=0.000, n=600
  - residual_stream_late: mean Δ recovery=0.781, median=0.939, n=360
  - selection_head_output_proxy: mean Δ recovery=0.360, median=0.348, n=120
  - qk_query_final: mean Δ recovery=0.112, median=0.109, n=120
  - qk_key_all_writes: mean Δ recovery=0.111, median=0.107, n=120
  - qk_key_current_write: mean Δ recovery=-0.213, median=-0.195, n=120
  - qk_key_stale_writes: mean Δ recovery=0.492, median=0.448, n=120
  - late_mlp_output: mean Δ recovery=0.132, median=0.105, n=960
  - selection_head_attention_pattern_proxy: mean Δ recovery=0.130, median=0.131, n=120
- QK geometry baseline: mean current-stale margin=7.835, query identity align=0.234, query recency align=0.037.
- QK geometry Arm L: mean current-stale margin=51.139, query identity align=0.259, query recency align=0.168.
- OV sanity: mean attended-value token logit=0.495, gold token=0.098, answered-stale token=0.222.
- Limitations: Attention-pattern patching is reported as an allocation proxy; residual stream patching uses actual activation patch hooks. Q/K geometry is measured on the first token of each value span.

## G-2 External Benchmarks
- Protocols:
  - RULER-VT: per-example per-variable recall (fraction of gold chain variables present in output, case-insensitive) is the headline; strict set equality is secondary. Error labels: omission, cross-chain inclusion, other. No stale/superseded language.
  - BABILong: normalized gold string contained in normalized response (`target_in_response`). For qa2/qa3 only, superseded iff the prediction equals a strictly earlier state of the queried entity.
  - Entity-tracking boxes: predicted contents are parsed as a normalized set; order-insensitive set equality is accuracy. Superseded iff the predicted set equals the queried box contents at an earlier timestep.
- Found 6 v2 external summary artifacts.
- `results/stage_g/external/full_ai/v2/arm_l/all/summary.json` arm=arm_l model=Qwen/Qwen2.5-7B-Instruct
  - babilong qa1 0k: n=200, headline=0.975, strict_acc=0.975, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa1 4k: n=200, headline=0.820, strict_acc=0.820, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa2 0k: n=200, headline=0.485, strict_acc=0.485, superseded-error-share=0.068, other-error-share=0.932
  - babilong qa2 4k: n=200, headline=0.175, strict_acc=0.175, superseded-error-share=0.012, other-error-share=0.988
  - babilong qa3 0k: n=200, headline=0.255, strict_acc=0.255, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 4k: n=200, headline=0.165, strict_acc=0.165, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 0: n=144, headline=0.951, strict_acc=0.951, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.660, strict_acc=0.660, superseded-error-share=0.531, other-error-share=0.469
  - entity_tracking boxes 2: n=123, headline=0.341, strict_acc=0.341, superseded-error-share=0.321, other-error-share=0.679
  - entity_tracking boxes 3: n=87, headline=0.230, strict_acc=0.230, superseded-error-share=0.254, other-error-share=0.746
  - entity_tracking boxes 4: n=31, headline=0.129, strict_acc=0.129, superseded-error-share=0.333, other-error-share=0.667
  - entity_tracking boxes 5: n=14, headline=0.286, strict_acc=0.286, superseded-error-share=0.300, other-error-share=0.700
  - entity_tracking boxes 6: n=9, headline=0.111, strict_acc=0.111, superseded-error-share=0.250, other-error-share=0.750
  - entity_tracking boxes 7: n=4, headline=0.250, strict_acc=0.250, superseded-error-share=0.000, other-error-share=1.000
  - ruler_vt variable_tracking synthetic: n=400, headline=0.360, strict_acc=0.000, cross-chain-error-share=1.000, other-error-share=0.000
- `results/stage_g/external/full_ai/v2/generic/all/summary.json` arm=generic model=Qwen/Qwen2.5-7B-Instruct
  - babilong qa1 0k: n=200, headline=0.955, strict_acc=0.955, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa1 4k: n=200, headline=0.865, strict_acc=0.865, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa2 0k: n=200, headline=0.485, strict_acc=0.485, superseded-error-share=0.019, other-error-share=0.981
  - babilong qa2 4k: n=200, headline=0.125, strict_acc=0.125, superseded-error-share=0.011, other-error-share=0.989
  - babilong qa3 0k: n=200, headline=0.190, strict_acc=0.190, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 4k: n=200, headline=0.175, strict_acc=0.175, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 0: n=144, headline=0.986, strict_acc=0.986, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.771, strict_acc=0.771, superseded-error-share=0.488, other-error-share=0.512
  - entity_tracking boxes 2: n=123, headline=0.341, strict_acc=0.341, superseded-error-share=0.321, other-error-share=0.679
  - entity_tracking boxes 3: n=87, headline=0.299, strict_acc=0.299, superseded-error-share=0.295, other-error-share=0.705
  - entity_tracking boxes 4: n=31, headline=0.258, strict_acc=0.258, superseded-error-share=0.304, other-error-share=0.696
  - entity_tracking boxes 5: n=14, headline=0.286, strict_acc=0.286, superseded-error-share=0.100, other-error-share=0.900
  - entity_tracking boxes 6: n=9, headline=0.000, strict_acc=0.000, superseded-error-share=0.333, other-error-share=0.667
  - entity_tracking boxes 7: n=4, headline=0.250, strict_acc=0.250, superseded-error-share=0.000, other-error-share=1.000
  - ruler_vt variable_tracking synthetic: n=400, headline=0.343, strict_acc=0.000, cross-chain-error-share=1.000, other-error-share=0.000
- `results/stage_g/external/full_ai/v2/llama_baseline/all/summary.json` arm=llama_baseline model=data/stage_g/models/llama31_8b_merged
  - babilong qa1 0k: n=200, headline=0.930, strict_acc=0.930, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa1 4k: n=200, headline=0.680, strict_acc=0.680, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa2 0k: n=200, headline=0.385, strict_acc=0.385, superseded-error-share=0.146, other-error-share=0.854
  - babilong qa2 4k: n=200, headline=0.160, strict_acc=0.160, superseded-error-share=0.018, other-error-share=0.982
  - babilong qa3 0k: n=200, headline=0.250, strict_acc=0.250, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 4k: n=200, headline=0.265, strict_acc=0.265, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 0: n=144, headline=0.097, strict_acc=0.097, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.122, strict_acc=0.122, superseded-error-share=0.097, other-error-share=0.903
  - entity_tracking boxes 2: n=123, headline=0.089, strict_acc=0.089, superseded-error-share=0.134, other-error-share=0.866
  - entity_tracking boxes 3: n=87, headline=0.069, strict_acc=0.069, superseded-error-share=0.111, other-error-share=0.889
  - entity_tracking boxes 4: n=31, headline=0.000, strict_acc=0.000, superseded-error-share=0.129, other-error-share=0.871
  - entity_tracking boxes 5: n=14, headline=0.071, strict_acc=0.071, superseded-error-share=0.077, other-error-share=0.923
  - entity_tracking boxes 6: n=9, headline=0.000, strict_acc=0.000, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 7: n=4, headline=0.000, strict_acc=0.000, superseded-error-share=0.000, other-error-share=1.000
  - ruler_vt variable_tracking synthetic: n=400, headline=0.092, strict_acc=0.000, cross-chain-error-share=0.537, other-error-share=0.000
- `results/stage_g/external/full_ai/v2/llama_baseline_boxes_rerun/entity_tracking/summary.json` arm=llama_baseline_boxes_rerun model=data/stage_g/models/llama31_8b_merged
  - entity_tracking boxes 0: n=144, headline=0.611, strict_acc=0.611, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.436, strict_acc=0.436, superseded-error-share=0.415, other-error-share=0.585
  - entity_tracking boxes 2: n=123, headline=0.171, strict_acc=0.171, superseded-error-share=0.402, other-error-share=0.598
  - entity_tracking boxes 3: n=87, headline=0.310, strict_acc=0.310, superseded-error-share=0.300, other-error-share=0.700
  - entity_tracking boxes 4: n=31, headline=0.194, strict_acc=0.194, superseded-error-share=0.240, other-error-share=0.760
  - entity_tracking boxes 5: n=14, headline=0.214, strict_acc=0.214, superseded-error-share=0.273, other-error-share=0.727
  - entity_tracking boxes 6: n=9, headline=0.111, strict_acc=0.111, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 7: n=4, headline=0.000, strict_acc=0.000, superseded-error-share=0.000, other-error-share=1.000
- `results/stage_g/external/full_ai/v2/qwen_baseline/all/summary.json` arm=qwen_baseline model=Qwen/Qwen2.5-7B-Instruct
  - babilong qa1 0k: n=200, headline=0.940, strict_acc=0.940, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa1 4k: n=200, headline=0.800, strict_acc=0.800, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa2 0k: n=200, headline=0.525, strict_acc=0.525, superseded-error-share=0.074, other-error-share=0.926
  - babilong qa2 4k: n=200, headline=0.150, strict_acc=0.150, superseded-error-share=0.012, other-error-share=0.988
  - babilong qa3 0k: n=200, headline=0.205, strict_acc=0.205, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 4k: n=200, headline=0.150, strict_acc=0.150, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 0: n=144, headline=0.847, strict_acc=0.847, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.532, strict_acc=0.532, superseded-error-share=0.398, other-error-share=0.602
  - entity_tracking boxes 2: n=123, headline=0.293, strict_acc=0.293, superseded-error-share=0.253, other-error-share=0.747
  - entity_tracking boxes 3: n=87, headline=0.195, strict_acc=0.195, superseded-error-share=0.214, other-error-share=0.786
  - entity_tracking boxes 4: n=31, headline=0.097, strict_acc=0.097, superseded-error-share=0.214, other-error-share=0.786
  - entity_tracking boxes 5: n=14, headline=0.357, strict_acc=0.357, superseded-error-share=0.111, other-error-share=0.889
  - entity_tracking boxes 6: n=9, headline=0.000, strict_acc=0.000, superseded-error-share=0.333, other-error-share=0.667
  - entity_tracking boxes 7: n=4, headline=0.250, strict_acc=0.250, superseded-error-share=0.000, other-error-share=1.000
  - ruler_vt variable_tracking synthetic: n=400, headline=0.367, strict_acc=0.000, cross-chain-error-share=0.995, other-error-share=0.000
- `results/stage_g/external/full_ai/v2/random_heads/all/summary.json` arm=random_heads model=Qwen/Qwen2.5-7B-Instruct
  - babilong qa1 0k: n=200, headline=0.905, strict_acc=0.905, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa1 4k: n=200, headline=0.800, strict_acc=0.800, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa2 0k: n=200, headline=0.475, strict_acc=0.475, superseded-error-share=0.114, other-error-share=0.886
  - babilong qa2 4k: n=200, headline=0.145, strict_acc=0.145, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 0k: n=200, headline=0.270, strict_acc=0.270, superseded-error-share=0.000, other-error-share=1.000
  - babilong qa3 4k: n=200, headline=0.200, strict_acc=0.200, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 0: n=144, headline=0.972, strict_acc=0.972, superseded-error-share=0.000, other-error-share=1.000
  - entity_tracking boxes 1: n=188, headline=0.617, strict_acc=0.617, superseded-error-share=0.500, other-error-share=0.500
  - entity_tracking boxes 2: n=123, headline=0.325, strict_acc=0.325, superseded-error-share=0.325, other-error-share=0.675
  - entity_tracking boxes 3: n=87, headline=0.253, strict_acc=0.253, superseded-error-share=0.215, other-error-share=0.785
  - entity_tracking boxes 4: n=31, headline=0.194, strict_acc=0.194, superseded-error-share=0.240, other-error-share=0.760
  - entity_tracking boxes 5: n=14, headline=0.214, strict_acc=0.214, superseded-error-share=0.182, other-error-share=0.818
  - entity_tracking boxes 6: n=9, headline=0.000, strict_acc=0.000, superseded-error-share=0.111, other-error-share=0.889
  - entity_tracking boxes 7: n=4, headline=0.250, strict_acc=0.250, superseded-error-share=0.000, other-error-share=1.000
  - ruler_vt variable_tracking synthetic: n=400, headline=0.331, strict_acc=0.000, cross-chain-error-share=1.000, other-error-share=0.000
- Figure F13 written to `results/stage_g/figures/F13_external_benchmarks.{png,pdf}`.

## Artifacts
- F11/F12: `results/stage_g/**/figures/` or `results/stage_g/figures/` depending on winning portal.
- F13: `results/stage_g/figures/F13_external_benchmarks.{png,pdf}` when external summaries are present.
