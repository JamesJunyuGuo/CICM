# Figure Manifest

- `F1_behavioral_accuracy`: P0 accuracy vs interference load for Qwen and Llama. Sources: results/p0_qwen7b.summary.json, results/stage_e/llama/p0_ai_local_40g.summary.json.
- `F2_failure_signature`: Stacked error-type bars for stale vs other failures. Sources: results/stage_e/qwen_p0_errors.summary.json, results/stage_e/llama/p0_errors_ai_local_40g.summary.json, results/stage_e/natural/qwen_baseline_ai.summary.json, results/stage_e/natural/llama_baseline_ai_local_relaxed_40g.summary.json.
- `F3_mechanism_p_last`: p_last heatmap and mean p_last vs uniform reference. Sources: results/stage_a/extract_stable.npz, results/stage_a/extract_stable_index.jsonl, results/stage_e/llama/extract_stable_ai_local.npz, results/stage_e/llama/extract_stable_index_ai_local.jsonl.
- `F4_late_resolution_logitlens`: Logit-lens gold minus best-stale gap by layer. Sources: results/stage_a/a2_stable_summary.json, results/stage_e/llama/a2_stable_median_ai_summary.json.
- `F5_causal_dose_response`: Stage-B dose-response and answered-stale positive control. Sources: results/stage_b/summary.json.
- `F6_repair_transfer`: Repair accuracy across Stage-C, Stage-D, and naturalistic groups. Sources: results/stage_c/summary.json, results/stage_d/transfer_summary.json, results/stage_e/natural/qwen_baseline_ai.summary.json, results/stage_e/natural/llama_baseline_ai_local_relaxed_40g.summary.json.
- `F7_arm_l_random_transfer_gap`: Arm L vs Random-heads transfer gap with approximate two-proportion z. Sources: results/stage_c/summary.json, results/stage_d/transfer_summary.json, results/stage_e/natural/transfer_summary_ai.json, results/stage_e/second_seed/natural/transfer_summary.json.
- `F8_openrouter_scale_sweep`: OpenRouter scale sweep: accuracy vs interference load and within-stale share of errors. Sources: results/stage_e/openrouter/sweep_final.summary.json.
