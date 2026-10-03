# Stage L Figure Captions

## F1

F1. Phenomenon and dose response. Left: controlled Paper-1 overwrite accuracy from results/overwrite_task/p0_qwen7b.summary.json, with Wilson intervals derived from summary n and accuracy. Right: Stage-L natural CICM accuracy and within-stale share among failures by overwrite count from results/cicm/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json.

## F2

F2. Multiple-choice masks stale-binding. Bars show within-stale fraction among failures for Dynamic Preference free-form, Dynamic Preference multiple-choice, and Instructional Forgetting free-form. Values and CIs are read from results/icf_bench/k1d_summary.json.

## F3

F3. Identity-vs-recency competition. Grouped bars show within-stale and cross-slot response rates per factorial cell, with Wilson intervals derived from per-cell counts in results/cicm/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json. Crossover labels mark cells where cross-slot rate meets or exceeds within-stale rate.

## F4

F4. Selection failure mechanism. Decodability bars use probe_true_current_score from results/cicm/natural_factorial_otherdist_l1/full/l1_probe_rows.jsonl with bootstrap intervals; the null band and attention delta are read from results/cicm/natural_factorial_otherdist_l1/full/l1_summary.json. The projection-independent quantitative result is that current values remain well above the value-label shuffle null on failure trials while stale attention is elevated.

## F5

F5. Illustrative projection of Qwen decision-position representations. The main panel projects saved final-layer activations onto two supervised discriminative axes fit from results/cicm/natural_factorial_otherdist_l1/full/l1_harvest.npz and l1_index.jsonl. Illustrative projection; all quantitative claims come from the cross-validated probe in F4, not from this plot.

## F5_appendix

F5 appendix. Naive PCA projection of the same saved Qwen decision-position representations from results/cicm/natural_factorial_otherdist_l1/full/l1_harvest.npz and l1_index.jsonl. This is an unsupervised visualization only.

## F6

F6. Qualified causal intervention result. Left: residual-normalized final-layer targeted-minus-random target-error reduction by alpha. Right: best-alpha layer localization shows large final-layer effects and null/tiny middle-layer effects. All values and CIs are read from results/cicm/validity_l2/summary.json; legacy absolute-alpha final-layer sweeps are stored in results/cicm/natural_factorial_otherdist_l2/full_alpha_sweep/l2_summary.json and the Llama equivalent, but the plotted comparison uses the normalized validity rerun.

## F7

F7. Cross-model L-1 replication. Bars show true-current probe score on failure pools for Qwen2.5-7B and Llama-3.1-8B, with bootstrap intervals from each model's l1_probe_rows.jsonl and shuffle-null bands from the corresponding l1_summary.json files.

## Source Files

- geometry: `results/cicm/geometry/summary.json`
- llama_l0: `results/cicm/natural_factorial_otherdist_llama/l0/openrouter_llama31_8b_summary.json`
- llama_l1: `results/cicm/natural_factorial_otherdist_llama/l1/full/l1_summary.json`
- llama_l1_rows: `results/cicm/natural_factorial_otherdist_llama/l1/full/l1_probe_rows.jsonl`
- llama_l2: `results/cicm/natural_factorial_otherdist_llama/l2/full_alpha_sweep/l2_summary.json`
- p0: `results/overwrite_task/p0_qwen7b.summary.json`
- qwen_l1: `results/cicm/natural_factorial_otherdist_l1/full/l1_summary.json`
- qwen_l1_index: `results/cicm/natural_factorial_otherdist_l1/full/l1_index.jsonl`
- qwen_l1_npz: `results/cicm/natural_factorial_otherdist_l1/full/l1_harvest.npz`
- qwen_l1_rows: `results/cicm/natural_factorial_otherdist_l1/full/l1_probe_rows.jsonl`
- qwen_l2: `results/cicm/natural_factorial_otherdist_l2/full_alpha_sweep/l2_summary.json`
- stage_k_k1d: `results/icf_bench/k1d_summary.json`
- stage_l_l0: `results/cicm/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json`
- validity_l2: `results/cicm/validity_l2/summary.json`