# Stage L Geometry Addendum

Pre-registered readings:

- G-1: geometric-argmax predicts the selected value above BOTH the shuffle null AND the stronger of the two single-factor baselines -> selection is a genuine geometric competition. Otherwise -> the simple alignment geometry does not capture selection.
- G-2: geometry predicts the held-out per-cell crossover -> selection is a geometric identity-vs-recency competition. Prediction fails -> the crossover is not explained by this alignment geometry; the mechanism stays at decodable-but-not-selected.

Anti-circularity: current-value alignment is a sanity check only. Headline claims below use only candidate competition and held-out-cell crossover prediction.

## qwen

- Sanity: layer 28, CV accuracy 0.9000, within-stale true-current score 0.7593, shuffle95 [0.033098669618678286, 0.06294803582081768].
- G-1: geometric argmax accuracy 0.4340; shuffle95 [0.12588028169014084, 0.16637323943661972]; recency-only 0.3451; identity-only 0.7923; gate pass `False`.
- G-2 raw: held-out-cell cross-share MAE 0.2417; crossover sign accuracy 0.6667.
- G-2 length-controlled: held-out-cell cross-share MAE 0.2417; crossover sign accuracy 0.6667.

| held-out cell | predicted cross share | observed cross share | abs error | predicted crossover | observed crossover/tie |
|---|---:|---:|---:|---:|---:|
| same_far__other_far | 0.0000 | 0.3333 | 0.3333 | False | False |
| same_far__other_mid | 0.0000 | 0.5897 | 0.5897 | False | True |
| same_far__other_near2 | 0.0000 | 0.5077 | 0.5077 | False | True |
| same_near__other_far | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_mid | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_near2 | 0.0000 | 0.0195 | 0.0195 | False | False |

G-3 layer scan top rows by G-1 accuracy:

| layer | G-1 acc | recency-only | identity-only | CV acc | within score | gate |
|---:|---:|---:|---:|---:|---:|---:|

## llama

- Sanity: layer 32, CV accuracy 0.8792, within-stale true-current score 0.7461, shuffle95 [0.03365627863521121, 0.06177385641392391].
- G-1: geometric argmax accuracy 0.4017; shuffle95 [0.12457627118644068, 0.1669491525423729]; recency-only 0.2424; identity-only 0.6737; gate pass `False`.
- G-2 raw: held-out-cell cross-share MAE 0.1738; crossover sign accuracy 1.0000.
- G-2 length-controlled: held-out-cell cross-share MAE 0.1738; crossover sign accuracy 1.0000.

| held-out cell | predicted cross share | observed cross share | abs error | predicted crossover | observed crossover/tie |
|---|---:|---:|---:|---:|---:|
| same_far__other_far | 0.0000 | 0.4545 | 0.4545 | False | False |
| same_far__other_mid | 0.0000 | 0.2959 | 0.2959 | False | False |
| same_far__other_near2 | 0.0000 | 0.1190 | 0.1190 | False | False |
| same_near__other_far | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_mid | 0.0000 | 0.1679 | 0.1679 | False | False |
| same_near__other_near2 | 0.0000 | 0.0056 | 0.0056 | False | False |

G-3 layer scan top rows by G-1 accuracy:

| layer | G-1 acc | recency-only | identity-only | CV acc | within score | gate |
|---:|---:|---:|---:|---:|---:|---:|

## Verdict

G-1 and G-2 are the only headline tests. A model passes G-1 only if geometric argmax beats both shuffle and the stronger single-factor baseline. A model passes G-2 only if the held-out identity/recency decomposition predicts the per-cell crossover pattern; otherwise the Stage-L mechanism remains selection/decodable-but-not-selected without this stronger geometric account.
