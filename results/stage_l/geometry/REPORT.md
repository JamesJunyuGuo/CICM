# Stage L Geometry Addendum

Pre-registered readings:

- G-1: geometric-argmax predicts the selected value above BOTH the shuffle null AND the stronger of the two single-factor baselines -> selection is a genuine geometric competition. Otherwise -> the simple alignment geometry does not capture selection.
- G-2: geometry predicts the held-out per-cell crossover -> selection is a geometric identity-vs-recency competition. Prediction fails -> the crossover is not explained by this alignment geometry; the mechanism stays at decodable-but-not-selected.

Anti-circularity: current-value alignment is a sanity check only. Headline claims below use only candidate competition and held-out-cell crossover prediction.

## qwen

- Sanity: layer 28, stored L-1 probe-row within-stale true-current score 0.8485; geometry refit CV accuracy 0.9158, refit within-stale true-current score 0.8273, shuffle95 [0.032211161931875516, 0.06401655098568763].
- G-1: geometric argmax accuracy 0.4305; shuffle95 [0.12588028169014084, 0.16637323943661972]; recency-only 0.3451; identity-only 0.7923; gate pass `False`. Reading: above shuffle, but below the stronger identity-only baseline, so simple geometric argmax does not pass the competition gate.
- G-2 raw: held-out-cell cross-share MAE 0.2417; crossover sign accuracy 0.6667.
- G-2 length-controlled: held-out-cell cross-share MAE 0.2417; crossover sign accuracy 0.6667. Reading: Prediction fails: the length-controlled identity/recency decomposition predicts stale-only competition in every held-out cell while observed cross-slot failures remain nonzero.

| held-out cell | predicted cross share | observed cross share | abs error | predicted crossover | observed crossover/tie |
|---|---:|---:|---:|---:|---:|
| same_far__other_far | 0.0000 | 0.3333 | 0.3333 | False | False |
| same_far__other_mid | 0.0000 | 0.5897 | 0.5897 | False | True |
| same_far__other_near2 | 0.0000 | 0.5077 | 0.5077 | False | True |
| same_near__other_far | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_mid | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_near2 | 0.0000 | 0.0195 | 0.0195 | False | False |

G-3 layer scan top rows by G-1 accuracy (ridge linear scan, descriptive):

| layer | G-1 acc | recency-only | identity-only | CV acc | within score | gate |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 0.4217 | 0.3451 | 0.7923 | 0.9658 | 0.2036 | False |
| 24 | 0.4217 | 0.3451 | 0.7923 | 0.9658 | 0.2033 | False |
| 26 | 0.4181 | 0.3451 | 0.7923 | 0.9633 | 0.2060 | False |
| 27 | 0.4181 | 0.3451 | 0.7923 | 0.9533 | 0.2036 | False |
| 25 | 0.4173 | 0.3451 | 0.7923 | 0.9625 | 0.2049 | False |
| 28 | 0.4173 | 0.3451 | 0.7923 | 0.9558 | 0.2019 | False |
| 22 | 0.4076 | 0.3451 | 0.7923 | 0.9242 | 0.1812 | False |
| 21 | 0.4067 | 0.3451 | 0.7923 | 0.9217 | 0.1807 | False |

G-4 optional intrinsic dimension (participation ratio):

- correct_current: n=471, PR=10.9130
- within_stale: n=589, PR=11.1639
- cross_slot: n=76, PR=13.4996

## llama

- Sanity: layer 32, stored L-1 probe-row within-stale true-current score 0.8220; geometry refit CV accuracy 0.8958, refit within-stale true-current score 0.8074, shuffle95 [0.03294662462696083, 0.06302263615056507].
- G-1: geometric argmax accuracy 0.3958; shuffle95 [0.12540254237288134, 0.16610169491525423]; recency-only 0.2424; identity-only 0.6737; gate pass `False`. Reading: above shuffle, but below the stronger identity-only baseline, so simple geometric argmax does not pass the competition gate.
- G-2 raw: held-out-cell cross-share MAE 0.1738; crossover sign accuracy 1.0000.
- G-2 length-controlled: held-out-cell cross-share MAE 0.1738; crossover sign accuracy 1.0000. Reading: Prediction fails: the length-controlled identity/recency decomposition predicts stale-only competition in every held-out cell while observed cross-slot failures remain nonzero.

| held-out cell | predicted cross share | observed cross share | abs error | predicted crossover | observed crossover/tie |
|---|---:|---:|---:|---:|---:|
| same_far__other_far | 0.0000 | 0.4545 | 0.4545 | False | False |
| same_far__other_mid | 0.0000 | 0.2959 | 0.2959 | False | False |
| same_far__other_near2 | 0.0000 | 0.1190 | 0.1190 | False | False |
| same_near__other_far | 0.0000 | 0.0000 | 0.0000 | False | False |
| same_near__other_mid | 0.0000 | 0.1679 | 0.1679 | False | False |
| same_near__other_near2 | 0.0000 | 0.0056 | 0.0056 | False | False |

G-3 layer scan top rows by G-1 accuracy (ridge linear scan, descriptive):

| layer | G-1 acc | recency-only | identity-only | CV acc | within score | gate |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.3915 | 0.2424 | 0.6737 | 0.9692 | 0.2016 | False |
| 22 | 0.3907 | 0.2424 | 0.6737 | 0.9542 | 0.1999 | False |
| 23 | 0.3907 | 0.2424 | 0.6737 | 0.9600 | 0.2018 | False |
| 17 | 0.3898 | 0.2424 | 0.6737 | 0.9375 | 0.2018 | False |
| 28 | 0.3898 | 0.2424 | 0.6737 | 0.9600 | 0.2012 | False |
| 19 | 0.3890 | 0.2424 | 0.6737 | 0.9417 | 0.2007 | False |
| 26 | 0.3890 | 0.2424 | 0.6737 | 0.9683 | 0.2030 | False |
| 29 | 0.3890 | 0.2424 | 0.6737 | 0.9583 | 0.2018 | False |

G-4 optional intrinsic dimension (participation ratio):

- correct_current: n=458, PR=10.0443
- within_stale: n=634, PR=9.0606
- cross_slot: n=88, PR=10.8600

## Verdict

G-1 and G-2 are the only headline tests. In both Qwen and Llama, G-1 is above the candidate-shuffle null but does not beat the stronger identity-only baseline. G-2 does not provide the requested cross-model geometric explanation of the identity-vs-recency crossover: Qwen misses the crossover signs in the same_far/mid and same_far/near2 cells, and both models' length-controlled decomposition predicts zero cross-slot share in every held-out cell. The defensible reading is therefore negative for this geometry addendum: Stage L remains a strong decodable-but-not-selected selection result, but this simple probe-direction identity/recency geometry does not explain why the wrong competitor wins.
