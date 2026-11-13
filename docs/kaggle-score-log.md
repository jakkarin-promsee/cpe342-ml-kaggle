# Kaggle leaderboard experiment log

This file records user-reported **overall** leaderboard scores. The competition has not supplied per-task scores or task weights, so an overall change is not a measured Task 5 F3. Test labels remain unavailable. All Task 5 fitting described here is unsupervised under the user's reported lecturer clarification; the written competition rule in `docs/kaggle-detial/rule.md` still says Test is not for fitting.

| Submission | Overall score | Controlled change | Interpretation |
| --- | ---: | --- | --- |
| `kaggle_t12345_current_v5.csv` | 0.66919 | Refreshed old submission after the dataset change | Historical reference; differs from v6 in Tasks 3–4. |
| `kaggle_t12345_gate_resnet_v6.csv` | 0.72700 | Current Tasks 1–4 and Task 5 OCSVM at about 5% alerts | Controlled reference for Task 5 probes. |
| `kaggle_t12345_gate_resnet_t5flip_v7.csv` | 0.66266 | Only Task 5 labels inverted from v6 | Overall score fell by 0.06434; retain `1 = anomaly`. |
| `kaggle_t12345_t5_alert15_v8.csv` | **0.74446** | Only Task 5 cutoff changed from v6; 1,365 to 3,971 alerts | Overall score rose by **0.01746** versus v6. The broader alert budget helped this fixed detector. |
| `kaggle_t12345_t5_alert25_v9.csv` | **0.74593** | Same OCSVM ranking at about 25% alerts; 6,531 alerts | Overall score rose **0.00147** versus v8 and **0.01893** versus v6; the marginal gain is smaller than v6-to-v8. |
| `kaggle_t12345_t5_alert35_v10.csv` | Pending | Same OCSVM ranking at about 35% alerts; 9,118 alerts | Prepared locally; adds 2,587 alerts beyond v9. No Kaggle score yet. |
| `kaggle_t12345_t5_gmm_match_v9_v11.csv` | 0.74521 | GMM replaces OCSVM ranking at exactly 6,531 Task 5 alerts | Controlled family swap versus v9: 1,154 alerts added and 1,154 removed; overall score fell **0.00072**. |

The pre-refresh v5 score was **0.60269** on a different dataset version. It is not a controlled comparison with the refreshed **0.66919** v5 score.

## Experiment protocol

Keep Tasks 1–4 exactly equal to v6 and keep Task 5's confirmed `1 = anomaly` orientation. Change one Task 5 setting per submission. Record the CSV/card lineage, changed-row count, alert count, and user-reported overall score. Compare controlled variants against both v6 and the immediately preceding probe. With no labels, detector stability, security-flag enrichment, and score differences cannot establish Task 5 precision, recall, or F3.

## Next probe

The 15%-to-25% cutoff change improved the overall score by only **0.00147** for 2,560 additional alerts, versus **0.01746** for the earlier 5%-to-15% change. The controlled v11 swap held the exact v9 alert count and changed only the detector ranking to GMM, swapping 1,154 accounts in each direction. Its user-reported overall score was **0.74521**, **0.00072 below v9**. The GMM-only accounts had many more supplied security flags, but those flags are model inputs, not labels; that descriptive signal did not improve the overall leaderboard score. **V9 remains the best scored submission at 0.74593.** The 35% OCSVM v10 draft remains unsubmitted. After the submission window resets, choose another controlled test using the new score evidence; no further current-window submission is available.

## Exact v9–v11 prediction overlap

The two submitted CSVs have **25,889** identical IDs and identical Tasks 1–4. Each predicts **6,531** Task 5 anomalies. Their exact Task 5 contingency table is:

| v9 prediction | v11 = 0 | v11 = 1 |
| --- | ---: | ---: |
| 0 | 18,204 | 1,154 |
| 1 | 1,154 | 5,377 |

Thus **2,308 / 25,889 (8.91%)** rows disagree, while **5,377 / 6,531 (82.33%)** of each model's positive alerts overlap. Positive-set Jaccard is **0.6997**; each model swaps **17.67%** of its alert budget. The small overall score difference (**0.00072**) despite 2,308 changed labels says neither exclusive set has a large measured advantage in this leaderboard comparison. It does **not** show whether the 5,377 shared alerts are correct or how many anomalies lie among the 18,204 shared negatives; hidden labels or a Task 5 score are required for that distinction.
