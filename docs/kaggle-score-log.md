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
| `kaggle_t12345_t2_tuned_v12.csv` | Pending | Only Task 2 replaced with the full-Train tuned weighted CatBoost; 1,088 Task 2 labels changed, other task fields copied exactly from v9 | Local candidate only. Task 2 CV and reused holdout gains do not establish an overall Kaggle gain. |
| `kaggle_t12345_t3_log_v13.csv` | **0.74686** | Only Task 3 replaced with the full-Train log-positive-amount mixture; 25,587 Task 3 values changed, Tasks 1, 2, 4, and 5 copied exactly from v9 | User-reported overall score rose **0.00093** versus v9. This is a controlled Task 3 submission difference, not a reported Task 3 NMAE. |
| `kaggle_t12345_t3log_t5_ocsvm35_v14.csv` | 0.74490 | Only Task 5 changed from v13: the same OCSVM ranking alerts 9,118 rather than 6,531 accounts | User-reported overall score fell **0.00196** versus v13. The broader alert budget hurt this controlled submission; no Task 5 F₃ is available. |
| `kaggle_t12345_t3log_t5_consensus_v15.csv` | 0.74595 | Only Task 5 changed from v13: OCSVM ∩ GMM alerts 5,377 accounts | User-reported overall score fell **0.00091** versus v13 and **0.00353** versus v17. The smaller consensus set did not beat the fixed-budget rank blend; no Task 5 F₃ is available. |
| `kaggle_t12345_t3log_t5_union_v16.csv` | 0.74700 | Only Task 5 changed from v13: OCSVM ∪ GMM alerts 7,685 accounts | User-reported overall score rose **0.00014** versus v13 but fell **0.00248** versus v17. The added detector-specific alerts did not beat the fixed-budget rank blend; no Task 5 F₃ is available. |
| `kaggle_t12345_t3log_t5_rankblend25_v17.csv` | **0.74948** | Only Task 5 changed from v13: equal OCSVM/GMM percentile-rank blend at the same 6,531-alert count, swapping 542 alerts in and 542 out | User-reported overall score rose **0.00262** versus v13. This is the highest reported overall score; no Task 5 F₃ is available. |

The pre-refresh v5 score was **0.60269** on a different dataset version. It is not a controlled comparison with the refreshed **0.66919** v5 score.

## Experiment protocol

For the 14 November slot series, keep Tasks 1–4 exactly equal to v13 and Task 5's confirmed `1 = anomaly` orientation. Change one Task 5 decision rule per submission and compare each result with v13. Record CSV/card lineage, changed-row count, alert count, and user-reported overall score. Earlier v6–v11 comparisons kept their own Tasks 1–4 reference; do not compare those scores directly with v14 to isolate Task 5. With no labels, detector stability, security-flag enrichment, and score differences cannot establish Task 5 precision, recall, or F3.

## Latest controlled results and remaining drafts

The Task 3 log-target v13 file raised the overall score from v9's **0.74593** to **0.74686** while keeping Tasks 1, 2, 4, and 5 identical. V14 widened only Task 5's OCSVM alert budget from 6,531 to 9,118 and fell to **0.74490** (**−0.00196** versus v13). V17 kept the 6,531-alert budget and changed only Task 5's ranking to an equal OCSVM/GMM percentile-rank blend, rising to **0.74948** (**+0.00262** versus v13). V16's OCSVM/GMM union scored **0.74700** (**−0.00248** versus v17); v15's smaller consensus set scored **0.74595** (**−0.00353** versus v17). V17 remains the highest user-reported score. The earlier v9-to-v11 GMM swap lost **0.00072** at the same alert budget, but it used the old Task 3 column and is not a direct v13 comparison. All five 14 November planned candidates now have reported scores; the old-Task-3 35% draft v10 remains unsubmitted. No result reveals per-task F₃, hidden labels, or a precise best alert threshold.

## Exact v9–v11 prediction overlap

The two submitted CSVs have **25,889** identical IDs and identical Tasks 1–4. Each predicts **6,531** Task 5 anomalies. Their exact Task 5 contingency table is:

| v9 prediction | v11 = 0 | v11 = 1 |
| --- | ---: | ---: |
| 0 | 18,204 | 1,154 |
| 1 | 1,154 | 5,377 |

Thus **2,308 / 25,889 (8.91%)** rows disagree, while **5,377 / 6,531 (82.33%)** of each model's positive alerts overlap. Positive-set Jaccard is **0.6997**; each model swaps **17.67%** of its alert budget. The small overall score difference (**0.00072**) despite 2,308 changed labels says neither exclusive set has a large measured advantage in this leaderboard comparison. It does **not** show whether the 5,377 shared alerts are correct or how many anomalies lie among the 18,204 shared negatives; hidden labels or a Task 5 score are required for that distinction.
