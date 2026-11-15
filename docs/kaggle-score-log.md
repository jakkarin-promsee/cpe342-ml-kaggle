# Kaggle leaderboard experiment log

This file records user-reported **overall** leaderboard scores. The competition has not supplied per-task scores or task weights, so an overall change is not a measured Task 5 F3. Test labels remain unavailable. The Task 5 fits recorded so far are unsupervised. The user now reports that the lecturer permits full use of the Task 5 test-only file for modeling and analysis, while warning against repeated submissions that harvest leaderboard feedback; the written rule in `docs/kaggle-detial/rule.md` still says Test is not for fitting.

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
| `kaggle_t12345_t3log_t5_rankblend25_v17.csv` | **0.74948** | Only Task 5 changed from v13: equal OCSVM/GMM percentile-rank blend at the same 6,531-alert count, swapping 542 alerts in and 542 out | User-reported overall score rose **0.00262** versus v13. It is the fixed reference for v20/v21; no Task 5 F₃ is available. |
| `kaggle_t12345_t5_peerchange_v18.csv` | **0.72500** | Fixed 50% v17 rank + 50% peer-change rank at 6,531 alerts, swapping 1,628 in/out versus v17 | User-reported overall score **−0.02448** versus v17. This later scored v17-based H1 probe also lost; it is distinct from Tail-based v22. |
| `kaggle_t12345_t5_peerchange_bot_v19.csv` | Unsubmitted, superseded draft | Adds a fixed bot-regularity score to v18, swapping 887 alerts in and out | Historical H2 file; the current plan instead uses Tail-based v23. |
| `kaggle_t12345_t5_tail_engineered_v20.csv` | **0.75338** | Only Task 5 changed from v17: saved engineered Empirical Tail ranking at the same 6,531-alert count; 2,885 alerts swapped each way | User-reported overall score **+0.00390** versus v17. This is the highest reported overall score; it supports this fixed Tail selection on the public split, not a measured Task 5 F₃. |
| `kaggle_t12345_t5_lof_engineered_v21.csv` | **0.70060** | Only Task 5 changed from v17: saved engineered LOF ranking at the same 6,531-alert count; 1,405 alerts swapped each way | User-reported overall score **−0.04888** versus v17 and **−0.05278** versus Tail. LOF's stable ranking was a poor public-split selection under this controlled setup. |
| `kaggle_t12345_t5_tail_peer_v22.csv` | **0.75004** | Fixed 80% Tail rank + 20% peer-change rank at 6,531 alerts, swapping 783 in/out versus scored v20 | User-reported overall score **−0.00334** versus v20; the fixed peer addition did not improve the Tail base on the public split. |
| `kaggle_t12345_t5_tail_peer_bot_v23.csv` | **0.74201** | Fixed 80% v22 score + 20% bot-regularity rank at 6,531 alerts, swapping 974 in/out versus v22 | User-reported overall score **−0.00803** versus v22 and **−0.01137** versus v20. The fixed bot addition further reduced public score. |

The pre-refresh v5 score was **0.60269** on a different dataset version. It is not a controlled comparison with the refreshed **0.66919** v5 score.

## 16 November Tail-based behavior checks

V20 Tail, v22 Tail+peer, and v23 Tail+peer+bot have identical IDs, Tasks 1–4, `1 = anomaly`, and **6,531** Task 5 alerts. The controlled overall score sequence is **0.75338 → 0.75004 → 0.74201**. V22 changes 783 alerts each way versus v20; v23 changes 974 each way versus v22. Those changes lower the public score by **0.00334** and another **0.00803**, respectively. The evidence is specific to these fixed score definitions and weights; it does not show that account change or bot activity are absent from the true anomalies. V20 remains the best user-reported combined submission. No per-task F₃ or hidden labels are available, so stop this blend branch rather than searching weights from the public results.

The user also submitted the earlier v17-based H1 file v18, scoring **0.72500** versus v17's **0.74948** (**−0.02448**) with 1,628 alerts swapped each way. This is a second controlled loss for a peer-change addition, now on the OCSVM/GMM base and at a larger 50% peer weight. The v18-versus-v22 difference cannot isolate detector base because both the base and peer weight change. It does not validate the old feature families as fraud truth; it shows the two tested peer blends did not improve their respective public baselines.

## Experiment protocol

For the 14 November slot series, keep Tasks 1–4 exactly equal to v13 and Task 5's confirmed `1 = anomaly` orientation. Change one Task 5 decision rule per submission and compare each result with v13. Record CSV/card lineage, changed-row count, alert count, and user-reported overall score. Earlier v6–v11 comparisons kept their own Tasks 1–4 reference; do not compare those scores directly with v14 to isolate Task 5. With no labels, detector stability, security-flag enrichment, and score differences cannot establish Task 5 precision, recall, or F3.

## Latest controlled results and remaining drafts

The Task 3 log-target v13 file raised the overall score from v9's **0.74593** to **0.74686** while keeping Tasks 1, 2, 4, and 5 identical. V14 widened only Task 5's OCSVM alert budget from 6,531 to 9,118 and fell to **0.74490** (**−0.00196** versus v13). V17 kept the 6,531-alert budget and changed only Task 5's ranking to an equal OCSVM/GMM percentile-rank blend, rising to **0.74948** (**+0.00262** versus v13). V16's OCSVM/GMM union scored **0.74700** (**−0.00248** versus v17); v15's smaller consensus set scored **0.74595** (**−0.00353** versus v17). The earlier v9-to-v11 GMM swap lost **0.00072** at the same alert budget, but it used the old Task 3 column and is not a direct v13 comparison. The later v20 Tail candidate now leads at **0.75338** and v21 LOF fell to **0.70060**, both at v17's alert count with exact frozen Tasks 1–4. These are controlled ranking comparisons, not direct F₃ or recovered Test labels. V18/v19 and the old-Task-3 35% draft v10 remain unsubmitted.

## Exact v9–v11 prediction overlap

The two submitted CSVs have **25,889** identical IDs and identical Tasks 1–4. Each predicts **6,531** Task 5 anomalies. Their exact Task 5 contingency table is:

| v9 prediction | v11 = 0 | v11 = 1 |
| --- | ---: | ---: |
| 0 | 18,204 | 1,154 |
| 1 | 1,154 | 5,377 |

Thus **2,308 / 25,889 (8.91%)** rows disagree, while **5,377 / 6,531 (82.33%)** of each model's positive alerts overlap. Positive-set Jaccard is **0.6997**; each model swaps **17.67%** of its alert budget. The small overall score difference (**0.00072**) despite 2,308 changed labels says neither exclusive set has a large measured advantage in this leaderboard comparison. It does **not** show whether the 5,377 shared alerts are correct or how many anomalies lie among the 18,204 shared negatives; hidden labels or a Task 5 score are required for that distinction.
