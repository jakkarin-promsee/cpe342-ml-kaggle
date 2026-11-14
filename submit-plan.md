# Submit slots for 14/11/2026

All five CSVs are prepared locally in `src/submission/` and **all have been uploaded and scored**. The previous reference `kaggle_t12345_t5_alert25_v9.csv` scored **0.74593** overall. The table keeps the original planned slot labels; the actual submission order was v13, v14, v17, v16, v15. Scores below are overall public scores, not per-task metrics or Task 5 F₃.

| Slot | CSV to upload | Single change and hypothesis | Task 5 alerts | Public score |
| --- | --- | --- | ---: | --- |
| 1 | [`kaggle_t12345_t3_log_v13.csv`](src/submission/kaggle_t12345_t3_log_v13.csv) | Replace Task 3 with the full-Train log-positive-amount model. Five-fold MAE improved, and the overall score rose **0.00093** versus v9. | 6,531 | **0.74686** |
| 2 | [`kaggle_t12345_t3log_t5_ocsvm35_v14.csv`](src/submission/kaggle_t12345_t3log_t5_ocsvm35_v14.csv) | Keep Tasks 1–4 from slot 1; widen the same OCSVM ranking from 25% to 35% alerts. The overall score fell **0.00196** versus slot 1. | 9,118 | 0.74490 |
| 3 | [`kaggle_t12345_t3log_t5_consensus_v15.csv`](src/submission/kaggle_t12345_t3log_t5_consensus_v15.csv) | Keep Tasks 1–4 from slot 1; alert only on the OCSVM ∩ GMM set. Overall score fell **0.00353** versus v17. | 5,377 | 0.74595 |
| 4 | [`kaggle_t12345_t3log_t5_union_v16.csv`](src/submission/kaggle_t12345_t3log_t5_union_v16.csv) | Keep Tasks 1–4 from slot 1; alert on the OCSVM ∪ GMM set. Overall score fell **0.00248** versus v17. | 7,685 | 0.74700 |
| 5 | [`kaggle_t12345_t3log_t5_rankblend25_v17.csv`](src/submission/kaggle_t12345_t3log_t5_rankblend25_v17.csv) | Keep Tasks 1–4 from slot 1; average OCSVM and GMM anomaly-score percentile ranks 50/50 and retain exactly 6,531 alerts. Overall score rose **0.00262** versus slot 1. | 6,531 | **0.74948** |

## Frozen fields

1. **Task 1:** keep v9's `t1_lgbm_combined_pruned_cv6_full_v1`. The class-weight follow-up did not pass the practical improvement threshold.
2. **Task 2:** keep v9's `t2_cat_original_no_noise_v2`. The tuned candidate and fixed ensemble gains did not justify changing this submission; the unscored v12 file is not part of this plan.
3. **Task 3:** slot 1 replaces v9's `t3_gate_blend_cv5_v2` with `t3_log_positive_amount_full_v6`; slots 2–5 keep that exact Task 3 column. The selected procedure improved mean five-fold MAE **2,895.92 → 2,814.97 THB**. Kaggle NMAE impact is unknown.
4. **Task 4:** keep v9's `t4_resnet18_geometry_full_v2` exactly.
5. **Task 5:** slot 1 keeps v9's `t5_r4_ocsvm_nu10_alert25_v4` decisions, with `1 = anomaly`. Slots 2–5 change only this field and use already fitted competition-data models or their fixed predictions.

## Comparison rule

- **Slot 1 versus v9:** **+0.00093** overall, with only Task 3 changed. Slot 1 freezes the Task 5 reference for the remaining tests.
- **Slot 2 versus slot 1:** **−0.00196** overall, with only the Task 5 alert budget changed. Compare any later slot with slot 1 to isolate its Task 5 decision rule. Do not compare these directly with old v10/v11 scores because those files still used the old Task 3 column.
- **V17 versus slot 1:** **+0.00262** overall, at the same Task 5 alert count with 542 alerts added and 542 removed. V17 is the highest user-reported result.
- **V16 versus v17:** **−0.00248** overall; the union alerts 7,685 accounts versus v17's 6,531.
- **V15 versus v17:** **−0.00353** overall; the consensus keeps 5,377 alerts, removing 1,154 from v17. The five planned candidates are now all scored; v17 remains the best.
- Record each overall score and difference in `docs/kaggle-score-log.md`. Without per-task labels or weights, a difference in overall score is evidence for that controlled submission comparison, not a measured F₃ or recovered Test labels.
- These five rules were fixed before today's scores. Do not tune more cutoffs or blend weights from the same public leaderboard feedback.

## File checks and lineage

[`assemble_submit_plan_20261114.ipynb`](src/submission/assemble_submit_plan_20261114.ipynb) generated and verified slots 2–5. Every CSV has **25,889** unique sample `ANS` IDs in order, the required six columns, and no missing values. Slots 2–5 match slot 1 exactly in `id`, `task1`, `task2`, `task3`, and `task4`; each has a JSON card beside its CSV with source SHA-256, output SHA-256, alert count, and changed-row count. Slot 1's [JSON card](src/submission/kaggle_t12345_t3_log_v13.json) records the v9 source hash and Task 3 model lineage.
