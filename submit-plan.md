# Submit slots for 14/11/2026

All five CSVs are prepared locally in `src/submission/`. **None has been uploaded.** The scored reference is `kaggle_t12345_t5_alert25_v9.csv` at overall public score **0.74593**. Record public scores below after submitting; they are overall scores, not per-task metrics or Task 5 F₃.

| Slot | CSV to upload | Single change and hypothesis | Task 5 alerts | Public score |
| --- | --- | --- | ---: | --- |
| 1 | [`kaggle_t12345_t3_log_v13.csv`](src/submission/kaggle_t12345_t3_log_v13.csv) | Replace Task 3 with the full-Train log-positive-amount model. Test whether its five-fold MAE improvement transfers to the leaderboard. | 6,531 | Pending |
| 2 | [`kaggle_t12345_t3log_t5_ocsvm35_v14.csv`](src/submission/kaggle_t12345_t3log_t5_ocsvm35_v14.csv) | Keep Tasks 1–4 from slot 1; widen the same OCSVM ranking from 25% to 35% alerts. Hypothesis: recall gain outweighs extra false alerts. | 9,118 | Pending |
| 3 | [`kaggle_t12345_t3log_t5_consensus_v15.csv`](src/submission/kaggle_t12345_t3log_t5_consensus_v15.csv) | Keep Tasks 1–4 from slot 1; alert only on the OCSVM ∩ GMM set. Hypothesis: model agreement improves precision enough to offset lower alert volume. | 5,377 | Pending |
| 4 | [`kaggle_t12345_t3log_t5_union_v16.csv`](src/submission/kaggle_t12345_t3log_t5_union_v16.csv) | Keep Tasks 1–4 from slot 1; alert on the OCSVM ∪ GMM set. Hypothesis: complementary coverage improves recall enough to offset extra alerts. | 7,685 | Pending |
| 5 | [`kaggle_t12345_t3log_t5_rankblend25_v17.csv`](src/submission/kaggle_t12345_t3log_t5_rankblend25_v17.csv) | Keep Tasks 1–4 from slot 1; average OCSVM and GMM anomaly-score percentile ranks 50/50 and retain exactly 6,531 alerts. Hypothesis: combined ordering improves detection at the same alert budget. | 6,531 | Pending |

## Frozen fields

1. **Task 1:** keep v9's `t1_lgbm_combined_pruned_cv6_full_v1`. The class-weight follow-up did not pass the practical improvement threshold.
2. **Task 2:** keep v9's `t2_cat_original_no_noise_v2`. The tuned candidate and fixed ensemble gains did not justify changing this submission; the unscored v12 file is not part of this plan.
3. **Task 3:** slot 1 replaces v9's `t3_gate_blend_cv5_v2` with `t3_log_positive_amount_full_v6`; slots 2–5 keep that exact Task 3 column. The selected procedure improved mean five-fold MAE **2,895.92 → 2,814.97 THB**. Kaggle NMAE impact is unknown.
4. **Task 4:** keep v9's `t4_resnet18_geometry_full_v2` exactly.
5. **Task 5:** slot 1 keeps v9's `t5_r4_ocsvm_nu10_alert25_v4` decisions, with `1 = anomaly`. Slots 2–5 change only this field and use already fitted competition-data models or their fixed predictions.

## Comparison rule

- Compare **slot 1 against scored v9** to isolate the Task 3 change. Slot 1 also freezes the Task 5 reference for this day's tests.
- Compare **each of slots 2–5 against slot 1** to isolate its Task 5 decision rule. Do not compare these directly with old v10/v11 scores because those files still used the old Task 3 column.
- Record each overall score and difference in `docs/kaggle-score-log.md`. Without per-task labels or weights, a difference in overall score is evidence for that controlled submission comparison, not a measured F₃ or recovered Test labels.
- These five rules were fixed before today's scores. Do not tune more cutoffs or blend weights from the same public leaderboard feedback.

## File checks and lineage

[`assemble_submit_plan_20261114.ipynb`](src/submission/assemble_submit_plan_20261114.ipynb) generated and verified slots 2–5. Every CSV has **25,889** unique sample `ANS` IDs in order, the required six columns, and no missing values. Slots 2–5 match slot 1 exactly in `id`, `task1`, `task2`, `task3`, and `task4`; each has a JSON card beside its CSV with source SHA-256, output SHA-256, alert count, and changed-row count. Slot 1's [JSON card](src/submission/kaggle_t12345_t3_log_v13.json) records the v9 source hash and Task 3 model lineage.
