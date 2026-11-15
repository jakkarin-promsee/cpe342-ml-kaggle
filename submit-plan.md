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
- **V15 versus v17:** **−0.00353** overall; the consensus keeps 5,377 alerts, removing 1,154 from v17. All five candidates in the 14 November round were scored; v17 was that round's best.
- Record each overall score and difference in `docs/kaggle-score-log.md`. Without per-task labels or weights, a difference in overall score is evidence for that controlled submission comparison, not a measured F₃ or recovered Test labels.
- These five rules were fixed before today's scores. Do not tune more cutoffs or blend weights from the same public leaderboard feedback.

## File checks and lineage

[`assemble_submit_plan_20261114.ipynb`](src/submission/assemble_submit_plan_20261114.ipynb) generated and verified slots 2–5. Every CSV has **25,889** unique sample `ANS` IDs in order, the required six columns, and no missing values. Slots 2–5 match slot 1 exactly in `id`, `task1`, `task2`, `task3`, and `task4`; each has a JSON card beside its CSV with source SHA-256, output SHA-256, alert count, and changed-row count. Slot 1's [JSON card](src/submission/kaggle_t12345_t3_log_v13.json) records the v9 source hash and Task 3 model lineage.

## Controlled Task 5 checks for 16/11/2026

The original detector reference was v17 (**0.74948 overall**); Tail v20 leads at **0.75338** with the same **6,531 Task 5 alerts**. All four planned detector/behavior comparisons have now been uploaded and scored. The behavior hypotheses used **Tail v20 as their frozen base**, per the user's updated plan. Tasks 1–4, sample IDs, `1 = anomaly`, and the alert budget were identical across these files. Version numbers identify artifacts, not upload order. No per-task F₃ or hidden labels are available.

| Submission order | Question | CSV | Comparison and changed alerts |
| --- | --- | --- | --- |
| **1 — scored** | Does a stable empirical-tail ranking beat v17's OCSVM/GMM blend on the same engineered features? | [`kaggle_t12345_t5_tail_engineered_v20.csv`](src/submission/kaggle_t12345_t5_tail_engineered_v20.csv) | **0.75338 overall, +0.00390 vs v17.** 2,885 alerts enter and 2,885 leave; top-25% rank-stability proxy 0.97044. |
| **2 — scored** | Does a local-density ranking do better under the same features and alert budget? | [`kaggle_t12345_t5_lof_engineered_v21.csv`](src/submission/kaggle_t12345_t5_lof_engineered_v21.csv) | **0.70060 overall, −0.04888 vs v17.** 1,405 alerts enter and 1,405 leave; top-25% stability proxy 0.91205. |
| **3 — scored** | Does peer-relative account change add information to the stronger Tail ranking? | [`kaggle_t12345_t5_tail_peer_v22.csv`](src/submission/kaggle_t12345_t5_tail_peer_v22.csv) | **0.75004 overall, −0.00334 vs Tail v20.** Fixed 80% Tail rank + 20% peer rank; 783 alerts enter and leave. Peer-score top-25% stability proxy 0.85874. |
| **4 — scored** | Does mechanical regularity add information beyond Tail plus peer change? | [`kaggle_t12345_t5_tail_peer_bot_v23.csv`](src/submission/kaggle_t12345_t5_tail_peer_bot_v23.csv) | **0.74201 overall, −0.00803 vs v22; −0.01137 vs v20.** Fixed 80% v22 score + 20% bot rank; 974 alerts enter and leave versus v22. Bot-score top-25% stability proxy 0.99402. |

Use each slot's overall public score as a controlled comparison, not as Task 5 F₃ or proof of which accounts are threats. The Task 5 test-only file has no labels. The user reports a lecturer clarification allowing full use of this file for Task 5 modeling and analysis, while the written rule forbids Test fitting; document that conflict in the final report. The lecturer warned against repeated leaderboard submissions to harvest feedback. Record each uploaded score and exact CSV hash in `docs/kaggle-score-log.md`; leave unused slots unused and do not tune cutoffs or blend weights from these scores.

The first two CSVs and lineage cards were built by [`assemble_task5_detector_alternatives_20261116.ipynb`](src/submission/assemble_task5_detector_alternatives_20261116.ipynb) from the saved `t5_r2_tail_engineered_v2` and `t5_r2_lof_engineered_v2` models. Their SHA-256 hashes are `2dd79b546af197688d3ff35211cc744c2e86a28360ea3a01343f743364c1505a` (v20 Tail) and `2bc372f7c2abaf71156166859dcf155aab4630f1d8ac8cb6b0f0e007616b7322` (v21 LOF). The Tail-based behavior files and cards were built by [`assemble_task5_tail_behavior_20261116.ipynb`](src/submission/assemble_task5_tail_behavior_20261116.ipynb); their hashes are `41a8a2eef3e488400b44653642f9cc579f67afa15f53e5d206a9dc618d368a89` (v22) and `538a09a07779b2ae46bf75606efb7e8f6135cb98318f2b5e3f42d1928de99682` (v23). All four scores are recorded in `docs/kaggle-score-log.md`.

The earlier [v18/v19 files](src/submission/assemble_task5_behavior_20261116.ipynb) remain intact as historical v17-based artifacts. **V18 was also submitted later and scored 0.72500**, **−0.02448** versus v17 at the same 6,531-alert budget with 1,628 swaps each way; v19 remains unsubmitted. V18 and v22 both add the peer-change score to their own base and both lose, but they use different bases and peer weights (50% versus 20%). V22 tested H1 against Tail v20, and v23 tested H2 against v22. V23 differs from Tail v20 by 1,169 alerts each way. Keep v20 as the scored reference. Do not tune weights or alert budget from this leaderboard feedback.

**Offline next step, when requested:** consider a tree surrogate of the saved Tail/v17 scores and alert decisions to rank feature families for clearer visualization. The proposed fidelity checks and interpretation limits are in [the Task 5 research note](docs/research-note-task5.md). No surrogate model or notebook has been created. The current four-slot submission experiment is complete; do not start a weight search from these scores.
