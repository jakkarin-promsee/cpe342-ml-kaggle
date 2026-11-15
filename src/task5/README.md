# Task 5 — Account Security Monitoring

This is the reading map for Task 5. The notebook version numbers describe when artifacts were created, **not** the current submission order. Use [submit-plan.md](../../submit-plan.md) for the upload order and [the research note](../../docs/research-note-task5.md) for exact feature names, metrics, and conclusions.

The supplied Task 5 data has only `data/raw/test.csv` and no anomaly labels. The user now reports that the lecturer permits full use of this file for Task 5 modeling and analysis; the written competition rule still prohibits Test fitting, so document this clarification and conflict in the final report. The lecturer's practical limit is to avoid repeated Kaggle submissions that harvest leaderboard feedback. Rank-stability proxies measure repeatability, **not macro F₃**. Public Kaggle scores below are overall scores across five tasks, not Task 5 scores.

## Read the work in this order

1. [Competition description](../../docs/kaggle-detial/) and [pipeline guide](../pipeline/README.md).
2. **Original-hypothesis branch:** v2 EDA → saved detector families → ranking diagnostics → OCSVM/GMM submissions → v17 → Tail/LOF alternatives.
3. **New-behavior branch:** v3 EDA → peer/bot models → v17-based v18/v19 → Tail-based v22/v23.
4. [Decision-line and 3D visualization](eda/inspect_decision_lines_v1.ipynb), after seeing which model scores feed each submission.
5. [Submit plan](../../submit-plan.md) for the current sequence; [score log](../../docs/kaggle-score-log.md) for actual uploaded results.

## Original-hypothesis branch: raw → features → detectors

| Step | Notebook / saved artifact | Purpose and next step |
| --- | --- | --- |
| 1 | [build_eda_original_v2.ipynb](eda/build_eda_original_v2.ipynb) → `data/eda/eda_original_v2.json` | Reads raw Test, audits missingness and coordinates, masks invalid coordinate pairs, and saves the 122 supplied predictors. |
| 2 | [build_eda_1_v2.ipynb](eda/build_eda_1_v2.ipynb) → `data/eda/eda_1_v2.json` | Reads raw Test independently and saves 122 supplied + 91 within-account engineered features. It does not depend on step 1's notebook variables. |
| 3 | [build_model_v2.ipynb](models/build_model_v2.ipynb) → `save/t5_r2_*.json` | Reads both EDA cards; fits Isolation Forest, LOF, OCSVM, GMM, PCA reconstruction, and Empirical Tail on original/engineered features. Each saved model has a top-5% rank-stability proxy. Engineered OCSVM/GMM feed v17; engineered Tail/LOF feed v20/v21. |
| 4 | [rank_diagnostics_v1.ipynb](models/rank_diagnostics_v1.ipynb) | Compares already saved engineered rankings at a common alert budget, especially OCSVM versus GMM. Analysis only; no new model or EDA artifact. |

The first-generation [original EDA](eda/build_eda_original.ipynb), [engineered EDA](eda/build_eda_1.ipynb), and [model notebook](models/build_model.ipynb) created `*_v1` artifacts before the refreshed v2 data. They are historical, not the source of current submissions.

## Original detector submissions: OCSVM → GMM → v17

| Step | Notebook / submission | Controlled change and observed result |
| --- | --- | --- |
| 5% baseline | `build_model_v2.ipynb` → v6; [assemble_v2.ipynb](../submission/assemble_v2.ipynb) → label-flip v7 | OCSVM `nu=0.10` at its original cutoff. The v7 comparison supports `1 = anomaly`. |
| 15% | [build_model_v3.ipynb](models/build_model_v3.ipynb) → v8 | Same saved OCSVM ranking, 3,971 alerts; overall **0.74446**. |
| 25% | [build_model_v4.ipynb](models/build_model_v4.ipynb) → v9 | Same ranking, 6,531 alerts; overall **0.74593**. This becomes the fixed alert budget. |
| 35% | [build_model_v5.ipynb](models/build_model_v5.ipynb) → v10 | Same ranking, 9,118 alerts; v10 remains unscored. |
| GMM at v9 volume | [build_model_v6.ipynb](models/build_model_v6.ipynb) → v11 | Saved engineered GMM with 6,531 alerts; overall **0.74521**. |
| Freeze Tasks 1–4 | [assemble_task3_log_v13.ipynb](../submission/assemble_task3_log_v13.ipynb) → v13 | Changes Task 3, then freezes Tasks 1–4 for later Task 5 comparisons; overall **0.74686**. |
| Budget, consensus, union, blend | [assemble_submit_plan_20261114.ipynb](../submission/assemble_submit_plan_20261114.ipynb) → v14–v17 | V17 averages saved OCSVM/GMM anomaly-score percentile ranks 50/50, selects 6,531 alerts, and scored **0.74948**. It was the best result in that round and is the fixed reference for v20/v21. V14–v16 are in the score log. |

The model JSON cards identify exact dataset and model files; submission JSON cards identify source CSVs and output hashes. No overall-score difference directly reveals Task 5 F₃.

## Before changing assumptions: Tail and LOF at v17's budget

[assemble_task5_detector_alternatives_20261116.ipynb](../submission/assemble_task5_detector_alternatives_20261116.ipynb) reloads the **saved engineered** Tail and LOF models from step 3. It keeps v17's IDs and Tasks 1–4, selects each model's top **6,531** anomaly scores, and writes local CSV/JSON pairs:

| Planned order | File / saved model | Difference from v17 | Status |
| --- | --- | ---: | --- |
| **1** | [v20 Tail CSV](../submission/kaggle_t12345_t5_tail_engineered_v20.csv) / `t5_r2_tail_engineered_v2` | 2,885 alerts enter and 2,885 leave | Scored **0.75338 overall**, +0.00390 vs v17; current best |
| **2** | [v21 LOF CSV](../submission/kaggle_t12345_t5_lof_engineered_v21.csv) / `t5_r2_lof_engineered_v2` | 1,405 enter and 1,405 leave | Scored **0.70060 overall**, −0.04888 vs v17 |

Compare **each directly with v17** to isolate detector choice under the original engineered-feature hypothesis. Matched top-25% rank-stability proxies are **0.97044** for Tail and **0.91205** for LOF; neither predicted the observed quality gap or estimates accuracy. Tail and LOF share only **3,048** alerts. The [research note](../../docs/research-note-task5.md) interprets these controlled public results without claiming a Task 5 F₃.

## New-behavior branch: raw → peer/bot models → Tail-based v22/v23

This branch starts again from raw Test rather than from `eda_1_v2`:

| Step | Notebook / artifact | Purpose and next step |
| --- | --- | --- |
| A | [build_eda_behavior_v3.ipynb](eda/build_eda_behavior_v3.ipynb) → `data/eda/eda_behavior_views_v3.json` | Saves 122 raw + 45 new within-account change, peer-matching, and regularity features. One test-only EDA artifact; no dependency on v2 notebook variables. |
| B | [build_model_v7.ipynb](models/build_model_v7.ipynb) → `t5_peer_change_v3`, `t5_bot_regular_v3` | Fits peer-relative change and sustained-activity plus regularity scorers. Saves separate rank-stability evaluations and analyzes their relation to v17. |
| C, v17-based check | [assemble_task5_behavior_20261116.ipynb](../submission/assemble_task5_behavior_20261116.ipynb) → v18/v19 | Reconstructs v17 from saved OCSVM/GMM models. V18 blends v17 with peer change and later scored **0.72500**, **−0.02448** versus v17; v19 adds bot regularity but remains unsubmitted. |
| D, scored follow-up | [assemble_task5_tail_behavior_20261116.ipynb](../submission/assemble_task5_tail_behavior_20261116.ipynb) → [v22 Tail + peer](../submission/kaggle_t12345_t5_tail_peer_v22.csv), [v23 Tail + peer + bot](../submission/kaggle_t12345_t5_tail_peer_bot_v23.csv) | Reproduces scored Tail v20 exactly. V22 fixes 80% Tail rank + 20% peer rank; v23 fixes 80% v22 score + 20% bot rank. Both keep 6,531 alerts and frozen Tasks 1–4. Overall scores: **0.75004** and **0.74201**. |

V22 swaps **783** alerts each way versus Tail v20 and falls **0.00334** overall. V23 swaps **974** each way versus v22 and falls another **0.00803**. The intended peer-change and regularity profiles appear in the added alerts, but neither fixed addition improves the scored Tail baseline. V18 also loses on its own v17 base after **1,628** swaps each way. Tail v20 remains the best reported submission at **0.75338**; the v19 file remains an unsubmitted historical draft.

## Visualization: after the model paths

[inspect_decision_lines_v1.ipynb](eda/inspect_decision_lines_v1.ipynb) reads saved EDA cards, models, and local v17–v19 CSVs. It creates **no** new EDA dataset or model. Read its views in order:

1. **OCSVM score cutoff:** exact one-dimensional operational boundary for the 6,531-alert OCSVM selection.
2. **OCSVM × GMM percentile-rank plane:** v17's exact diagonal rank-blend boundary. Similar score planes show v18 and v19.
3. **Raw-feature and PCA views:** descriptive projections. Three clipped PCA components explain only **12.93%** of the 252-feature union's variance, so they cannot show the full OCSVM boundary.
4. **Interactive 3D cell:** follows the same sampled IDs through OCSVM, GMM, v17, v18, and v19 score ranks, with fixed ID colors and a red v17 boundary. This shows rank movement, not physical movement in feature space.

For feature definitions and detailed limitations, read the [research note](../../docs/research-note-task5.md). For the next upload and its correct comparison baseline, read the [submit plan](../../submit-plan.md).
