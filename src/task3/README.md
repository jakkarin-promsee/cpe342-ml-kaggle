# Task 3 — Monthly Spending Prediction

The 12 November data refresh did **not** change Task 3. The current selected model is `t3_log_positive_amount_full_v6`, used in the scored v13 and later combined submissions. The competition metric is **NMAE**; notebook MAE in THB is a local comparison and must not be labeled NMAE. Read the [research note](../../docs/research-note-task3.md) for exact features, scores, and limits.

## Main reading path

| Step | Notebook → saved artifact | What it establishes |
| --- | --- | --- |
| 1 | [build_eda_original.ipynb](eda/build_eda_original.ipynb) → `data/eda/eda_original_v1.json` | Reads raw Train/Test, checks quality/leakage, saves original predictors and labeled Train split. |
| 2 | [build_eda_1.ipynb](eda/build_eda_1.ipynb) → `data/eda/eda_combined_v1.json` | Reads raw files independently and saves 16 deterministic row-wise features on top of the original predictors. |
| 3 | [build_model.ipynb](models/build_model.ipynb) | Reads EDA cards, compares regression families/feature sets and establishes the development split, holdout, and five-fold baseline. |
| 4 | [build_model_v2.ipynb](models/build_model_v2.ipynb) → `t3_gate_blend_cv5_v2` | Tests a fixed CatBoost/LightGBM blend, a zero-aware occurrence gate plus positive amount model, and their fixed mixture. The selected early reference scores **2,895.92 THB** mean five-fold MAE. |
| 5 | [build_model_v3.ipynb](models/build_model_v3.ipynb) | Trains three PyTorch tabular networks from scratch on the same EDA card and paired folds. All lose to step 4; this branch stops. |
| 6 | [analyze_top_model_errors_1.ipynb](eda/analyze_top_model_errors_1.ipynb) | Reads saved OOF predictions, fits nothing, and locates costly false-nonzero decisions and high-spend tail errors. |
| 7 | [build_eda_2.ipynb](eda/build_eda_2.ipynb) → `data/eda/eda_spend_persistence_v2.json` | Starts again from raw Train/Test, reproduces prior features, and adds eight row-wise persistence/tail candidates motivated by step 6. One new EDA dataset. |
| 8 | [build_model_v4.ipynb](models/build_model_v4.ipynb) | Tests new gate, amount-tail features, and a cost-weighted gate separately on the same folds. No useful improvement; keep step 4 as reference for the next idea. |
| 9 | [build_model_v5.ipynb](models/build_model_v5.ipynb) | Changes only the positive-amount target to `log1p`, then inverts with `expm1`. Mean five-fold MAE improves **2,895.92 → 2,814.97 THB**, winning all five paired folds. |
| 10 | [build_model_v6.ipynb](models/build_model_v6.ipynb) → `t3_log_positive_amount_full_v6` | Refits the selected v5 procedure on all labeled Train rows and saves its per-task Test prediction. The evaluation card carries the earlier CV procedure estimate, not a new score on the full fit. |
| 11 | [build_model_v7.ipynb](models/build_model_v7.ipynb) | Tries log-RMSE for the positive amount model and a fixed half blend with log-MAE. Log-RMSE loses; the half blend gains only **3.81 THB** mean fold MAE with mixed group tradeoffs, so v6 remains selected. |

Do not infer a direct EDA dependency from the step numbers: the two initial EDA notebooks each read raw files, and `build_eda_2.ipynb` reproduces prior preparation from raw rather than importing notebook variables. Model notebooks read named saved datasets through JSON cards. The [pipeline guide](../pipeline/README.md) explains that contract.

## Submission lineage

[assemble_task3_log_v13.ipynb](../submission/assemble_task3_log_v13.ipynb) built `kaggle_t12345_t3_log_v13.csv` from the full-data v6 prediction, holding Tasks 1, 2, 4, and 5 fixed against v9. Its user-reported **overall** Kaggle score rose **0.74593 → 0.74686**. V14–v21 keep this Task 3 column while testing Task 5 changes; the v20/v21 files are still local and unsubmitted. This overall difference is not a reported Task 3 NMAE. The [score log](../../docs/kaggle-score-log.md) records uploaded results, and [submit-plan.md](../../submit-plan.md) identifies the frozen columns for current comparisons.
