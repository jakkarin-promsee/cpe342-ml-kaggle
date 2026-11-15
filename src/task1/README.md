# Task 1 — Anti-Cheat Pre-Filter

Read this as an artifact path, not as a command to rerun every notebook. The 12 November data refresh **did not change Task 1**. The selected full-data model is `t1_lgbm_combined_pruned_cv6_full_v1`; the competition metric is **macro F₂**. See the [research note](../../docs/research-note-task1.md) for exact feature names, fold scores, and limitations, and the [pipeline guide](../pipeline/README.md) for the saved-card contract.

## Main reading path

| Step | Notebook → saved artifact | What to read next |
| --- | --- | --- |
| 1 | [build_eda_original.ipynb](eda/build_eda_original.ipynb) → `data/eda/eda_original_v1.json` | Raw Train/Test checks; removes unlabeled Train rows and saves the 31 supplied predictors. |
| 2 | [build_eda_1.ipynb](eda/build_eda_1.ipynb) → `data/eda/eda_combined_v1.json` | Reads raw files and saves the original predictors plus 14 deterministic row-wise features. This is a separate EDA artifact, not a dependency on step 1's notebook state. |
| 3 | [build_model.ipynb](models/build_model.ipynb) → `save/*.json` | Reads both EDA cards. Follow its four rounds: baseline scikit-learn models → boosted trees and thresholds → six paired outer folds and feature ablation → errors/blends. The selected 38-feature LightGBM is refitted on all labeled Train as `t1_lgbm_combined_pruned_cv6_full_v1`. Its **0.77733** macro F₂ is the procedure's six-fold estimate, not a fresh score for the full-data fit. |
| 4 | [analyze_top_model_errors_1.ipynb](eda/analyze_top_model_errors_1.ipynb) | Reads saved OOF predictions; studies persistent false positives/negatives, margins, raw-feature tails, and error slices. Analysis only; no new dataset/model. |
| 5 | [build_eda_2.ipynb](eda/build_eda_2.ipynb) → `data/eda/eda_error_context_v2.json` | Starts again from raw Train/Test, reproduces earlier row-wise features, and adds eight context/behavior candidates motivated by step 4. One new EDA dataset; no notebook-variable dependency. |
| 6 | [build_model_v3.ipynb](models/build_model_v3.ipynb) | Reads the v2 error-context card and compares new feature groups, LightGBM capacity settings, matched thresholds, and exploratory blends against the frozen 38-feature reference on paired folds. Improvements were too small to replace it. |
| 7 | [build_model_v4.ipynb](models/build_model_v4.ipynb) | Tests the class-imbalance hint with weighted LightGBM on the selected feature set. No material gain; the original full-data model remains selected. |

The OOF analysis and v2 error-context EDA explain *why* later tests were tried. Their saved artifacts do not supersede the earlier selected model. Thresholds explored on reused OOF labels are exploratory, not independent validation.

## Submission lineage

The Task 1 column in the scored combined submissions is frozen from `t1_lgbm_combined_pruned_cv6_full_v1`. The Task 5 and Task 3 comparisons in [submit-plan.md](../../submit-plan.md) keep that column unchanged. Read the model's JSON card in `save/` for its EDA link and the combined submission's adjacent JSON card for the exact source and hash. [The score log](../../docs/kaggle-score-log.md) contains overall Kaggle results; it does not provide a separate Task 1 F₂.
