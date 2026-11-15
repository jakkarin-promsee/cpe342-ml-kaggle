# Task 2 — Player Segment Classification

The 12 November refresh replaced Task 2 CSVs. Use the **v2 EDA/model cards** for current work; `*_v1` artifacts and `*-outdate-12-11-2026.csv` files trace the old data. The metric is **macro F₁**. The scored combined submission still uses `t2_cat_original_no_noise_v2`; the stronger local full-data candidate `t2_cat_tuned_weighted_full_v8` is in an **unscored** v12 file. Read the [research note](../../docs/research-note-task2.md) for exact features, metrics, and temporal-provenance concerns.

## Main reading path

| Step | Notebook → artifact | What it establishes |
| --- | --- | --- |
| 1 | [build_eda_original_v2.ipynb](eda/build_eda_original_v2.ipynb) → `data/eda/eda_original_v2.json` | Reads refreshed raw Train/Test, audits quality/leakage, saves the original predictor set. |
| 2 | [build_eda_1_v2.ipynb](eda/build_eda_1_v2.ipynb) → `data/eda/eda_combined_v2.json` | Independently reads refreshed raw files and saves additional row-wise features. It does not import step 1's variables. |
| 3 | [build_model_v2.ipynb](models/build_model_v2.ipynb) → `save/t2_*_v2.json` | Compares original/combined features and model families on the refreshed data. Saved `t2_cat_original_no_noise_v2` becomes the conservative scored-submission reference: **0.75256** mean five-fold CV and **0.75041** reused holdout macro F₁. |
| 4 | [build_model_v3.ipynb](models/build_model_v3.ipynb) | Audits class-level errors and tests a probability-bias decision rule. Its holdout gain is small; it does not replace the reference. |
| 5 | [build_model_v4.ipynb](models/build_model_v4.ipynb) | Tests fixed Social/Whale class weights `[1, 1, 1.25, 1.25]` on five paired development folds; **0.75310** mean CV versus **0.75256** reference. |
| 6 | [analyze_top_model_errors_1.ipynb](eda/analyze_top_model_errors_1.ipynb) | Reads cached OOF predictions for row/group errors, confidence, tier combinations, and labels rescued or broken by weighting. Analysis only. |
| 7 | [build_eda_tier_cross_v2.ipynb](eda/build_eda_tier_cross_v2.ipynb) → one tier-cross EDA card; [build_model_v5.ipynb](models/build_model_v5.ipynb) | Tests one `loyalty_engagement_cross` feature motivated by the audit. The mean CV gain is negligible and holdout falls, so this branch stops. |
| 8 | [build_model_v6.ipynb](models/build_model_v6.ipynb) → [build_model_v7.ipynb](models/build_model_v7.ipynb) | Bounded CatBoost tuning, then the same fixed class weights on the tuned configuration. Tuned weighted mean five-fold CV is **0.754156**, only **0.000061** above tuned unweighted. |
| 9 | [build_model_v8.ipynb](models/build_model_v8.ipynb) | Refits the tuned weighted procedure on all labeled Train rows as `t2_cat_tuned_weighted_full_v8`. Its evaluation records the earlier CV **procedure estimate**, not a score on the full-data fit. |
| 10 | [build_model_v9.ipynb](models/build_model_v9.ipynb) | Checks fixed CatBoost/LightGBM/XGBoost soft ensembles on paired folds. Best ensemble **0.752148**, below tuned weighted CatBoost on all five folds; no ensemble replaces v8. |

The earlier [v1 original EDA](eda/build_eda_original.ipynb), [v1 combined EDA](eda/build_eda_1.ipynb), and [v1 model notebook](models/build_model.ipynb) belong to the outdated CSVs. They are useful for history, not for current inference.

## Submission and interpretation

[assemble_task2_tuned_v12.ipynb](../submission/assemble_task2_tuned_v12.ipynb) builds `kaggle_t12345_t2_tuned_v12.csv` from the full-data v8 candidate, changing only Task 2 relative to scored v9. It is **local and unscored**; scored v17 and the current Task 5 candidate slots retain the older `t2_cat_original_no_noise_v2` column. Do not treat the v8 CV gain or reused holdout gain as a measured Kaggle improvement. The creation times of `vip_tier`, `engagement_level`, and `loyalty_tier` relative to `segment` remain undocumented; the [leakage audit](../../docs/leakage-audit-top-models.md) records the limit. Use the [pipeline guide](../pipeline/README.md) for the EDA/model card interface and the [score log](../../docs/kaggle-score-log.md) for uploaded overall results.
