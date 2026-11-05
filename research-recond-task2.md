# Task 2: Player Segment Classification

**Goal:** Classify each player as Casual (0), Grinder (1), Social (2), or Whale (3). Competition metric: **macro F₁**.

## Data and EDA

- Raw Train: **104,000 rows**, 44 features, `player_id`, and complete `segment` labels. Test: **25,889 rows**, the same 44 features and `player_id`, without labels.
- Class counts: Casual **40,934**, Grinder **26,034**, Social **21,014**, Whale **16,018**. Use a stratified split and inspect every class, not accuracy alone.
- Features: **34 numeric** and **10 string categorical**. Many columns have missing values; `payment_method` and `region` are missing most often. There are no duplicate or overlapping IDs, infinite numeric values, constant columns, or unseen Test categories at this stage.
- `player_type_tag` does not directly encode the label. `engagement_level` and `loyalty_tier` correlate with the label but do not determine it. Their exact availability time is not documented, so operational leakage cannot be ruled out from this static dataset alone.
- `src/task2/eda/eda.ipynb` saves all 44 original features in `eda_original_v1`. Imputation and encoding are fitted inside model Pipelines, never in EDA or on Test.

## Round 1 models

All scores use the same stratified **80/20** development/validation split (`SEED=42`). The HGB hyperparameter search uses 3-fold CV on development only. Every saved model has its own JSON card and evaluation.

| Model | Macro F₁ | Casual | Grinder | Social | Whale |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_hgb_no_noise_v1` | **0.74944** | 0.8207 | 0.7394 | 0.7133 | 0.7243 |
| `t2_hgb_original_v1` | 0.74654 | 0.8182 | 0.7368 | 0.7102 | 0.7210 |
| `t2_hgb_no_noise_tuned_v1` | 0.74555 | 0.8200 | 0.7382 | 0.7065 | 0.7175 |
| `t2_logreg_original_v1` | 0.74376 | 0.7985 | 0.7375 | 0.7022 | **0.7369** |
| `t2_rf_original_v1` | 0.72933 | 0.7964 | 0.7154 | 0.6917 | 0.7138 |
| `t2_dummy_original_v1` | 0.14122 | 0.5649 | 0 | 0 | 0 |

Removing the three `random_metric_*` columns documented as noise increased HGB validation macro F₁ by **0.00290** at matched baseline settings. The small random search did not improve HGB on the holdout. The winner's permutation importance is led by `loyalty_tier` (drop **0.2195**), `engagement_level` (**0.0657**), and `vip_tier` (**0.0335**), followed by login recency, playtime, and spending. Importance describes this fitted model's reliance on columns, not causation.

**Next:** Treat this as a first selection estimate. Compare shortlisted models on repeated folds, review Social and Whale errors, and test stronger handling of categorical features. Do not refit on all labels or produce a final Task 2 submission until the model choice is more stable.
