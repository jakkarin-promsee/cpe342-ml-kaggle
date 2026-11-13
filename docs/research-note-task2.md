# Task 2 — Player Segments

**Current sources:** `src/task2/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, and `src/task2/models/build_model_v2.ipynb` plus `build_model_v3.ipynb`. Older Task 2 notebooks are excluded. Metric: validation macro F₁.

## EDA and features

`eda_original_v2` keeps all **104,000** labeled rows and 44 raw predictors (34 numeric, 10 categorical); Train/Test IDs and schemas are checked. Missingness stays for development-fitted preprocessing. `random_metric_1`, `random_metric_2`, and `random_metric_3` are documented noise and are ablated in modeling. `eda_combined_v2` retains the raw inputs and adds **all 14** features: `session_time_proxy_hours`, `play_hours_per_account_day`, `login_recency_over_age`, `spend_per_play_hour`, `spend_per_account_day`, `spend_per_purchase_proxy`, `purchase_to_play_frequency`, `social_actions_per_friend`, `team_chat_interaction`, `social_preference_interaction`, `ranked_time_proxy_hours`, `ranked_win_interaction`, `cosmetic_vs_performance`, `source_missing_count`.

`play_frequency` exceeds seven although the datacard describes days/week, so derived time products are relative proxies. `loyalty_tier`, `engagement_level`, and `vip_tier` have unknown creation times.

## Models and results

**Journey.** Round 1 held out the same 20% of rows and compared majority (**0.14127** macro F₁), logistic regression, random forest, and HGB. Dropping the three documented noise fields and a small HGB search did not beat original-feature logistic regression (**0.74749**). Round 2 compared LightGBM, XGBoost, and CatBoost with noise removed on both datasets; original-feature CatBoost became the holdout leader (**0.75041**). Round 3 removed the two tier fields one at a time and together because their availability at prediction time was unclear. Round 4 used five paired development folds to check CatBoost against HGB and fold-local feature pruning. The v3 follow-up inspected class errors and tested probability offsets without changing the feature set.

| Model | Features | Holdout macro F₁ |
| --- | --- | ---: |
| CatBoost `t2_cat_original_no_noise_v2` | Original, excluding three noise columns | **0.75041** |
| CatBoost `t2_cat_combined_no_noise_v2` | Combined, excluding noise | 0.74976 |
| Logistic regression | Original / combined | 0.74749 / 0.74666 |
| Tuned HGB, excluding noise | Original / combined | 0.74589 / 0.74612 |
| LightGBM, excluding noise | Original / combined | 0.74558 / 0.74485 |
| XGBoost, excluding noise | Original / combined | 0.74416 / 0.74386 |
| Random forest | Original / combined | 0.72733 / 0.72786 |

On **five paired development folds**, full CatBoost scores **0.75256 ± 0.00471**, full HGB **0.74478 ± 0.00345**, and fold-local pruned CatBoost **0.75220 ± 0.00314**. CatBoost wins every paired fold against HGB; pruning improves only 2/5 folds. The notebook interpretation now matches these executed results.

Removing `loyalty_tier` drops CatBoost holdout F₁ to **0.6544**, removing `engagement_level` to **0.7170**, and removing both to **0.6196**; combined no-tier features score **0.6190**. The no-tier model relies most on `vip_tier`, `spending_frequency`, `avg_monthly_spending`, `days_since_last_login`, and `total_spending_thb`.

## Class-bias decision check

`build_model_v3.ipynb` reproduces the saved CatBoost holdout score exactly, then refits its configuration on the same five development folds to collect out-of-fold probabilities. The raw fold scores exactly reproduce the prior CV record. Class 0 has holdout F1 **0.8179**, while classes 1, 2, and 3 have **0.7454**, **0.7031**, and **0.7352**; class 2 recall is **0.6679** and class 3 recall is **0.6924**. These classes are undercalled relative to class 0.

A small log-probability offset was selected for classes 1-3 on development folds, with class 0 fixed at zero. Leave-one-fold-out selection improved mean checked-fold macro F1 by only **0.00020**, winning **3/5** folds. The all-development offsets `[0, 0.15, 0.30, 0.50]` raised same-OOF F1 from **0.75257** to **0.75425**; this tuned-on-the-same-rows difference is optimistic. The separately checked, reused 20% holdout improved from **0.75041** to **0.75222**, changing **935/20,800** predictions. Class 3 holdout F1 rose from **0.7352** to **0.7383**, but class 0 fell from **0.8179** to **0.8166**. A wider offset grid on the same cached development probabilities gains only **0.00042** mean macro F1 over raw predictions in a leave-one-fold-out check, winning **3/5** folds. The gain is small, so this is exploratory evidence, not a robust new submission choice. The saved wrapper is `t2_cat_class_bias_cv5_v3` and still contains the development-trained baseline; a full-train refit would be needed for submission.

## Insight and next step

The 14 engineered features do not consistently improve matched models. Keep full original-feature CatBoost as the competition-data leader. Resolve when the three tier fields are created before using this as a deployment model; the ablation proves reliance, not leakage. If tiers are unavailable at prediction time, compare a separately validated no-tier model. Further class-bias tuning has diminishing returns; a class-weight check would require new fold fits and should follow higher-value experiments in other tasks.
