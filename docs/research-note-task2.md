# Task 2 — Player Segments

**Current sources:** `src/task2/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, and `src/task2/models/build_model_v2.ipynb`. Older Task 2 notebooks are excluded. Metric: validation macro F₁.

## EDA and features

`eda_original_v2` keeps all **104,000** labeled rows and 44 raw predictors (34 numeric, 10 categorical); Train/Test IDs and schemas are checked. Missingness stays for development-fitted preprocessing. `random_metric_1`, `random_metric_2`, and `random_metric_3` are documented noise and are ablated in modeling. `eda_combined_v2` retains the raw inputs and adds **all 14** features: `session_time_proxy_hours`, `play_hours_per_account_day`, `login_recency_over_age`, `spend_per_play_hour`, `spend_per_account_day`, `spend_per_purchase_proxy`, `purchase_to_play_frequency`, `social_actions_per_friend`, `team_chat_interaction`, `social_preference_interaction`, `ranked_time_proxy_hours`, `ranked_win_interaction`, `cosmetic_vs_performance`, `source_missing_count`.

`play_frequency` exceeds seven although the datacard describes days/week, so derived time products are relative proxies. `loyalty_tier`, `engagement_level`, and `vip_tier` have unknown creation times.

## Models and results

| Model | Features | Holdout macro F₁ |
| --- | --- | ---: |
| CatBoost `t2_cat_original_no_noise_v2` | Original, excluding three noise columns | **0.75041** |
| CatBoost `t2_cat_combined_no_noise_v2` | Combined, excluding noise | 0.74976 |
| Logistic regression | Original / combined | 0.74749 / 0.74666 |
| Tuned HGB, excluding noise | Original / combined | 0.74589 / 0.74612 |
| LightGBM, excluding noise | Original / combined | 0.74558 / 0.74485 |
| XGBoost, excluding noise | Original / combined | 0.74416 / 0.74386 |
| Random forest | Original / combined | 0.72733 / 0.72786 |

On **five paired development folds**, full CatBoost scores **0.75256 ± 0.00471**, full HGB **0.74478 ± 0.00345**, and fold-local pruned CatBoost **0.75220 ± 0.00314**. CatBoost wins every paired fold against HGB; pruning improves only 2/5 folds. These values are from the executed output in the current notebook; its final Markdown paragraph still has older numbers and should be updated.

Removing `loyalty_tier` drops CatBoost holdout F₁ to **0.6544**, removing `engagement_level` to **0.7170**, and removing both to **0.6196**; combined no-tier features score **0.6190**. The no-tier model relies most on `vip_tier`, `spending_frequency`, `avg_monthly_spending`, `days_since_last_login`, and `total_spending_thb`.

## Insight and next step

The 14 engineered features do not consistently improve matched models. Keep full original-feature CatBoost as the competition-data leader. Resolve when the three tier fields are created before using this as a deployment model; the ablation proves reliance, not leakage. If tiers are unavailable at prediction time, compare a separately validated no-tier model. Update the stale Round 2–4 prose in the notebook to match its executed tables.
