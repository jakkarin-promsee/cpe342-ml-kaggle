# Task 3 — Monthly Spending

**Current sources:** `src/task3/eda/build_eda_original.ipynb`, `build_eda_1.ipynb`, and `src/task3/models/build_model.ipynb`. Scores below are **MAE in THB, lower is better**: the supplied competition documents do not define the NMAE denominator.

## EDA and features

`eda_original_v1` keeps **104,000** labeled rows, 25,889 Test rows, and 32 raw numeric predictors. Nearly half the targets are zero; the positive tail is long. `eda_combined_v1` adds **all 16** row-wise features: `purchase_value_proxy`, `recent_to_historical_spend`, `historical_spend_per_account_month`, `transactions_per_account_month`, `historical_spend_per_transaction`, `discount_purchase_share`, `discount_value_interaction`, `purchase_recency_over_age`, `login_purchase_recency_gap`, `session_hours_proxy_per_week`, `play_hours_per_account_day`, `social_interactions_per_session`, `cross_game_activity_per_session`, `achievements_per_play_hour`, `event_social_interaction`, `source_missing_count`. Raw values and missingness remain; model preprocessing is fitted on development data.

## Models and results

| Model | Features | Holdout MAE (THB) |
| --- | --- | ---: |
| CatBoost, absolute-error objective | 48 combined | **2,928.48** |
| LightGBM, absolute-error objective | 48 combined / 32 original | 2,957.98 / 2,960.34 |
| HGB | 32 original / 48 combined | 2,959.57 / 2,997.50 |
| CatBoost | 32 original | 3,022.91 |
| Positive-only CatBoost, scored on every holdout row | 48 combined | 2,934.48 |
| Median baseline | 32 original | 10,431.34 |

Across **five development folds**, positive-only CatBoost averages **2,978.86** MAE and wins 4/5 paired folds against full CatBoost (**3,059.59**), but loses slightly on the original holdout. Full LightGBM averages **3,016.06**, reversing its holdout ordering against full CatBoost. Fold-local top-24 CatBoost averages **3,038.20**; top-8 averages **3,149.04**. Clipping negative CatBoost predictions to zero improves holdout MAE by about **0.93 THB**.

## Insight and next step

`historical_spending`, `segment`, `prev_month_spending`, `days_since_last_purchase`, `discount_value_interaction`, and `vip_status` appeared in every fold's top eight. They are predictive anchors, but their availability before the prediction cutoff is unverified. Keep full combined CatBoost as the current saved holdout leader. Test a zero-aware two-stage model or a fixed CatBoost/LightGBM blend on paired development folds; obtain the official NMAE denominator before reporting competition scores or refitting a final model.
