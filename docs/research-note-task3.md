# Task 3 — Monthly Spending

**Current sources:** `src/task3/eda/build_eda_original.ipynb`, `build_eda_1.ipynb`, `src/task3/models/build_model.ipynb`, and `build_model_v2.ipynb`. Scores below are **MAE in THB, lower is better**. The supplied competition documents do not define the NMAE denominator, so these are not official NMAE scores.

## EDA and features

`eda_original_v1` keeps **104,000** labeled rows, 25,889 Test rows, and 32 raw numeric predictors. Nearly half the targets are zero and the positive tail is long. `eda_combined_v1` retains those raw columns and adds exactly **16** row-wise features: `purchase_value_proxy`, `recent_to_historical_spend`, `historical_spend_per_account_month`, `transactions_per_account_month`, `historical_spend_per_transaction`, `discount_purchase_share`, `discount_value_interaction`, `purchase_recency_over_age`, `login_purchase_recency_gap`, `session_hours_proxy_per_week`, `play_hours_per_account_day`, `social_interactions_per_session`, `cross_game_activity_per_session`, `achievements_per_play_hour`, `event_social_interaction`, and `source_missing_count`. Missing-value imputation and categorical encoding are fitted within development or CV training rows.

## Models and results

| Procedure | Five development folds, mean MAE | Original 20% holdout MAE |
| --- | ---: | ---: |
| Full combined-feature CatBoost, absolute-error objective | 3,059.59 | 2,928.48 raw; **2,927.55 clipped** |
| Full combined-feature LightGBM, absolute-error objective | 3,015.85 | 2,957.77 clipped |
| Positive-only combined-feature CatBoost | 2,978.86 | 2,934.38 clipped |
| Fixed CatBoost/LightGBM blend, 40%/60% | **2,989.46** | **2,907.13** |
| Nonzero CatBoost classifier + positive-only CatBoost amount, probability cutoff 0.50 | **2,903.13** | **2,857.64** |
| LightGBM nonzero classifier + same positive-only CatBoost amount, cutoff 0.50 | **2,903.70** | **2,858.11** |
| Fixed 80% zero-aware model / 20% CatBoost-LightGBM blend | **2,895.92** | **2,845.10** |

The v2 notebook selected the blend weight and nonzero probability cutoff on five stratified development folds, then checked each once on the existing holdout. The gate threshold is near the fold-local optima (0.50 or 0.55), and the two-stage candidate beats the CatBoost/LightGBM blend in mean development-fold MAE by **86.33 THB**. A follow-up fixed mixture of those two selected candidates improves mean CV MAE by another **7.21 THB** against the zero-aware model alone, winning on **4/5** paired folds. On the holdout it improves over clipped full CatBoost by **82.45 THB**. The holdout has already informed earlier Task 3 work, so this remains exploratory evidence rather than an untouched final test.

The two-stage model trades worse zero-target MAE (**1,847.19** versus **1,736.84** for the CatBoost/LightGBM blend) for better positive-target MAE (**3,795.04** versus **3,992.82**) and lower MAE among targets at least ฿50,000 (**18,959.42** versus **20,088.78**). The final 80/20 mixture reduces zero-target MAE to **1,825.12** and positive-target MAE to **3,791.35** on the holdout. Existing fold-local feature rankings repeatedly placed `historical_spending`, `segment`, `prev_month_spending`, `days_since_last_purchase`, `discount_value_interaction`, and `vip_status` in the top eight. Their availability before prediction time remains unverified.

A later check multiplied the gate's positive amount by a single scale selected on four development folds and evaluated on the fifth. This **worsened** MAE on all **5/5** outer folds, by **3.26 THB** on average relative to the fixed mixture. Selecting on all development rows chose **0.975** and changed its same-row OOF MAE only from **2,895.92** to **2,895.85**. On the reused holdout it worsened MAE from **2,845.10** to **2,845.22**. It lowered zero-target MAE but increased error for targets at least THB 50,000, so a global amount correction is not useful.

A fold-paired positive-amount regressor swap also failed. Replacing the CatBoost positive-only amount model with LightGBM L1 worsened the fixed mixture's mean development-fold MAE from **2,895.92** to **2,940.83 THB**, losing on **5/5** folds; its tail (target at least THB 50,000) MAE rose from **18,923.12** to **19,600.94 THB**. A fixed 50/50 CatBoost/LightGBM amount blend scored **2,902.35 THB**, losing on **4/5** folds. These comparisons reused the same fold-local occurrence probabilities and mixture settings, so the change isolates the amount model. No holdout score or saved model was created for these losing variants.

In the selected mixture's out-of-fold development audit, targets at least ฿50,000 are **8.16%** of rows but contribute **53.30%** of total absolute error. Zero targets are **48.12%** of rows and contribute **32.04%**. The occurrence gate's false-nonzero cases are **13.04%** of rows but account for **31.96%** of total absolute error; false-zero cases are **15.24%** of rows but only **2.52%** of error. This points to reducing costly false spending predictions and modeling the large positive tail, not merely improving binary occurrence accuracy.

## Insight and next experiment

The fixed mixture `t3_gate_blend_cv5_v2` is the strongest measured Task 3 candidate. Its components are trained only on development rows, with fitted preprocessing included in the saved models. The simpler `t3_positive_gate_cv5_v2` is a close backup. Replacing only its occurrence classifier with LightGBM changes CV and holdout MAE by less than **1 THB**; classifier family is not the current bottleneck. Before reporting competition NMAE, obtain the official denominator. The simple amount scale and positive-only LightGBM alternatives failed. Any later Task 3 experiment should target the false-nonzero and high-spend errors jointly, using paired development folds and avoiding further tuning on the reused holdout.
