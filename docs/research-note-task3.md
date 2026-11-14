# Task 3 — Monthly Spending

**Current sources:** `src/task3/eda/build_eda_original.ipynb`, `build_eda_1.ipynb`, `analyze_top_model_errors_1.ipynb`, `build_eda_2.ipynb`, `src/task3/models/build_model.ipynb`, `build_model_v2.ipynb`, `build_model_v3.ipynb`, `build_model_v4.ipynb`, `build_model_v5.ipynb`, `build_model_v6.ipynb`, and `build_model_v7.ipynb`. Scores below are **MAE in THB, lower is better**. The instruction-slide extract gives `1 / (1 + Σ|y−ŷ| / Σy)` as NMAE, but the official Kaggle implementation and task weights have not been confirmed. MAE preserves model ordering under that formula on a fixed validation set; the values below are not official Kaggle scores.

## EDA and features

`eda_original_v1` keeps **104,000** labeled rows, 25,889 Test rows, and 32 raw numeric predictors. Nearly half the targets are zero and the positive tail is long. `eda_combined_v1` retains those raw columns and adds exactly **16** row-wise features: `purchase_value_proxy`, `recent_to_historical_spend`, `historical_spend_per_account_month`, `transactions_per_account_month`, `historical_spend_per_transaction`, `discount_purchase_share`, `discount_value_interaction`, `purchase_recency_over_age`, `login_purchase_recency_gap`, `session_hours_proxy_per_week`, `play_hours_per_account_day`, `social_interactions_per_session`, `cross_game_activity_per_session`, `achievements_per_play_hour`, `event_social_interaction`, and `source_missing_count`. Missing-value imputation and categorical encoding are fitted within development or CV training rows.

## Models and results

**Journey.** Round 1 compared a median dummy (**10,431.34 THB** holdout MAE), linear/ridge (**about 5,708–5,712**), median quantile regression (**about 4,459–4,464**), HGB (**2,959.57** original), LightGBM (**2,957.98** combined), and CatBoost on matched rows. Combined-feature CatBoost led that holdout at **2,928.48 THB** before clipping. Round 2 used five paired development folds: positive-only CatBoost improved mean fold MAE but slightly lost on the holdout; LightGBM beat full CatBoost on folds despite losing on the holdout. Fold-local top-24 feature selection improved four folds but lost badly on one; top-8 lost on every fold. Clipping negative predictions gave a small consistent gain. The v2 notebook then tested a fixed CatBoost/LightGBM blend, an occurrence gate plus positive amount model, their mixture, a different gate classifier, amount scaling, and positive-amount regressor swaps in that order.

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

## Deep-learning round (v3)

`build_model_v3.ipynb` trains three PyTorch tabular networks **from scratch** on the saved `eda_combined_v1` dataset: plain MLP with direct MAE objective, residual MLP with direct MAE, and residual MLP with separate nonzero-probability and positive-amount heads. Numeric inputs use median imputation, missing indicators, signed-log transformation, and scaling; four nominal inputs use one-hot encoding. Preprocessing is fitted inside training rows, each network uses an internal 12% early-stopping split, and every model has distinct saved model/evaluation cards. The zero-aware network chooses its gate threshold by leave-one-development-fold-out checks, then fits on the full development split. No pre-trained weights, external data, or Test rows are used.

The notebook reproduces the existing mixture's **2,895.92 THB mean MAE** on the same five development folds from saved OOF predictions. All neural candidates lose on every paired fold:

| Model | Five-fold development mean MAE (THB) | Reused 20% holdout MAE (THB) | Δ CV vs existing mixture |
| --- | ---: | ---: | ---: |
| Existing gate/blend mixture | **2,895.92** | **2,845.10** | — |
| Plain direct MLP | 3,168.78 | 3,039.08 | +272.86 |
| Residual zero-aware MLP | 3,215.60 | 3,091.39 | +319.68 |
| Residual direct MLP | 3,257.12 | 3,219.51 | +361.20 |

The plain MLP is the best neural option, yet on the reused holdout targets ≥฿50,000 contribute **56.66%** of its total absolute error. The gap is large and consistent across folds, so further broad neural hyperparameter search is low priority. These MAE values are not competition NMAE; the holdout remains reused, not independent final confirmation. The next step is an OOF error audit of the existing mixture, then one targeted EDA/model experiment if that audit reveals an actionable pattern.

## Selected-model OOF error audit

`src/task3/eda/analyze_top_model_errors_1.ipynb` reconstructs the selected 80/20 gate/blend mixture from saved model cards and all five development OOF files, exactly reproducing **2,895.92 THB mean fold MAE**. It fits nothing and reads no Test rows. Targets ≥฿50,000 are **6,786/83,200 rows (8.16%)** but contribute **53.30%** of total absolute error; they are underpredicted **58.8%** of the time, with median predicted/actual ratio **0.96**. Zero targets are **48.12%** of rows and contribute **32.04%** of error.

The gate's **10,849 false-nonzero cases (13.04% of development rows)** contribute **31.96%** of final-mixture error and cost **7,097 THB MAE** within that group. False-zero cases are **15.24%** of rows but only **2.52%** of error, because their true spend is usually small. Gate-probability bins are fairly calibrated descriptively, so another global threshold search lacks a clear rationale. The final mixture improves the 50k+ group over the continuous CatBoost/LightGBM blend (**18,923 versus 20,376 THB MAE**) but is slightly worse than the gate alone there (**18,722 THB**). It lowers zero-target MAE relative to the gate (**1,928 versus 1,953 THB**) but remains worse than the continuous blend (**1,829 THB**).

Within the false-nonzero group, the highest previous-month-spending quintile (roughly **฿1,823–63,064**) contains **2,084** cases but accounts for **74.5% of that group's error**; the highest historical-spending quintile similarly accounts for **74.1%**. These feature slices overlap and must not be summed. This points to a testable *persistence versus lapse* hypothesis: large prior spend alone can produce a very costly false positive when current spending stops. A new EDA dataset should encode recent spending together with purchase recency/frequency, then compare paired development folds. High-spend positive underprediction remains a separate issue; no global amount scale or positive-only LightGBM swap helped earlier.

## Spending-persistence EDA (v2)

`build_eda_2.ipynb` reads raw competition Train/Test, reproduces the 16 prior deterministic features, and saves exactly one dataset, `eda_spend_persistence_v2`, with eight new row-wise candidates. **Persistence family:** `recent_spend_per_transaction` = previous-month spend/(transactions + 1); `spend_persistence_proxy` = previous-month spend × purchase frequency/(days since purchase + 1); `prior_spend_staleness` = previous-month spend × days since purchase/(account age + 30); `purchase_frequency_per_recency` = purchase frequency/(days since purchase + 1). **Tail family:** `historical_prev_spend_gap` = historical − previous-month spend; `prior_spend_per_account_month` = previous-month spend/(account age/30 + 1); `discounted_prev_spend_proxy` = previous-month spend × discount purchase share; `prev_spend_login_recency` = previous-month spend/(days since login + 1). None uses target, ID, or fitted statistics. Input availability before prediction remains unverified for the same source columns as earlier EDA.

In the highest previous-month-spend quintile (cutoff ~฿1,823 on development rows), **13.3%** have zero next-month spending; the existing gate predicts nonzero for all of those zero cases. Within this high-prior-spend group, the observed zero rate rises from **10.4%** to **17.6%** across purchase-recency quartiles and falls from **16.1%** to **10.8%** across purchase-frequency quartiles. These are descriptive associations and motivate testing the new persistence inputs in the occurrence gate. The next model round will separately test new gate and new positive-amount features against the existing fixed 80/20 mixture on paired development folds.

## Persistence and tail model ablation (v4)

`build_model_v4.ipynb` changes one saved-mixture component at a time, using the same five development folds, 0.50 gate threshold, 80/20 final mixture, and other fitted components. Both candidates have distinct saved model/evaluation cards linked to `eda_spend_persistence_v2`. The new gate fits CatBoost Logloss on the original 48 features plus the four persistence features; the new positive-only amount model fits CatBoost MAE on the original 48 plus four tail features. Preprocessing is fitted only inside each fold's training rows. All holdout scores are on the already reused 20% split.

| Procedure | Five-fold development mean MAE (THB) | Reused holdout MAE (THB) | Paired fold result |
| --- | ---: | ---: | --- |
| Existing `t3_gate_blend_cv5_v2` | **2,895.92** | 2,845.10 | Reference |
| New persistence gate `t3_gate_persistence_cv5_v4` | 2,895.81 | 2,845.08 | Better 4/5 folds, mean −0.12 THB |
| New tail amount `t3_amount_tail_cv5_v4` | 2,910.15 | 2,852.17 | Worse 4/5 folds, mean +14.23 THB |

The new gate fixes **242** old false-nonzero decisions but creates **232** new ones among 83,200 development rows; false-nonzero count falls from **10,849** to **10,839** and high-prior-spend false-nonzero count from **2,084** to **2,082**. Its high-prior-spend zero-target MAE improves only **฿2.20** (27,525.97 → 27,523.77), with unchanged 50k+ MAE. The new amount model worsens 50k+ MAE from **18,923.14** to **19,040.98 THB**, positive-target MAE from **3,793.93** to **3,818.74**, and zero-target MAE from **1,927.89** to **1,930.72**. Thus the added ratios do not unlock the two dominant error groups. The tiny gate gain is exploratory and below a useful practical threshold; keep the old mixture as the main Task 3 model. The evaluation index numerically selects the new gate under `MAE_thb` because its reused-holdout score is lower by **฿0.015**, but that difference is not a meaningful model-selection reason. Further arbitrary row-wise ratios, DL hyperparameter sweeps, or global amount scaling are low priority. New competition-provided timing evidence or an organizer clarification about target generation would be needed to justify another large Task 3 search; Test and leaderboard probing remain off limits.

## Cost-weighted gate follow-up

One focused follow-up in `build_model_v4.ipynb` kept the original 48 predictors and CatBoost occurrence settings but gave **3× training weight** to zero-target accounts above each training fold's 80th percentile of previous-month spending. The rule was fixed before fold results; the positive-only amount model, continuous blend, 0.50 gate threshold, and final 80/20 weight stayed unchanged. `t3_weighted_gate_cv5_v4` scores **2,926.42 THB mean development-fold MAE** versus **2,895.92** for the reference, losing all **5/5** paired folds by **30.50 THB** on average; reused-holdout MAE is **2,876.51** versus **2,845.10**. False-nonzero decisions fall from **10,849** to **9,993**, but false-zero decisions rise from **12,676** to **14,352**. Zero-target MAE improves from **1,927.89** to **1,849.84 THB**, while positive-target MAE worsens from **3,793.93** to **3,925.12**. The target tail ≥฿50,000 stays unchanged because the gate already passes those cases. This demonstrates the gate tradeoff directly: reducing false-nonzero cost by suppressing positive predictions loses more elsewhere. A weight/threshold grid on the same OOF data would risk tuning this tradeoff to noise. Stop this branch and keep the original mixture as the practical reference.

## Log-positive-target hint check (v5) and full-data refit (v6)

`build_model_v5.ipynb` changes **only** the positive-only CatBoost amount model: it fits the same MAE pipeline on `log1p(spending_30d)` for positive training rows and inverts predictions with `expm1`. The existing occurrence gate, probability cutoff 0.50, continuous CatBoost/LightGBM 40/60 blend, and final 80/20 mixture stay fixed. All preprocessing is fitted inside the same five development folds as before; no Test rows or leaderboard scores were used to select this model.

| Procedure | Five-fold mean MAE (THB) | Reused 20% holdout MAE (THB) |
| --- | ---: | ---: |
| Existing gate/blend mixture | 2,895.92 | 2,845.10 |
| Log-positive-amount mixture `t3_log_positive_amount_cv5_v5` | **2,814.97** | **2,780.94** |

The log-target variant improves MAE on **all 5/5 paired folds**, by **80.95 THB** on average, and by **64.16 THB** on the reused holdout. In pooled development OOF rows, zero-target MAE changes **1,927.89 → 1,923.42**, positive-target MAE **3,793.93 → 3,642.03**, and target ≥฿50,000 MAE **18,923.14 → 18,294.04**. The fixed gate's false-zero rows are unaffected; its false-nonzero rows improve only **16.50 THB** within that group. Most benefit comes from predicting positive amounts and the tail more accurately. This is a meaningful local improvement, though the holdout has been consulted repeatedly and the competition NMAE gain remains unknown.

`build_model_v6.ipynb` refits the selected occurrence, log-positive-amount, and both continuous regressors on all 104,000 labeled Train rows as `t3_log_positive_amount_full_v6`. Its `MAE_thb_cv5_procedure` evaluation records the earlier development-fold mean, **not** a fresh score for the full-data model. The linked per-task Test prediction draft is `src/task3/submission/t3_log_positive_amount_full_v6.csv`. The controlled combined submission v13 changes only Task 3 relative to scored v9, keeping Tasks 1, 2, 4, and 5 identical. The user-reported overall Kaggle score rose **0.74593 → 0.74686 (+0.00093)**. This supports a small leaderboard improvement from the Task 3 change; it is not a reported Task 3 NMAE, and the official task weights remain unknown.

## Log-target loss and tail follow-up (v7, 14 November)

`build_model_v7.ipynb` tests one targeted hypothesis: after `log1p` improved the positive amount model, change only its CatBoost objective from MAE to RMSE to see whether it reduces the remaining high-spend underprediction. The occurrence gate, positive-only training rows, preprocessing, 0.50 cutoff, continuous 40/60 blend, and final 80/20 mixture remain fixed. It also checks a **prespecified 50/50 blend** of the independently fold-fitted log-MAE and log-RMSE positive amount predictions. All five development folds are paired with v5; the original holdout is reused and secondary. Both development-fitted candidates have distinct saved model/evaluation cards.

| Positive amount model | Five-fold mean MAE (THB) | Reused holdout MAE (THB) | Paired fold wins vs log-MAE |
| --- | ---: | ---: | ---: |
| Selected log-MAE v5 | **2,814.97** | 2,780.94 | — |
| Log-RMSE `t3_log_rmse_amount_cv5_v7` | 2,833.45 | 2,794.84 | 0/5 |
| Fixed half blend `t3_log_amount_halfblend_cv5_v7` | 2,811.16 | **2,777.26** | 4/5 |

Log-RMSE alone loses on every fold by **18.48 THB** mean MAE. The half blend improves mean MAE by only **3.81 THB** and reused holdout by **3.68 THB** relative to log-MAE. Its OOF tail ≥฿50,000 MAE improves **18,294.04 → 18,123.58 THB**, but zero-target MAE worsens **1,923.42 → 1,932.42** and the ฿10,000–50,000 group worsens **4,358.16 → 4,483.44**. This is a small, mixed tradeoff on already reused folds. Keep the full-data v6 log-MAE model and scored v13 combined submission as the practical Task 3 choice; do not refit the half blend for submission or tune blend weights on these folds. The remaining tail bias is descriptive evidence for a separate future experiment, not proof that a global upward correction will help; prior global scaling already failed.
