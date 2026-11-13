# Task 1 — Anti-Cheat

**Current sources:** `src/task1/eda/build_eda_original.ipynb`, `build_eda_1.ipynb`, and `src/task1/models/build_model.ipynb`. Metric: macro F₂; reported holdout and repeated-fold scores are separate comparisons.

## EDA and features

`eda_original_v1` drops 13,184 unlabeled Train rows, retains 90,816 labeled rows, all 31 raw numeric predictors and `player_id`, and leaves missing values for model-stage handling. `eda_combined_v1` adds **all 14** row-wise features: `missing_feature_count`, `account_age_days_missing`, `reports_received_missing`, `avg_session_length_min_missing`, `aiming_smoothness_missing`, `reports_per_100_days`, `daily_play_hours`, `headshot_accuracy_gap`, `aim_precision_mean`, `aim_precision_spread`, `aim_precision_observed`, `reaction_adjusted_aim`, `aim_strategy_gap`, `kd_per_teamplay`. The formulas use neither target nor Test-fitted statistics.

## Models and results

**Journey.** Round 1 used the same 20% holdout for majority, logistic regression, random forest, and HGB on original versus combined features. Macro F₂ rose from **0.45152** (majority) and **0.67733** (original logistic regression) to **0.74157** (original forest) and **0.75285** (original HGB). Small development-only searches gave **0.76680** for the tuned combined forest, but choosing an HGB decision threshold on development folds raised its holdout score to **0.77684**. Baseline engineered features alone did not help consistently.

Round 2 compared LightGBM, XGBoost, and CatBoost on both feature sets, then tuned promising configurations and their thresholds. Their best thresholded holdout scores were **0.77847** (combined LightGBM), **0.77530** (combined XGBoost), and **0.77318** (combined CatBoost); original LightGBM scored **0.77762**. The small LightGBM feature-set gap after many holdout comparisons prompted Round 3: six paired outer folds comparing the finalists and a 38-feature ablation. Round 4 checked paired errors, a fixed 50/50 blend, and remaining threshold headroom before stopping broad search.

| Comparison | Feature set | Macro F₂ |
| --- | --- | ---: |
| Thresholded HGB, best scikit-learn holdout | 45 combined | 0.77684 |
| Thresholded LightGBM, best holdout | 45 combined | 0.77847 |
| Thresholded LightGBM, next holdout | 31 original | 0.77762 |
| LightGBM, six paired outer folds | 38 selected from combined | 0.77733 mean |
| LightGBM, same outer folds | 45 combined / 31 original | 0.77679 / 0.77676 mean |
| HGB, same outer folds | 45 combined | 0.77536 mean |
| Fixed 50/50 blend, same outer folds | Pruned LightGBM + original LightGBM | 0.77841 mean |

The 38-feature model removes `account_age_days_missing`, `reports_received_missing`, `avg_session_length_min_missing`, `aiming_smoothness_missing`, `aim_precision_observed`, `missing_feature_count`, and `map_knowledge` from the combined set. The saved full-data model is `t1_lgbm_combined_pruned_cv6_full_v1`; its **0.77733** evaluation is the selected procedure's outer-CV estimate, not a new holdout score. The fixed blend adds only **0.00108** on those folds.

## Insight and next step

Threshold selection helped more than the small original-versus-engineered feature gap. `reports_received`, `friend_network_size`, `account_age_days`, and `crosshair_placement` led validation permutation importance; this indicates reliance, not causation. The notebook's Round 4 decision is to stop broad Task 1 search, use the saved 38-feature model, and revisit only a fixed-weight blend if other tasks are finished. The repeated folds reuse the same accounts, so the blend gain is a stability signal rather than independent confirmation.
