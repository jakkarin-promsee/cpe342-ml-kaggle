# Task 1: Anti-Cheat Pre-Filter

**Problem:** Automated monitoring flags 2,000 suspicious accounts daily, but 60–70% are false positives involving skilled legitimate players. Manual review can handle only 200 cases per day, creating a backlog of more than 8,000 accounts.

**Goal:** Build a binary classifier that separates true cheaters from false positives among flagged accounts.

**Metric:** Macro F₂ score.

**Data**:
- **train (with labels):** 32 columns / 104,000 rows /  399,347 cell missing 
- **test:** 31 columns / 25,889 rows / 64,190 cell missing 

Default Clean:
- Drop rows with missing `is_cheater` target from labeled training: 104,000 -> 90,816. Keep the 13,184 unlabeled rows out of supervised training and validation.

## Explore

![[Pasted image 20261105200358.png]]


![[Pasted image 20261105200627.png]]

![[Pasted image 20261105200652.png]]

## Feature Extraction 1 (14 feature)

1. Missing flag (true/false) for: ["account_age_days", "reports_received", "avg_session_length_min", "aiming_smoothness"]
2. reports_per_100_days
3. daily_play_hours
4. `headshot_accuracy_gap = headshot_percentage - accuracy_score`
5. Aim score from ["aiming_smoothness", "spray_control_score", "crosshair_placement", "input_consistency_score"]
	1. `aim_precision_mean` (mean of the four available values)
	2. `aim_precision_spread` (max minus min of the four available values)
	3. `aim_precision_observed` (non-null count among the four values)
6. reaction_adjusted_aim (aiming_smoothness / reaction_time_ms)
7. aim_strategy_gap 


## Train model 1
> 80/20 holdout splitting
> `is_cheater` "0/1" : "65.07% / 34.93%"

**Metric:** macro F_2

**Model (All try):**
- DummyClassifier
- LogisticRegression
- RandomForestClassifier
- HistGradientBoostingClassifier
- LightGBM
- CatBoost
- XGBoost

**Tune Target (with RandomizedSearchCV and decision threshold)**:
- RandomForestClassifier
- HistGradientBoostingClassifier

**Model Ranking:**

|     | name                                  | data_name       | macro_F2 |
| --- | ------------------------------------- | --------------- | -------- |
| 0   | t1_lgbm_combined_matched_threshold_v1 | eda_combined_v1 | 0.7785   |
| 1   | t1_lgbm_original_threshold_v1         | eda_original_v1 | 0.7776   |
| 2   | t1_hgb_combined_threshold_v1          | eda_combined_v1 | 0.7768   |
| 3   | t1_xgb_combined_threshold_v1          | eda_combined_v1 | 0.7753   |
| 4   | t1_xgb_original_matched_threshold_v1  | eda_original_v1 | 0.7742   |
| 5   | t1_cat_combined_threshold_v1          | eda_combined_v1 | 0.7732   |
| 6   | t1_hgb_original_threshold_v1          | eda_original_v1 | 0.7732   |
| 7   | t1_cat_original_matched_threshold_v1  | eda_original_v1 | 0.7727   |
| 8   | t1_rf_original_matched_v1             | eda_original_v1 | 0.7708   |
| 9   | t1_rf_combined_tuned_v1               | eda_combined_v1 | 0.7668   |
| 10  | t1_lgbm_original_tuned_v1             | eda_original_v1 | 0.7577   |
| 11  | t1_lgbm_combined_matched_v1           | eda_combined_v1 | 0.7567   |
| 12  | t1_xgb_original_matched_v1            | eda_original_v1 | 0.7564   |
| 13  | t1_xgb_combined_tuned_v1              | eda_combined_v1 | 0.7551   |
| 14  | t1_lgbm_original_v1                   | eda_original_v1 | 0.7549   |
| 15  | t1_hgb_original_matched_v1            | eda_original_v1 | 0.7542   |
| 16  | t1_cat_combined_v1                    | eda_combined_v1 | 0.7536   |
| 17  | t1_cat_combined_tuned_v1              | eda_combined_v1 | 0.7531   |
| 18  | t1_hgb_combined_tuned_v1              | eda_combined_v1 | 0.7529   |
| 19  | t1_hgb_original_v1                    | eda_original_v1 | 0.7528   |
| 20  | t1_lgbm_combined_v1                   | eda_combined_v1 | 0.7525   |
| 21  | t1_hgb_combined_v1                    | eda_combined_v1 | 0.7525   |
| 22  | t1_cat_original_v1                    | eda_original_v1 | 0.7522   |
| 23  | t1_cat_original_matched_v1            | eda_original_v1 | 0.7509   |
| 24  | t1_xgb_combined_v1                    | eda_combined_v1 | 0.7491   |
| 25  | t1_xgb_original_v1                    | eda_original_v1 | 0.7458   |
| 26  | t1_rf_original_v1                     | eda_original_v1 | 0.7416   |
| 27  | t1_rf_combined_v1                     | eda_combined_v1 | 0.7412   |
| 28  | t1_logreg_original_v1                 | eda_original_v1 | 0.6773   |
| 29  | t1_logreg_combined_v1                 | eda_combined_v1 | 0.6766   |
| 30  | t1_dummy_original_v1                  | eda_original_v1 | 0.4515   |

The top two models differ by only **0.00085** macro F₂ on this holdout. Because the same validation split was used to compare 31 saved experiments, this is a model-selection estimate, not evidence of a robust improvement. Next, compare a short list on repeated folds with threshold selection inside each training fold; then save a newly named final model fitted on all labeled rows before building the Task 1 submission.

**Feature Ranking (From winner t1_lgbm_combined_matched_threshold_v1):

|     | feature                        | macro_F2_drop | std    | source     |
| --- | ------------------------------ | ------------- | ------ | ---------- |
| 10  | reports_received               | 0.0911        | 0.0015 | original   |
| 9   | friend_network_size            | 0.0804        | 0.0053 | original   |
| 6   | account_age_days               | 0.0705        | 0.0005 | original   |
| 30  | crosshair_placement            | 0.0578        | 0.0021 | original   |
| 20  | game_sense_score               | 0.0180        | 0.0018 | original   |
| 1   | headshot_percentage            | 0.0172        | 0.0019 | original   |
| 7   | level                          | 0.0159        | 0.0008 | original   |
| 0   | kill_death_ratio               | 0.0149        | 0.0019 | original   |
| 3   | accuracy_score                 | 0.0090        | 0.0017 | original   |
| 4   | kill_consistency               | 0.0082        | 0.0018 | original   |
| 36  | reports_per_100_days           | 0.0078        | 0.0020 | engineered |
| 18  | aiming_smoothness              | 0.0074        | 0.0019 | original   |
| 19  | spray_control_score            | 0.0073        | 0.0004 | original   |
| 40  | aim_precision_spread           | 0.0063        | 0.0004 | engineered |
| 12  | input_consistency_score        | 0.0051        | 0.0010 | original   |
| 28  | damage_per_round               | 0.0049        | 0.0011 | original   |
| 2   | win_rate                       | 0.0049        | 0.0026 | original   |
| 21  | communication_rate             | 0.0045        | 0.0014 | original   |
| 27  | survival_time_avg              | 0.0042        | 0.0018 | original   |
| 17  | movement_pattern_score         | 0.0042        | 0.0009 | original   |
| 26  | first_blood_rate               | 0.0041        | 0.0005 | original   |
| 11  | device_changes_count           | 0.0039        | 0.0010 | original   |
| 37  | daily_play_hours               | 0.0038        | 0.0015 | engineered |
| 22  | team_play_score                | 0.0037        | 0.0012 | original   |
| 14  | sessions_per_day               | 0.0033        | 0.0011 | original   |
| 29  | utility_usage_rate             | 0.0031        | 0.0017 | original   |
| 44  | kd_per_teamplay                | 0.0030        | 0.0007 | engineered |
| 23  | buy_decision_score             | 0.0029        | 0.0007 | original   |
| 16  | weapon_switch_speed            | 0.0028        | 0.0003 | original   |
| 5   | reaction_time_ms               | 0.0028        | 0.0008 | original   |
| 15  | night_play_ratio               | 0.0026        | 0.0015 | original   |
| 25  | clutch_success_rate            | 0.0026        | 0.0013 | original   |
| 8   | level_progression_speed        | 0.0022        | 0.0018 | original   |
| 13  | avg_session_length_min         | 0.0019        | 0.0020 | original   |
| 42  | reaction_adjusted_aim          | 0.0019        | 0.0013 | engineered |
| 43  | aim_strategy_gap               | 0.0018        | 0.0010 | engineered |
| 39  | aim_precision_mean             | 0.0014        | 0.0013 | engineered |
| 38  | headshot_accuracy_gap          | 0.0012        | 0.0010 | engineered |
| 31  | missing_feature_count          | 0.0009        | 0.0005 | engineered |
| 24  | map_knowledge                  | 0.0007        | 0.0009 | original   |
| 32  | account_age_days_missing       | 0.0000        | 0.0000 | engineered |
| 35  | aiming_smoothness_missing      | 0.0000        | 0.0000 | engineered |
| 34  | avg_session_length_min_missing | 0.0000        | 0.0000 | engineered |
| 33  | reports_received_missing       | 0.0000        | 0.0000 | engineered |
| 41  | aim_precision_observed         | -0.0003       | 0.0003 | engineered |

![[Pasted image 20261105211632.png]]

**Feature Ranking (From 3rd winner t1_hgb_combined_threshold_v1):

|     | feature                        | macro_F2_drop | std    | source     |
| --- | ------------------------------ | ------------- | ------ | ---------- |
| 10  | reports_received               | 0.0846        | 0.0017 | original   |
| 9   | friend_network_size            | 0.0789        | 0.0059 | original   |
| 6   | account_age_days               | 0.0716        | 0.0003 | original   |
| 30  | crosshair_placement            | 0.0587        | 0.0036 | original   |
| 7   | level                          | 0.0162        | 0.0003 | original   |
| 1   | headshot_percentage            | 0.0110        | 0.0028 | original   |
| 20  | game_sense_score               | 0.0106        | 0.0037 | original   |
| 0   | kill_death_ratio               | 0.0090        | 0.0028 | original   |
| 3   | accuracy_score                 | 0.0062        | 0.0015 | original   |
| 19  | spray_control_score            | 0.0038        | 0.0012 | original   |
| 25  | clutch_success_rate            | 0.0037        | 0.0001 | original   |
| 37  | daily_play_hours               | 0.0026        | 0.0006 | engineered |
| 4   | kill_consistency               | 0.0024        | 0.0040 | original   |
| 16  | weapon_switch_speed            | 0.0023        | 0.0019 | original   |
| 17  | movement_pattern_score         | 0.0023        | 0.0008 | original   |
| 27  | survival_time_avg              | 0.0023        | 0.0005 | original   |
| 22  | team_play_score                | 0.0021        | 0.0014 | original   |
| 13  | avg_session_length_min         | 0.0018        | 0.0006 | original   |
| 11  | device_changes_count           | 0.0018        | 0.0007 | original   |
| 8   | level_progression_speed        | 0.0016        | 0.0011 | original   |
| 42  | reaction_adjusted_aim          | 0.0015        | 0.0017 | engineered |
| 26  | first_blood_rate               | 0.0012        | 0.0011 | original   |
| 5   | reaction_time_ms               | 0.0010        | 0.0011 | original   |
| 36  | reports_per_100_days           | 0.0008        | 0.0028 | engineered |
| 44  | kd_per_teamplay                | 0.0005        | 0.0011 | engineered |
| 24  | map_knowledge                  | 0.0004        | 0.0005 | original   |
| 18  | aiming_smoothness              | 0.0003        | 0.0011 | original   |
| 43  | aim_strategy_gap               | 0.0000        | 0.0009 | engineered |
| 35  | aiming_smoothness_missing      | 0.0000        | 0.0000 | engineered |
| 34  | avg_session_length_min_missing | 0.0000        | 0.0000 | engineered |
| 32  | account_age_days_missing       | 0.0000        | 0.0000 | engineered |
| 33  | reports_received_missing       | 0.0000        | 0.0000 | engineered |
| 38  | headshot_accuracy_gap          | -0.0001       | 0.0010 | engineered |
| 41  | aim_precision_observed         | -0.0003       | 0.0002 | engineered |
| 29  | utility_usage_rate             | -0.0006       | 0.0006 | original   |
| 31  | missing_feature_count          | -0.0007       | 0.0011 | engineered |
| 12  | input_consistency_score        | -0.0009       | 0.0019 | original   |
| 39  | aim_precision_mean             | -0.0009       | 0.0007 | engineered |
| 23  | buy_decision_score             | -0.0009       | 0.0031 | original   |
| 28  | damage_per_round               | -0.0010       | 0.0025 | original   |
| 15  | night_play_ratio               | -0.0010       | 0.0003 | original   |
| 40  | aim_precision_spread           | -0.0014       | 0.0007 | engineered |
| 21  | communication_rate             | -0.0016       | 0.0013 | original   |
| 14  | sessions_per_day               | -0.0022       | 0.0012 | original   |
| 2   | win_rate                       | -0.0030       | 0.0022 | original   |

![[Pasted image 20261105211352.png]]

## Round 3: stability and feature ablation

Compared three finalists on the same **3-fold × 2-repeat** outer CV splits. For each outer training fold, a separate 3-fold CV selected the decision threshold; the outer fold was used only for scoring. The model families and settings had already been shortlisted using the earlier holdout, so these CV means are exploratory rather than a new untouched validation estimate.

| Model | Features | Mean macro F₂ | Fold std | Threshold range |
| --- | ---: | ---: | ---: | ---: |
| LightGBM combined, pruned | 38 | **0.77733** | 0.00127 | 0.375–0.400 |
| LightGBM combined | 45 | 0.77679 | 0.00130 | 0.375–0.400 |
| LightGBM original | 31 | 0.77676 | 0.00051 | 0.375–0.400 |
| HGB combined | 45 | 0.77536 | 0.00110 | 0.350–0.400 |

The pruned variant drops `account_age_days_missing`, `reports_received_missing`, `avg_session_length_min_missing`, `aiming_smoothness_missing`, `aim_precision_observed`, `missing_feature_count`, and `map_knowledge`, chosen from the low end of the previous winner's permutation-importance ranking. Its paired gain over full LightGBM combined averaged **0.00054** macro F₂ and occurred in 4 of 6 folds. This is a small exploratory gain, not strong evidence that all seven features are harmful.

The highest-mean procedure was retrained on all **90,816 labeled rows** with inner-CV threshold tuning. It was saved as `t1_lgbm_combined_pruned_cv6_full_v1` with threshold **0.400**. The `macro_F2_outer_cv6` evaluation card records the six-fold procedure estimate (**0.77733**), not a validation score measured on that full-data model. `src/task1/submission/t1_round3_full_v1.csv` contains 25,889 predictions and a predicted cheater rate of **38.41%**. Test was used only for prediction and format checks.
