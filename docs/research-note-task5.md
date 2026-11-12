# Task 5 — Account Security Monitoring

**Current sources:** `src/task5/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, and `src/task5/models/build_model_v2.ipynb`. These notebooks contain **unlabeled Test only**. Their model fitting is described as instructor-permitted in the notebooks; this needs reconciliation with the written competition rule against Test-based training. No offline macro F₃ is available.

## EDA and features

`eda_original_v2` retains **122** supplied numeric features and masks invalid `login_lat_1`–`login_lat_4` / `login_lon_1`–`login_lon_4` coordinate pairs. `player_id` is only for joining predictions; `is_anomaly` is absent. `eda_1_v2` adds **91** row-wise features, completely specified by these names and patterns:

- For **each** of `login_count`, `location_changes`, `device_count`, `ip_hash_entropy`, `login_time_variance`, `session_duration_mean`, `play_pattern_consistency`, `activity_level`, `skill_rating`, `reaction_time_mean`, `purchase_frequency`, `purchase_amount_mean`, `currency_transfers`, `communication_frequency`: `<family>_mean4`, `<family>_std4`, `<family>_range4`, `<family>_delta41`, `<family>_recent_shift`, `<family>_max_step` (14 × 6 = 84 names).
- The other seven are `missing_feature_count`, `login_move_km_12`, `login_move_km_23`, `login_move_km_34`, `login_move_max_km`, `login_move_total_km`, and `security_flag_count`.

The four security inputs counted by `security_flag_count` are `password_changed`, `vpn_usage`, `suspicious_login_time`, and `mass_item_sale`; none is a confirmed anomaly label. Geographic displacement is descriptive because timestamps are absent.

## Models and results

The notebooks compare Isolation Forest, Local Outlier Factor, RBF One-Class SVM, Gaussian Mixture, PCA reconstruction, and an empirical tail detector, plus parameter variants, on original versus engineered features. The score is **top-5% rank-stability Jaccard** between overlapping 12,000-row and 10,000-row fits, measured on a 4,000-row holdout. It measures repeatability, **not fraud detection quality or F₃**.

| Detector | Engineered stability | Original stability |
| --- | ---: | ---: |
| One-Class SVM, `nu=0.10` | **1.000** | 0.990 |
| One-Class SVM, base | 0.990 | 0.980 |
| Gaussian Mixture, base | 0.980 | 0.717 |
| Empirical tail | 0.961 | 0.896 |
| Isolation Forest, 512-row subsamples | 0.905 | 0.521 |
| Local Outlier Factor | 0.860 | 0.869 |
| Isolation Forest, base | 0.770 | 0.493 |

## Insight and next step

Engineering improves stability for several detector families, but the different families flag different accounts: top-5% overlap ranges from **0.143** (LOF versus tail) upward. Stability cannot choose the best fraud detector. First resolve the Test-fitting rule with the organizers and obtain labeled validation or an approved evaluation path. Then compare actual detection precision/recall at an operational alert budget; do not call the current proxy macro F₃.
