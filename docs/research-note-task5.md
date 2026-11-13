# Task 5 — Account Security Monitoring

**Current sources:** `src/task5/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, `src/task5/models/build_model_v2.ipynb` through `build_model_v6.ipynb`, and `rank_diagnostics_v1.ipynb`. The earlier v1 EDA/model notebooks used prior artifacts and are historical only. These notebooks contain **unlabeled Test only**. The user reports a lecturer clarification permitting unsupervised fitting on this Task 5 file because no labeled training set is supplied; the written rule in `docs/kaggle-detial/rule.md` still says Test cannot be used for fitting, so keep that exception explicit in any final report. No offline macro F₃ is available.

## EDA and features

`eda_original_v2` retains **122** supplied numeric features and masks invalid `login_lat_1`–`login_lat_4` / `login_lon_1`–`login_lon_4` coordinate pairs. `player_id` is only for joining predictions; `is_anomaly` is absent. `eda_1_v2` adds **91** row-wise features, completely specified by these names and patterns:

- For **each** of `login_count`, `location_changes`, `device_count`, `ip_hash_entropy`, `login_time_variance`, `session_duration_mean`, `play_pattern_consistency`, `activity_level`, `skill_rating`, `reaction_time_mean`, `purchase_frequency`, `purchase_amount_mean`, `currency_transfers`, `communication_frequency`: `<family>_mean4`, `<family>_std4`, `<family>_range4`, `<family>_delta41`, `<family>_recent_shift`, `<family>_max_step` (14 × 6 = 84 names).
- The other seven are `missing_feature_count`, `login_move_km_12`, `login_move_km_23`, `login_move_km_34`, `login_move_max_km`, `login_move_total_km`, and `security_flag_count`.

The four security inputs counted by `security_flag_count` are `password_changed`, `vpn_usage`, `suspicious_login_time`, and `mass_item_sale`; none is a confirmed anomaly label. Geographic displacement is descriptive because timestamps are absent.

## Models and results

**Journey.** The v2 baseline notebook compared six detector families on raw versus engineered features, plus larger Isolation Forest subsamples, higher One-Class SVM `nu`, and more GMM components. Engineered `nu=0.10` One-Class SVM had the highest top-5% *stability* proxy (**1.000**), so its original 5% cutoff became the v6 submission baseline. A controlled label flip (v7) lowered the overall Kaggle score, confirming the original `1 = anomaly` orientation. Next, the same fitted ranking was tested at about 15% (v8), 25% (v9), and 35% (unscored v10) alerts. Overall score improved strongly from 5% to 15%, then only slightly to 25%. Offline ranking diagnostics showed GMM would select a different set; v11 matched v9's exact alert count and swapped only the ranking. Its lower overall score left v9 as the best *scored* submission. No step measured Task 5 F₃ directly.

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

Engineering improves stability for several detector families, but the different families flag different accounts: top-5% overlap ranges from **0.143** (LOF versus tail) upward. Stability cannot choose the best fraud detector. Under the reported lecturer exception, unsupervised fitting on the unlabeled file is permitted for this task, but without labels the next useful evidence is a sanctioned leaderboard result or labeled validation. Compare actual detection precision/recall at an operational alert budget if labels become available; do not call the current proxy macro F₃.

`src/submission/assemble_v2.ipynb` also writes an optional Task 5 label-inversion submission (`kaggle_t12345_gate_resnet_t5flip_v7.csv`) from the same saved predictions. It changes no fitted model or anomaly ranking; it only swaps binary output labels. There is no offline F₃ result for either orientation.

**Submission label mapping and leaderboard feedback:** The user confirmed that the original Task 5 0/1 mapping in `kaggle_t12345_gate_resnet_v6.csv` is correct. Keep `1 = anomaly`; the flipped v7 file is only a comparison artifact. Overall leaderboard scores supplied by the user are **0.72700 for v6** and **0.66266 for v7**. These files differ only by Task 5 label inversion, so the **0.06434 overall-score drop** supports the original orientation. The refreshed old v5 scores **0.66919** (the pre-refresh old-dataset score was **0.60269**), but v5 and v6 also differ in Tasks 3–4, so their score difference does not isolate Task 5. The competition's task weighting and per-task scores remain unknown; none of these overall scores is an offline or directly reported Task 5 F3.

## Kaggle probe 1: alert budget

`build_model_v3.ipynb` reconstructs v6's **1,365** Task 5 alerts exactly from the saved engineered-feature `t5_r2_ocsvm_nu10_engineered_v2` model. It keeps that model's fitted preprocessing, detector and ranking, and changes only the cutoff from the 95th to the 85th percentile of its original 12,000-row fit sample. The saved candidate is `t5_r3_ocsvm_nu10_alert15_v3` on the same `eda_1_v2` dataset, with **no new engineered features**. It produces **3,971 / 25,889** alerts (**15.34%**) and adds **2,606** alerts; all v6 alerts remain positive. Its top-15% rank-stability Jaccard on the prior overlapping-fit diagnostic is **0.98675**. This is a repeatability proxy, not F3. The added alerts have a **26.13%** rate for any of the four provided security flags, close to the full-file **25.44%** rate; the old alerts have **51.36%**. Those flags are model inputs and do not validate anomaly labels.

The candidate Kaggle file is `src/submission/kaggle_t12345_t5_alert15_v8.csv`. Its card records the v6 SHA-256, model and data lineage, and the 2,606 changed Task 5 rows. All Task 1–4 values and ANS IDs are verified identical to v6, and Task 5 still uses `1 = anomaly`. The user reported an **overall Kaggle score of 0.74446**, up **0.01746** from v6's **0.72700**. This controlled change supports a broader alert budget for the fixed OCSVM ranking; it does not reveal Task 5 F3. The score history and comparison limits are recorded separately in `docs/kaggle-score-log.md`. The next probe changes only this budget to 25%.

## Kaggle probe 2: 25% alert budget

`build_model_v4.ipynb` keeps the same `eda_1_v2` features and saved OCSVM ranking, moving the cutoff to the 75th percentile of the original fit sample. Its saved model `t5_r4_ocsvm_nu10_alert25_v4` produces **6,531 / 25,889** alerts (**25.23%**), adding **2,560** beyond v8 and **5,166** beyond v6. Top-25% overlapping-fit rank-stability Jaccard is **0.98216**, again only a proxy. The additional v8-to-v9 alerts have a **25.78%** security-flag rate, near the whole-file **25.44%** rate; this is descriptive and cannot establish fraud recall.

`src/submission/kaggle_t12345_t5_alert25_v9.csv` is prepared with Tasks 1–4 and ANS IDs identical to v6, the same `1 = anomaly` orientation, and a lineage card referencing v6 and v8. The user reported an **overall Kaggle score of 0.74593**, up **0.00147** from v8 and **0.01893** from v6. The marginal gain from the second 10-percentage-point alert-budget increase is much smaller than the first. This suggests checking one wider cutoff before switching detector families; it does not identify Task 5 F3 or a precise optimum.

## Kaggle probe 3: 35% alert budget

`build_model_v5.ipynb` changes only the same OCSVM score cutoff to the 65th percentile of the original fit sample. The saved `t5_r5_ocsvm_nu10_alert35_v5` model produces **9,118 / 25,889** alerts (**35.22%**), adding **2,587** beyond v9 and **7,753** beyond v6. The top-35% overlapping-fit rank-stability Jaccard is **0.97880**, an unlabeled proxy. The additional v9-to-v10 alerts have a **24.00%** security-flag rate, close to the whole-file **25.44%**; this is not a detection-quality measure.

`src/submission/kaggle_t12345_t5_alert35_v10.csv` is prepared. Its Tasks 1–4 and ANS IDs match v6 exactly; Task 5 keeps `1 = anomaly`, and its card links to v6 and v9. **v10 has no Kaggle score yet.** The earlier v9 overall score was **0.74593**; compare v10 with it before changing the detector family or alert budget again.

## Offline ranking diagnostics before another submission

With one submission remaining in the user's current window, `rank_diagnostics_v1.ipynb` compares **already fitted** engineered-feature detector scores without submitting or fitting another model. At a fixed top-25% count of **6,472** rows (slightly different from v9's fit-sample cutoff count of **6,531**), OCSVM and GMM select **5,321** accounts in common, giving Jaccard **0.698**; each has **1,151** exclusive accounts. GMM's prior top-5% fit-stability proxy is **0.980**, versus OCSVM's **1.000**. At top 25%, their score Spearman correlation is **0.927**, so the swap is distinct but not a wholesale ranking reversal.

At that fixed budget, any of the four provided security flags appears in **70.5%** of GMM-only accounts but **12.0%** of OCSVM-only accounts. GMM-only versus OCSVM-only rates are **20.8% vs 0.3%** for `password_changed`, **21.0% vs 0.4%** for `suspicious_login_time`, and **22.5% vs 0.0%** for `mass_item_sale`; `vpn_usage` is **21.5% vs 11.3%**. GMM and OCSVM both use these fields as inputs, so this difference describes their ranking behavior and **does not establish fraud quality**. The GMM-only group has median `security_flag_count` **1**, versus **0** for OCSVM-only; other selected median feature contrasts are shown in the notebook.

The next high-information controlled probe after the submission reset is a GMM-derived top-25% candidate against v9, holding Tasks 1–4 and the `1 = anomaly` mapping fixed. Keep the unscored 35% OCSVM v10 draft as an alternative. Do not spend the remaining current-window submission merely to continue a cutoff sweep with diminishing observed gains.

## Controlled GMM ranking candidate at v9 alert volume

`build_model_v6.ipynb` reuses the saved engineered-feature diagonal GMM `t5_r2_gmm_engineered_v2` and selects an unlabeled full-file score cutoff that yields exactly **6,531** alerts, matching scored OCSVM v9. The saved `t5_r6_gmm_match_v9_alerts_v6` model retains the fitted imputer, scaler, and GMM; no new feature set is created. Its overlapping-fit top-25% rank-stability Jaccard is **0.98413**, an unlabeled proxy. Compared with v9, it adds **1,154** accounts and removes **1,154**, leaving **5,377** common alerts. Any supplied security flag appears in **70.28%** of GMM-only accounts and **12.22%** of OCSVM-only accounts. These flags are model inputs, not validation labels.

`src/submission/kaggle_t12345_t5_gmm_match_v9_v11.csv` is prepared with Tasks 1–4 and ANS IDs identical to v9, the same `1 = anomaly` mapping, exactly the same Task 5 alert count, and a lineage card. The user reported an **overall Kaggle score of 0.74521**, **0.00072 below v9**. Because v9 and v11 have equal Task 5 alert counts and identical Tasks 1–4, this controlled result favors the OCSVM ranking over GMM at this budget on the leaderboard, despite GMM-only accounts showing more supplied security flags. It still does not provide a Task 5 F3 or establish a general detector ranking. **V9 remains the highest scored submission at 0.74593.** The 35% OCSVM v10 draft remains unscored for a later window.

## Exact v9–v11 prediction overlap

The two submitted CSVs have **25,889** identical IDs and identical Tasks 1–4. Each predicts **6,531** Task 5 anomalies. Their exact Task 5 contingency table is:

| v9 prediction | v11 = 0 | v11 = 1 |
| --- | ---: | ---: |
| 0 | 18,204 | 1,154 |
| 1 | 1,154 | 5,377 |

Thus **2,308 / 25,889 (8.91%)** rows disagree, while **5,377 / 6,531 (82.33%)** of each model's positive alerts overlap. Positive-set Jaccard is **0.6997**; each model swaps **17.67%** of its alert budget. The small overall score difference (**0.00072**) despite 2,308 changed labels says neither exclusive set has a large measured advantage in this leaderboard comparison. It does **not** show whether the 5,377 shared alerts are correct or how many anomalies lie among the 18,204 shared negatives; hidden labels or a Task 5 score are required for that distinction.
