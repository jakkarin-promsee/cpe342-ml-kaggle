# Task 2 — Player Segments

**Current sources:** `src/task2/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, `analyze_top_model_errors_1.ipynb`, `build_eda_tier_cross_v2.ipynb`, and `src/task2/models/build_model_v2.ipynb` through `build_model_v9.ipynb`. Older Task 2 notebooks are excluded. Metric: validation macro F₁.

## EDA and features

`eda_original_v2` keeps all **104,000** labeled rows and 44 raw predictors (34 numeric, 10 categorical); Train/Test IDs and schemas are checked. Missingness stays for development-fitted preprocessing. `random_metric_1`, `random_metric_2`, and `random_metric_3` are documented noise and are ablated in modeling. `eda_combined_v2` retains the raw inputs and adds **all 14** features: `session_time_proxy_hours`, `play_hours_per_account_day`, `login_recency_over_age`, `spend_per_play_hour`, `spend_per_account_day`, `spend_per_purchase_proxy`, `purchase_to_play_frequency`, `social_actions_per_friend`, `team_chat_interaction`, `social_preference_interaction`, `ranked_time_proxy_hours`, `ranked_win_interaction`, `cosmetic_vs_performance`, `source_missing_count`.

`play_frequency` exceeds seven although the datacard describes days/week, so derived time products are relative proxies. `loyalty_tier`, `engagement_level`, and `vip_tier` have unknown creation times.

`eda_tier_cross_v2` starts from the refreshed raw files, keeps all 44 raw predictors, and adds exactly one feature: `loyalty_engagement_cross = loyalty_tier + " | " + engagement_level`, substituting `<missing>` for absent components. It is a deterministic categorical cross with no fitted statistics. The feature inherits both tiers' unknown temporal availability.

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

### Class-weight experiment (v4, 13 November)

`build_model_v4.ipynb` used the same `eda_original_v2` card, stratified development/holdout split, five paired development folds, CatBoost configuration, and original predictors with the three documented noise columns excluded by the saved preprocessing recipe. It changed only `class_weights` to `[1.0, 1.0, 1.25, 1.25]` for Casual, Grinder, Social, and Whale. Each fold refitted preprocessing on its training part. The model `t2_cat_weighted_23_cv5_v4` was fitted on development and evaluated on the **reused** 20% holdout.

| Check | Baseline macro F₁ | Weighted macro F₁ | Difference |
| --- | ---: | ---: | ---: |
| Five development folds, mean | 0.75256 | **0.75310** | +0.00054 (3/5 folds) |
| Reused 20% holdout | 0.75041 | **0.75285** | +0.00243 |

The development OOF confusion matrix shows Social→Casual **3,155** and Whale→Social **1,813** errors for the baseline. Of 83,200 OOF rows, **2,722** are wrong with maximum predicted probability over 0.8. Weighting raises Social recall **0.6782→0.7064** and Whale recall **0.6988→0.7157**, while Social precision falls **0.7449→0.7179** and Whale precision **0.7810→0.7671**. Social F₁ rises **0.7100→0.7121** and Whale **0.7376→0.7405**; Casual **0.8215→0.8201** and Grinder **0.7412→0.7397** fall. This is a real but small tradeoff, not evidence of a large hidden gain. Holdout is repeatedly consulted, so its larger difference is exploratory.

### Detailed OOF error audit

`src/task2/eda/analyze_top_model_errors_1.ipynb` aligns the existing five-fold OOF caches to the 83,200 development rows and checks the same model without touching Test. There are **19,315** wrong OOF predictions (23.22%). Largest directions: Grinder→Casual **4,098**, Social→Casual **3,155**, Casual→Grinder **2,779**, Whale→Social **1,813**. The **2,722** wrong predictions above 0.8 top probability are **14.1% of errors** and **3.27% of all rows**. Median top probability is **0.585** for errors versus **0.810** for correct rows. Error rate rises from **4.61%** for top probability above 0.9 to **60.47%** at or below 0.4. Most errors are in low-confidence overlap; a smaller confident-error group remains.

Among true Social rows, Social→Casual has similar medians to correctly classified Social for friends (**85 vs 86**) and team play (**48.83 vs 48.71**), but lower play frequency (**9.55 vs 9.93**). Among true Whale rows, Whale→Social has lower total spend (**42,212 vs 48,378 THB**), monthly spend (**4,369 vs 4,813 THB**) and purchase frequency (**21.74 vs 24.73**) than correctly classified Whale. These are descriptive within-class contrasts, not causal effects. Across the 41 model inputs there are **no exact duplicate feature rows** in development. Missing input count has a mild error gradient (20.3% with zero missing fields; 24.3% with six), but sparse high-missing groups are too small to interpret.

`loyalty_tier` contributes **31.85%** of the saved CatBoost's feature importance, `engagement_level` **10.61%**, and `vip_tier` **6.50%**; `player_type_tag` contributes **0%**. The joint loyalty/engagement category has strongly different class mixes. For example, Bronze+Low is **86% Casual**, Bronze+High **70% Grinder**, Diamond+Low **84% Social**, and Diamond+High **75% Whale** (rounded from labeled development). Loyalty also tracks spending, so this association does not establish target leakage. The tier creation times remain undocumented.

Comparing baseline and weighted OOF predictions, weighting switches **2,815** labels, rescues **1,154** wrong rows and breaks **1,223** previously correct rows. Its macro F₁ gain comes from better Social/Whale balance, not higher overall accuracy.

### Tier-cross model check

`build_model_v5.ipynb` trained `t2_cat_tier_cross_cv5_v5` on `eda_tier_cross_v2`: the original **41** non-noise model inputs plus the one new cross. CatBoost hyperparameters and split positions match the original baseline; each development fold refits preprocessing. The cross changes **1,861/83,200** OOF predictions, but improvement is inconsistent.

| Check | Baseline macro F₁ | Tier-cross macro F₁ | Difference |
| --- | ---: | ---: | ---: |
| Five paired development folds, mean | 0.75256 | 0.75265 | +0.00008 (2/5 wins) |
| Reused 20% holdout | **0.75041** | 0.74958 | −0.00084 |

Pooled OOF per-class F₁ barely changes: Casual **0.82148→0.82177**, Grinder **0.74120→0.74080**, Social **0.71000→0.71015**, Whale **0.73761→0.73788**. The explicit cross therefore adds almost no reproducible information beyond CatBoost's existing splits. Do not expand this into many tier combinations on the same holdout.

### Conclusion through v5

The 14 broad engineered features and the targeted tier cross did not consistently improve matched models. At this stage, weighted CatBoost had the highest recorded Task 2 holdout score, but its paired CV advantage was only **0.00054** with 3/5 wins; the original CatBoost remained the conservative submission choice. Further class-bias, weight, and tier-cross tuning on these splits had low expected value. The next information check was whether `loyalty_tier`, `engagement_level`, and `vip_tier` were available at prediction time. Their ablation proved strong reliance, not leakage; if unavailable, a separately chosen no-tier model would need validation.

### Tier provenance audit (13 November)

The current competition file `src/task2/data/raw/datacard.txt` describes `vip_tier` as the **current VIP tier level**, `engagement_level` as an engagement tier classification, and `loyalty_tier` as a loyalty tier **based on spending**. The old and refreshed datacards contain the same descriptions. `docs/kaggle-detial/data.md`, `overview.md`, and the local instruction-slide extract `hint_canva_extracted.txt` (Task 2, page 17) do not provide feature creation timestamps, aggregation windows, or tier formulas. The three fields occur in the supplied Train and Test schema, with Train missingness of about **7.65%**, **7.85%**, and **7.72%**, respectively. Every observed loyalty and engagement tier contains all four target classes, so neither is a direct deterministic copy of the label. None of this establishes that the values were computed before the segment label or would be available in a real prospective deployment. The provenance question remains unresolved; request the organizer's feature timing or generation specification before making a leakage claim. The v6 tuning notebook records a no-tier robustness check alongside the tier-included search.

### CatBoost hyperparameter tuning (v6)

`build_model_v6.ipynb` reads `eda_original_v2` by its JSON card and keeps the same 41 non-noise model inputs, seed, and 80/20 split. A random parameter sampler drew 12 combinations of depth, iterations, learning rate, L2 regularization, random strength, and bagging temperature. A runtime screen retained nine combinations with depth at most 6 or at most 300 iterations; one excluded combination completed one fold but was ignored. The nine candidates and baseline were scored on three folds of a stratified 40,000-row **development subset**. The top three and baseline were then scored on five matched folds of the full 83,200 development rows, fitting preprocessing inside each fold. The previously reused holdout was not used for parameter selection.

| Full-development five-fold check | Mean macro F₁ | Δ vs baseline | Paired wins |
| --- | ---: | ---: | ---: |
| Original CatBoost baseline | 0.752564 | — | — |
| `search_01` (depth 4, 700 iterations, learning rate .06, L2 20, random strength 3, bagging temperature .5) | **0.754095** | **+0.001531** | 4/5 |
| `search_12` | 0.753358 | +0.000794 | 5/5 |
| `search_05` | 0.753007 | +0.000443 | 3/5 |
| Baseline recipe without all three tiers | 0.587355 | −0.165209 | 0/5 |

The baseline's five-fold score exactly reproduces the earlier 0.75256 result. The selected `search_01` configuration is saved as `t2_cat_tuned_cv5_v6`, fitted on development rows. Its **reused** 20% holdout macro F₁ is **0.751095**, versus **0.750410** for the original baseline; this is a small exploratory gain, not untouched confirmation. The no-tier ablation establishes strong model reliance on tier values, not target leakage. The strongest feature set remains the original non-noise inputs; the engineered 14-feature set and explicit tier cross did not improve matched models earlier. The next controlled check combines these tuned parameters with the already tested fixed Social/Whale class weights, with no new weight grid.

### Fixed class weights on the tuned model (v7)

`build_model_v7.ipynb` keeps the v6 winner's preprocessing and CatBoost parameters and changes only class weights to the prior `[1, 1, 1.25, 1.25]`. On the same five paired development folds it scores **0.754156** mean macro F₁ versus **0.754095** for unweighted v6: **+0.000061**, with 3/5 wins. Its saved development-fitted model `t2_cat_tuned_weighted_cv5_v7` scores **0.752895** on the reused 20% holdout versus **0.751095** for v6 and **0.752849** for the older weighted baseline. The last difference is only **+0.000045** on repeatedly inspected rows. Weighting and v6 tuning together yield the highest numerical CV and holdout values among these local variants, but their incremental gain is too small to establish a reliable improvement. Keep both v6 and v7 candidates; do not treat the evaluation-index winner on reused holdout as an independent ranking.

`build_model_v8.ipynb` refits the v7 configuration once on all **104,000** labeled Train rows as `t2_cat_tuned_weighted_full_v8`. Its `macro_F1_cv5_procedure` value **0.754156** is the earlier five-fold development estimate, not an evaluation of the full-data fit. The full model predicts all 25,889 saved EDA Test rows with no missing values and labels 0–3. `src/submission/assemble_task2_tuned_v12.ipynb` produces the **unscored** combined candidate `kaggle_t12345_t2_tuned_v12.csv` from scored v9, changing **1,088 Task 2 predictions** by saved Test row position. The other four task fields remain exactly equal as CSV strings; the JSON card records lineage and hash. Do not infer a leaderboard gain from its CV or reused-holdout differences. The next valuable evidence is organizer clarification of the tier computation window or a genuinely independent Task 2 result.

### Fixed ensemble check (v9)

`build_model_v9.ipynb` tested the slide's ensemble hint without using Test or the repeatedly consulted holdout. It reused the baseline and class-weighted CatBoost OOF probabilities and refitted the saved LightGBM and XGBoost pipelines on the **same five development folds**. Preprocessing was fitted within each fold. The fixed candidates were 50/50 CatBoost/LightGBM and equal-thirds CatBoost/LightGBM/XGBoost, each with original or weighted CatBoost. No blend weights or meta-model were tuned on OOF labels. These are fixed soft ensembles, not learned stacking.

| Procedure | Five-fold mean macro F₁ | Fold wins vs tuned weighted CatBoost v7 |
| --- | ---: | ---: |
| Tuned weighted CatBoost v7 reference | **0.754156** | — |
| Original weighted CatBoost | 0.753100 | 1/5 |
| Best fixed ensemble: weighted CatBoost + LightGBM, 50/50 | 0.752148 | 0/5 |
| Weighted CatBoost + LightGBM + XGBoost, equal thirds | 0.751655 | 0/5 |
| LightGBM alone / XGBoost alone | 0.747151 / 0.747503 | 0/5 |

The best fixed ensemble loses to the selected CatBoost configuration on **all five folds** by **0.002008** mean F₁. This closes the simple ensemble hint; a learned stack would require nested cross-fitting and has no current evidence of useful headroom. Keep the full-data v8 CatBoost as the Task 2 freeze candidate. The v12 Kaggle candidate remains unscored.
