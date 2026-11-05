# Task 2: Player Segment Classification

**Goal:** Classify each player as Casual (0), Grinder (1), Social (2), or Whale (3). Competition metric: **macro F₁**.

## Data and EDA

- Raw Train: **104,000 rows**, 44 features, `player_id`, and complete `segment` labels. Test: **25,889 rows**, the same 44 features and `player_id`, without labels.
- Class counts: Casual **40,934**, Grinder **26,034**, Social **21,014**, Whale **16,018**. Use a stratified split and inspect every class, not accuracy alone.
- Features: **34 numeric** and **10 string categorical**. Many columns have missing values; `payment_method` and `region` are missing most often. There are no duplicate or overlapping IDs, infinite numeric values, constant columns, or unseen Test categories at this stage.
- `player_type_tag` does not directly encode the label. `engagement_level` and `loyalty_tier` correlate with the label but do not determine it. Their exact availability time is not documented, so operational leakage cannot be ruled out from this static dataset alone.
- `src/task2/eda/build_eda_original.ipynb` saves all 44 original features in `eda_original_v1`. `src/task2/eda/build_eda_1.ipynb` saves `eda_combined_v1`: the same 44 columns plus 14 deterministic row-level behavior features. Imputation and encoding are fitted inside model Pipelines, never in EDA or on Test.
- The added feature groups cover spending and recency, play intensity, social interaction, and ranked participation. Their motivation comes from [RFM analysis of a freemium game](https://scholar.smu.edu/datasciencereview/vol6/iss2/1/), [playtime behavior research](https://arxiv.org/abs/1912.06972), and [competitive behavior modeling](https://arxiv.org/abs/2112.04379). These sources motivate feature families, not the exact formulas; no external data enters the dataset.
- `play_frequency` exceeds seven in the actual data despite its days/week description. Features using it are relative activity proxies rather than literal weekly hours.

## Round 1 models

All scores use the same stratified **80/20** development/validation split (`SEED=42`). The HGB hyperparameter search uses 3-fold CV on development only. Every saved model has its own JSON card and evaluation.

| Model | Macro F₁ | Casual | Grinder | Social | Whale |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_hgb_no_noise_v1` | **0.74944** | 0.8207 | 0.7394 | 0.7133 | 0.7243 |
| `t2_hgb_original_v1` | 0.74654 | 0.8182 | 0.7368 | 0.7102 | 0.7210 |
| `t2_hgb_no_noise_tuned_v1` | 0.74555 | 0.8200 | 0.7382 | 0.7065 | 0.7175 |
| `t2_logreg_original_v1` | 0.74376 | 0.7985 | 0.7375 | 0.7022 | **0.7369** |
| `t2_rf_original_v1` | 0.72933 | 0.7964 | 0.7154 | 0.6917 | 0.7138 |
| `t2_dummy_original_v1` | 0.14122 | 0.5649 | 0 | 0 | 0 |

The same split and model settings were then run on the combined dataset. Matched differences isolate the 14 new features:

| Configuration | Original macro F₁ | Combined macro F₁ | Combined − original |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.74376 | 0.74355 | −0.00020 |
| Random forest | 0.72933 | 0.72956 | +0.00023 |
| HGB, all raw columns | 0.74654 | 0.74559 | −0.00095 |
| HGB, noise columns dropped | **0.74944** | 0.74808 | −0.00137 |
| HGB, noise dropped and tuned | 0.74555 | 0.74636 | +0.00081 |

The combined HGB with noise removed is second overall but does not beat the original winner. The tiny gains in the forest and tuned HGB are too small to call robust from one holdout. Keep `t2_hgb_no_noise_v1` as the provisional leader. The combined features may still merit a targeted ablation on development folds later, especially for Social errors, but the present comparison does not justify replacing the leader.

Removing the three `random_metric_*` columns documented as noise increased HGB validation macro F₁ by **0.00290** at matched baseline settings. The small random search did not improve HGB on the holdout. The winner's permutation importance is led by `loyalty_tier` (drop **0.2195**), `engagement_level` (**0.0657**), and `vip_tier` (**0.0335**), followed by login recency, playtime, and spending. Importance describes this fitted model's reliance on columns, not causation.

The next Round 1 question was whether stronger categorical learners would show the same original-versus-combined feature effect. Round 2 below tests that question.

## Round 2 boosted-tree comparison

LightGBM, CatBoost, and XGBoost were trained on the **same 83,200 development / 20,800 validation rows** as Round 1. Every pair removes the three documented `random_metric_*` noise columns and keeps model settings fixed between original and combined. LightGBM uses ordinal-encoded native categorical splits, CatBoost uses string categorical inputs, and XGBoost uses the Round 1 one-hot Pipeline. These are initial family baselines, not tuned searches.

| Family | Original macro F₁ | Combined macro F₁ | Combined − original |
| --- | ---: | ---: | ---: |
| HGB (Round 1) | 0.74944 | 0.74808 | −0.00137 |
| LightGBM | 0.74851 | 0.74685 | −0.00166 |
| CatBoost | **0.75025** | 0.74923 | −0.00102 |
| XGBoost | 0.74805 | 0.74679 | −0.00126 |

All four families favor the original features at these settings on this holdout. CatBoost original leads HGB original by only **0.00081** macro F₁. Its class F₁ is Casual **0.8209**, Grinder **0.7408**, Social **0.7089**, Whale **0.7304**; relative to HGB it gains on Grinder and Whale but loses on Social. This is a small, model-dependent tradeoff, not yet a stable winner.

The leading CatBoost model's validation permutation importance is led by `loyalty_tier` (macro F₁ drop **0.2245**), `engagement_level` (**0.0744**), and `vip_tier` (**0.0264**), followed closely by average monthly spending, login recency, and total playtime. It relies on the same tier signals that dominated the earlier HGB. These importances describe the fitted model, not causal effects.

The Round 2 comparison raised a feature-availability question for the two dominant tiers. Round 3 below measures how strongly the leading models depend on them before multi-fold validation.

## Round 3 tier ablation and feature audit

The datacard calls `loyalty_tier` a spending-based tier and `engagement_level` an engagement classification, but provides no creation time. Their high permutation importance prompted a controlled retrain: remove each tier separately, then both, while retaining the Round 2 settings, the original stratified 80/20 rows, and the noise-column removal. The combined dataset was also tried without both tiers. Macro F₁:

| Available features | HGB | CatBoost |
| --- | ---: | ---: |
| Original, both tiers | 0.74944 | **0.75025** |
| No `loyalty_tier` | 0.64839 | 0.65255 |
| No `engagement_level` | 0.71125 | 0.71809 |
| Neither tier, original | 0.61262 | 0.62544 |
| Neither tier, combined | 0.61259 | 0.62212 |

Removing `loyalty_tier` costs about **0.10** macro F₁ in both families; removing `engagement_level` costs **0.03–0.04**. Removing both costs **0.12–0.14**. The 14 engineered behavior features do not recover the no-tier performance: HGB changes by **−0.00003** and CatBoost by **−0.00332** when moving from original to combined. In CatBoost, Social F₁ falls from **0.7089** with both tiers to **0.4906** without them. This is evidence of strong predictive reliance, **not proof of target leakage**.

The no-tier CatBoost permutation ranking shifts to `vip_tier` (macro F₁ drop **0.0741**), `spending_frequency` (**0.0495**), `avg_monthly_spending` (**0.0443**), `days_since_last_login` (**0.0345**), and `total_spending_thb` (**0.0319**). `vip_tier` is another derived spending-related tier and should be included in the provenance audit. Raw spending, recency, social, and competitive inputs remain available but do not encode the removed tiers as strongly at these settings.

### What to do before final k-fold validation

1. **Establish the prediction cutoff and feature lineage.** Ask when `loyalty_tier`, `engagement_level`, and `vip_tier` were computed, which observations they use, and whether they exist when a new player's segment must be predicted. The [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls) defines unavailable-at-prediction information as leakage; a high score or a large ablation drop alone cannot answer the timing question.
2. **Choose the allowed feature set before further selection.** If the tiers are available at the prediction cutoff, keep the original CatBoost and HGB as the main multi-fold candidates. If they are unavailable, exclude them and treat **0.62544** CatBoost as the current two-tier-free baseline, then also audit/ablate `vip_tier` and other inferred fields. Do not use Test to infer or reconstruct any tiers.
3. **Avoid another broad ratio sweep for now.** The combined dataset already tests RFM-style spending/recency, play intensity, social, and ranked interactions, motivated by [freemium-game RFM research](https://scholar.smu.edu/datasciencereview/vol6/iss2/1/) and [competitive behavior modeling](https://arxiv.org/abs/2112.04379); it did not help with or without tiers. A narrower next test would remove groups of raw spending, social, and competitive features within development data to see which family explains remaining Social/Whale errors. [Correlated-feature guidance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html) explains why group ablations can be more informative than single-column permutation rankings.

The holdout has already supported several model decisions. Once the allowed feature set is fixed, run paired stratified folds for the shortlisted configurations and report per-fold macro and per-class F₁; do not claim stability from these holdout deltas.

## Round 4 source audit and paired validation

The project has raw Train/Test, the Task 2 datacard, and competition description/rules, but no tier-generation script, formula, or timestamp. The datacard defines `loyalty_tier` as a tier based on spending, `engagement_level` as an engagement classification, and `vip_tier` as the current VIP level. The competition data description says Test follows Train's feature structure; the rules permit competition-provided data and do not prohibit these fields. This supports using them for the **provided competition prediction task**. It does **not** establish that the same tiers would exist before an operational segment prediction. A real-world use case still needs source-system lineage and an explicit prediction cutoff.

Round 4 in `src/task2/models/model.ipynb` tests the competition feature set with five paired stratified folds on the 83,200-row development portion. The 20,800-row holdout and competition Test are excluded. Each fold fits the original CatBoost, selects eight low-importance raw features **from that fold's training data only**, refits a pruned CatBoost, and fits the original HGB as a reference. The temporary fold fits are summarized in `src/task2/models/round4_cv_v1.json` rather than added to `evaluation_index.json`, whose scores are held-out model evaluations.

| Development fold | CatBoost full | CatBoost pruned | HGB full | Pruned − full CatBoost |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.74973 | 0.74996 | 0.74504 | +0.00023 |
| 2 | 0.75098 | 0.75059 | 0.74461 | −0.00039 |
| 3 | 0.75306 | 0.75240 | 0.74836 | −0.00066 |
| 4 | 0.75111 | 0.75115 | 0.74474 | +0.00004 |
| 5 | 0.74572 | 0.74586 | 0.74031 | +0.00013 |
| **Mean ± sample SD** | **0.75012 ± 0.00273** | 0.74999 ± 0.00248 | 0.74461 ± 0.00286 | −0.00013 mean |

CatBoost full beats HGB full on **all five** paired folds, averaging **+0.00551** macro F₁. Its average class F₁ is Casual **0.82112**, Grinder **0.73921**, Social **0.70924**, Whale **0.73092**. Pruning wins 3/5 folds by tiny amounts but loses more in the other two; it does not improve mean score. `device_type`, `language`, `player_type_tag`, and `account_status` are selected for pruning in all five folds, while the rest of the eight vary. Keep the full CatBoost as the provisional competition model. A smaller model could be considered for simplicity, but this experiment gives no accuracy reason to prune it.

**Decision:** The competition inputs support a CatBoost selection, but the tier provenance remains unresolved for operational use. Obtain the source-system formula and timestamp for all three tiers before deployment. If they were computed after the intended prediction cutoff, use a separate tier-free feature set and rerun validation rather than treating these CV scores as valid for that use case.
