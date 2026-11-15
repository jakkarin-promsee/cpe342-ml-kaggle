# Deep Perception analysis

## Why I am changing the question

- **What we have already done:** Schema and missing-value checks, duplicate and ID checks, feature engineering, ablations, model comparisons, and error analysis. Repeating those steps broadly is unlikely to reveal the main remaining blind spot.
- **The new question:** Do our labels, feature measurement windows, validation populations, and final decision rules match the real prediction problem? This is a hypothesis-driven audit, not a claim that the first EDA was poor.
- **How to judge an idea:** State what observation would disprove it before fitting. The lecturer slides offer techniques worth trying, but they do not prove those techniques help this dataset or expose Kaggle's hidden implementation.

Three forms of alignment help us organize the discussion:

- **Objective alignment — are we optimizing the scored decision?** A training loss can be convenient without producing the best final prediction under the competition metric. Task 3's zero-versus-positive amount decision is the clearest example.
- **Signal alignment — is the model using the behavior we mean to detect?** Importance tells us which variables the fitted model used, not whether those variables will generalize or reflect the intended mechanism. Task 1's reports and Task 2's tiers deserve this check.
- **Population alignment — does validation resemble final prediction?** Missing labels, reused folds, and related screenshots can make a local score describe a narrower or easier population than expected.

A domain story can generate a useful hypothesis. It still has to beat the full baseline under a matched experiment; we should not force a model to use features merely because their names sound right.

## What the current results actually measure

| Task | Local result | How to read it |
| --- | ---: | --- |
| [1 — Anti-Cheat](task-1-p-note.md) | **0.77733 macro F₂** | Selected 38-feature procedure on six paired outer folds; later gains have been tiny. |
| [2 — Player Segment](task-2-p-note.md) | **0.75416 macro F₁** | Tuned weighted CatBoost on five development folds used for selection. |
| [3 — Monthly Spending](task-3-p-note.md) | **฿2,814.97 MAE** | Log-positive-amount mixture on five development folds; this is not official NMAE. |
| [4 — Game Detection](task-4-p-note.md) | **0.95447 macro F₁** | Three ResNet18 geometry folds; folds 1–2 helped choose the setup. |
| [5 — Account Security](task-5-p-note.md) | **No labeled CV/F₃** | Stability and combined public scores cannot measure anomaly F₃. |

- **Best reported combined public score:** v20 scored **0.75338 overall**. The slides say Kaggle combines independently scored tasks with weights, but the individual task scores and weights are unavailable.
- **What the controlled Task 5 change shows:** v17 → v20 kept Tasks 1–4 and the **6,531-alert** budget fixed. The Task 5 ranking change raised the **overall** public score by **0.00390**. That supports this particular selection on the public split; it does not disclose Task 5 F₃, hidden anomaly types, or which task has the most remaining headroom.
- **Why the local numbers need care:** Feature/model selection reused related folds or holdouts, and Task 5 used public feedback. A selected best score may be optimistic even when every individual evaluation was computed correctly.

## Shared blind spots worth discussing

### 1. How were the data and labels produced?

- **Feature provenance:** A column name or data card may tell us what a feature represents without telling us *when* it was measured. A correctly coded pipeline can still use a source feature measured after the prediction cutoff. This is an unresolved risk, not demonstrated leakage.
- **Label provenance:** Task 2 tiers may be valid historical summaries or may overlap the segment-label rule. Task 3's prior spending and recency columns need a timing check. Task 1's missing labels may reflect who gets reviewed.
- **Rule for our explanations:** Separate **organizer-stated fact**, **measured pattern**, and **working hypothesis**. For example, removing Task 2 tiers causes a large score drop; that proves reliance, not leakage.

### 2. What does validation actually estimate?

- **Repeated selection:** Task 2 and 3 holdouts were reused. Task 4's first two folds selected augmentation and remain in its reported mean. Task 5's public leaderboard informed detector choices. These results are useful for development, but they are not independent final estimates.
- **Dependence can survive duplicate checks:** Exact row checks and image hashes catch some repeats. They do not establish source-feature timing or catch edited screenshots from one original image.
- **Better discipline:** Before fitting, fix the hypothesis, folds, metric, threshold/decision procedure, and acceptance condition. Use paired fold comparisons honestly as exploratory evidence. A strict final estimate would need an untouched set or a nested evaluation where resources permit.

### 3. Which signal does the model actually reward?

- **Importance is descriptive:** It means the fitted model used a variable. It does not mean the variable is causal, safe across populations, or the best proxy for the task.
- **Matched feature-family test:** Keep the same split, metric, preprocessing, model settings, and threshold procedure while comparing:
  1. the full validated baseline;
  2. the proposed family alone;
  3. the baseline without that family; and
  4. the baseline plus one conditional representation.
- **How to interpret results:** A family-only model close to baseline shows that the family carries signal; it does not make other features worthless. Little loss after removing a family can reflect correlated substitutes. A consistent gain from one conditional representation is stronger evidence, though reused folds still limit certainty.
- **Inspect before building:** Check conditional distributions, errors, and calibration first. State what result would refute the proposed mechanism so that we do not replace random feature engineering with a persuasive but untested story.

### 4. Does the prediction rule match the target?

- **Task 3:** The log-positive-amount model improved all five paired folds, whereas several extra ratios did not. A conditional-quantile hurdle might better approximate the whole zero-plus-positive conditional median under MAE. High-spender underprediction by itself does not justify raising predictions.
- **Task 5:** No supplied anomaly labels exist. Rank stability, security-flag enrichment, and a good-looking PCA plot cannot establish F₃. First explain what the saved Tail detector is counting, including whether several features represent one underlying behavior.

## Where teammates may reasonably disagree

- **“More features should solve it.”** A new feature can help if it captures a missing mechanism. Recent Task 1 context/behavior features and Task 3 persistence/tail ratios did not produce a meaningful matched-protocol gain. A conditional pattern and ablation should motivate the next feature, not another long list.
- **“The most realistic security story should win.”** Domain knowledge helps form hypotheses. The tested peer-change and bot-regularity blends selected plausible profiles but reduced controlled public overall scores. This rejects those *specific score rules and weights* on that split, not the existence of account takeover or bots.
- **“Task 4 is already at 0.95, so stop.”** The score is strong, but exact-hash grouping may miss crops and related captures. Check near duplicates and source grouping before treating the result as fully independent. A new backbone is not the first response.
- **“Tier importance proves Task 2 leakage.”** Strong tier reliance warrants a formula and timestamp audit. Spending and engagement can legitimately be predictive of segments, so reliance alone is insufficient.
- **“The overall leaderboard reveals our weakest task.”** Controlled column changes show whether one submitted change helped overall. Unknown weights and missing per-task scores prevent us from ranking the remaining opportunity across all tasks.

## Decision rule for each new experiment

1. **Name one mechanism and one failure condition.** Say why a change should help and which observation would weaken the claim.
2. **Match the comparison.** Use the same folds, metric, preprocessing, model settings, and decision rule; fit any learned preprocessing only inside training folds.
3. **Record more than the mean.** Save OOF predictions, paired fold scores, class/segment error slices, and settings. Look for a meaningful repeatable gain, not a tiny maximum after many attempts.
4. **Keep the data boundary.** Use competition-provided data. Follow the written Train/Validation-only fitting rule; Test is for final prediction unless the Task 5 exception is explicitly documented.
5. **Stop a failed branch.** Keep the current reference when separation is absent or gains are unstable. An unlabeled Task 5 proxy is not F₃, and repeated public submissions should not become a hidden-label search.

## Priority order and why

1. **Confirm official scoring and Task 5 permission.** The Task 3 NMAE expression appears in the slides, but implementation and overall task weights are not confirmed. Written rules prohibit Test fitting while Task 5 has only test.csv and project notes report a lecturer exception. Record the allowed workflow.
2. **Task 5 — explain v20 Tail.** Reconstruct scores, group its top-eight contributions by behavior family, and inspect v20-versus-v17 disagreements. This tells us what the strongest reported combined submission selects, without pretending to know anomaly truth.
3. **Task 3 — test the conditional-quantile hurdle.** Compare it with the **฿2,814.97 mean CV MAE** log-target mixture on paired folds. Whole-population MAE decides whether the new rule helps.
4. **Task 1 — audit the labeled population and conditional mechanics.** Ask why 13,184 Train labels are missing and whether mechanical behavior separates known classes after matching performance.
5. **Task 4 — inspect perceptual near duplicates.** Determine whether the current fold unit is defensible before another architecture search.
6. **Task 2 — establish tier and feature meaning.** Check tier windows, label overlap, and the unexpected play_frequency range before another tier cross or blend.

This is an order by **information gained**, not a promise of leaderboard improvement. Each linked task note supplies its evidence, competing explanations, and a specific test.

*Sources: the two attached audits; repository research notes for Tasks 1–5, leakage audit, Kaggle score log, and competition overview/rules/instruction slides. The slide text in each task note remains source material, including copied metric-label inconsistencies.*

