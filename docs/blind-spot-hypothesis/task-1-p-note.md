# Task 1: Anti-Cheat Pre-Filter

## 1. Actual Background (slide ref)

### Situation
Karena Thailand receives over 10,000 cheat reports daily across all games, but the manual review team can only handle 500 cases per day, creating a massive backlog of 45,000+ pending reports. Many reports are false positives from frustrated players who lost matches fairly, while real cheaters continue to damage game experience and drive legitimate players away. The current automated system has a 70% false positive rate, meaning innocent players get flagged and sometimes banned, leading to customer service nightmares and permanent player loss. Meanwhile, sophisticated cheaters who use subtle aimbots [1] or wallhacks [2] slip through undetected. Player retention has dropped 15% in competitive game modes due to cheating complaints, directly impacting revenue by approximately ฿15 million daily. The company needs an intelligent pre-filtering system that can accurately identify high-probability cheaters for priority manual review, reducing workload while catching real threats

### Problem
[1] Aimbot: Cheating software that automatically aims at enemies
[2] Wallhack: Exploit allowing players to see through walls

## 2. Information

### Task: Binary Classification
Build a binary classifier to automatically pre-filter reports and prioritize high-confidence suspicious accounts for manual review.
### Metric: F2
F2 = 5 · (P·R) / (4P + R)
Target: is_cheater (0=Legitimate, 1=Cheater)

### Hint:
- Handle missing values — impute or drop carefully
- Handle class imbalance: resampling, class weights, SMOTE
- Feature engineering — build interaction features
- Use cross-validation for reliable performance
- Try decision-threshold tuning
- Ensemble models for robustness

### Features Example
- kill_death_ratio (float): kills-to-deaths ratio
- headshot_percentage (float): % kills that are headshots
- win_rate (float): overall win rate %
- accuracy_score (float): shot accuracy %
- reaction_time_ms (float): avg. reaction time (ms)
- 33 features total (32 predictive + target)

---

# Deep Perception analysis

## What is established

- **Current reference:** The selected 38-feature LightGBM scores **0.77733 mean macro F₂** on six paired outer folds, with its threshold selected inside each training fold. This is the appropriate comparison for another threshold-tuned candidate.
- **Recent alternatives barely moved the result:** The behavior-feature variant scores **0.77690** and the shallower, longer-boosted variant **0.77748** under that matched procedure. Earlier original 31-feature, combined 45-feature, and selected 38-feature procedures scored **0.77676**, **0.77679**, and **0.77733**. After several comparisons, a difference below 0.001 is weak evidence for replacing the reference.
- **Persistent errors deserve explanation:** One OOF repeat had **11,846 false positives** and **7,470 false negatives**. On the same accounts in a second repeat, **85.7%** of FPs and **79.7%** of FNs stayed wrong; far-from-threshold errors recurred at about **99%**. These repeats are not independent validation, but a simple threshold-only story is less persuasive.
- **Protocol matters:** Fixed-threshold scores and inner-fold threshold-tuned scores answer different questions. We should not rank them as if they came from the same procedure.

## Three possible sources of the remaining errors

- **Population selection:** The original EDA retained **90,816 labeled Train accounts** and excluded **13,184 with missing is_cheater**. Excluding them from supervised fitting is correct. The missing-label mechanism is unknown; perhaps the unlabeled accounts were reviewed less often or have different histories. That would make validation on labeled accounts less representative, but it is a hypothesis.
- **Signal selection:** Reports, friend network size, account age, and crosshair placement lead validation importance. Trees choose useful splits for their objective, so importance shows reliance rather than the best transferable explanation of cheating. Reports might capture exposure, player frustration, or real suspicion; the task is to distinguish these possibilities conditionally.
- **Class or label overlap:** Repeated confident mistakes might reflect mixed behaviors, missing measurements, model bias, or uncertain labels. A nearest-neighbor diagnostic in five raw features found some false negatives resembling legitimate accounts. Five dimensions are insufficient to call the labels wrong or claim an irreducible error floor.

## Hypothesis: judge performance against supporting behavior

- **Mechanism:** A high K/D and headshot percentage can be legitimate if aiming, reaction, movement, input consistency, and tactical behavior support it. Similar performance with contradictory mechanics may deserve more scrutiny.
- **Counterexample:** Some cheating methods could make the mechanical measurements look coherent. The hypothesis is testable, not a universal definition of cheating.
- **First observation:** Compare known cheaters and legitimate players **within similar performance and skill ranges**. Check distributions of aiming smoothness, reaction time, input consistency, and related signals, with group sizes and missingness. If the groups do not separate, do not invent an inconsistency feature to save the story.

## A matched experiment

1. **Population diagnostic:** Compare labeled and unlabeled **Train** rows for account age, reports, level progression, friends, aiming smoothness, and missingness. A fold-validated labeled-vs-unlabeled classifier can measure population difference; it cannot infer missing labels. Keep Test out.
2. **Feature-family comparison:** If conditional mechanics show a pattern, predefine one performance-conditioned representation. Compare the full 38-feature baseline, performance/mechanics alone, baseline without mechanics, and baseline plus that single representation.
3. **Evaluation:** Use the same six outer folds and inner threshold-selection procedure. Report mean and fold variation for macro F₂ and both class F₂ scores, plus the persistent FP/FN slices.
4. **Failure condition:** If the conditional separation is absent or paired gains are unstable, keep the 38-feature model. More ratios, class-weight/SMOTE grids, or tiny blends are less justified than understanding the error mechanism.

*Measured results: repository Task 1 research note. Missing-label and mechanical-consistency explanations remain hypotheses.*

