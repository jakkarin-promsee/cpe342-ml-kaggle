# Task 2: Player Segment Classification

## 1. Actual Background (slide ref)

### Situation
Karena Thailand's marketing team currently runs generic campaigns across all players, resulting in extremely low engagement rates of only 2-5%. They spend approximately ฿60 million monthly on promotional campaigns, but most fall flat because they're not personalized. For example, competitive esports tournament announcements are shown to casual mobile players who just want to relax, cosmetic item sales are advertised to players who have never made a purchase, and FiveN roleplay event invitations go to hardcore PUBE competitive grinders. This misalignment wastes massive advertising budgets and annoys players with irrelevant content. The top 5% of players (whales) generate 60% of total revenue, but the company can't identify the reliably using simple metrics like playtime alone. A high-playtime player could be a competitive grinder seeking glory, a social player enjoying friend time, or a whale collecting items—each needs completely different messaging. Proper segmentation could increase campaign effectiveness by 3-4x, translating to millions of baht in additional revenue and improved player satisfaction.

## 2. Information

### Task: Multi-class Classification
Classify each player into one of four behavioral segments to enable targeted marketing and personalized experiences.
### Metric: F2
F1 = 2 · (P·R) / (P + R)
Target: segment (0=Casual, 1=Grinder, 2=Social, 3=Whale)

### Hint:
- Leverage Task 1 tips — missing values & imbalance
- Use Stratified K-Fold to preserve class distribution
- Try XGBoost, LightGBM, or CatBoost
- Stack these models together for extra robustness
- Engineer interaction & ratio features

### Features Example
- play_frequency (float): avg. days/week played
- avg_session_duration (float): avg. session length (min)
- total_spending_thb (float): lifetime spending
- avg_monthly_spending (float): average monthly spend
- friend_count (float): number of in-game friends
- 45 features total (44 predictive + target)

---

# Deep Perception analysis

## What the experiments establish

- **Current local result:** Baseline CatBoost scores **0.75256 mean macro F₁** on five development folds. The tuned weighted version scores **0.75416**, about 0.0016 higher after multiple comparisons. The full-data refit is a submission candidate, not a new validation run.
- **Tier reliance is substantial:** Removing loyalty_tier, engagement_level, and vip_tier together lowered the baseline from **0.75256 to 0.58736** in a separate five-fold check. Removing loyalty_tier alone lowered a reused holdout result from **0.75041 to 0.6544**. This proves predictive reliance, not leakage.
- **Previously tried model changes have limited support:** Class weighting, probability offsets, a loyalty × engagement tier cross, tuning, and fixed blends were tested. The best fixed CatBoost + LightGBM blend scored **0.75215**, below tuned CatBoost, losing all five paired folds. Another broad ensemble search has little rationale.
- **Metric correction below the slide:** The copied source says “Metric: F2” while displaying an F₁ formula. The competition uses **macro F₁** for this task.

## The deeper question is how the segments were defined

- **Tier provenance:** Were loyalty, engagement, and VIP tiers computed from historical behavior available at prediction time? Or do their formulas/windows overlap the process that made the segment label? All four classes appear within observed loyalty and engagement tiers, so the tiers do not simply encode one class. Timing and formula evidence is needed.
- **Feature semantics:** The data card describes play_frequency as days per week, yet some observed values exceed seven. Any engineered “hours” derived from it should be called a proxy until its unit is clarified. Numeric-looking vip_tier, competitive_rank, and preferred_game_mode may be ordered levels or category codes.
- **Possible behavioral conflicts:** High spending with low engagement, strong social activity with little team play, and old accounts with little playtime could create ambiguous profiles. These are proposed error slices, not a proven geometry of the four classes. Social behavior may form an axis distinct from spending and engagement.

## A focused provenance and error audit

1. **Measure what the tiers know:** Inspect class mix inside loyalty × engagement groups, report group sizes, and fit a simple tier-only model on matched folds. Then check whether spending, play, and social variables still separate labels *within* each combination.
2. **Check near-matching profiles:** Examine accounts with similar tier and raw behavioral values but different labels. Look at probabilities and confusion patterns before attributing the disagreement to data or model error.
3. **Validate units and types:** Inspect play_frequency's range and relationship to playtime and session duration; confirm whether tier/rank/game-mode codes have a meaningful order.
4. **Reopen modeling only for a specific finding:** If a prespecified conflicting-axis group concentrates errors, compare one conditional feature or class-specific decision adjustment against the full baseline on paired folds. Report per-class F₁. If the apparent overlap persists, more ratios cannot create missing label information.

**Decision:** Freeze broad tier-cross and ensemble work. A provenance correction can improve the team's interpretation even if the macro F₁ score does not move.

*Measured results: repository Task 2 research note. Tier timing and label-generation rules remain unknown.*

