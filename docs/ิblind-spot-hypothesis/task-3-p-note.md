# Task 3: Player Monthly Spending Prediction

## 1. Actual Background (slide ref)

### Situation
Karena Thailand's finance team struggles with revenue forecasting, consistently missing quarterly targets by 35-40%, which impacts budgeting, hiring decisions, and investor confidence. Without accurate player-level spending predictions, multiple departments are operating blindly. The VIPsupport team of 20 people provides white-glove service but doesn't know which players actually deserve priority treatment—last quarter they spent significant resources on 150 high-playtime players who collectively spent only ฿12,000, while a whale who spent ฿255,000 waited 3 days for ticket response and subsequently left for a competitor. The marketing team sends promotional discount codes randomly, often giving 50% off to whales who would have paid full price anyway, resulting in ฿6 million+ monthly revenue loss. Customer service wastes time on players unlikely to ever monetize while high-value customers receive generic treatment. Additionally, the company needs to optimize their limited-time event scheduling and new content releases based on when high-spending players are most active. Accurate spending predictions would enable proper resource allocation, targeted offers, revenue forecasting, and VIP program optimization across the organization.

## 2. Information

### Task: Regression
Predict how much (in THB) each player will spend on in-game currency and items in the next 30 days
### Metric: F2
NMAE = 1 / (1 + Σ|y−ŷ| / Σy)
Target: spending_30d (continuous, THB)

### Hint:
- Zero-inflated data — try a two-stage model: (1) will spend? (2) how much?
- Feature engineering & log transformation of target
- Try XGBoost / LightGBM / CatBoost Regressors
- Ensemble / stacking for extra accuracy
- Watch for common mistakes in target scaling

### Features Example
- friend_count (float): number of in-game friends
- event_participation_rate (float): % limited-time events joined
- daily_login_streak (float): current login strea
- avg_session_length (float): avg. session duration (min)
- historical_spending (float): past spending
- 33 features total (32 predictive + target)

---

# Deep Perception analysis

## The target distribution shapes the decision

- **Zero spend is common:** In the selected mixture's OOF audit, **48.12%** of development targets are zero and account for **32.04%** of total absolute error. A model that predicts a positive amount for many of these players pays a large MAE cost.
- **The upper tail is expensive:** Targets of at least **฿50,000** are **8.16%** of rows but contribute **53.30%** of absolute error. These are error shares, not a claim that all high spenders are misclassified or equally predictable.
- **The two-stage model already addresses both groups:** It estimates whether spending will occur, estimates a positive amount, and blends that answer with a continuous regressor. Still, **10,849 false-nonzero OOF predictions** on true zero spenders account for **31.96%** of total error. High previous spending followed by no new spending is one costly pattern.
- **Why not just raise high-spender predictions?** MAE rewards a conditional **median**, not a mean. Correcting dramatic whale underpredictions can increase error on more numerous ordinary cases. Judge the whole-population MAE and error slices together.

## What has and has not improved

- **Clear improvement:** Changing only the positive-amount model to log1p(target), then inverting its predictions, moved five-fold mean development MAE from **฿2,895.92 to ฿2,814.97**. All five paired folds improved. The reused holdout improved by **฿64.16**, but it is secondary because it was consulted in prior work.
- **Controlled public check:** A combined submission changing only Task 3 improved overall public score **0.74593 → 0.74686**. That supports this submitted change on the public split, not a measured Task 3 NMAE.
- **Weak or negative alternatives:** Persistence features changed mean MAE by about **−฿0.12**; tail features by **+฿14.23**; a cost-weighted gate by **+฿30.50** (positive means worse). Log-RMSE alone lost on all five folds; a fixed half blend with log-MAE gained only **฿3.81** on reused folds. Another broad ratio or amount-scaling search is hard to justify.
- **Theory versus finite models:** A true conditional median is invariant under a monotone target transformation. The log-target gain therefore does not make log scale theoretically optimal for MAE. With finite trees and imperfect fits, a transformed target can still help in practice.

## Hypothesis: the final hurdle rule can better match MAE

- **Current behavior:** The gate uses roughly p = 0.50 to decide whether to output a positive amount. The positive amount is then blended with a continuous prediction. A median among positive spenders is generally not the median of the complete zero-plus-positive distribution.
- **Proposed rule:** Let p be a calibrated probability that spending exceeds zero. Under a well-estimated conditional positive-spending distribution:
  - Predict **zero** when p ≤ 0.5.
  - When p > 0.5, predict positive-spending quantile **q = 1 − 0.5/p**. If p = 0.60, q ≈ 0.17; q approaches the positive-spender median only as p approaches 1.
- **Why it might fail:** Gate probabilities may be miscalibrated; quantile models may be noisy; interpolation may be rough; the existing continuous blend may already compensate. This is a decision-theory hypothesis, not a measured improvement.

## One fold-safe test

1. **Preselect a small quantile set.** On each existing development fold, fit the gate on that fold's training rows and quantile models on its positive training rows. Fit preprocessing only there.
2. **Make validation predictions without peeking.** Use each fold's validation probability to choose or interpolate the positive quantile. Compare original-scale THB MAE with the current log-target mixture on exactly the same folds.
3. **Inspect tradeoffs:** Report zero, ordinary-positive, and high-spend errors, paired fold differences, and the overall MAE. A whale-only gain that worsens total MAE rejects the branch.
4. **Check source timing:** Historical/previous spending, purchase recency, segment, and VIP status need extraction windows relative to the next-30-day target. Task 3's segment is supplied directly; it was not joined from Task 2 predictions.

**Metric boundary:** The copied slide says “Metric: F2” while showing an NMAE formula. This is regression. Confirm the official NMAE denominator/implementation and task weights before translating local THB MAE into a competition claim.

*Measured results: repository Task 3 research note. The conditional-quantile hurdle is proposed, not yet tested.*

