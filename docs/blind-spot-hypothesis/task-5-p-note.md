# Task 5: Account Security Monitoring

## 1. Actual Background (slide ref)

### Situation
Karena Thailand faces over 5,000 confirmed account thefts monthly (actual number likely 2x higher), making it the #1 customer support issue consuming 60% of team resources. Beyond theft, sophisticated account selling rings destroy competitive integrity, bot farms operate 24/7 causing game economy inflation, and players exploit VPNs for regional pricing (70% discounts), costing millions of baht annually. The current rule-based security system has 60% false positive rate—flagging legitimate travelers while missing smart attackers who wait 24 hours between actions. Simple hresholds flag players who legitimately improve while missing gradual botting patterns. The security team drowns in alerts while real threats slip through. The company needs an intelligent unsupervised anomaly detection system that learns normal behavior patterns and flags truly unusual activities without rigid rules or labeled data.

## 2. Information

### Task: Anomaly Detection (Unsupervised)
Build an unsupervised anomaly detection model to flag accounts with unusual behavior, using techniques like Isolation Forest or Autoencoders — without relying on labeled training data.
### Metric: F2
F3 = 10 · (P·R) / (9P + R)
Target: is_anomaly (0=Normal, 1=Anomalous)

### Hint:
- Try Isolation Forest, One-Class SVM, or Auto-Encoder
- Even unsupervised models can be ensembled
- Read up on soft vs. hard voting techniques
- Tune contamination / threshold carefully

### Features Example
- login_count_1/2/3/4 (float): logins per time period
- login_lat/lon_1/2/3/4 (float): login coordinates
- location_changes_1/2/3/4 (float): significant location changes
- device_count_1/2/3/4 (float): unique devices used
- 33 features total (32 predictive + target)
- Note: no labeled training set provided — test.csv only

---

# Deep Perception analysis

## Start with the evaluation limit

- **No supervised check is available:** Task 5 supplies **25,889 accounts** without is_anomaly labels. We can inspect detector scores, rankings, alert overlap, and selected profiles, but cannot compute offline macro F₃, precision, or recall. The supplied security flags are model inputs, **not** ground truth.
- **A stable ranking can still be wrong:** Stability asks whether a detector repeats its choices, not whether the selected accounts are truly anomalous. Likewise, flag enrichment or a persuasive 2D plot cannot validate labels.
- **Slide label correction:** The copied source says “Metric: F2” while displaying the F₃ formula; this task's competition metric is **macro F₃**.

## What the controlled public comparisons show

The table reports **combined overall public scores**, not Task 5 F₃. Tasks 1–4, 1 = anomaly, and the **6,531-alert** budget were fixed.

| Task 5 ranking | Overall public score |
| --- | ---: |
| v17 OCSVM/GMM percentile-rank blend | 0.74948 |
| v20 engineered Empirical Tail | **0.75338** |
| v21 LOF | 0.70060 |

- **Detector change:** Tail raised overall score **0.00390** versus v17. LOF fell **0.04888** below v17, even though its alert set overlapped v17 more than Tail's did. Closeness to the older winner and rank stability did not predict public performance.
- **Behavior additions:** v20 Tail plus peer-change scored **0.75004**; adding bot-regularity scored **0.74201**. A separate 50% peer addition to v17 scored **0.72500**. The tested score definitions and weights lost on the public split. This does **not** prove that takeover, within-account change, or bots are absent from the hidden positives.
- **What we still cannot infer:** Hidden labels and task weights are unavailable. The controlled changes isolate which predictions changed, but cannot tell us the true anomaly types, Task 5 F₃, or why each swapped alert helped or hurt.

## Hypothesis: Tail may count one behavior several times

- **How v20 works:** Empirical Tail sums the **eight largest two-sided marginal ECDF-tail surprises** for each account.
- **Potential duplication:** Four period-specific login counts and engineered mean, range, spread, or largest step can all be unusual because of one underlying login pattern. Eight large contributions may therefore represent fewer than eight independent behaviors. We have **not yet measured** how often this happens among alerts.
- **An alternative anomaly idea:** One account may have very high but internally consistent activity; another may show moderate device, location, transfer, and security changes that are unusual *together*. Marginal rarity and unusual combinations need not produce the same ranking. We do not know which definition the hidden labels reward.
- **Candidate mechanisms:** Identity instability, geographic inconsistency, economic anomalies, behavioral change, automation-like regularity, and security events are investigation families, **not** organizer-confirmed anomaly classes.

## Explain the saved winner before changing it

1. **Reproduce the artifact:** Recover v20's saved continuous scores and all **25,889 binary decisions exactly**. Otherwise, an explanation might describe a different detector than the scored submission.
2. **Decompose the eight contributions:** For every account, record the contributing features and amounts. Group raw and engineered columns into behavior families; measure distinct-family count and the largest family's share, especially among alerts.
3. **Inspect detector disagreements:** At the same budget, compare Tail-only, v17-only, shared alerts, and shared non-alerts. Show actual detector scores and feature values, including missing/imputed fields and security flags. These describe detector preference, not anomaly truth.
4. **Keep visual and surrogate limits explicit:** A 2D PCA plot cannot reconstruct a detector over hundreds of inputs. A tree surrogate should first show held-out score/rank and alert fidelity before anyone reads its importance. Fidelity to v20 is **not** F₃.
5. **Only then consider a new score:** If repeated contributions from one family are common, predefine a family-balanced candidate such as maximum surprise per family before aggregating families. A cleaner explanation is not evidence of better F₃; do not search public submissions for weights.

**Additional context:** Earlier PCA results changed sharply with scaling: the first three components explained **76.70%** before clipping versus **12.93%** after robust scaling/clipping. This shows visualization sensitivity, not a reason to delete extreme values. The four periods are sequential, but the newest endpoint is unconfirmed, so directional change claims are limited.

**Rule boundary:** The written rules prohibit fitting on Test, while Task 5 supplies only test.csv and project notes report a lecturer clarification permitting use of that file. Record the exact allowed workflow before another fit. Public leaderboard feedback must not become hidden-label tuning.

*Measured results: repository Task 5 research note and Kaggle score log. Family double counting remains a hypothesis.*

