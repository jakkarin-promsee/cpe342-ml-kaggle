# Player Intelligence System — CPE342 ML

Part of **CPE342 Machine Learning (1/2026)**, Computer Engineering, KMUTT.

## Overview

Welcome to the **Karena Thailand Player Intelligence Competition**. This machine learning challenge covers five tasks that simulate gaming industry problems. The goal is to build an end-to-end AI-powered Player Intelligence System for Karena Thailand, a gaming platform that develops and publishes games including PUBE Mobile, Free Fried, FiveN, Roblock, and 7-Eleven Knight.

| Task | Problem type | Evaluation metric |
| --- | --- | --- |
| 1. Anti-Cheat Pre-Filter | Binary classification | Macro F₂ score |
| 2. Player Segment Classification | Multiclass classification | Macro F₁ score |
| 3. Monthly Spending Prediction | Regression | Normalized MAE (NMAE) |
| 4. Game Title Detection | Image classification | Macro F₁ score |
| 5. Account Security Monitoring | Anomaly detection | Macro F₃ score |

Each task is a component of the Player Intelligence System.

## Background

Karena Thailand is a gaming platform in Southeast Asia with over 50 million registered players. The company faces challenges in maintaining game integrity, understanding player behavior, optimizing monetization, and improving retention.

## Tasks

### Task 1: Anti-Cheat Pre-Filter

**Problem:** Automated monitoring flags 2,000 suspicious accounts daily, but 60–70% are false positives involving skilled legitimate players. Manual review can handle only 200 cases per day, creating a backlog of more than 8,000 accounts.

**Goal:** Build a binary classifier that separates true cheaters from false positives among flagged accounts.

**Metric:** Macro F₂ score.

### Task 2: Player Segment Classification

**Problem:** Generic marketing campaigns achieve only 2–5% engagement. The company spends ฿60 million per month on promotions but cannot identify the top 5% of players (whales), who generate 60% of revenue.

**Goal:** Classify players into four behavioral segments.

| Class | Segment | Description |
| --- | --- | --- |
| 0 | Casual Player | Relaxed play, low spending |
| 1 | Competitive Grinder | High playtime, ranked focus |
| 2 | Social Player | Friend-focused, cosmetic spending |
| 3 | Whale | High spending, status-driven |

**Metric:** Macro F₁ score.

### Task 3: Monthly Spending Prediction

**Problem:** Finance misses revenue targets by 35–40%. VIP support spends resources on non-spenders while players spending ฿255,000 wait days for responses. Marketing offers 50% discounts to whales who would pay full price, losing more than ฿6 million monthly.

**Goal:** Predict each player's spending in THB over the next 30 days.

**Metric:** Normalized MAE (NMAE).

### Task 4: Game Title Detection

**Problem:** Customer support receives 2,000 image-based tickets daily and spends more than 10 minutes identifying the game before routing each ticket. The “Play Multiple Games for Rewards” campaign receives more than 50,000 screenshots and can manually verify only 3% of them.

**Goal:** Identify the game shown in a screenshot.

| Class | Game |
| --- | --- |
| 0 | PUBE Mobile |
| 1 | Free Fried |
| 2 | FiveN |
| 3 | Roblock |
| 4 | 7-Eleven Knight |

**Metric:** Macro F₁ score.

### Task 5: Account Security Monitoring

**Problem:** More than 5,000 account thefts are confirmed each month, and the actual number may be twice as high. Bot farms operate around the clock and cause economy inflation. Some players use VPNs to obtain 70% regional pricing discounts. The current rule-based system has a 60% false positive rate.

**Goal:** Use unsupervised anomaly detection to flag unusual behavior patterns without labeled data.

**Metric:** Macro F₃ score.

## Getting help

If a submission fails:

1. Check the error message for the specific issue.
2. Validate the CSV format against the sample submission.
3. Post the error details in the competition Discussion forum.
4. Contact the competition organizers if the problem persists.

## Materials

- Instruction Slide
- Group Registry (use a KMUTT email address)

> The source description uses relative competition dates (“Start: 4 days ago” and “Close: 20 days to go”). Check Kaggle for the current schedule.
