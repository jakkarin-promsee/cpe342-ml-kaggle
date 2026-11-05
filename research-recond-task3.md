# Task 3: Next-30-day spending prediction

**Goal:** Predict `spending_30d` in THB. Competition metric is called NMAE, but the provided `docs/kaggle-detial/` pages and Task 3 datacard do **not** define its denominator. Round 1 records **MAE_thb** only; these values are not official NMAE scores.

## Data and EDA

- Train has **104,000** labeled rows, 32 numeric features, and unique `player_id`; Test has **25,889** rows with the same inputs and no target. There are no overlapping IDs, negative targets, infinite values, or duplicate rows.
- **50,049** Train targets (**48.1%**) are zero. Median spending is about **฿79**, the 99th percentile about **฿150,032**, and the maximum **฿300,000**. This makes zero/positive errors and the upper tail worth inspecting separately.
- Missingness is around 6% for many inputs and is retained in the EDA artifacts. The model Pipelines fit imputation on development rows only.
- `src/task3/eda/build_eda_original.ipynb` saves `eda_original_v1` with all 32 supplied features. `src/task3/eda/build_eda_1.ipynb` saves `eda_combined_v1` with those columns plus **16** deterministic row-level features.
- The engineered groups cover purchase value and recency, spend and transaction intensity relative to account age, discount exposure, session intensity, and social/cross-game activity. These are task-specific formulas informed by [freemium-game RFM analysis](https://scholar.smu.edu/datasciencereview/vol6/iss2/1/), [gaming spending prediction using past transactions and feature interactions](https://arxiv.org/abs/2409.17077), and [playtime behavior research](https://arxiv.org/abs/1912.06972). No external data enters training or feature construction.
- The datacard describes `historical_spending` and `prev_month_spending` as past values and `segment` as the Task 2 classification. It supplies no row-level cutoff timestamps, so operational availability of these features cannot be proven from this static dataset.

## Round 1 models

All five models use the same 80/20 split stratified by zero and positive-spending bands (`SEED=42`). Categorical codes (`primary_game`, `platform`, `seasonal_spending_pattern`, `segment`) are one-hot encoded. Numeric values use median imputation with missing indicators inside saved Pipelines. The two tree families use an absolute-error objective.

| Model | Validation MAE (THB) | Zero-target MAE | Positive-target MAE |
| --- | ---: | ---: | ---: |
| LightGBM combined | **2,957.98** | 1,733.35 | 4,094.08 |
| HGB original | 2,959.57 | 1,733.18 | 4,097.30 |
| LightGBM original | 2,960.34 | 1,741.54 | 4,091.02 |
| HGB combined | 2,997.50 | 1,730.54 | 4,172.87 |
| Global median | 10,431.34 | 78.61 | 20,035.67 |

The 16 features improve LightGBM by only **฿2.36** MAE and worsen HGB by **฿37.93** at matched settings. The provisional leader, `t3_lgb_combined_v1`, beats `t3_hgb_original_v1` by only **฿1.59** on this holdout. Its permutation importance is dominated by `prev_month_spending` (MAE increase about **฿6,407**), `historical_spending` (**฿2,700**), and `segment` (**฿1,430**). This is model reliance, not causation.

The leader predicts median zero for actual spenders in the ฿1–1,000 band, while actual spending above ฿50,000 has MAE around **฿20,507**. Tree models can also output small negative amounts (the leader has **480** negative predictions among 20,800 validation rows). Before final prediction, explicitly handle nonnegative outputs and reassess the same validation metric. A later round can test a two-part spender/amount approach or targeted tail handling; this Round 1 does not claim those would help.

**Next:** Obtain the competition's exact NMAE denominator, then compute that metric without guessing. Check the small LightGBM–HGB difference on multiple development folds, examine spending bands, and test a nonnegative prediction step. Do not use competition Test for tuning or pseudo-labeling.
