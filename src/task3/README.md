# Task 3 — Monthly Spending Prediction

The instructor's 12 November dataset update did not change Task 3. The existing `data/raw/train.csv`, `data/raw/test.csv`, and `data/raw/datacard.txt` remain current. The EDA notebooks in `eda/` save `eda_original_v1` and `eda_combined_v1` with their JSON cards and Train/Test Parquet splits.

`models/build_model.ipynb` contains the original model comparisons and five development-fold analyses. `models/build_model_v2.ipynb` reads `eda_combined_v1` directly from its card, uses the same development/holdout split, and tests a fixed CatBoost/LightGBM blend, a zero-aware classifier plus positive-amount model, and a fixed mixture of those candidates. Each has a distinct saved model and evaluation card. Fold predictions are cached in `models/results/` for review and guarded reruns.

The strongest measured holdout candidate is `t3_gate_blend_cv5_v2`, with **2,845.10 THB MAE**. Later cells in `build_model_v2.ipynb` show that globally scaling positive amounts and replacing the positive-only CatBoost with LightGBM do not improve paired development-fold MAE. The competition's NMAE denominator is not documented in the supplied materials, so these are not NMAE scores. See [`docs/research-note-task3.md`](../../docs/research-note-task3.md) for the paired-fold results, error breakdown, and limitations.
