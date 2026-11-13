# Task 2 — Player Segment Classification

The refreshed competition CSVs are in `data/raw/train.csv` and `data/raw/test.csv`. The files ending in `-outdate-12-11-2026.csv` and all `v1` EDA and model artifacts remain available for tracing the previous dataset.

Run `eda/build_eda_original_v2.ipynb` and `eda/build_eda_1_v2.ipynb` for the refreshed data. They save `eda_original_v2` and `eda_combined_v2`. `models/build_model_v2.ipynb` reads those cards and saves distinct `v2` model artifacts. The notebook retains the original feature and model logic while changing input paths and artifact names.

`models/build_model_v3.ipynb` reuses `eda_original_v2` and the saved CatBoost baseline. It checks class-level errors, fits probability biases on five development folds, and evaluates one saved decision-rule wrapper on the reused holdout. Holdout macro F1 increases from **0.75041** to **0.75222**, but the leave-one-fold-out gain is only **0.00020** on average, so it has not replaced the scored Task 2 submission. See [`docs/research-note-task2.md`](../../docs/research-note-task2.md) for the comparison and limitations.
