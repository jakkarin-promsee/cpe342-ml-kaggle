# Task 2 — Player Segment Classification

The refreshed competition CSVs are in `data/raw/train.csv` and `data/raw/test.csv`. The files ending in `-outdate-12-11-2026.csv` and all `v1` EDA and model artifacts remain available for tracing the previous dataset.

Run `eda/build_eda_original_v2.ipynb` and `eda/build_eda_1_v2.ipynb` for the refreshed data. They save `eda_original_v2` and `eda_combined_v2`. `models/build_model_v2.ipynb` reads those cards and saves distinct `v2` model artifacts. The notebook retains the original feature and model logic while changing input paths and artifact names.
