# Task 5 — Account Security Monitoring

The refreshed competition data is `data/raw/test.csv`. The old test file and all `v1` EDA and model artifacts remain available. There is no labeled train file.

Run `eda/build_eda_original_v2.ipynb` and `eda/build_eda_1_v2.ipynb` to create test-only EDA cards. `models/build_model_v2.ipynb` follows the previously user-directed unsupervised workflow on the current Test data and saves distinct `v2` models. Its top-5% rank stability Jaccard is an unlabeled proxy, not competition macro F₃.
