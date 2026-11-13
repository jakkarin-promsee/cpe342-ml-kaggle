# Task 5 — Account Security Monitoring

The refreshed competition data supplies only `data/raw/test.csv` and no anomaly labels. The user reports a lecturer clarification permitting unsupervised fitting on this file; the written competition rule still says Test must not be used for fitting. Keep that exception explicit in any report. No offline macro F3 can be measured.

Run `eda/build_eda_original_v2.ipynb` and `eda/build_eda_1_v2.ipynb` to create the current test-only EDA cards. `models/build_model_v2.ipynb` fits detector families and records an unlabeled rank-stability proxy. It cannot identify the best anomaly detector.

The later model notebooks use one controlled Kaggle change at a time, each with a new model, per-task submission, combined submission, and JSON lineage card:

| Notebook | Task 5 change | Combined submission |
| --- | --- | --- |
| `models/build_model_v3.ipynb` | Saved OCSVM ranking at about 15% alerts | `kaggle_t12345_t5_alert15_v8.csv` |
| `models/build_model_v4.ipynb` | Same ranking at about 25% alerts | `kaggle_t12345_t5_alert25_v9.csv` |
| `models/build_model_v5.ipynb` | Same ranking at about 35% alerts | `kaggle_t12345_t5_alert35_v10.csv`, unscored |
| `models/build_model_v6.ipynb` | Saved GMM ranking with exactly v9's 6,531 alerts | `kaggle_t12345_t5_gmm_match_v9_v11.csv` |

These notebooks keep Tasks 1–4 and the confirmed `1 = anomaly` mapping fixed. `models/rank_diagnostics_v1.ipynb` compares saved detector rankings and the exact v9/v11 predictions without labels. The best user-reported overall Kaggle score is **0.74593 for v9**; v11 scores **0.74521** at the same alert count. These overall scores do not reveal Task 5 F3 or the task weights. See [the Kaggle score log](../../docs/kaggle-score-log.md) for the complete score history and [the research note](../../docs/research-note-task5.md) for feature names, proxy results, and limitations.
