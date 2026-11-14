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

The v6–v11 comparisons kept Tasks 1–4 and the confirmed `1 = anomaly` mapping fixed. `models/rank_diagnostics_v1.ipynb` compares saved detector rankings and the exact v9/v11 predictions without labels. The newer v13 submission changed Task 3 only and scored **0.74686**. V14 then kept v13's Tasks 1–4 fixed, widened the same OCSVM Task 5 alert budget from **6,531 to 9,118**, and fell to **0.74490**. V17 kept v13's Tasks 1–4 and 6,531 alerts, blended OCSVM/GMM percentile ranks 50/50, and rose to **0.74948**, the highest user-reported overall result. V16's 7,685-alert union scored **0.74700** and v15's 5,377-alert consensus scored **0.74595**, both below v17. These overall scores do not reveal Task 5 F3 or the task weights. See [the Kaggle score log](../../docs/kaggle-score-log.md) for the complete score history and [the research note](../../docs/research-note-task5.md) for feature names, proxy results, and limitations.
