# Task 4 — Game Title Detection

The current `data/raw/test.csv` has `id,file_name,label` and uses `ANS...` IDs in a new order. `train.csv` has the same schema. The image files and train labels match the previous dataset; the old manifests and `v1` artifacts remain available.

Run `eda/build_eda_original_v2.ipynb` and `eda/build_eda_1_v2.ipynb` to create the refreshed EDA cards. They map `file_name` to the existing image features, verify SHA-256 for every current image, and preserve the new manifest IDs and order. `models/build_model_v2.ipynb` copies all 28 fitted candidates into new `r2` model artifacts linked to the `v2` EDA cards and checks each validation score. It does not retrain, because the train images and labels are unchanged.

The old and new winners have identical validation scores, so `best_model(4)` still returns the older `r1` entry under the current tie rule. For predictions on the refreshed test manifest, explicitly select `model_name="t4_lgbm_31_tuned_features_r2"`.
