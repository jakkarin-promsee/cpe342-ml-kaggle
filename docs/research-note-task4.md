# Task 4 — Game Title Detection

**Current sources only:** `src/task4/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, and `src/task4/models/build_model_v2.ipynb`. The older notebooks are excluded; the current notebooks reuse artifacts from them after checking image hashes, labels, and feature schema.

## EDA and features

The new manifest has **31,546** labeled Train images and **25,889** Test images. Both current EDA notebooks check IDs, filenames, labels, every image's SHA-256, and Train/Test hash separation, then remap the verified old features to new IDs/order. `eda_original_v2` contains **384** image features; `eda_1_v2` contains **1,228**. The current notebooks explicitly name `id`, `class`, `image_sha256`, `width`, and `height` as non-model columns, but **do not enumerate the 384 or 1,228 feature names or reimplement their extraction**. An exhaustive feature-name list cannot be sourced from current notebooks alone.

## Models and results

The current model notebook verifies that old and new Train image hashes, labels, and feature columns match, reuses the already fitted candidates, and re-evaluates on the same stratified image-hash fold. This is a **validated copy, not a new training run**.

| Copied model | Dataset | Validation macro F₁ |
| --- | --- | ---: |
| `t4_lgbm_31_tuned_features_r2` | `eda_1_v2` | **0.77164** |
| `t4_lgbm_31_features_r2` | Image-feature variant | 0.76267 |
| `t4_hgb_tuned_features_r2` | Image-feature variant | 0.75287 |
| `t4_lgbm_15_tuned_features_r2` | Image-feature variant | 0.74278 |
| `t4_logreg_features_r2` | Image-feature variant | 0.72128 |
| `t4_extratrees_features_r2` | Image-feature variant | 0.61219 |
| `t4_lgbm_tuned_pixels_r2` | Pixel variant | 0.59073 |

## Insight and next step

Engineered image features beat the listed pixel representation, and tuned LightGBM leads this single validation fold. Make the v2 EDA self-contained: regenerate its image features from current raw files and print or save the full feature-name list. Then train and compare models on the current artifact directly, ideally across more than one grouped fold, before treating the copied winner as final.
