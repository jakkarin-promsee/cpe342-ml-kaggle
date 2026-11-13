# Task 4 — Game Title Detection

**Current sources:** `src/task4/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`, and `src/task4/models/build_model_v2.ipynb`. The v1 EDA and model notebook document the historical tabular search summarized below. Metric: macro F₁. The instructor clarified that only Task 4 `test.csv` changed; the current EDA notebooks verified all image hashes and train labels before rekeying the existing feature artifacts. No new EDA dataset was needed for the image experiments.

## Data and features

The current manifest has **31,546** labeled Train images and **25,889** Test images. All Train files decode as RGB at **230 × 120**. The EDA cards preserve image hashes and check Train/Test exact-hash separation. Grouped validation keeps identical image hashes in one split. The selected ResNet18 uses the supplied RGB images directly, with a 112 × 224 resize, vertical padding to 224 × 224, and ImageNet normalization; it creates no tabular engineered features.

The rekeyed `eda_1_v2` artifact has **1,228** numeric image features, fully identified by these names and exact families: `px_000`–`px_383` (384); `r_mean`, `r_std`, `g_mean`, `g_std`, `b_mean`, `b_std` (6); `r_pool_<row>_<col>`, `g_pool_<row>_<col>`, and `b_pool_<row>_<col>` for rows 0–1 and columns 0–3 (24); `h_hist_00`–`h_hist_15`, `s_hist_00`–`s_hist_15`, and `v_hist_00`–`v_hist_15` (48); `hog_000`–`hog_755` (756); and `lbp_00`–`lbp_09` (10). `eda_original_v2` contains the 384 `px_*` features. `id`, `class`, `image_sha256`, `width`, and `height` are not model inputs. The v2 EDA notebooks rekey and hash-check those features rather than re-extracting them from raw images.

## Models and results

**Journey.** The first model notebook tried majority, logistic regression, SGD linear SVM, ExtraTrees, HGB, and LightGBM on raw pooled pixels versus engineered image features, then small searches on grouped development folds. The original holdout scores include **0.11305** for majority, **0.41192** for pixel logistic regression, **0.72128** for engineered logistic regression, and **0.77164** for tuned engineered LightGBM. This established the value of handcrafted color, texture, and spatial features but left substantial errors. After only Test images changed, the v2 notebook rekeyed and hash-checked existing feature artifacts, copied unchanged candidates, and compared pretrained ResNet18 with frozen versus fine-tuned weights. It then tested geometry, extra blur/noise, and full rotation on paired image-hash folds, confirmed the selected geometry setup on a third fold, and trained a seven-epoch full-data model.

All image models start from the same pretrained ResNet18 weights, which are permitted for Task 4. Training augmentation is applied only to Train rows; validation photos are untouched. Geometry uses rotation up to ±50°, perspective distortion, and scale changes. The noise variant adds Gaussian blur and noise. The full-rotation arm changes the rotation range to ±180°.

| Model / training procedure | Grouped fold 1 macro F₁ | Grouped fold 2 macro F₁ |
| --- | ---: | ---: |
| Copied feature LightGBM / retrained matched LightGBM | 0.77164 | 0.77532 |
| Frozen pretrained ResNet18, trained classifier | 0.70728 | — |
| Fine-tuned ResNet18, original photos | 0.94637 | 0.94962 |
| Fine-tuned ResNet18, geometry | **0.95246** | **0.95528** |
| Fine-tuned ResNet18, geometry + blur/noise | 0.95533 | 0.94346 |
| Fine-tuned ResNet18, ±180° rotation | 0.94936 | — |

Geometry beats plain fine-tuning by **0.00609** and **0.00566** on the two paired folds. Blur/noise gains only **0.00286** on fold 1 and loses **0.01182** against geometry on fold 2, so geometry alone was selected. A fixed geometry run on grouped fold 3 scored **0.95568**; its per-class F₁ values were **0.97506** (`7-11k`), **0.95243** (`fiveN`), **0.96934** (`freefried`), **0.96221** (`pube`), and **0.91937** (`roblock`). The selected geometry procedure's three scores average **0.95447**. This is an exploratory three-fold estimate because folds 1–2 informed the variant choice; it is not an independent competition score.

The saved full-data model `t4_resnet18_geometry_full_v2` was trained for seven fixed epochs on all labeled Train images, using the median best epoch from the three geometry folds (7, 8, 7). Its `macro_F1_grouped_cv3` evaluation card records the **procedure estimate**, not a score measured on the full-data fit. `build_sample_submission` generated `src/task4/submission/t4_resnet18_geometry_full_v2.csv`; its 25,889 IDs match the current sample submission in order. Test was used only for prediction and format checks.

## Insight and next experiment

End-to-end fine-tuning is the main gain over handcrafted image features. Geometry augmentation adds a smaller but repeated improvement, while heavier noise/blur and ±180° rotation do not improve natural validation consistently. The old LightGBM's weakest class, `roblock`, rose from about **0.68** F₁ on fold 1 to about **0.92** on geometry folds 2–3. Hash grouping prevents exact-image leakage but cannot rule out differently photographed copies of the same source screenshot; this remains a possible source of optimistic validation. Stop broad Task 4 search for now. If later leaderboard feedback shows a Task 4 gap, inspect residual errors by perspective, brightness, blur, and near-duplicate source images before trying a stronger backbone or test-time geometric averaging. The existing v2 EDA feature extraction remains non-self-contained and should be regenerated from raw files if that tabular route is revisited.
