# Task 4 — Game Title Detection

Task 4 classifies supplied screenshots. The 12 November refresh changed the **Test manifest IDs/order**, while the image files and Train labels stayed the same. Use the **v2 EDA cards** and current sample order for submissions. The selected model is `t4_resnet18_geometry_full_v2`; the metric is **macro F₁**. See the [research note](../../docs/research-note-task4.md) for feature families, fold results, and limits.

## Read the two paths together

| Step | Notebook → artifact | What it establishes |
| --- | --- | --- |
| 1, historical feature extraction | [build_eda_original.ipynb](eda/build_eda_original.ipynb) → `eda_original_v1`; [build_eda_1.ipynb](eda/build_eda_1.ipynb) → `eda_1_v1` | Extracts pooled pixels and then broader color, texture, and spatial image features from supplied images. These explain the early tabular baseline; their old Test IDs are stale. |
| 2, current manifest | [build_eda_original_v2.ipynb](eda/build_eda_original_v2.ipynb) → `data/eda/eda_original_v2.json`; [build_eda_1_v2.ipynb](eda/build_eda_1_v2.ipynb) → `data/eda/eda_1_v2.json` | Reads refreshed Train/Test manifests, rekeys the existing feature tables to current IDs/order, and verifies image SHA-256 hashes and labels. Both v2 EDA notebooks produce separate saved datasets. |
| 3, tabular baseline | [build_model.ipynb](models/build_model.ipynb) | Compares majority, linear, tree, HGB, and LightGBM on v1 pooled pixels versus engineered image features using image-hash grouped validation. Tuned engineered LightGBM scored **0.77164** holdout macro F₁. |
| 4, current model notebook | [build_model_v2.ipynb](models/build_model_v2.ipynb) | First copies the 28 unchanged fitted tabular candidates to v2 cards after hash/label checks; it does not retrain them. Then compares frozen and fine-tuned pretrained ResNet18, geometry augmentation, geometry plus blur/noise, and wider rotation on grouped image folds. |
| 5, selected full-data fit | Later cells of [build_model_v2.ipynb](models/build_model_v2.ipynb) → `save/t4_resnet18_geometry_full_v2.json` and `.pt` | Geometry wins across paired folds. A third grouped fold confirms the fixed setup; three-fold mean macro F₁ is **0.95447**. The selected seven-epoch model is refitted on all labeled Train images. Its evaluation card stores the **procedure estimate**, not a score measured on the full-data checkpoint. |
| 6, per-task prediction | Later cells of [build_model_v2.ipynb](models/build_model_v2.ipynb) → `submission/t4_resnet18_geometry_full_v2.csv` | Loads the saved checkpoint through [image_training.py](models/image_training.py), predicts current Test images, and writes the Task 4 file with current `ANS` IDs. The combined submissions use this frozen Task 4 column. |

The direct-image ResNet branch uses the supplied RGB images and the saved manifest; it does **not** require a new tabular EDA feature artifact. Pretrained weights are allowed for Task 4. The v2 tabular feature cards depend on previously extracted image features and are not a fresh end-to-end extraction from raw images; regenerate them if that route is revisited. The model checkpoint includes weights and inference preprocessing metadata, so prediction does not download weights.

## Validation and submission checks

Image-hash grouping keeps exact duplicate screenshots in one validation split. It cannot rule out different photos of the same underlying screenshot. Geometry augmentation is applied only to training images; validation images are untouched. The selected procedure's grouped-fold macro F₁ is **not** a public Kaggle score. The [pipeline guide](../pipeline/README.md) explains cards and artifact paths; [the score log](../../docs/kaggle-score-log.md) and [submit plan](../../submit-plan.md) track combined submissions and their frozen Task 4 column.
