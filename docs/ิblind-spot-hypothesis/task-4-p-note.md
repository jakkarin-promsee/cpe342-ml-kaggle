# Task 4: Game Title Detection

## 1. Actual Background (slide ref)

### Situation
Karena Thailand operates five distinct games but can’t automatically identify which game is being played from screenshots. The customer support team receives 2,000 daily tickets with images and wastes 10+ minutes per ticket figuring out which game before routing to specialists. The “Play Multiple Games for Rewards” campaign can't verify 50,000+ screenshot submissions, forcing manual checks of only 3% (leading to fraud). Players post 100,000+ gameplay images weekly across platforms that can't be auto-tagged for contests or moderation. Marketing can't track which games are trending without manual review. An automated detection system would enable faster support, verified campaigns, scalable content moderation, and cross-game insights.

## 2. Information

### Task: Image Classification (Multi-Class)
Build an image classifier that identifies which Karena game is being played from a single screenshot.
### Metric: F2
F1 = 2 · (P·R) / (P + R)
Target: game_title (5 classes)

### Hint:
- Review image classification basics & data augmentation
- Watch for class imbalance in images
- Try VGG16/VGG19, Vision Transformer (ViT), Swin Transformer
- Use cross-validation for CNNs; ensemble methods help
- You may need a GPU — Colab/Kaggle offer free T4 GPUs
- Kaggle: ~30 GPU hrs/week; Colab: dynamic, no fixed limit

### Features Example
- Gameplay screenshots (144p, centered crop to 230x120)
- Sources: live streams, submissions, forums, tickets
- Variety: different maps, phases, weather, UI states
- 5 classes: 7-11K, fiveN, freefried, pube, roblock
- Pre-trained models allowed for this task

---

# Deep Perception analysis

## Why the validation unit matters more than a new backbone

- **Current result:** Handcrafted image features with LightGBM reached about **0.772 macro F₁** on a grouped holdout. A pretrained ResNet18 reached roughly **0.946–0.950** on the first two grouped folds. Geometry augmentation then scored **0.95568** on a fixed third fold, giving a **0.95447 three-fold mean**.
- **Selection limit:** The first two folds helped select the geometry setup and are included in that mean. The result is a strong **exploratory procedure estimate**, not an untouched final score. On fold 3, roblock was the weakest class at **0.91937 F₁**.
- **Slide label correction:** The source block says “Metric: F2” but shows the F₁ formula. The competition uses **macro F₁**. Its suggested VGG/ViT/Swin models are possible experiments, not evidence that they will outperform the current model.

## Hypothesis: exact file hashes are not enough

- **What SHA-256 grouping guarantees:** Identical image files stay within one fold.
- **What it does not guarantee:** Separate crops, resizes, compression changes, brightness edits, or nearby captures from the same stream can have different hashes and cross folds. The draft audit suggests no repeated exact Train hashes; verify that count. Near-duplicate leakage remains **possible, not demonstrated**.
- **A second possible shortcut:** The network may use UI layout, brightness, background, or screenshot-source patterns correlated with a game. Those cues can work within this sample. The question is whether they hold under the sources and transformations relevant to final prediction.

## Focused audit and decision

1. **Find candidate pairs using Train only:** Start with pHash/dHash rather than a large embedding search. Manually inspect a manageable sample to distinguish shared-source images from similar scenes or common UI.
2. **Change the split only if warranted:** If credible source groups exist, keep each group inside one fold and re-evaluate the **same ResNet18 geometry procedure**. Compare macro F₁, per-class F₁, and residual errors against the existing folds.
3. **Inspect failure slices:** Measure errors by blur, brightness, crop/perspective, and UI density, especially for roblock. A new augmentation should answer a specific failure pattern.
4. **Interpret either result carefully:** A material drop would lower confidence in the old validation estimate. Similar performance would increase confidence but could not prove coverage of every future image source.

External example images are illustrations rather than competition data or duplicate evidence. Pretrained models are allowed for Task 4. If this audit finds no material issue, freeze the current model instead of starting a broad backbone search.

*Measured results: repository Task 4 research note. Source-level dependence and shortcut use remain hypotheses.*

