# Notebook artifact pipeline

This is the single usage guide for `src/pipeline/`. For the flow between components, see [ARCHITECTURE.md](ARCHITECTURE.md). The examples follow the working Task 1 notebooks: [`build_eda_original.ipynb`](../task1/eda/build_eda_original.ipynb) and [`build_model.ipynb`](../task1/models/build_model.ipynb). Their local helpers are shown below; they are **not** functions exported by `src.pipeline`.

## Start a notebook

Notebooks may run from the repository root or their own directory. The first cell locates the project and sets the shared seed:

```python
from pathlib import Path
import sys

ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "src" / "pipeline" / "save_data.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"

from src.pipeline.random_seed import set_seed
SEED = set_seed()
```

Pass `SEED` to every supported `random_state`. For PyTorch, use `set_seed(include_torch=True)`.

## 1. Save one EDA dataset

EDA reads competition files from `src/taskN/data/raw/`, checks quality and leakage, and prepares DataFrames. One EDA notebook writes one named dataset with `save_data`:

```python
from src.pipeline.save_data import save_data

assert len(feature_cols) == 31
assert train_clean.columns.tolist() == test_clean.columns.tolist() + [TARGET]
assert train_clean[TARGET].notna().all()
assert test_clean.equals(test_raw)

original_info = save_data(
    task=1,
    name="eda_original_v1",
    train=train_clean,
    test=test_clean,
    card={
        "source": "task1/data/raw/train.csv and test.csv",
        "preparation": "Excluded 13,184 unlabeled train rows; cast is_cheater to int8; kept all 31 raw features and player_id; no imputation or fitted transforms.",
    },
    format="parquet",
)
print(original_info["splits"]["train"]["path"])
print(original_info["splits"]["test"]["path"])
```

`save_data` receives DataFrames, not source paths. It writes `src/task1/data/eda/eda_original_v1-train.parquet`, `eda_original_v1-test.parquet`, a primary `eda_original_v1.json` card, and a JSON card for each split. Use `format="csv"` when needed. The primary card includes the format, train/test paths, row counts, columns, and supplied description.

The Task 1 model notebook loads the saved dataset **from the primary card**:

```python
import json
import pandas as pd

def load_dataset(name):
    card_path = SRC / "task1" / "data" / "eda" / f"{name}.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    def read_split(split):
        path = SRC / card["splits"][split]["path"]
        return pd.read_parquet(path) if card["format"] == "parquet" else pd.read_csv(path)
    return card, read_split("train"), read_split("test")

datasets = {
    "original": load_dataset("eda_original_v1"),
    "combined": load_dataset("eda_combined_v1"),
}
```

This is a notebook-local `load_dataset`, not a pipeline API. EDA notebooks can use an earlier saved EDA artifact when repeating preparation is costly, but must record that dependency and avoid a cycle. Task 5's genuine test-only artifact is supported with `save_data(task=5, name="eda_original_v1", test=test_df, card="Unlabeled Task 5 test")`; its card has no train split.

## 2. Guard training, then save the model

**Check for an existing result before calling `fit`.** `save_model` rejects a duplicate name, but checking only at save time would still waste the training run. The Task 1 notebook first prepares its development and held-out validation data, then defines `saved_result` and `record_model` in a shared helper cell:

```python
import joblib
import numpy as np

from src.pipeline.save_model import save_model
from src.pipeline.evaluate_model import evaluate_model

SAVE_DIR = SRC / "task1" / "save"

def model_paths(name):
    return (SAVE_DIR / f"{name}.joblib", SAVE_DIR / f"{name}.json",
            SAVE_DIR / f"{name}-evaluation.json")

def load_model(name):
    model_path, card_path, _ = model_paths(name)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["model_path"] == f"task1/save/{name}.joblib"
    return joblib.load(model_path)

def saved_result(name, dataset):
    model_path, card_path, eval_path = model_paths(name)
    if not any(path.exists() for path in (model_path, card_path, eval_path)):
        return None
    if not model_path.exists() or not card_path.exists():
        raise RuntimeError(f"Incomplete artifact for {name}; inspect files before retrying")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    expected = datasets[dataset][0]["name"]
    if card["data_name"] != expected:
        raise ValueError(f"{name} is linked to a different dataset")
    if not eval_path.exists():
        model = load_model(name)
        pred = np.asarray(model.predict(X_valid[dataset])).reshape(-1)
        tuning = card["card"].get("tuning", "none")
        return evaluate_model(
            task=1, name=name, y_true=y_valid, y_pred=pred,
            card={"split": "held-out validation 20%", "seed": SEED,
                  "tuning": tuning},
        )
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    if result["data_name"] != expected or result["metric"] != "macro_F2":
        raise ValueError(f"Unexpected saved evaluation for {name}")
    return result

def record_model(name, dataset, model, idea, tuning="none"):
    save_model(
        task=1, name=name, model=model,
        data_name=datasets[dataset][0]["name"],
        card={"idea": idea, "seed": SEED,
              "split": "stratified 80/20", "tuning": tuning},
    )
    pred = np.asarray(model.predict(X_valid[dataset])).reshape(-1)
    return evaluate_model(
        task=1, name=name, y_true=y_valid, y_pred=pred,
        card={"split": "held-out validation 20%", "seed": SEED,
              "tuning": tuning},
    )
```

These helpers assume the notebook has already defined `datasets`, `X_valid`, `y_valid`, and `SEED`. `saved_result` returns `None` only when no artifact exists, checks the linked dataset, and evaluates a saved model if only its evaluation is missing. It raises on an incomplete or mismatched artifact instead of retraining over it. Only load model files you trust.

Put each model's train, save, and evaluation in **one distinct cell**. This is the tuned HGB cell from Task 1; setup cells also define `RandomizedSearchCV`, `boosting`, `hgb_params`, `SCORE`, `CV`, `X_dev`, and `y_dev`:

```python
name = "t1_hgb_combined_tuned_v1"
if saved_result(name, "combined") is None:
    search = RandomizedSearchCV(
        estimator=boosting(), param_distributions=hgb_params,
        n_iter=6, scoring=SCORE, cv=CV, refit=True,
        random_state=SEED, n_jobs=1,
    )
    search.fit(X_dev["combined"], y_dev)
    print("Best development CV macro F2:", round(search.best_score_, 5))
    print("Best parameters:", search.best_params_)
    record_model(name, "combined", search.best_estimator_, "Development CV HGB search",
                 tuning={"cv": "3-fold stratified", "searches": 6,
                         "best_params": search.best_params_,
                         "best_cv_macro_F2": search.best_score_})
```

`save_model` writes a `.joblib` file and `<name>.json` model card linked to `data_name`. Save any fitted imputer, scaler, encoder, or threshold logic with the model when inference needs it (for example, in a scikit-learn `Pipeline`). For another model format, provide both `saver=` and `extension=`.

## 3. Evaluate each saved model

`record_model` above calls `evaluate_model` after `save_model`, using held-out validation predictions. The evaluation name must match the saved model name. The helper writes `<name>-evaluation.json` and automatically updates `src/taskN/save/evaluation_index.json`.

| Task | Metric recorded by `evaluate_model` | Extra input |
| --- | --- | --- |
| 1 | `macro_F2` | Validation labels and predictions |
| 2 and 4 | `macro_F1` | Validation labels and predictions |
| 3 | `NMAE` | `normalizer=` from the competition's definition |
| 5 | `macro_F3` only with labeled validation | Otherwise an explicitly named proxy score |

For Task 3, supply the confirmed positive NMAE denominator; do not guess it. For an unlabeled Task 5 proxy, use `score=...`, `metric_name="proxy_name"`, and `higher_is_better=True` or `False`. A proxy is **not** the competition F₃ score. Competition rules prohibit using Test for training-related work; clarify Task 5's permitted fitting workflow with the organizers.

## 4. Read the main helper results

`save_data`, `save_model`, `evaluate_model`, and the submission helpers return metadata with saved paths. Every saved artifact requires a nonempty `card`: a short sentence is enough, or use a small dictionary for source, preparation, and tuning details. The helpers fill in paths, timestamps, row counts, and links where applicable.

Names use letters, digits, `_`, and `-`. Duplicate data, model, evaluation, and submission artifacts raise before writing unless `force=True` is passed. Use `force=True` only while refining an artifact that no real result has used yet. After that, create a new name. The automatically updated `evaluation_index.json` is the deliberate exception.

```python
from src.pipeline.read_evalutate_summarize import read_evaluation_index, best_model

index = read_evaluation_index(1)
print(index["evaluations"])
print(index["best_by_metric"])
print(best_model(1, metric="macro_F2"))
```

If an index contains several metrics, pass `metric=` to `best_model`; scores from different metrics cannot be ranked together.

## 5. Compare saved models

The Task 1 notebook defines `show_rankings` in a shared cell. The core comparison is below; the notebook also displays a chart. It checks for missing evaluations and sorts **only Task 1 macro F₂** scores:

```python
from src.pipeline.read_evalutate_summarize import read_evaluation_index

def show_rankings(names, title):
    evaluations = read_evaluation_index(1)["evaluations"]
    missing = [name for name in names if name not in evaluations]
    if missing:
        raise ValueError(f"Missing evaluations: {missing}")
    rows = []
    for name in names:
        result = evaluations[name]
        if result["metric"] != "macro_F2":
            raise ValueError(f"Unexpected metric for {name}: {result['metric']}")
        card = json.loads(model_paths(name)[1].read_text(encoding="utf-8"))
        rows.append({"name": name, "dataset": card["data_name"],
                     "macro_F2": result["score"]})
    ranking = pd.DataFrame(rows).sort_values("macro_F2", ascending=False).reset_index(drop=True)
    print(title)
    display(ranking.round(4))
    return ranking
```

Call it with the saved models from the round:

```python
round1_names = [
    "t1_dummy_original_v1",
    "t1_logreg_original_v1", "t1_logreg_combined_v1",
    "t1_rf_original_v1", "t1_rf_combined_v1",
    "t1_hgb_original_v1", "t1_hgb_combined_v1",
    "t1_hgb_combined_tuned_v1", "t1_rf_combined_tuned_v1",
    "t1_hgb_combined_threshold_v1",
    "t1_hgb_original_matched_v1", "t1_rf_original_matched_v1",
    "t1_hgb_original_threshold_v1",
]
round1_ranking = show_rankings(round1_names, "Round 1 — scikit-learn models")
round1_winner = round1_ranking.iloc[0]["name"]
```

`show_rankings` is notebook-local, not a pipeline export. It uses the `json`, `pd`, `display`, and `model_paths` setup shown earlier. For another task or metric, adjust the metric check and sort direction. A round's winner is a validation comparison; final selection can also consider later cross-validation and error analysis.

## 6. Assemble the Kaggle submission

Use the complete [submission assembly notebook](../submission/assemble.ipynb) for the current five-task workflow. It loads each selected model and its linked EDA test split, applies the task-specific prediction steps, checks IDs and output columns against `src/submission/sample_submission.csv`, then writes the final file and a JSON lineage card. You do **not** need to create five per-task CSVs to run this notebook.

1. Make sure the five selected model files, model cards, evaluation cards, and linked EDA test files exist. Keep the downloaded `sample_submission.csv` in `src/submission/`.
2. Open `src/submission/assemble.ipynb`. In its first code cell, set a **new** `NAME` for this submission and review `MODEL_NAMES` and `EXPECTED_METRICS` for tasks 1–5. The current selections are Task 1 pruned LightGBM, Task 2 CatBoost, Task 3 CatBoost, Task 4 LightGBM, and Task 5 One-Class SVM.
3. Run the notebook from top to bottom. Its prediction cells restore Task 2 categorical nulls, clip negative Task 3 spending predictions to zero, map Task 4 class names to numeric labels, and produce Task 5 binary alerts. The assembly cells verify 25,889 `ANS` IDs, row counts, column order, missing values, selected artifact lineage, and the saved file hash.
4. Use `src/submission/<NAME>.csv` as the Kaggle upload file. The matching `<NAME>.json` records the selected models, scores, dataset links, and prediction adjustments. The final CSV is **comma-separated** with columns `id,task1,task2,task3,task4,task5`.

The current notebook also checks a fixed inventory of outdated models, reads `src/task4/data/raw/train-outdate-12-11-2026.csv` to verify the Task 4 label mapping, and compares Tasks 1 and 3 with `src/submission/kaggle_t12345_best_v4.csv`. Those files are local prerequisites for this version of the notebook; a fresh clone with only the new raw download cannot run it unchanged. Review or update these checks when preparing a new submission. The notebook assembles and verifies the file locally; upload it to the Kaggle competition separately.

For the later Task 3 and Task 4 experiments, `src/submission/assemble_v2.ipynb` creates `kaggle_t12345_gate_resnet_v6.csv` from the verified `kaggle_t12345_current_v5.csv`. It preserves Tasks 1, 2, and 5, predicts Task 3 with `t3_gate_blend_cv5_v2` using its linked EDA Test split, and joins the Task 4 per-task `t4_resnet18_geometry_full_v2.csv` by current `ANS` ID. It verifies the base SHA-256, all 25,889 IDs, unchanged columns, prediction ranges, and a new output SHA-256. Its JSON card records every source. The Task 3 evaluation is MAE in THB, the Task 4 value is a grouped-fold procedure estimate, and Task 5 remains an unlabeled stability proxy.

The same notebook also writes `kaggle_t12345_gate_resnet_t5flip_v7.csv`, an optional local comparison that changes only Task 5 predictions from `0` to `1` and `1` to `0`. Its separate JSON card records the exact v6 base hash and the transform. The user subsequently confirmed that **v6's original Task 5 0/1 mapping is correct**; keep it for future submissions. No numeric Task 5 F₃ score was supplied with that confirmation.

The later Task 5 notebooks (`src/task5/models/build_model_v3.ipynb` through `build_model_v6.ipynb`) use that scored v6 file as a fixed Tasks 1–4 reference and vary only Task 5. Their combined CSVs are created locally and uploaded separately. The user-reported overall Kaggle results are recorded in [the score log](../../docs/kaggle-score-log.md); **v9 scored 0.74593**, the highest reported score. The 35% OCSVM v10 draft has no Kaggle score. No per-task F3 was supplied.

The generic pipeline APIs remain available for other workflows: `build_sample_submission` writes a pipe-separated `id|taskN` per-task file from a saved model, and `combine_submissions` joins five such files against a **pipe-separated** sample submission. The current `src/submission/sample_submission.csv` is **comma-separated**, so that generic combine path does not work with it unchanged. `assemble.ipynb` contains the current task-specific inference and writes the final comma-separated CSV and JSON card directly; it is the workflow used here.
