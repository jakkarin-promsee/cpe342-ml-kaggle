# Notebook artifact pipeline

For a notebook-first walkthrough in Thai, see [CALL.md](CALL.md).

Use `from src.pipeline.random_seed import set_seed` and `SEED = set_seed()` at the start of a notebook. Pass `SEED` explicitly to `random_state` parameters; use `set_seed(include_torch=True)` for PyTorch notebooks.

Each EDA notebook produces one named dataset. A model notebook can produce many named models. The files on disk connect the steps; notebooks do not need to share variables.

## Names and files

| Step | Main function | Output |
| --- | --- | --- |
| EDA | `save_data` | `taskN/data/eda/<name>-train.parquet`, `<name>-test.parquet`, `<name>.json`, and split cards |
| Model | `save_model` | `taskN/save/<name>.joblib` and `<name>.json` |
| Validation | `evaluate_model` | `taskN/save/<name>-evaluation.json` and `evaluation_index.json` |
| Prediction | `build_sample_submission` | `taskN/submission/<name>.csv` and `<name>.json` |

`evaluation_index.json` is the **mutable score index**. It lists results and the best model per metric. Each model's evaluation file records that model's result. Data, model, evaluation, and submission artifacts reject duplicate names unless `force=True`. Use `force=True` while refining a new artifact; once a result is relied on, give revisions a new name.

The `card` input is required for every saved artifact. A short sentence is enough; use a dictionary when more detail is useful. Paths, dates, row counts, and model/data links are filled in automatically.

## Example: EDA notebook

Run from the repository root, or add the repository root to `sys.path` in the notebook.

```python
from src.pipeline.save_data import save_data

save_data(
    task=1,
    name="eda_v1",
    train=train_df,
    test=test_df,
    card="Removed post-outcome columns; kept account activity features.",
)
```

Use `format="csv"` for CSV instead of the default Parquet.

Task 5 also supports a genuine test-only card: `save_data(task=5, name="eda_original_v1", test=test_df, card="Unlabeled Task 5 test")`. Its master card has only `splits.test`; no train split is invented. Other tasks still require `train`. Fitting on Task 5 Test conflicts with the repository's general competition rule unless explicitly directed. A test-only artifact does not provide offline F3 labels.

## Example: model notebook

```python
from src.pipeline.save_model import save_model
from src.pipeline.evaluate_model import evaluate_model
from src.pipeline.read_evalutate_summarize import best_model, read_evaluation_index
from src.pipeline.random_seed import set_seed

SEED = set_seed()

save_model(
    task=1,
    name="random_forest_v1",
    model=model,
    data_name="eda_v1",
    card={"idea": "Random forest baseline", "seed": SEED},
)
evaluate_model(
    task=1, name="random_forest_v1", card="Held-out validation split",
    y_true=y_valid, y_pred=valid_pred,
)

print(best_model(1))
print(read_evaluation_index(1))
```

Task 3 needs `normalizer=<positive competition NMAE denominator>`. Do not assume a denominator: confirm it from the competition scoring definition. Task 5 has no labeled training data, so offline F3 cannot be computed without validation labels. You may record a proxy with `score=...`, `metric_name="..."`, and `higher_is_better=True/False`; label it clearly as a proxy.

## Example: per-task prediction

```python
from src.pipeline.build_sample_submission import build_sample_submission

build_sample_submission(task=1, name="rf_submission_v1", card="First Task 1 prediction draft")
```

This reads the selected model and its linked EDA test file, then writes a pipe-separated `id|task1` file. It selects the best model from `evaluation_index.json` unless `model_name` is supplied. For image models, supply custom `loader` and `predict_fn` callbacks.

Kaggle's final submission needs all five task columns. Once each task has a prediction file, use `combine_submissions(name=..., sample_path=..., task_files={1: ..., 2: ..., 3: ..., 4: ..., 5: ...}, card=...)`. It checks every ID against the sample and writes `src/submission/<name>.csv`.

The standard workflow never uses test data in training or evaluation. Task 5 has a user-directed unsupervised exception in `src/task5/models/build_model.ipynb`; its Test-derived rank stability is only a proxy and cannot estimate macro F3. Keep fitted preprocessing inside each saved model pipeline.
