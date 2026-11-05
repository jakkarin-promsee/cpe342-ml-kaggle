# Project working instructions

This repository is the **CPE342 Player Intelligence System** competition project. Work in the existing notebook-first MLOps style. Read `src/pipeline/CALL.md` for callable examples and `docs/kaggle-detial/` for the competition description, data layout, and rules before changing task logic.

## Core workflow

1. A task's EDA notebook reads `src/taskN/data/raw/`, checks data quality and leakage, performs feature work, and calls `save_data` to create a named dataset in `src/taskN/data/eda/`.
2. A task's model notebook reads that saved dataset **by the paths in its JSON card**, trains and validates models, then calls `save_model` and `evaluate_model` for each model.
3. `evaluation_index.json` in `src/taskN/save/` lists model scores and the best model for each metric. `read_evaluation_index` and `best_model` read it.
4. `build_sample_submission` loads a selected saved model and its linked EDA test split to write `src/taskN/submission/<name>.csv`. `combine_submissions` joins all five task files against `sample_submission.csv` into `src/submission/<name>.csv`.

Notebook variables must not be a dependency between stages. The saved files and JSON cards are the interface. Do not make one notebook import variables or execute cells from another notebook.

## Notebook layout

- **EDA:** one linear, reviewable `.ipynb` per named dataset. Its cells should show how that dataset was made and where leakage was checked.
- **Models:** one `.ipynb` may create many models and configurations. Each saved model needs its own distinct `name` and a short, useful `card`.
- Use `SEED = set_seed()` from `src.pipeline.random_seed` at the start of a notebook and pass `SEED` explicitly to `random_state` parameters. For PyTorch, use `set_seed(include_torch=True)`.
- Fit preprocessing only on Train/Validation data. Save fitted preprocessing with the model when inference needs it (for example, a scikit-learn `Pipeline`).

## Artifact contract

- `save_data`, `save_model`, `evaluate_model`, `build_sample_submission`, and `combine_submissions` all require a nonempty `card`. A one-sentence string is valid; use a small dict only when it helps. Do not add a long mandatory form.
- `task` and `name` choose the output directory and filename. `name` uses letters, digits, `_`, and `-`. The functions return metadata with the saved paths. The caller supplies DataFrames to `save_data` and trained models to `save_model`; the submission builder produces predictions from a saved model. Callers do not choose arbitrary artifact destination paths.
- `src/taskN/data/eda/<name>.json` is the primary dataset card: it must identify the train/test files and include enough information to inspect the dataset without opening split cards. `<name>-train.json` and `<name>-test.json` hold split details.
- `src/taskN/save/<name>.json` identifies a model's data artifact and model file. A model notebook may save multiple such artifacts.
- A saved artifact name is normally written **once**. A duplicate must raise unless the caller explicitly passes `force=True`. Treat `force=True` as a rare way to finish a new EDA/model/submission artifact before relying on it. Once an artifact is used for a real result, use a new name for revisions so old paths keep their meaning.
- `evaluation_index.json` is the deliberate exception: evaluation updates it automatically. The per-model `<name>-evaluation.json` is an artifact and follows the duplicate/`force=True` rule. Never rank scores from different metrics as if they were comparable.
- When changing a function signature, file layout, or JSON fields, update `src/pipeline/CALL.md` and `src/pipeline/README.md` in the same change.

## Competition constraints

- Use only competition-provided data. No external datasets or web scraping. Augmentation of provided training data is allowed.
- Use only Train and Validation for fitting, tuning, and evaluation. Do not use Test for pseudo-labeling, fine-tuning, probing, or any other training-related step. Test is for final prediction and format checks.
- Pre-trained models are allowed for **Task 4 only**. Tasks 1, 2, 3, and 5 must train from scratch with the provided data. Ensembling and feature engineering are allowed. AutoML frameworks are prohibited; hyperparameter search tools are allowed.
- Metrics: Task 1 macro F₂; Tasks 2 and 4 macro F₁; Task 3 NMAE; Task 5 macro F₃. Task 3's NMAE denominator must come from the competition definition, not a guessed formula.
- Task 5 currently has only `test.csv` in the provided data description. The current `save_data` API expects train and test. Do not relabel Test as Train or invent a training set to make the API run. Clarify the permitted Task 5 workflow before implementing it. Without labeled validation data, an offline proxy is **not** the competition F₃ score.

## Implementation and checks

- Keep reusable artifact and scoring logic in `src/pipeline/`; keep task-specific EDA, feature choices, training, and analysis in `src/taskN/` notebooks or task modules.
- Prefer path construction with `pathlib` and the existing project layout. Do not hard-code a user's absolute Windows path in reusable code.
- Make duplicate checks happen before writing. Keep cards readable JSON and include enough lineage to trace an output back to its dataset and model.
- Before calling a task complete, run a focused check of changed functions with small synthetic data. Do not run expensive full training simply to verify a pipeline edit.
