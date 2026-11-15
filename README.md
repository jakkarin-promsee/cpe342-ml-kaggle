# CPE342 Player Intelligence System

This repository contains the notebook-first workflow for the five tasks in the Karena Thailand Player Intelligence Competition. Git shares code and documentation; each contributor downloads the competition data and creates local data, model, and submission artifacts. Read the [overview](docs/kaggle-detial/overview.md), [data layout](docs/kaggle-detial/data.md), [rules](docs/kaggle-detial/rule.md), and [extracted instruction-slide hints](docs/kaggle-detial/hint_canva_extracted.txt) before running experiments. The hints suggest experiments; the research notes record which ones have already been tested and whether they helped.

For results and next steps, see the concise research notes for [Task 1](docs/research-note-task1.md), [Task 2](docs/research-note-task2.md), [Task 3](docs/research-note-task3.md), [Task 4](docs/research-note-task4.md), and [Task 5](docs/research-note-task5.md). [The Kaggle score log](docs/kaggle-score-log.md) records user-reported overall scores and controlled submission comparisons. The highest reported score is **0.75338** for v20 Empirical Tail; per-task scores and competition weights are unavailable.

## 1. How to load the dataset

Use Python 3.11. Join the Kaggle competition, accept its rules, and authenticate the Kaggle CLI. Replace `<slug>` with the last part of its URL (`https://www.kaggle.com/competitions/<slug>`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
kaggle auth login
python src/load-raw-dataset.py --competition <slug>
```

The requirement files are cumulative. Install the highest tier needed for the work; each higher tier includes the one below it:

| Tier | Install command | Adds |
| --- | --- | --- |
| Core | `python -m pip install -r requirements.txt` | Data handling, pipeline, notebooks, plots, and Kaggle CLI |
| Models | `python -m pip install -r requirements-model.txt` | CatBoost, LightGBM, XGBoost, Pillow, and scikit-image |
| This machine's CUDA | `python -m pip install -r requirements-cuda.txt` | Matching PyTorch and torchvision CUDA 12.8 wheels for Task 4 image training |

`requirements-cuda.txt` records the tested CUDA build on this machine. Teammates with another GPU, CUDA setup, or CPU should install the model tier and choose a compatible PyTorch/torchvision build from the [official PyTorch installation options](https://pytorch.org/get-started/previous-versions/) for their environment. Task 4 image EDA uses Pillow and scikit-image from the model tier; its ResNet18 notebook additionally needs PyTorch.

The loader unpacks each task's CSV files and `datacard.txt` into `src/taskN/data/raw/`, Task 4 images into `src/task4/data/raw/train/` and `test/`, and `sample_submission.csv` into `src/submission/`. Task 5 has only `test.csv` in the provided data description; the loader does not create a training file. The loader refuses to replace existing raw files. Its `--force` option is for an intentional replacement of a current download, never a normal experiment rerun.

## 2. How to rerun previous experiments

The `.gitignore` excludes task data, trained model files, fold-prediction caches, per-task submission files, and combined submission CSVs. It keeps `src/task*/save/evaluation_index.json` and combined submission JSON lineage cards for inspection. Those indexes and cards can refer to local artifacts that a fresh clone does not contain. Install the pinned dependencies and download the competition files first. Inspect the exact notebook order:

```powershell
python src/re-run-all-ipynb.py --dry-run
python src/re-run-all-ipynb.py --task-n 1 --dry-run
```

Run the full plan or one task:

```powershell
python src/re-run-all-ipynb.py
python src/re-run-all-ipynb.py --task-n 1
```

The runner executes current EDA notebooks before model notebooks, with original EDA datasets before derived ones. It skips explicitly listed outdated notebooks in `IGNORE_NOTEBOOKS`. It saves outputs into a notebook only after that notebook succeeds; a failed cell stops the run. Full training can take a long time. The dry run also shows any local, untracked notebooks that will run.

Rerunning against existing artifacts does **not** authorize an overwrite. Notebooks can load completed artifacts and skip training where they implement that behavior. For a complete reproduction, use a fresh clone, newly downloaded raw data, and empty generated artifact directories. To revise a result that has already been used, give the new dataset, model, or submission a new name and update downstream references. Do not delete prior results merely to make a rerun pass.

## 3. Pipeline workflow philosophy

Each stage reads a saved artifact and writes the next one. Files and JSON cards connect notebooks, so a later notebook never needs variables or cells from an earlier notebook.

| Stage | Reads | Produces |
| --- | --- | --- |
| Raw data | Competition download | Original files in `src/taskN/data/raw/` |
| EDA | Raw files, or an earlier EDA artifact when repeating preparation is costly | **One dataset per notebook** in `src/taskN/data/eda/`, with train/test files and a JSON card |
| Modeling | Dataset paths from the EDA JSON card | **Many models per notebook**, saved in `src/taskN/save/` with model and evaluation cards |
| Submission | A saved model and its linked test split | A per-task CSV in `src/taskN/submission/`, then a combined CSV in `src/submission/` |

### Rules that make runs traceable

1. **Keep each step self-contained.** EDA checks data quality and leakage. If an EDA notebook builds on an earlier EDA artifact, record that source in its card and avoid circular dependencies. Give each model its own cell for fitting, saving, and evaluation; shared setup and ranking cells can sit outside those cells. Save fitted preprocessing with the model when prediction needs it.
2. **Save through the pipeline.** Pass DataFrames to `save_data` and trained models to `save_model`; the helpers choose output paths. Every save or evaluation call needs a nonempty `card` explaining the artifact. The model card links to its dataset, and `evaluation_index.json` tracks the best model for each metric separately.
3. **Treat names as permanent.** An existing data, model, evaluation, or submission artifact cannot be written again unless `force=True` is passed explicitly. Use `force=True` only while exploring an artifact that has not yet been used by a downstream result. Once used, create a new name for revisions.

**Task 5:** The pipeline can save its genuine test-only dataset. The user reports a lecturer clarification permitting full use of that file for Task 5 modeling and analysis, while the written competition rule still prohibits Test fitting; the lecturer warned against harvesting leaderboard feedback through repeated submissions. The Task 5 stability measures are unlabeled proxies, not competition macro F₃. See [the Task 5 README](src/task5/README.md) for the current experiment sequence.

## 4. Example notebook cells

These examples follow Task 1. For pipeline calls, notebook helpers, model comparison, and submission creation, see [src/pipeline/README.md](src/pipeline/README.md). Start notebooks with `SEED = set_seed()` from `src.pipeline.random_seed` and pass `SEED` to `random_state` parameters.

### Save one EDA dataset

The EDA notebook prepares `train_clean` and `test_clean`, checks its assumptions, and calls `save_data` once for this named dataset:

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

The primary `src/task1/data/eda/eda_original_v1.json` card records the train and test paths, shape, columns, format, and description. The pipeline also writes split cards.

### Train and save one model in one cell

In `src/task1/models/build_model.ipynb`, setup cells load the EDA card and data and define `X_dev`, `y_dev`, `SCORE`, `CV`, `hgb_params`, and the notebook-local `saved_result` and `record_model` helpers. `record_model` calls pipeline `save_model` and `evaluate_model`, each with a card. A model cell can then be:

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

`saved_result` checks whether the named artifact already exists and is linked to the expected EDA dataset. An existing result is loaded instead of silently rewritten.

### Compare saved models

`show_rankings` is another Task 1 notebook helper. It reads saved evaluations and sorts models by Task 1 validation macro F₂:

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

The model card links back to its EDA card. The evaluation card records the validation metric and score. `src/taskN/save/evaluation_index.json` tracks the best saved model **per metric**. For the current five-task Kaggle file, run [src/submission/assemble.ipynb](src/submission/assemble.ipynb) as described in the [pipeline guide](src/pipeline/README.md).
