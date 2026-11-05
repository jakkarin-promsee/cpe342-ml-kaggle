"""Predict test rows with a saved model and write a task submission draft."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import pandas as pd

from ._common import SRC, checked_card, checked_name, ensure_free, read_json, relative, task_dir, timestamp, write_json
from .read_evalutate_summarize import best_model


def build_sample_submission(
    *,
    task: int | str,
    name: str,
    card: str | dict[str, Any],
    model_name: str | None = None,
    metric: str | None = None,
    id_column: str = "id",
    predict_fn: Callable[[Any, pd.DataFrame], Any] | None = None,
    loader: Callable[[str], Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create `id|taskN` predictions in taskN/submission.

    Defaults to the best indexed model, joblib.load, and model.predict. Supply
    callbacks for image models or frameworks with different loading/prediction.
    This is a per-task file; combine all five task columns for Kaggle's final CSV.
    """
    checked_name(name)
    card = checked_card(card)
    task_name = task_dir(task).name
    model_name = model_name or best_model(task, metric)
    checked_name(model_name)
    model_card = read_json(task_dir(task) / "save" / f"{model_name}.json")
    data_card = read_json(SRC / model_card["data_card_path"])
    test_path = SRC / data_card["splits"]["test"]["path"]
    test = pd.read_parquet(test_path) if test_path.suffix == ".parquet" else pd.read_csv(test_path)
    if id_column not in test.columns:
        raise KeyError(f"Missing ID column {id_column!r} in {test_path}")
    if test[id_column].isna().any() or test[id_column].duplicated().any():
        raise ValueError("Test IDs must be non-null and unique")

    folder = task_dir(task) / "submission"
    csv_path = folder / f"{name}.csv"
    card_path = folder / f"{name}.json"
    ensure_free([csv_path, card_path], force)
    model_path = SRC / model_card["model_path"]
    model = (loader or joblib.load)(str(model_path))
    features = test.drop(columns=[id_column])
    predictions = predict_fn(model, features) if predict_fn else model.predict(features)
    if len(predictions) != len(test):
        raise ValueError("Prediction count does not match test row count")
    output = pd.DataFrame({"id": test[id_column].to_numpy(), task_name: predictions})
    if output[task_name].isna().any():
        raise ValueError("Predictions contain missing values")
    folder.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, sep="|", index=False)
    detail = {
        "name": name,
        "task": task_name,
        "created_at": timestamp(),
        "card": card,
        "model_name": model_name,
        "model_card_path": relative(task_dir(task) / "save" / f"{model_name}.json"),
        "data_name": model_card["data_name"],
        "submission_path": relative(csv_path),
        "rows": len(output),
        "columns": ["id", task_name],
    }
    write_json(card_path, detail, force=force)
    return detail


def combine_submissions(
    *,
    name: str,
    sample_path: str | Path,
    task_files: Mapping[int, str | Path],
    card: str | dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Combine five per-task files into Kaggle's `id|task1|...|task5` format."""
    checked_name(name)
    card = checked_card(card)
    if set(task_files) != {1, 2, 3, 4, 5}:
        raise ValueError("task_files must contain exactly tasks 1 through 5")
    sample = pd.read_csv(sample_path, sep="|")
    expected_columns = ["id", "task1", "task2", "task3", "task4", "task5"]
    if sample.columns.tolist() != expected_columns:
        raise ValueError("Sample submission must have columns id|task1|task2|task3|task4|task5")
    if sample["id"].isna().any() or sample["id"].duplicated().any():
        raise ValueError("Sample submission IDs must be non-null and unique")
    output = sample[["id"]].copy()
    for task in range(1, 6):
        column = f"task{task}"
        part = pd.read_csv(task_files[task], sep="|")
        if part.columns.tolist() != ["id", column]:
            raise ValueError(f"Task {task} file must have columns id|{column}")
        if part["id"].isna().any() or part["id"].duplicated().any():
            raise ValueError(f"Task {task} IDs must be non-null and unique")
        if len(part) != len(output) or set(part["id"]) != set(output["id"]):
            raise ValueError(f"Task {task} IDs do not match sample submission")
        output = output.merge(part, on="id", how="left", validate="one_to_one", sort=False)
        if output[column].isna().any():
            raise ValueError(f"Task {task} has missing predictions")
    folder = SRC / "submission"
    csv_path = folder / f"{name}.csv"
    card_path = folder / f"{name}.json"
    ensure_free([csv_path, card_path], force)
    folder.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, sep="|", index=False)
    detail = {
        "name": name,
        "created_at": timestamp(),
        "card": card,
        "submission_path": relative(csv_path),
        "rows": len(output),
        "columns": expected_columns,
        "task_files": {str(task): str(path) for task, path in task_files.items()},
    }
    write_json(card_path, detail, force=force)
    return detail
