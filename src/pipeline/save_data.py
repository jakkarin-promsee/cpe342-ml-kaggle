"""Save one EDA dataset and its readable card."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._common import checked_card, checked_name, ensure_free, relative, task_dir, timestamp, write_json


def save_data(
    *,
    task: int | str,
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    card: str | dict[str, Any],
    format: str = "parquet",
    force: bool = False,
) -> dict[str, Any]:
    """Write train/test plus name.json and split cards to taskN/data/eda.

    `card` may be one sentence or a small dict. `force` is for refining a new
    EDA artifact before it is relied upon; use a new name after that point.
    """
    checked_name(name)
    card = checked_card(card)
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames")
    if format not in {"csv", "parquet"}:
        raise ValueError("format must be 'csv' or 'parquet'")
    folder = task_dir(task) / "data" / "eda"
    train_path = folder / f"{name}-train.{format}"
    test_path = folder / f"{name}-test.{format}"
    master_path = folder / f"{name}.json"
    train_card_path = folder / f"{name}-train.json"
    test_card_path = folder / f"{name}-test.json"
    ensure_free([train_path, test_path, master_path, train_card_path, test_card_path], force)
    folder.mkdir(parents=True, exist_ok=True)

    # Write data first and the master card last, so the master card marks completion.
    writer = "to_csv" if format == "csv" else "to_parquet"
    options = {"index": False}
    getattr(train, writer)(train_path, **options)
    getattr(test, writer)(test_path, **options)
    splits = {}
    for split, frame, path, card_path in [
        ("train", train, train_path, train_card_path),
        ("test", test, test_path, test_card_path),
    ]:
        detail = {
            "name": name,
            "split": split,
            "path": relative(path),
            "rows": len(frame),
            "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
        write_json(card_path, detail, force=force)
        splits[split] = {
            "path": detail["path"],
            "card_path": relative(card_path),
            "rows": detail["rows"],
            "columns": detail["columns"],
        }
    master = {
        "name": name,
        "task": task_dir(task).name,
        "created_at": timestamp(),
        "card": card,
        "format": format,
        "splits": splits,
    }
    write_json(master_path, master, force=force)
    return master
