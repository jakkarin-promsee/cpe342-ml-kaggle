"""Save one EDA dataset and its readable card."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._common import checked_card, checked_name, ensure_free, relative, task_dir, timestamp, write_json


def save_data(
    *,
    task: int | str,
    name: str,
    test: pd.DataFrame,
    card: str | dict[str, Any],
    train: pd.DataFrame | None = None,
    format: str = "parquet",
    force: bool = False,
) -> dict[str, Any]:
    """Write train/test, or Task 5 test-only, plus readable dataset cards.

    `card` may be one sentence or a small dict. `force` is for refining a new
    EDA artifact before it is relied upon; use a new name after that point.
    """
    checked_name(name)
    card = checked_card(card)
    if not isinstance(test, pd.DataFrame) or (train is not None and not isinstance(train, pd.DataFrame)):
        raise TypeError("test and any supplied train must be pandas DataFrames")
    if train is None and task_dir(task).name != "task5":
        raise ValueError("train is required except for Task 5 test-only datasets")
    if format not in {"csv", "parquet"}:
        raise ValueError("format must be 'csv' or 'parquet'")
    folder = task_dir(task) / "data" / "eda"
    test_path = folder / f"{name}-test.{format}"
    master_path = folder / f"{name}.json"
    test_card_path = folder / f"{name}-test.json"
    split_items = []
    if train is not None:
        split_items.append(("train", train, folder / f"{name}-train.{format}", folder / f"{name}-train.json"))
    split_items.append(("test", test, test_path, test_card_path))
    ensure_free([master_path] + [path for _, _, data_path, card_path in split_items for path in (data_path, card_path)], force)
    folder.mkdir(parents=True, exist_ok=True)

    # Write data first and the master card last, so the master card marks completion.
    writer = "to_csv" if format == "csv" else "to_parquet"
    options = {"index": False}
    for _, frame, path, _ in split_items:
        getattr(frame, writer)(path, **options)
    splits = {}
    for split, frame, path, card_path in split_items:
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
