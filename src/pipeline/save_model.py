"""Save a trained model and a short model card."""

from __future__ import annotations

from typing import Any, Callable

import joblib

from ._common import checked_card, checked_name, ensure_free, relative, task_dir, timestamp, write_json


def save_model(
    *,
    task: int | str,
    name: str,
    model: Any,
    card: str | dict[str, Any],
    data_name: str,
    force: bool = False,
    saver: Callable[[Any, str], None] | None = None,
    extension: str = "joblib",
) -> dict[str, Any]:
    """Save a model to taskN/save. Default serialization is joblib.

    For a framework-specific model, provide both `saver(model, path)` and an
    `extension` such as "keras" or "pt". Do not load untrusted model files.
    """
    checked_name(name)
    checked_name(data_name)
    card = checked_card(card)
    if not extension.isalnum():
        raise ValueError("extension must contain only letters and digits")
    data_card_path = task_dir(task) / "data" / "eda" / f"{data_name}.json"
    if not data_card_path.exists():
        raise FileNotFoundError(f"Save the EDA dataset first: {data_card_path}")
    folder = task_dir(task) / "save"
    model_path = folder / f"{name}.{extension}"
    card_path = folder / f"{name}.json"
    ensure_free([model_path, card_path], force)
    folder.mkdir(parents=True, exist_ok=True)
    if saver is None:
        if extension != "joblib":
            raise ValueError("Provide saver when extension is not 'joblib'")
        joblib.dump(model, model_path)
    else:
        saver(model, str(model_path))
    detail = {
        "name": name,
        "task": task_dir(task).name,
        "created_at": timestamp(),
        "card": card,
        "data_name": data_name,
        "data_card_path": relative(data_card_path),
        "model_path": relative(model_path),
        "format": extension,
    }
    write_json(card_path, detail, force=force)
    return detail
