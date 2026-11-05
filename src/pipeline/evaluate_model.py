"""Record validation results and maintain a per-task evaluation index."""

from __future__ import annotations

import math
from typing import Any, Sequence

from sklearn.metrics import fbeta_score, mean_absolute_error

from ._common import checked_card, checked_name, ensure_free, read_json, relative, task_dir, timestamp, write_json


def evaluate_model(
    *,
    task: int | str,
    name: str,
    card: str | dict[str, Any],
    y_true: Sequence[Any] | None = None,
    y_pred: Sequence[Any] | None = None,
    normalizer: float | None = None,
    score: float | None = None,
    metric_name: str | None = None,
    higher_is_better: bool | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved model on validation data and update evaluation_index.json.

    For Task 3, supply the competition's NMAE denominator as `normalizer`.
    For unlabeled Task 5, supply an explicitly named validation proxy `score`;
    its score is not the leaderboard F3 score.
    """
    checked_name(name)
    card = checked_card(card)
    task_name = task_dir(task).name
    folder = task_dir(task) / "save"
    model_card_path = folder / f"{name}.json"
    if not model_card_path.exists():
        raise FileNotFoundError(f"Save the model first: {model_card_path}")
    model_card = read_json(model_card_path)
    result_path = folder / f"{name}-evaluation.json"
    ensure_free([result_path], force)

    if score is not None:
        if y_true is not None or y_pred is not None:
            raise ValueError("Pass either score or y_true/y_pred, not both")
        if not metric_name or higher_is_better is None:
            raise ValueError("score requires metric_name and higher_is_better")
        value = float(score)
        metric = metric_name
        maximize = higher_is_better
    else:
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred are required unless score is supplied")
        if task_name == "task3":
            if normalizer is None or normalizer <= 0:
                raise ValueError("Task 3 requires a positive NMAE normalizer")
            value = float(mean_absolute_error(y_true, y_pred) / normalizer)
            metric, maximize = "NMAE", False
        else:
            beta = 2 if task_name == "task1" else 3 if task_name == "task5" else 1
            value = float(fbeta_score(y_true, y_pred, beta=beta, average="macro", zero_division=0))
            metric, maximize = f"macro_F{beta}", True

    if not math.isfinite(value):
        raise ValueError("Evaluation score must be finite")

    result = {
        "name": name,
        "task": task_name,
        "evaluated_at": timestamp(),
        "split": "validation",
        "card": card,
        "metric": metric,
        "score": value,
        "higher_is_better": maximize,
        "model_card_path": relative(model_card_path),
        "model_path": model_card["model_path"],
        "data_name": model_card["data_name"],
    }
    write_json(result_path, result, force=force)

    index_path = folder / "evaluation_index.json"
    index = read_json(index_path) if index_path.exists() else {"task": task_name, "evaluations": {}}
    index["evaluations"][name] = {
        "score": value,
        "metric": metric,
        "higher_is_better": maximize,
        "evaluated_at": result["evaluated_at"],
        "evaluation_path": relative(result_path),
        "model_path": result["model_path"],
    }
    # Scores with different metrics cannot be ranked against one another.
    groups = {}
    for model_name, item in index["evaluations"].items():
        groups.setdefault(item["metric"], []).append((model_name, item))
    index["best_by_metric"] = {
        metric_key: (max if items[0][1]["higher_is_better"] else min)(items, key=lambda pair: pair[1]["score"])[0]
        for metric_key, items in groups.items()
    }
    index["updated_at"] = timestamp()
    write_json(index_path, index, force=True)
    return result
