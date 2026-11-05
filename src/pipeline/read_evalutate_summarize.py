"""Read the per-task evaluation index. Kept at the requested module path."""

from __future__ import annotations

from typing import Any

from ._common import read_json, task_dir


def read_evaluation_index(task: int | str) -> dict[str, Any]:
    """Return all recorded model scores and best model names by metric."""
    path = task_dir(task) / "save" / "evaluation_index.json"
    return read_json(path)


def best_model(task: int | str, metric: str | None = None) -> str:
    """Return the best model name; specify metric if multiple metrics exist."""
    best = read_evaluation_index(task).get("best_by_metric", {})
    if metric is not None:
        return best[metric]
    if len(best) != 1:
        raise ValueError(f"Specify metric; available metrics: {list(best)}")
    return next(iter(best.values()))
