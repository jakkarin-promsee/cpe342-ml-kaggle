"""Small shared helpers for the notebook artifact pipeline."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SRC = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def task_dir(task: int | str) -> Path:
    value = str(task)
    if value.startswith("task"):
        value = value[4:]
    if value not in {"1", "2", "3", "4", "5"}:
        raise ValueError("task must be 1, 2, 3, 4, or 5")
    return SRC / f"task{value}"


def checked_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError("name must contain only letters, digits, hyphens, and underscores")
    return name


def checked_card(card: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(card, str) and card.strip():
        return card.strip()
    if isinstance(card, dict) and card:
        json.dumps(card, ensure_ascii=False)
        return card
    raise ValueError("card must be a nonempty string or JSON-serializable dict")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict[str, Any], *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Already exists: {path}. Use a new name or force=True.")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, default=str)
            file.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def ensure_free(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError("Artifact already exists: " + ", ".join(existing))


def relative(path: Path) -> str:
    return path.resolve().relative_to(SRC).as_posix()
