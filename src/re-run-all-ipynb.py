"""Run current EDA notebooks first, then their model notebooks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = range(1, 6)

# Per-task READMEs mark these notebooks/artifacts as based on the old dataset.
# Add new notebooks for the refreshed data instead of reusing these names.
IGNORE_NOTEBOOKS = {
    2: {"eda/build_eda_original.ipynb", "eda/build_eda_1.ipynb", "models/build_model.ipynb"},
    4: {"eda/build_eda_original.ipynb", "eda/build_eda_1.ipynb", "models/build_model.ipynb"},
    5: {"eda/build_eda_original.ipynb", "eda/build_eda_1.ipynb", "models/build_model.ipynb"},
}


def notebook_order(path: Path) -> tuple[int, str]:
    return (0 if "original" in path.stem.split("_") else 1, path.name)


def notebook_plan(tasks: list[int], root: Path = ROOT) -> tuple[list[Path], list[str]]:
    """Put every selected EDA before every selected model notebook."""
    plan: list[Path] = []
    skipped: list[str] = []
    runnable_eda: set[int] = set()
    for stage in ("eda", "models"):
        for task in tasks:
            task_dir = root / "src" / f"task{task}"
            notebooks = sorted((task_dir / stage).glob("*.ipynb"), key=notebook_order)
            for notebook in notebooks:
                relative = notebook.relative_to(task_dir).as_posix()
                if relative in IGNORE_NOTEBOOKS.get(task, set()):
                    skipped.append(f"task{task}/{relative}: outdated dataset")
                    continue
                if stage == "models" and task not in runnable_eda:
                    skipped.append(f"task{task}/{relative}: no current EDA notebook")
                    continue
                if stage == "eda":
                    runnable_eda.add(task)
                plan.append(notebook)
    return plan, skipped


def execute_notebook(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-n", type=int, choices=TASKS, help="Run only this task (1-5); default: all")
    parser.add_argument("--dry-run", action="store_true", help="Show the notebook order without running it")
    args = parser.parse_args()
    tasks = [args.task_n] if args.task_n else list(TASKS)
    plan, skipped = notebook_plan(tasks)
    for reason in skipped:
        print(f"SKIP {reason}")
    for notebook in plan:
        print(f"RUN  {notebook.relative_to(ROOT)}")
    if not plan:
        raise SystemExit("No current notebooks to run for the selected task; see src/taskN/README.md")
    if args.dry_run:
        return

    # The python3 kernelspec uses `python`; resolve it to this script's venv.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    for notebook in plan:
        print(f"Executing {notebook.relative_to(ROOT)}", flush=True)
        execute_notebook(notebook)
    print(f"Completed {len(plan)} notebooks")


if __name__ == "__main__":
    main()
