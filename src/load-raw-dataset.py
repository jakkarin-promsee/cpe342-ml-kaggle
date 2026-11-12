"""Download the competition's raw files into src/taskN/data/raw."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    task: (("train.csv", "test.csv", "datacard.txt") if task < 5 else ("test.csv", "datacard.txt"))
    for task in range(1, 6)
}


def competition_slug(value: str) -> str:
    """Accept a Kaggle competition slug or its competition URL."""
    value = value.strip().rstrip("/")
    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.netloc not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError("Expected a kaggle.com competition URL")
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "competitions":
            raise ValueError("Expected https://www.kaggle.com/competitions/<slug>")
        value = parts[1]
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in value):
        raise ValueError("Competition must be a Kaggle URL suffix such as player-intelligence")
    return value


def kaggle_command() -> str:
    executable = shutil.which("kaggle")
    if executable:
        return executable
    sibling = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError("Kaggle CLI is missing. Run: python -m pip install -r requirements.txt")


def extract_archives(folder: Path) -> None:
    """Unpack Kaggle's outer ZIP and any ZIP files nested inside it."""
    processed: set[Path] = set()
    while True:
        archives = [path for path in folder.rglob("*.zip") if path not in processed]
        if not archives:
            return
        for archive in archives:
            processed.add(archive)
            destination = archive.with_suffix("")
            with ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    name = PurePosixPath(member.filename.replace("\\", "/"))
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError(f"Unsafe ZIP member in {archive}: {member.filename}")
                zipped.extractall(destination)


def find_task_folder(folder: Path, task: int) -> Path:
    required = EXPECTED[task]
    candidates = [
        path for path in folder.rglob(f"task{task}")
        if path.is_dir() and all((path / name).is_file() for name in required)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one task{task} folder with {', '.join(required)}; found {len(candidates)}")
    return candidates[0]


def collect_files(folder: Path, root: Path = ROOT) -> list[tuple[Path, Path]]:
    """Map validated competition files to this repository's raw layout."""
    planned: list[tuple[Path, Path]] = []
    for task in range(1, 6):
        source = find_task_folder(folder, task)
        destination = root / "src" / f"task{task}" / "data" / "raw"
        planned.extend((source / name, destination / name) for name in EXPECTED[task])
        if task == 4:
            for split in ("train", "test"):
                images = source / split
                if not images.is_dir() or not any(images.rglob("*.jpg")) and not any(images.rglob("*.png")):
                    raise ValueError(f"Missing Task 4 {split} images in {images}")
                planned.extend(
                    (image, destination / split / image.relative_to(images))
                    for image in images.rglob("*") if image.is_file()
                )
    samples = list(folder.rglob("sample_submission.csv"))
    if len(samples) != 1:
        raise ValueError(f"Expected one sample_submission.csv; found {len(samples)}")
    planned.append((samples[0], root / "src" / "submission" / "sample_submission.csv"))
    return planned


def install_files(files: list[tuple[Path, Path]], force: bool) -> None:
    collisions = [destination for _, destination in files if destination.exists()]
    if collisions and not force:
        raise FileExistsError(f"{len(collisions)} destination files already exist (first: {collisions[0]}). Use --force to replace current raw files.")
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", required=True, type=competition_slug, help="Kaggle competition slug or URL")
    parser.add_argument("--force", action="store_true", help="Replace current raw files; archived *-outdate-* files are untouched")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="player-intelligence-kaggle-") as temporary:
        download = Path(temporary)
        command = [kaggle_command(), "competitions", "download", args.competition, "-p", str(download)]
        print("Downloading competition files from Kaggle...", flush=True)
        subprocess.run(command, check=True)
        extract_archives(download)
        files = collect_files(download)
        install_files(files, force=args.force)
        print(f"Installed {len(files)} files into src/task*/data/raw and src/submission/sample_submission.csv")


if __name__ == "__main__":
    main()
