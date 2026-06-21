"""Download and safely extract the official MovieLens 1M dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ARCHIVE_NAME = "ml-1m.zip"


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def download_dataset(output_directory: str | Path) -> Path:
    output_directory = Path(output_directory).resolve()
    ratings_path = output_directory / "ml-1m" / "ratings.dat"
    archive_path = output_directory / ARCHIVE_NAME
    temporary_path = archive_path.with_suffix(".download")

    if ratings_path.is_file():
        return ratings_path

    output_directory.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        temporary_path.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=180) as response:
                with temporary_path.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file, length=1024 * 1024)
            if not zipfile.is_zipfile(temporary_path):
                raise ValueError("Downloaded MovieLens archive is not a valid ZIP file")
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    _safe_extract(archive_path, output_directory)
    if not ratings_path.is_file():
        raise FileNotFoundError(
            f"Expected ratings file was not found after extraction: {ratings_path}"
        )
    return ratings_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens 1M.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/raw"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(download_dataset(arguments.output_directory))


if __name__ == "__main__":
    main()
