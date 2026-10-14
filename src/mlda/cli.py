"""Command-line entry points declared in pyproject.toml.

    uv run mlda-download   fetch and verify the MSHA source files
    uv run mlda-train      rebuild processed data, models and report artefacts
    uv run mlda-zip        build the intranet submission archive
"""

from __future__ import annotations

import argparse
import logging


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def download() -> None:
    from mlda.data.download import ensure_raw_files

    _configure_logging()
    parser = argparse.ArgumentParser(description="Download and verify the MSHA source files.")
    parser.add_argument("--force", action="store_true", help="re-download even if the checksum matches")
    args = parser.parse_args()
    for name, path in ensure_raw_files(force=args.force).items():
        print(f"{name:20} -> {path}")


def train_all() -> None:
    from mlda.pipeline import train_all as _train_all

    _configure_logging()
    _train_all()


def zip_submission() -> None:
    from mlda.submission import build_zip

    _configure_logging()
    print(build_zip())
