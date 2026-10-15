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
    from mlda.models import MODEL_ZOO
    from mlda.pipeline import train_all as _train_all

    _configure_logging()
    parser = argparse.ArgumentParser(description="Rebuild processed data, models and report artefacts.")
    parser.add_argument("--quick", action="store_true", help="two-point grids, 10,000 training rows, 3 folds")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_ZOO), metavar="NAME",
        help=f"subset of the model zoo (default: all of {', '.join(MODEL_ZOO)})",
    )
    args = parser.parse_args()
    manifest = _train_all(quick=args.quick, models=args.models)
    print(f"finished in {manifest['total_seconds']} s; best model {manifest['best_model']}")


def zip_submission() -> None:
    """Build the submission archive, or say plainly that the builder does not exist yet.

    ``mlda.submission`` is outside the module contracts of docs/interfaces.md (sections 1-11).
    Until it is written, the command exits with a one-line message instead of a traceback.
    """
    from mlda.submission import main as _main

    _configure_logging()
    _main()
