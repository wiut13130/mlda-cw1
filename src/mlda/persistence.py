"""Save and load fitted pipelines with a manifest that makes every artefact traceable.

``models/<name>.joblib`` holds the pipeline (joblib, compress=3). ``models/manifest.json``
holds one entry per model: library versions, the git commit, a digest of the pinned data
snapshot, the save time and the best hyper-parameters. ``load_model`` warns when the
installed scikit-learn differs from the one that produced the file, because pickled
estimators are only guaranteed to load under the same version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from mlda.config import MODELS_DIR, PROJECT_ROOT, RAW_DIR, STUDENT_ID

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SNAPSHOT_PATH = RAW_DIR / "SNAPSHOT.json"


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars, tuples and nested containers into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def git_sha(repo: Path = PROJECT_ROOT) -> str | None:
    """Current commit hash, or None when git or the repository is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def data_checksum(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, str | None]:
    """Digest of the pinned raw-data snapshot: sha256 over the per-file hashes in name order.

    Returns ``{"sha256": None, "snapshot_utc": None}`` when the snapshot file is missing, so a
    manifest can still be written in an environment without the raw data.
    """
    if not snapshot_path.exists():
        return {"sha256": None, "snapshot_utc": None}
    with snapshot_path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    digest = hashlib.sha256()
    for filename in sorted(k for k, v in snapshot.items() if isinstance(v, dict) and "sha256" in v):
        digest.update(f"{filename}:{snapshot[filename]['sha256']}\n".encode())
    return {"sha256": digest.hexdigest(), "snapshot_utc": snapshot.get("snapshot_utc")}


def _manifest_path(models_dir: Path) -> Path:
    return models_dir / MANIFEST_NAME


def read_manifest(models_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """The manifest as a dict keyed by model name (empty when no manifest exists yet)."""
    path = _manifest_path(models_dir or MODELS_DIR)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(manifest: dict[str, dict[str, Any]], models_dir: Path | None = None) -> Path:
    """Write the manifest with sorted keys so re-runs produce a stable diff."""
    path = _manifest_path(models_dir or MODELS_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def environment_record() -> dict[str, Any]:
    """Versions and provenance fields shared by every manifest entry."""
    checksum = data_checksum()
    return {
        "student_id": STUDENT_ID,
        "saved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "data_checksum": checksum["sha256"],
        "data_snapshot_utc": checksum["snapshot_utc"],
    }


def save_model(
    pipeline: Pipeline,
    name: str,
    best_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    models_dir: Path | None = None,
) -> Path:
    """Persist ``pipeline`` as ``<models_dir>/<name>.joblib`` and record it in the manifest.

    ``extra`` (for example test metrics or the number of training rows) is stored verbatim
    under ``"extra"``; ``best_params`` are the tuned hyper-parameters from ``TuneResult``.
    """
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, path, compress=3)
    estimator = pipeline.steps[-1][1] if isinstance(pipeline, Pipeline) else pipeline
    entry = environment_record()
    entry.update(
        {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "estimator": type(estimator).__name__,
            "best_params": _jsonable(best_params or {}),
            "extra": _jsonable(extra or {}),
        }
    )
    manifest = read_manifest(models_dir)
    manifest[name] = entry
    write_manifest(manifest, models_dir)
    log.info("saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_model(name: str, models_dir: Path | None = None) -> Pipeline:
    """Load ``<models_dir>/<name>.joblib``; warn when the scikit-learn version differs."""
    models_dir = models_dir or MODELS_DIR
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no saved model at {path}")
    entry = read_manifest(models_dir).get(name)
    if entry is None:
        warnings.warn(f"{name}: no manifest entry; provenance unknown", UserWarning, stacklevel=2)
    elif entry.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{name} was saved with scikit-learn {entry.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed; predictions may differ or loading may fail",
            UserWarning,
            stacklevel=2,
        )
    return joblib.load(path)


__all__ = [
    "MANIFEST_NAME",
    "data_checksum",
    "environment_record",
    "git_sha",
    "load_model",
    "read_manifest",
    "save_model",
    "write_manifest",
]
