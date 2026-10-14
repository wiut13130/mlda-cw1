"""Save/load round trip and manifest provenance, written to a temporary models directory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlda.config import RANDOM_STATE
from mlda.models import MODEL_ZOO, build_pipeline
from mlda.persistence import (
    MANIFEST_NAME,
    data_checksum,
    git_sha,
    load_model,
    read_manifest,
    save_model,
)


@pytest.fixture(scope="module")
def fitted() -> tuple[Pipeline, pd.DataFrame]:
    X, y = make_classification(n_samples=200, n_features=6, random_state=RANDOM_STATE)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    pre = ColumnTransformer([("num", numeric, list(frame.columns))])
    pipeline = build_pipeline(MODEL_ZOO["logreg"], pre).fit(frame, y)
    return pipeline, frame


def test_round_trip_and_manifest_fields(fitted, tmp_path: Path) -> None:
    pipeline, X = fitted
    params = {"clf__C": np.float64(0.1), "clf__class_weight": None}
    path = save_model(pipeline, "logreg_test", best_params=params, extra={"n_train": 200}, models_dir=tmp_path)
    assert path == tmp_path / "logreg_test.joblib" and path.exists()

    manifest = read_manifest(tmp_path)
    assert set(manifest) == {"logreg_test"}
    entry = manifest["logreg_test"]
    for field in (
        "saved_at_utc", "python_version", "sklearn_version", "numpy_version", "pandas_version",
        "git_sha", "data_checksum", "data_snapshot_utc", "best_params", "file", "size_bytes",
        "estimator", "student_id",
    ):
        assert field in entry, field
    assert entry["sklearn_version"] == sklearn.__version__
    assert entry["student_id"] == "13130"
    assert entry["best_params"] == {"clf__C": 0.1, "clf__class_weight": None}
    assert entry["extra"] == {"n_train": 200}
    assert entry["estimator"] == "LogisticRegression"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", entry["saved_at_utc"])
    assert entry["git_sha"] is None or re.fullmatch(r"[0-9a-f]{40}", entry["git_sha"])
    # the manifest is plain JSON on disk
    with (tmp_path / MANIFEST_NAME).open(encoding="utf-8") as handle:
        assert json.load(handle)["logreg_test"]["file"] == "logreg_test.joblib"

    loaded = load_model("logreg_test", models_dir=tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(X), pipeline.predict_proba(X))


def test_second_save_adds_an_entry_without_dropping_the_first(fitted, tmp_path: Path) -> None:
    pipeline, _ = fitted
    save_model(pipeline, "a", models_dir=tmp_path)
    save_model(pipeline, "b", models_dir=tmp_path)
    assert set(read_manifest(tmp_path)) == {"a", "b"}


def test_load_warns_on_sklearn_version_mismatch(fitted, tmp_path: Path) -> None:
    pipeline, X = fitted
    save_model(pipeline, "m", models_dir=tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["m"]["sklearn_version"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.warns(UserWarning, match="0.0.0"):
        loaded = load_model("m", models_dir=tmp_path)
    assert loaded.predict(X).shape == (200,)


def test_load_warns_without_manifest_entry_and_fails_when_missing(fitted, tmp_path: Path) -> None:
    pipeline, _ = fitted
    save_model(pipeline, "m", models_dir=tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()
    with pytest.warns(UserWarning, match="no manifest entry"):
        load_model("m", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model("does_not_exist", models_dir=tmp_path)


def test_data_checksum_is_deterministic_and_tolerates_missing_snapshot(tmp_path: Path) -> None:
    snapshot = {
        "A.zip": {"sha256": "aa", "bytes": 1},
        "B.zip": {"sha256": "bb", "bytes": 2},
        "snapshot_utc": "2026-10-14T12:06:01Z",
    }
    path = tmp_path / "SNAPSHOT.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    first = data_checksum(path)
    assert first["snapshot_utc"] == "2026-10-14T12:06:01Z"
    assert re.fullmatch(r"[0-9a-f]{64}", first["sha256"])
    # order of keys in the file does not matter
    path.write_text(json.dumps(dict(reversed(list(snapshot.items())))), encoding="utf-8")
    assert data_checksum(path) == first
    assert data_checksum(tmp_path / "missing.json") == {"sha256": None, "snapshot_utc": None}


def test_git_sha_is_hex_or_none() -> None:
    sha = git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha)
    assert git_sha(Path("/definitely/not/a/repo")) is None
