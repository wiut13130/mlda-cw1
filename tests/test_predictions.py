"""Stored predictions keep the precision the thresholds were computed from (float64)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlda.evaluation import threshold_for_specificity
from mlda.pipeline import ArtefactDirs, ScoredModel, StudyData, write_predictions
from mlda.split import SplitIndices

N_ROWS = 400


def scored(name: str, oof: np.ndarray, test: np.ndarray) -> ScoredModel:
    return ScoredModel(name, oof, test, thresholds={}, evaluation={}, curves={})


def test_write_predictions_stores_float64_scores_that_reproduce_thresholds(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, N_ROWS)
    train_idx, test_idx = np.arange(0, 300), np.arange(300, N_ROWS)
    meta = pd.DataFrame(
        {"MINE_ID": [f"{i:07d}" for i in range(N_ROWS)], "ACCIDENT_DT": pd.Timestamp("2018-01-01")}
    )
    data = StudyData(
        X=pd.DataFrame({"a": np.zeros(N_ROWS)}),
        y=y,
        groups=np.array([f"g{i}" for i in range(N_ROWS)]),
        meta=meta,
        years=np.full(N_ROWS, 2018),
        sector=np.array(["C"] * N_ROWS),
        split=SplitIndices(train_idx=train_idx, test_idx=test_idx),
    )
    # Scores that differ from their float32 rounding by about 1e-9, as real probabilities do.
    oof = rng.uniform(0.05, 0.95, len(train_idx)) + 3e-9
    test = rng.uniform(0.05, 0.95, len(test_idx)) + 3e-9
    assert not np.array_equal(oof, oof.astype("float32").astype("float64"))
    dirs = ArtefactDirs(*(tmp_path / n for n in ("processed", "models", "metrics", "cv", "predictions", "figures")))
    write_predictions(data, [scored("m", oof, test)], {"base": scored("base", oof * 0 + 0.5, test * 0 + 0.5)}, dirs)

    stored_oof = pd.read_parquet(dirs.predictions / "oof_predictions.parquet")
    stored_test = pd.read_parquet(dirs.predictions / "test_predictions.parquet")
    assert stored_oof["proba_m"].dtype == np.float64 and stored_test["proba_base"].dtype == np.float64
    np.testing.assert_array_equal(stored_oof["proba_m"].to_numpy(), oof)  # bit for bit
    np.testing.assert_array_equal(stored_test["proba_m"].to_numpy(), test)
    assert stored_oof["row"].tolist() == train_idx.tolist() and stored_test["row"].tolist() == test_idx.tolist()
    # A threshold computed in memory flags exactly the same rows when applied to the stored scores.
    in_memory = threshold_for_specificity(y[train_idx], oof, 0.40)
    from_file = threshold_for_specificity(stored_oof["y_true"].to_numpy(), stored_oof["proba_m"].to_numpy(), 0.40)
    assert from_file == in_memory
    np.testing.assert_array_equal(stored_oof["proba_m"].to_numpy() >= in_memory, oof >= in_memory)
