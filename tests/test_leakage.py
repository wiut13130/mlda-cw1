"""Leakage guards: deterministic levels, forbidden columns and the shape of the feature frame.

The real-data tests skip when the raw extracts are absent. They guard the finding that
NATURE_INJURY_CD = 100 (amputation, 925 rows) is 100 % lost-time: that level is documented in
docs/data_profile.md section 9, so any *other* pure level, or a change to this one, fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlda.config import LEAKAGE_COLUMNS, MINES_CURRENT_STATE_COLUMNS
from mlda.features import FEATURE_COLUMNS, FEATURE_SPEC, FeatureFrame
from mlda.sensitivity import (
    DOCUMENTED_PURE_LEVELS,
    PURE_LEVEL_MIN_ROWS,
    excluding_levels,
    pure_levels,
)

LEAKAGE_TOKENS = ("DEGREE", "DAYS_LOST", "DAYS_RESTRICT", "SCHEDULE_CHARGE", "TRANS_TERM", "RETURN_TO_WORK", "CLOSED_DOC", "NARRATIVE")


def small_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Three categorical columns: one with a pure level of 120 rows, one pure level of 60, one impure."""
    n = 600
    rng = np.random.default_rng(5)
    nature = np.array(["100"] * 120 + ["330"] * 60 + ["180"] * 420)
    y = np.where(nature == "100", 1, np.where(nature == "330", 0, (rng.random(n) < 0.5).astype(int)))
    X = pd.DataFrame({column: "a" for column in FEATURE_SPEC["categorical"]}, index=range(n))
    for column in FEATURE_SPEC["binary"]:
        X[column] = np.int8(0)
    X["NATURE_INJURY_CD"] = nature
    X["is_night"] = (rng.random(n) < 0.3).astype("int8")
    return X, y


def test_pure_levels_reports_only_large_one_class_levels() -> None:
    X, y = small_frame()
    found = pure_levels(X, y)
    pairs = set(zip(found["column"], found["level"], strict=True))
    assert ("NATURE_INJURY_CD", "100") in pairs  # 120 rows, all positive
    assert ("NATURE_INJURY_CD", "330") not in pairs  # pure but only 60 rows
    assert ("NATURE_INJURY_CD", "180") not in pairs  # large but mixed
    row = found[(found["column"] == "NATURE_INJURY_CD") & (found["level"] == "100")].iloc[0]
    assert (row["n"], row["positives"], row["positive_rate"], bool(row["documented"])) == (120, 120, 1.0, True)
    assert ("SUBUNIT_CD", "a") not in pairs  # a constant column holds both classes
    lowered = pure_levels(X, y, min_rows=50)
    assert ("NATURE_INJURY_CD", "330") in set(zip(lowered["column"], lowered["level"], strict=True))


def test_excluding_levels_scores_the_remaining_rows_with_fixed_thresholds() -> None:
    X, y = small_frame()
    proba = np.where(y == 1, 0.8, 0.2)
    proba[X["NATURE_INJURY_CD"].to_numpy() == "100"] = 0.9
    thresholds = {"m": {"specificity_0.40": 0.5, "flag_rate_0.50": 0.5}}
    levels = pure_levels(X, y).query("column == 'NATURE_INJURY_CD'")
    out = excluding_levels(X, y, {"m": proba}, thresholds, levels)
    assert out["n_test_excluded"] == 120 and out["levels"] == ["NATURE_INJURY_CD=100"]
    keep = X["NATURE_INJURY_CD"].to_numpy() != "100"
    assert out["prevalence_test_remaining"] == pytest.approx(y[keep].mean())
    model = out["models"]["m"]
    recall_with = ((proba >= 0.5) & (y == 1)).sum() / (y == 1).sum()
    recall_without = ((proba[keep] >= 0.5) & (y[keep] == 1)).sum() / (y[keep] == 1).sum()
    assert model["with_levels"]["recall_at_0.5"] == pytest.approx(recall_with)
    assert model["without_levels"]["recall_at_0.5"] == pytest.approx(recall_without)
    assert model["delta"]["recall_at_0.5"] == pytest.approx(recall_without - recall_with)


def test_feature_names_hold_no_label_or_outcome_columns() -> None:
    assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS)
    assert set(FEATURE_COLUMNS).isdisjoint(MINES_CURRENT_STATE_COLUMNS)
    for name in FEATURE_COLUMNS:
        assert not any(token in name.upper() for token in LEAKAGE_TOKENS), name
    assert "lost_time" not in FEATURE_COLUMNS and "group_id" not in FEATURE_COLUMNS


def test_the_only_pure_level_in_the_real_data_is_the_documented_amputation_code(real_features: FeatureFrame) -> None:
    """Fails loudly if a new level with at least 100 rows is 100 % one class, or if this one changes."""
    X, y = real_features.X, real_features.y.to_numpy()
    found = pure_levels(X, y, min_rows=PURE_LEVEL_MIN_ROWS)
    assert set(zip(found["column"], found["level"], strict=True)) == {("NATURE_INJURY_CD", "100")}
    assert set(zip(found["column"], found["level"], strict=True)) == set(DOCUMENTED_PURE_LEVELS)
    row = found.iloc[0]
    assert (row["n"], row["positive_rate"]) == (925, 1.0)  # 923 permanent disability (02) + 2 deaths (01)


def test_no_single_feature_predicts_the_label_almost_perfectly(real_features: FeatureFrame) -> None:
    """Level-mean encoding of every categorical feature: the best single-feature AUC stays far from 1."""
    from sklearn.metrics import roc_auc_score

    X, y = real_features.X, real_features.y.to_numpy()
    rng = np.random.default_rng(0)
    half = rng.random(len(y)) < 0.5  # learn the level means on one half, score the other
    best = 0.0
    for column in FEATURE_SPEC["categorical"]:
        rate = pd.Series(y[half]).groupby(X[column].to_numpy()[half]).mean()
        score = X[column].map(rate).fillna(y[half].mean()).to_numpy()[~half]
        best = max(best, roc_auc_score(y[~half], score))
    assert best < 0.75, best  # INJ_BODY_PART_CD is the strongest at about 0.71
