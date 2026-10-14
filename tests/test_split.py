"""Grouped stratified split: disjointness, nesting, prevalence and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _synthetic import synthetic_base
from sklearn.model_selection import StratifiedGroupKFold

from mlda.config import RANDOM_STATE, TEMPORAL_SPLIT_YEAR
from mlda.features import FeatureFrame, feature_frame
from mlda.split import (
    SplitIndices,
    check_split,
    cv_splitter,
    load_split_indices,
    make_split,
    save_split_indices,
    temporal_split,
)


@pytest.fixture(scope="module")
def parts() -> FeatureFrame:
    return feature_frame(synthetic_base(2000))


@pytest.fixture(scope="module")
def split(parts: FeatureFrame) -> SplitIndices:
    return make_split(parts.X, parts.y, parts.groups, accident_groups=parts.meta["accident_group"])


def test_row_positions_are_disjoint_and_complete(parts: FeatureFrame, split: SplitIndices) -> None:
    assert len(np.intersect1d(split.train_idx, split.test_idx)) == 0
    assert np.array_equal(np.union1d(split.train_idx, split.test_idx), np.arange(len(parts.y)))
    assert split.train_idx.dtype == np.int64 and split.n_rows == len(parts.y)


def test_group_ids_never_straddle_train_and_test(parts: FeatureFrame, split: SplitIndices) -> None:
    groups = parts.groups.to_numpy()
    assert set(groups[split.train_idx]).isdisjoint(groups[split.test_idx])
    assert parts.groups.nunique() > 100  # the synthetic frame has many mine-years


def test_accident_groups_never_straddle_train_and_test(parts: FeatureFrame, split: SplitIndices) -> None:
    accident_groups = parts.meta["accident_group"].to_numpy()
    assert set(accident_groups[split.train_idx]).isdisjoint(accident_groups[split.test_idx])
    assert parts.meta["accident_group"].duplicated().any()  # multi-victim groups exist in the fixture


def test_prevalence_within_one_percentage_point(parts: FeatureFrame, split: SplitIndices) -> None:
    y = parts.y.to_numpy()
    assert abs(y[split.test_idx].mean() - y.mean()) <= 0.01
    assert abs(y[split.train_idx].mean() - y.mean()) <= 0.01


def test_test_fraction_is_close_to_one_fifth(parts: FeatureFrame, split: SplitIndices) -> None:
    assert len(split.test_idx) / len(parts.y) == pytest.approx(0.2, abs=0.03)
    assert split.n_splits == 5 and split.test_fold == 0 and split.random_state == RANDOM_STATE


def test_same_seed_reproduces_the_split(parts: FeatureFrame, split: SplitIndices) -> None:
    again = make_split(parts.X, parts.y, parts.groups, random_state=RANDOM_STATE)
    assert np.array_equal(again.train_idx, split.train_idx) and np.array_equal(again.test_idx, split.test_idx)
    other = make_split(parts.X, parts.y, parts.groups, random_state=RANDOM_STATE + 1)
    assert not np.array_equal(other.test_idx, split.test_idx)


def test_saved_indices_reload_identically(split: SplitIndices, tmp_path: Path) -> None:
    path = save_split_indices(split, tmp_path / "split_indices.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["random_state"] == RANDOM_STATE
    assert payload["n_rows"] == split.n_rows and payload["student_id"] == "13130"
    loaded = load_split_indices(path)
    assert np.array_equal(loaded.train_idx, split.train_idx) and np.array_equal(loaded.test_idx, split.test_idx)
    assert (loaded.random_state, loaded.n_splits, loaded.test_fold) == (RANDOM_STATE, 5, 0)
    with pytest.raises(FileNotFoundError):
        load_split_indices(tmp_path / "absent.json")


def test_straddling_accident_groups_fail_the_assertions(parts: FeatureFrame) -> None:
    n = len(parts.y)
    row_groups = pd.Series(np.arange(n).astype(str))  # every row its own group
    one_accident = pd.Series(["same_day"] * n)  # one accident spanning every row
    with pytest.raises(AssertionError, match="accident_group"):
        make_split(parts.X, parts.y, row_groups, accident_groups=one_accident)


def test_check_split_detects_overlap_and_prevalence_drift(parts: FeatureFrame, split: SplitIndices) -> None:
    overlapping = SplitIndices(train_idx=split.train_idx, test_idx=np.append(split.test_idx, split.train_idx[0]))
    with pytest.raises(AssertionError, match="overlap"):
        check_split(overlapping, parts.y, parts.groups)
    y = parts.y.to_numpy()
    positives = np.flatnonzero(y == 1)
    skewed = SplitIndices(train_idx=np.setdiff1d(np.arange(len(y)), positives[:300]), test_idx=positives[:300])
    with pytest.raises(AssertionError, match="prevalence"):
        check_split(skewed, y, np.arange(len(y)).astype(str))


def test_invalid_test_fraction_raises(parts: FeatureFrame) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        make_split(parts.X, parts.y, parts.groups, test_fraction=0.3)
    with pytest.raises(ValueError, match="same number"):
        make_split(parts.X.iloc[:10], parts.y, parts.groups)


def test_cv_splitter_folds_respect_groups_on_training_rows(parts: FeatureFrame, split: SplitIndices) -> None:
    cv = cv_splitter()
    assert isinstance(cv, StratifiedGroupKFold)
    assert cv.get_n_splits() == 5 and cv.shuffle is True and cv.random_state == RANDOM_STATE
    X_train = parts.X.iloc[split.train_idx]
    y_train = parts.y.to_numpy()[split.train_idx]
    groups_train = parts.groups.to_numpy()[split.train_idx]
    seen = np.zeros(len(y_train), dtype=int)
    for fold_train, fold_valid in cv.split(X_train, y_train, groups_train):
        assert set(groups_train[fold_train]).isdisjoint(groups_train[fold_valid])
        seen[fold_valid] += 1
    assert (seen == 1).all()


def test_temporal_split_cuts_at_the_configured_year(parts: FeatureFrame) -> None:
    train_idx, test_idx = temporal_split(parts.meta)
    years = parts.meta["CAL_YR"].to_numpy(dtype=int)
    assert (years[train_idx] < TEMPORAL_SPLIT_YEAR).all() and (years[test_idx] >= TEMPORAL_SPLIT_YEAR).all()
    assert np.array_equal(np.union1d(train_idx, test_idx), np.arange(len(years)))
    assert len(train_idx) > 0 and len(test_idx) > 0
    with pytest.raises(ValueError, match="CAL_YR"):
        temporal_split(pd.DataFrame({"CAL_YR": [2015, None]}))
