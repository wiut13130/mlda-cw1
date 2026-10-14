"""Follow-up checks (mlda.followup) on synthetic data: no raw extract or full-run artefact is needed.

The checks score stored predictions and refit logistic regression with one extra one-hot block, so
the tests pin down what each check counts: the share flagged at a capacity, the planned and the
scored form of S5, the return-to-work grouping by filing date, and that the joint-cell block adds
columns without touching the test rows during fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_base

from mlda.cleaning import apply_error_rules
from mlda.config import RANDOM_STATE
from mlda.features import FEATURE_COLUMNS, feature_frame
from mlda.followup import (
    CELL_COLUMN,
    capacity_curve,
    equal_workload,
    interaction_preprocessor,
    joint_cell,
    logreg_interaction,
    recall_at_share,
    return_to_work_groups,
    s5_top_half,
    with_joint_cell,
)
from mlda.operating import FLAG_KEY, OPERATING_KEY
from mlda.split import make_split

N_ROWS = 1_200


@pytest.fixture(scope="module")
def parts():
    return feature_frame(apply_error_rules(synthetic_base(N_ROWS, seed=RANDOM_STATE))[0])


def _scores(n: int = 1_000, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    proba = np.clip(0.5 * y + rng.random(n) * 0.6, 0, 1)
    return y, proba


def test_recall_at_share_flags_the_requested_share_and_counts_hits() -> None:
    y, proba = _scores()
    point = recall_at_share(y, proba, 0.5)
    flagged = proba >= point["threshold"]
    assert point["flag_rate"] == pytest.approx(flagged.mean())
    assert point["flag_rate"] <= 0.5 + 1e-12
    assert point["recall"] == pytest.approx(flagged[y == 1].mean())


def test_capacity_curve_recall_rises_with_capacity() -> None:
    y, proba = _scores()
    recalls = [p["recall"] for p in capacity_curve(y, proba, (0.25, 0.5, 0.75))]
    assert recalls == sorted(recalls)


def test_s5_two_ways_and_equal_workload_use_the_stored_thresholds() -> None:
    y, proba = _scores()
    table = pd.DataFrame({"y_true": y, "proba_a": proba, "proba_b": proba[::-1]})
    thresholds = {"a": {FLAG_KEY: 0.4, OPERATING_KEY: 0.3}, "b": {FLAG_KEY: 0.6, OPERATING_KEY: 0.2}}
    s5 = s5_top_half(table, thresholds)
    assert s5["a"]["out_of_fold_threshold"]["flag_rate"] == pytest.approx((proba >= 0.4).mean())
    assert s5["a"]["top_half_of_test_scores"]["flag_rate"] <= 0.5 + 1e-12
    workload = equal_workload(table, thresholds, ["a", "b"])
    rates = workload["flag_rate_at_capacity_threshold"]
    assert workload["common_flag_rate"] == pytest.approx(np.median(list(rates.values())))
    assert set(workload["models"]) == {"a", "b"}


def test_return_to_work_groups_split_on_the_filing_date() -> None:
    dates = pd.to_datetime(["2020-01-10"] * 4)
    base = pd.DataFrame(
        {
            "lost_time": [0, 1, 1, 0],
            "filing_date": dates,
            "RETURN_TO_WORK_DT": pd.to_datetime(["2020-01-05", "2020-02-01", None, "2020-01-10"]),
        }
    )
    test_idx = np.array([0, 1, 2, 3])
    scores = pd.DataFrame({"y_true": [0, 1, 1, 0], "proba_m": [0.1, 0.9, 0.8, 0.2]})
    out = return_to_work_groups(base, test_idx, scores, {"m": {OPERATING_KEY: 0.5}}, models=("m",))
    summary = out["all_in_scope_rows"]
    assert summary["back_at_work_rows"] == 2          # on or before the filing date; a blank date counts as open
    assert summary["lost_time_rate_open_at_filing"] == pytest.approx(1.0)
    assert out["test_rows"]["open_at_filing"]["n"] == 2
    assert out["test_rows"]["open_at_filing"]["m"]["recall_at_capacity_threshold"] == pytest.approx(1.0)


def test_joint_cell_combines_nature_body_part_and_sector(parts) -> None:
    cells = joint_cell(parts.X)
    first = parts.X.iloc[0]
    assert cells.iloc[0] == f"{first['NATURE_INJURY_CD']}|{first['INJ_BODY_PART_CD']}|{first['COAL_METAL_IND']}"
    assert CELL_COLUMN in with_joint_cell(parts.X).columns


def test_interaction_preprocessor_adds_the_cell_block_learnt_on_training_rows(parts) -> None:
    X = with_joint_cell(parts.X)
    plain = interaction_preprocessor().fit(X)
    width = plain.transform(X.head(5)).shape[1]
    base_width = plain.named_transformers_["base"].transform(X[list(FEATURE_COLUMNS)].head(5)).shape[1]
    assert width > base_width
    unseen = X.head(3).copy()
    unseen[CELL_COLUMN] = "never|seen|cell"
    assert np.isfinite(plain.transform(unseen)).all()


def test_logreg_interaction_reports_both_variants_on_the_same_folds(parts) -> None:
    y = parts.y.to_numpy(dtype=int)
    groups = parts.groups.to_numpy(dtype=str)
    split = make_split(parts.X, y, groups, accident_groups=parts.meta["accident_group"])
    out = logreg_interaction(
        parts.X, y, groups, split.train_idx, split.test_idx, {"clf__C": 1.0, "clf__class_weight": None}, n_jobs=1
    )
    for name in ("logreg_selected", "logreg_with_joint_cell"):
        assert len(out[name]["cv_average_precision_folds"]) == 5
        assert 0.0 <= out[name]["test_pr_auc"] <= 1.0
    assert out["logreg_with_joint_cell"]["encoded_columns"] > out["logreg_selected"]["encoded_columns"]
