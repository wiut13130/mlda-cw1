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
    CAPACITY_POLICY,
    CELL_COLUMN,
    POLICY_COLUMNS,
    POLICY_GROUPS,
    capacity_curve,
    capacity_policy_table,
    equal_workload,
    flag_policies,
    interaction_preprocessor,
    joint_cell,
    logreg_interaction,
    recall_at_share,
    return_to_work_groups,
    s5_top_half,
    top_share_flags,
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


# Ten reports: four score 0.7, so the top half (five rows) has to split a tie, which no threshold can.
TEN_SCORES = np.array([0.9, 0.2, 0.7, 0.7, 0.1, 0.7, 0.4, 0.95, 0.3, 0.7])
TEN_LABELS = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1])


def test_top_share_flags_takes_exactly_k_rows_and_breaks_ties_by_row_order() -> None:
    top_half = top_share_flags(TEN_SCORES, 0.5)
    assert top_half.sum() == 5
    assert set(np.flatnonzero(top_half)) == {7, 0, 2, 3, 5}  # row 9 also scores 0.7 but comes last
    top_three_quarters = top_share_flags(TEN_SCORES, 0.75)
    assert top_three_quarters.sum() == 8  # 7.5 rows round half up
    assert set(np.flatnonzero(~top_three_quarters)) == {1, 4}
    assert top_share_flags(TEN_SCORES, 0.0).sum() == 0 and top_share_flags(TEN_SCORES, 1.0).all()
    with pytest.raises(ValueError):
        top_share_flags(TEN_SCORES, 1.5)


def test_flag_policies_count_the_hits_of_each_policy_on_ten_rows() -> None:
    out = flag_policies(TEN_LABELS, TEN_SCORES, threshold=0.5)
    assert set(out) == {CAPACITY_POLICY, "top_50pct", "top_75pct"}
    # rows 0 2 3 5 7 9 reach 0.5: TP 5, FP 1, FN 1 (row 6), TN 3
    assert out[CAPACITY_POLICY] == pytest.approx({
        "n": 10, "prevalence": 0.6, "flag_rate": 0.6, "recall": 5 / 6, "specificity": 3 / 4,
        "precision": 5 / 6, "cutoff": 0.5, "missed_per_1000": 100.0,
    })
    # rows 7 0 2 3 5: TP 4, FP 1 (row 3), FN 2 (rows 6 and 9), TN 3
    assert out["top_50pct"] == pytest.approx({
        "n": 10, "prevalence": 0.6, "flag_rate": 0.5, "recall": 4 / 6, "specificity": 3 / 4,
        "precision": 4 / 5, "cutoff": 0.7, "missed_per_1000": 200.0,
    })
    # every row but 1 and 4: TP 6, FP 2 (rows 3 and 8), FN 0, TN 2
    assert out["top_75pct"] == pytest.approx({
        "n": 10, "prevalence": 0.6, "flag_rate": 0.8, "recall": 1.0, "specificity": 2 / 4,
        "precision": 6 / 8, "cutoff": 0.3, "missed_per_1000": 0.0,
    })


def test_return_to_work_policies_cover_the_two_groups_and_flatten_to_the_table() -> None:
    filing = pd.to_datetime(["2020-01-10"] * 10)
    back = ["2020-01-05", None, "2020-02-01", "2020-01-10", "2020-03-01", None, "2020-01-01", "2020-01-20", "2020-01-02", None]
    base = pd.DataFrame({"lost_time": TEN_LABELS, "filing_date": filing, "RETURN_TO_WORK_DT": pd.to_datetime(back)})
    scores = pd.DataFrame({"y_true": TEN_LABELS, "proba_m": TEN_SCORES, "proba_r": (TEN_SCORES > 0.5).astype(float)})
    thresholds = {"m": {OPERATING_KEY: 0.5}, "r": {OPERATING_KEY: 1.0}}
    out = return_to_work_groups(base, np.arange(10), scores, thresholds, models=("m", "r"))
    rows = out["test_rows"]
    assert "policies" not in rows["back_at_work"]["m"]
    assert rows["all_test_rows"]["m"]["policies"] == flag_policies(TEN_LABELS, TEN_SCORES, 0.5)
    open_rows = ~(base["RETURN_TO_WORK_DT"] <= base["filing_date"]).to_numpy()
    assert rows["open_at_filing"]["n"] == int(open_rows.sum()) == 6
    assert rows["open_at_filing"]["m"]["policies"]["top_50pct"]["flag_rate"] == pytest.approx(0.5)
    for group in POLICY_GROUPS:
        for name in ("m", "r"):
            entry = rows[group][name]
            assert entry["policies"][CAPACITY_POLICY]["flag_rate"] == pytest.approx(entry["flag_rate_at_capacity_threshold"])
            assert entry["policies"][CAPACITY_POLICY]["recall"] == pytest.approx(entry["recall_at_capacity_threshold"])

    table = capacity_policy_table(rows)
    assert list(table.columns) == ["group", "model", "policy", *POLICY_COLUMNS]
    assert len(table) == len(POLICY_GROUPS) * 2 * 3 and set(table["group"]) == set(POLICY_GROUPS)
    cell = table.set_index(["group", "model", "policy"]).loc[("all_test_rows", "m", "top_75pct")]
    assert cell["recall"] == pytest.approx(1.0) and cell["cutoff"] == pytest.approx(0.3)
    # the two-valued rule: at 50 % five of its six tied top scores are flagged, the first five by row
    rule = table.set_index(["group", "model", "policy"]).loc[("all_test_rows", "r", "top_50pct")]
    assert rule["flag_rate"] == pytest.approx(0.5) and rule["cutoff"] == pytest.approx(1.0)
    np.testing.assert_array_equal(np.flatnonzero(top_share_flags(scores["proba_r"], 0.5)), [0, 2, 3, 5, 7])


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
