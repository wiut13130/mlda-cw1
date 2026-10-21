"""Follow-up checks (mlda.followup) on synthetic data: no raw extract or full-run artefact is needed.

The checks score stored predictions and refit logistic regression with one extra one-hot block, so
the tests pin down what each check counts: the share flagged at a capacity, the planned and the
scored form of S5, the return-to-work grouping by filing date, that the joint-cell block adds
columns without touching the test rows during fitting, and the workload-rule block (out-of-fold
cut-offs, open-cohort calibration, the paired mine-year bootstrap and the open-only refit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_base
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, brier_score_loss

from mlda.cleaning import apply_error_rules
from mlda.config import RANDOM_STATE
from mlda.features import FEATURE_COLUMNS, feature_frame
from mlda.followup import (
    CALIBRATION_COLUMNS,
    CAPACITY_POLICY,
    CELL_COLUMN,
    POLICY_COLUMNS,
    POLICY_GROUPS,
    WORKLOAD_COLUMNS,
    capacity_curve,
    capacity_policy_table,
    equal_frequency_bins,
    equal_workload,
    flag_policies,
    interaction_preprocessor,
    joint_cell,
    logreg_interaction,
    oof_cutoff,
    open_at_filing,
    open_calibration,
    open_only_refit,
    paired_workload_bootstrap,
    ranked_recall_and_average_precision,
    recall_at_share,
    reliability_bins,
    return_to_work_groups,
    s5_top_half,
    top_share_flags,
    with_joint_cell,
    workload_rule,
    workload_rule_table,
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


# ---------------------------------------------------------------------------
# Workload rule on the open-at-filing reports
# ---------------------------------------------------------------------------

# Twelve training rows: eight open with a score, one open with a blank score (row 3) and three
# back at work (rows 4, 5 and 10), whose high scores would move both cut-offs if they were counted.
OOF_M = np.array([0.05, 0.9, 0.3, np.nan, 0.6, 0.99, 0.2, 0.7, 0.45, 0.8, 0.98, 0.35])
TRAIN_OPEN = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
# Ten test rows, the first six open; row 6 scores exactly the top-75 % cut-off of 0.3.
TEST_M = np.array([0.95, 0.31, 0.29, 0.5, 0.1, 0.46, 0.3, 0.44, 0.2, 0.8])
TEST_Y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
TEST_OPEN = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)


def test_open_at_filing_counts_a_blank_return_date_as_open() -> None:
    base = pd.DataFrame({
        "filing_date": pd.to_datetime(["2020-01-10"] * 3),
        "RETURN_TO_WORK_DT": pd.to_datetime(["2020-01-10", "2020-01-11", None]),
    })
    np.testing.assert_array_equal(open_at_filing(base), [False, True, True])


def test_oof_cutoff_is_the_kth_highest_score_and_flags_exactly_k_open_training_rows() -> None:
    open_scores = OOF_M[TRAIN_OPEN]
    scored = open_scores[~np.isnan(open_scores)]
    for share, k, cutoff in ((0.75, 6, 0.3), (0.5, 4, 0.45)):
        out = oof_cutoff(open_scores, share)
        assert out["rows_scored"] == 8 and out["rows_blank"] == 1
        assert out["k"] == k == int(np.floor(share * scored.size + 0.5))
        assert out["cutoff_oof"] == pytest.approx(cutoff)
        assert out["cutoff_oof"] == pytest.approx(np.sort(scored)[::-1][k - 1])
        assert int((scored >= out["cutoff_oof"]).sum()) == k == out["rows_at_or_above_cutoff"]
        assert int(top_share_flags(scored, share).sum()) == k


def test_workload_rule_applies_the_training_cut_off_to_the_test_rows() -> None:
    oof = pd.DataFrame({"proba_m": OOF_M})
    test = pd.DataFrame({"y_true": TEST_Y, "proba_m": TEST_M})
    block = workload_rule(oof, TRAIN_OPEN, test, TEST_OPEN, models=("m",))
    entry = block["models"]["m"]
    assert entry["train_open_rows"] == 9 and entry["train_open_rows_blank_oof"] == 1
    assert entry["train_open_share_blank_oof"] == pytest.approx(1 / 9)
    top75 = entry["policies"]["top_75pct"]
    assert top75["cutoff_oof"] == pytest.approx(0.3)  # 0.35 if the back-at-work rows counted
    assert entry["policies"]["top_50pct"]["cutoff_oof"] == pytest.approx(0.45)  # 0.6 likewise
    # open test rows 0 1 3 5 reach 0.3: TP 3, FP 1 (row 3), FN 1 (row 2), TN 1 (row 4)
    assert top75["test"]["open_at_filing"] == pytest.approx({
        "n": 6, "prevalence": 4 / 6, "flag_rate": 4 / 6, "recall": 3 / 4, "specificity": 1 / 2,
        "precision": 3 / 4, "cutoff": 0.3, "missed_per_1000": 1000 / 6,
    })
    # all rows: 0 1 3 5 6 7 9 reach 0.3 (row 6 sits on the cut-off): TP 4, FP 3, FN 1, TN 2
    assert top75["test"]["all_test_rows"] == pytest.approx({
        "n": 10, "prevalence": 0.5, "flag_rate": 0.7, "recall": 4 / 5, "specificity": 2 / 5,
        "precision": 4 / 7, "cutoff": 0.3, "missed_per_1000": 100.0,
    })
    # top 50 %, cut-off 0.45, open rows 0 3 5: TP 2, FP 1, FN 2, TN 1
    top50 = entry["policies"]["top_50pct"]["test"]["open_at_filing"]
    assert top50["flag_rate"] == pytest.approx(0.5) and top50["recall"] == pytest.approx(0.5)
    assert top50["precision"] == pytest.approx(2 / 3) and top50["missed_per_1000"] == pytest.approx(2000 / 6)

    table = workload_rule_table(block)
    assert list(table.columns) == list(WORKLOAD_COLUMNS) and len(table) == 4
    assert set(table["train_open_rows"]) == {8}
    row = table.set_index(["policy", "test_group"]).loc[("top_75pct", "all_test_rows")]
    assert row["cutoff_oof"] == pytest.approx(0.3) and row["recall"] == pytest.approx(0.8)


# Eight test rows for the reliability table; four bins of two rows when ranked by score.
CAL_P = np.array([0.9, 0.1, 0.4, 0.2, 0.8, 0.3, 0.6, 0.5])
CAL_Y = np.array([1, 0, 1, 0, 1, 0, 0, 1])


def test_reliability_bins_partition_the_rows_and_match_a_hand_built_table() -> None:
    bins = equal_frequency_bins(CAL_P, 4)
    np.testing.assert_array_equal(bins, [4, 1, 2, 1, 4, 2, 3, 3])
    table = reliability_bins(CAL_Y, CAL_P, 4)
    assert list(table["bin"]) == [1, 2, 3, 4] and list(table["n"]) == [2, 2, 2, 2]
    np.testing.assert_allclose(table["mean_predicted"], [0.15, 0.35, 0.55, 0.85])
    np.testing.assert_allclose(table["observed_rate"], [0.0, 0.5, 0.5, 1.0])

    rng = np.random.default_rng(RANDOM_STATE)
    proba = rng.random(37)
    labels = (rng.random(37) < proba).astype(int)
    bins = equal_frequency_bins(proba, 10)
    sizes = np.bincount(bins, minlength=11)[1:]
    assert sizes.sum() == 37 and sizes.max() - sizes.min() <= 1
    for b in range(1, 10):  # bins follow the score order
        assert proba[bins == b].max() <= proba[bins == b + 1].min()
    table = reliability_bins(labels, proba, 10)
    assert table["n"].sum() == 37
    assert np.average(table["mean_predicted"], weights=table["n"]) == pytest.approx(proba.mean())
    assert np.average(table["observed_rate"], weights=table["n"]) == pytest.approx(labels.mean())


def test_open_calibration_summaries_on_a_hand_built_example() -> None:
    open_rows = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    test = pd.DataFrame({"y_true": CAL_Y, "proba_m": CAL_P})
    summaries, table = open_calibration(test, open_rows, models=("m",), n_bins=2)
    assert summaries["all_test_rows"]["m"] == pytest.approx({
        "n": 8, "mean_predicted": 0.475, "observed_rate": 0.5, "predicted_minus_observed": -0.025, "brier": 0.145,
    })
    assert summaries["all_test_rows"]["m"]["brier"] == pytest.approx(brier_score_loss(CAL_Y, CAL_P))
    # open rows 0-5: scores 0.9 0.1 0.4 0.2 0.8 0.3, labels 1 0 1 0 1 0
    assert summaries["open_at_filing"]["m"]["mean_predicted"] == pytest.approx(2.7 / 6)
    assert summaries["open_at_filing"]["m"]["observed_rate"] == pytest.approx(0.5)
    assert summaries["back_at_work"]["m"]["n"] == 2
    assert summaries["back_at_work"]["m"]["brier"] == pytest.approx((0.6**2 + 0.5**2) / 2)
    assert list(table.columns) == list(CALIBRATION_COLUMNS)
    assert table.groupby("group")["n"].sum().to_dict() == {"all_test_rows": 8, "back_at_work": 2, "open_at_filing": 6}
    lowest = table.set_index(["group", "bin"]).loc[("open_at_filing", 1)]
    assert lowest["mean_predicted"] == pytest.approx(0.2) and lowest["observed_rate"] == pytest.approx(0.0)


def test_ranked_recall_and_average_precision_match_the_library_and_top_share_flags() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    cases = [(TEN_LABELS, TEN_SCORES), (TEN_LABELS, (TEN_SCORES > 0.5).astype(float))]
    cases.append(((rng.random(40) < 0.4).astype(int), np.round(rng.random(40), 1)))  # many ties
    for y, proba in cases:
        for share in (0.5, 0.75):
            flagged = top_share_flags(proba, share)
            recall, ap = ranked_recall_and_average_precision(y, proba, int(flagged.sum()))
            assert recall == pytest.approx(flagged[y == 1].mean())
            assert ap == pytest.approx(average_precision_score(y, proba))


def _bootstrap_inputs() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    y = (rng.random(40) < 0.5).astype(int)
    scores = {
        "ref": np.clip(0.45 * y + rng.random(40) * 0.6, 0, 1),
        "lookup": np.round(np.clip(0.2 * y + rng.random(40) * 0.8, 0, 1), 1),
        "net": np.clip(0.3 * y + rng.random(40) * 0.7, 0, 1),
    }
    groups = np.repeat([f"mine{i}_2020" for i in range(10)], 4)
    return y, scores, groups


def test_paired_bootstrap_intervals_contain_the_point_estimate_and_repeat_with_the_seed() -> None:
    y, scores, groups = _bootstrap_inputs()
    kwargs = {"reference": "ref", "comparators": ("lookup", "net"), "n_boot": 300}
    out = paired_workload_bootstrap(y, scores, groups, **kwargs, random_state=RANDOM_STATE)
    assert out["mine_years"] == 10 and out["n"] == 40 and 0 < out["n_boot_used"] <= 300
    for name in ("ref", "lookup", "net"):
        point = out["point"][name]
        assert point["recall_top_75pct"] == pytest.approx(top_share_flags(scores[name], 0.75)[y == 1].mean())
        assert point["pr_auc"] == pytest.approx(average_precision_score(y, scores[name]))
    for pair in ("ref_minus_lookup", "ref_minus_net"):
        for metric in ("recall_top_75pct", "pr_auc"):
            cell = out["differences"][pair][metric]
            assert cell["lower"] <= cell["estimate"] <= cell["upper"]
            assert 0.0 <= cell["share_of_resamples_above_zero"] <= 1.0
    estimate = out["point"]["ref"]["pr_auc"] - out["point"]["net"]["pr_auc"]
    assert out["differences"]["ref_minus_net"]["pr_auc"]["estimate"] == pytest.approx(estimate)
    assert paired_workload_bootstrap(y, scores, groups, **kwargs, random_state=RANDOM_STATE) == out
    other = paired_workload_bootstrap(y, scores, groups, **kwargs, random_state=RANDOM_STATE + 1)
    assert other["differences"] != out["differences"]


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Scores each row by its ``score`` column and records the index of every frame it is fitted on."""

    fitted_on: list[np.ndarray] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _RecordingClassifier:
        type(self).fitted_on.append(X.index.to_numpy())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["score"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_open_only_refit_fits_on_the_open_training_rows_only() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 30
    y = (rng.random(n) < 0.5).astype(int)
    X = pd.DataFrame({"score": np.clip(0.4 * y + rng.random(n) * 0.6, 0, 1)})
    open_rows = rng.random(n) < 0.6
    test_idx = np.arange(0, n, 3)
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    stored = rng.random(test_idx.size)
    _RecordingClassifier.fitted_on = []
    out = open_only_refit(_RecordingClassifier(), X, y, train_idx, test_idx, open_rows, stored)

    expected_rows = train_idx[open_rows[train_idx]]
    assert len(_RecordingClassifier.fitted_on) == 1
    np.testing.assert_array_equal(_RecordingClassifier.fitted_on[0], expected_rows)
    assert out["training_rows_open"] == expected_rows.size and out["training_rows_all"] == train_idx.size
    open_test = test_idx[open_rows[test_idx]]
    assert out["test_rows_open"] == open_test.size
    refit_scores = X["score"].to_numpy()[open_test]
    assert out["open_only_refit"]["pr_auc"] == pytest.approx(average_precision_score(y[open_test], refit_scores))
    flagged = top_share_flags(refit_scores, 0.75)
    assert out["open_only_refit"]["top_75pct"]["recall"] == pytest.approx(flagged[y[open_test] == 1].mean())
    assert out["open_only_refit"]["top_75pct"]["n"] == open_test.size
    stored_open = stored[open_rows[test_idx]]
    assert out["all_rows_model"]["pr_auc"] == pytest.approx(average_precision_score(y[open_test], stored_open))
    assert out["refit_minus_all_rows"]["pr_auc"] == pytest.approx(
        out["open_only_refit"]["pr_auc"] - out["all_rows_model"]["pr_auc"]
    )
