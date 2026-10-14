"""EDA helpers checked against numpy / scipy on synthetic frames built to docs/interfaces.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mlda.config import RANDOM_STATE
from mlda.eda import (
    NUMERIC_STATS,
    CorrelationResult,
    class_rate_by_level,
    correlations,
    cramers_v,
    cramers_v_table,
    describe_categorical,
    describe_numeric,
    missingness_table,
    point_biserial,
    vif_table,
)

NUMERIC = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "prev_hours", "hours_into_shift"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Skewed experience and exposure columns, two categoricals and a 0/1 target with NaN gaps."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 600
    nature = rng.choice(["330", "180", "220", "320", "160"], size=n, p=[0.35, 0.25, 0.15, 0.1, 0.15])
    sector = rng.choice(["C", "M"], size=n)
    rate = pd.Series(nature).map({"330": 0.7, "180": 0.2, "220": 0.6, "320": 0.1, "160": 0.5}).to_numpy()
    y = (rng.random(n) < rate + np.where(sector == "C", 0.1, 0.0)).astype("int8")
    df = pd.DataFrame(
        {
            "TOT_EXPER": rng.exponential(8, n),
            "MINE_EXPER": rng.exponential(4, n),
            "JOB_EXPER": rng.exponential(3, n),
            "prev_hours": rng.lognormal(11, 1.2, n),
            "hours_into_shift": rng.uniform(0, 12, n),
            "NATURE_INJURY_CD": nature,
            "COAL_METAL_IND": sector,
            "lost_time": y,
        }
    )
    df.loc[rng.choice(n, 30, replace=False), "TOT_EXPER"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "NATURE_INJURY_CD"] = np.nan
    return df


# --- describe_numeric -------------------------------------------------------


def test_describe_numeric_columns_in_contract_order(frame: pd.DataFrame) -> None:
    table = describe_numeric(frame, NUMERIC)
    assert list(table.columns) == NUMERIC_STATS
    assert list(table.index) == NUMERIC
    assert table.index.name == "variable"


def test_describe_numeric_matches_numpy_and_scipy(frame: pd.DataFrame) -> None:
    table = describe_numeric(frame, NUMERIC)
    for col in NUMERIC:
        x = frame[col].dropna().to_numpy()
        row = table.loc[col]
        assert row["n"] == len(x)
        assert row["missing_pct"] == pytest.approx(100 * frame[col].isna().mean())
        assert row["mean"] == pytest.approx(x.mean())
        assert row["median"] == pytest.approx(np.median(x))
        assert row["trimmed_mean_10"] == pytest.approx(stats.trim_mean(x, 0.1))
        assert row["variance"] == pytest.approx(np.var(x, ddof=1))
        assert row["std"] == pytest.approx(np.std(x, ddof=1))
        assert row["min"] == pytest.approx(x.min())
        assert row["q1"] == pytest.approx(np.percentile(x, 25))
        assert row["q3"] == pytest.approx(np.percentile(x, 75))
        assert row["max"] == pytest.approx(x.max())
        assert row["iqr"] == pytest.approx(np.percentile(x, 75) - np.percentile(x, 25))
        assert row["skewness"] == pytest.approx(stats.skew(x, bias=False))
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(x, fisher=True, bias=False))


def test_describe_numeric_agrees_with_pandas_shape_statistics(frame: pd.DataFrame) -> None:
    # pandas .skew() / .kurt() are the bias-corrected estimators the report quotes
    table = describe_numeric(frame, ["TOT_EXPER"])
    assert table.loc["TOT_EXPER", "skewness"] == pytest.approx(frame["TOT_EXPER"].skew())
    assert table.loc["TOT_EXPER", "kurtosis"] == pytest.approx(frame["TOT_EXPER"].kurt())


def test_describe_numeric_trimmed_mean_resists_an_outlier() -> None:
    df = pd.DataFrame({"x": [1.0] * 19 + [1000.0]})
    row = describe_numeric(df, ["x"]).loc["x"]
    assert row["mean"] > 50
    assert row["trimmed_mean_10"] == pytest.approx(1.0)
    assert row["median"] == pytest.approx(1.0)


def test_describe_numeric_handles_empty_and_tiny_columns() -> None:
    df = pd.DataFrame({"empty": [np.nan, np.nan, np.nan], "two": [1.0, 3.0, np.nan]})
    table = describe_numeric(df)
    assert table.loc["empty", "n"] == 0
    assert table.loc["empty", "missing_pct"] == pytest.approx(100.0)
    assert np.isnan(table.loc["empty", "mean"])
    assert table.loc["two", "variance"] == pytest.approx(2.0)
    assert np.isnan(table.loc["two", "skewness"])


def test_describe_numeric_unknown_column_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        describe_numeric(frame, ["no_such_column"])


# --- describe_categorical ---------------------------------------------------


def test_describe_categorical_levels_mode_and_missing() -> None:
    df = pd.DataFrame({"c": ["a", "a", "b", None, "c", "a"]})
    row = describe_categorical(df, ["c"]).loc["c"]
    assert row["n"] == 5
    assert row["n_levels"] == 3
    assert row["top_level"] == "a"
    assert row["top_n"] == 3
    assert row["top_share_pct"] == pytest.approx(60.0)
    assert row["missing_pct"] == pytest.approx(100 / 6)


def test_describe_categorical_defaults_to_non_numeric_columns(frame: pd.DataFrame) -> None:
    table = describe_categorical(frame)
    assert set(table.index) == {"NATURE_INJURY_CD", "COAL_METAL_IND"}


# --- correlations -----------------------------------------------------------


def test_correlation_matrices_symmetric_with_unit_diagonal(frame: pd.DataFrame) -> None:
    result = correlations(frame, NUMERIC)
    assert isinstance(result, CorrelationResult)
    for matrix in (result.pearson, result.spearman):
        assert matrix.shape == (len(NUMERIC), len(NUMERIC))
        assert list(matrix.index) == NUMERIC and list(matrix.columns) == NUMERIC
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0)
        assert ((matrix >= -1) & (matrix <= 1)).all().all()


def test_correlations_match_pandas_and_scipy(frame: pd.DataFrame) -> None:
    result = correlations(frame, ["MINE_EXPER", "JOB_EXPER"])
    x, y = frame["MINE_EXPER"], frame["JOB_EXPER"]
    assert result.pearson.loc["MINE_EXPER", "JOB_EXPER"] == pytest.approx(stats.pearsonr(x, y).statistic)
    assert result.spearman.loc["MINE_EXPER", "JOB_EXPER"] == pytest.approx(stats.spearmanr(x, y).statistic)


def test_correlations_flag_monotone_non_linear_pair() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.uniform(0, 5, 400)
    df = pd.DataFrame({"x": x, "exp_x": np.exp(2 * x), "noise": rng.normal(size=400)})
    result = correlations(df, threshold=0.1)
    pair = result.disagreements.set_index(["var_a", "var_b"])
    assert ("x", "exp_x") in pair.index
    assert pair.loc[("x", "exp_x"), "spearman"] == pytest.approx(1.0)
    assert pair.loc[("x", "exp_x"), "pearson"] < 0.9
    assert (result.disagreements["abs_diff"] > 0.1).all()
    assert result.disagreements["abs_diff"].is_monotonic_decreasing
    # each unordered pair appears once
    assert not result.disagreements.duplicated(["var_a", "var_b"]).any()


# --- point_biserial ---------------------------------------------------------


def test_point_biserial_equals_pearson_on_binary_coding(frame: pd.DataFrame) -> None:
    table = point_biserial(frame, NUMERIC, "lost_time").set_index("feature")
    for col in NUMERIC:
        mask = frame[col].notna()
        expected = np.corrcoef(frame.loc[mask, "lost_time"].astype(float), frame.loc[mask, col])[0, 1]
        assert table.loc[col, "r"] == pytest.approx(expected)
        assert table.loc[col, "n"] == mask.sum()
        assert 0 <= table.loc[col, "p_value"] <= 1
    assert table["r"].abs().is_monotonic_decreasing


def test_point_biserial_rejects_non_binary_target(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        point_biserial(frame.assign(lost_time=[0, 1, 2] * 200), NUMERIC, "lost_time")


# --- cramers_v --------------------------------------------------------------


def test_cramers_v_is_zero_for_a_balanced_independent_design() -> None:
    # every (level, class) cell has the same count, so chi-square is exactly zero
    a = np.repeat(["p", "q", "r"], 40)
    b = np.tile([0, 1], 60)
    assert cramers_v(pd.Series(a), pd.Series(b)) == 0.0


def test_cramers_v_is_near_zero_for_random_independent_columns() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    a = pd.Series(rng.choice(list("abcdef"), 5000))
    b = pd.Series(rng.integers(0, 2, 5000))
    assert cramers_v(a, b) < 0.05


def test_cramers_v_is_one_for_a_copy() -> None:
    a = pd.Series(np.repeat(["x", "y", "z", "w"], 25))
    assert cramers_v(a, a.copy()) == pytest.approx(1.0)
    assert cramers_v(a, a.map({"x": 0, "y": 1, "z": 2, "w": 3})) == pytest.approx(1.0)


def test_cramers_v_undefined_for_a_constant_column() -> None:
    a = pd.Series(["only"] * 50)
    b = pd.Series([0, 1] * 25)
    assert np.isnan(cramers_v(a, b))


def test_cramers_v_table_levels_and_ordering(frame: pd.DataFrame) -> None:
    table = cramers_v_table(frame, ["COAL_METAL_IND", "NATURE_INJURY_CD"], "lost_time")
    assert list(table.columns) == ["feature", "n", "n_levels", "cramers_v", "chi2", "p_value"]
    by_feature = table.set_index("feature")
    assert by_feature.loc["NATURE_INJURY_CD", "n_levels"] == 6  # five codes plus "missing"
    assert by_feature.loc["COAL_METAL_IND", "n_levels"] == 2
    assert table["cramers_v"].is_monotonic_decreasing
    assert table.iloc[0]["feature"] == "NATURE_INJURY_CD"
    assert (table["n"] == len(frame)).all()


def test_cramers_v_table_can_drop_missing(frame: pd.DataFrame) -> None:
    table = cramers_v_table(frame, ["NATURE_INJURY_CD"], "lost_time", missing_as_level=False)
    assert table.loc[0, "n_levels"] == 5
    assert table.loc[0, "n"] == frame["NATURE_INJURY_CD"].notna().sum()


# --- vif_table --------------------------------------------------------------


def _manual_vif(X: np.ndarray, j: int) -> float:
    others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
    beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
    residual = X[:, j] - others @ beta
    r_sq = 1 - residual.var() / X[:, j].var()
    return 1 / (1 - r_sq)


def test_vif_matches_manual_regression_formula() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": b, "ab_mix": a + 0.5 * b + rng.normal(scale=0.3, size=500)})
    table = vif_table(df).set_index("feature")
    X = df.to_numpy()
    for j, col in enumerate(df.columns):
        assert table.loc[col, "vif"] == pytest.approx(_manual_vif(X, j), rel=1e-6)
    assert table.loc["ab_mix", "vif"] > 5
    assert table["vif"].is_monotonic_decreasing


def test_vif_independent_columns_near_one_and_collinear_column_explodes(frame: pd.DataFrame) -> None:
    table = vif_table(frame[NUMERIC]).set_index("feature")
    assert (table["vif"] < 1.5).all()
    with_copy = vif_table(frame[NUMERIC].assign(dup=2 * frame["TOT_EXPER"] + 1)).set_index("feature")
    assert with_copy.loc["dup", "vif"] > 100
    assert with_copy.loc["TOT_EXPER", "vif"] > 100


def test_vif_imputes_missing_and_marks_constant_columns(frame: pd.DataFrame) -> None:
    df = frame[NUMERIC].assign(const=1.0, all_nan=np.nan)
    table = vif_table(df).set_index("feature")
    assert np.isfinite(table.loc["TOT_EXPER", "vif"])  # TOT_EXPER has NaN gaps
    assert np.isnan(table.loc["const", "vif"])
    assert "all_nan" not in table.index


# --- missingness_table ------------------------------------------------------


def test_missingness_table_counts_and_order(frame: pd.DataFrame) -> None:
    table = missingness_table(frame)
    assert list(table.columns) == ["column", "dtype", "n_missing", "n_present", "missing_pct"]
    assert table.iloc[0]["column"] == "TOT_EXPER"
    assert table.iloc[0]["n_missing"] == 30
    assert table.iloc[0]["missing_pct"] == pytest.approx(100 * 30 / len(frame))
    assert table.iloc[1]["column"] == "NATURE_INJURY_CD"
    assert (table["n_missing"] + table["n_present"] == len(frame)).all()
    assert table["missing_pct"].is_monotonic_decreasing


# --- class_rate_by_level ----------------------------------------------------


def test_class_rate_by_level_rates_pooling_and_lift() -> None:
    df = pd.DataFrame(
        {
            "c": ["a"] * 10 + ["b"] * 6 + ["c"] * 2 + ["d"] * 1 + [None] * 5,
            "y": [1] * 8 + [0] * 2 + [1] * 3 + [0] * 3 + [1, 0] + [1] + [0] * 5,
        }
    )
    table = class_rate_by_level(df, "c", "y", min_n=5).set_index("level")
    assert list(table.index) == ["a", "b", "missing", "other (n < 5)"]
    assert table.loc["a", "n"] == 10
    assert table.loc["a", "positive_rate_pct"] == pytest.approx(80.0)
    assert table.loc["b", "positive_rate_pct"] == pytest.approx(50.0)
    assert table.loc["missing", "positive_rate_pct"] == pytest.approx(0.0)
    assert table.loc["other (n < 5)", "n"] == 3
    assert table.loc["other (n < 5)", "positive_rate_pct"] == pytest.approx(200 / 3)
    assert table["share_pct"].sum() == pytest.approx(100.0)
    overall = df["y"].mean()
    assert table.loc["a", "lift"] == pytest.approx(0.8 / overall)


def test_class_rate_by_level_without_rare_levels_has_no_pooled_row(frame: pd.DataFrame) -> None:
    table = class_rate_by_level(frame, "COAL_METAL_IND", "lost_time", min_n=10)
    assert list(table["level"]) == sorted(table["level"], key=lambda lvl: -table.set_index("level").loc[lvl, "n"])
    assert not table["level"].astype(str).str.startswith("other").any()
    assert table["n"].sum() == len(frame)
