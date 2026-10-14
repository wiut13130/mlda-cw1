"""ColumnTransformer options, outlier clipper and train-only fitting."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from _synthetic import synthetic_base
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from mlda.features import FEATURE_SPEC, FeatureEngineer, engineer_features
from mlda.preprocessing import (
    OPTIONS,
    OutlierClipper,
    ScaledKNNImputer,
    build_preprocessor,
    get_feature_names,
)

N_NUMERIC = len(FEATURE_SPEC["numeric"])
N_BINARY = len(FEATURE_SPEC["binary"])


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return synthetic_base(400)


@pytest.fixture(scope="module")
def frames(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train rows and test rows, the latter with one level never seen in training."""
    X = engineer_features(base)
    train, test = X.iloc[:320].copy(), X.iloc[320:].copy()
    test.iloc[0, test.columns.get_loc("SUBUNIT_CD")] = "ZZ"
    test.iloc[1, test.columns.get_loc("FIPS_STATE_CD")] = "99"
    return train, test


ALL_COMBINATIONS = list(itertools.product(*(list(values) for values in OPTIONS.values())))


@pytest.mark.parametrize(("numeric_imputer", "scaler", "encoder", "outlier_rule"), ALL_COMBINATIONS)
def test_every_option_combination_builds_and_transforms_without_nan(
    frames: tuple[pd.DataFrame, pd.DataFrame], numeric_imputer: str, scaler: str, encoder: str, outlier_rule: str | None
) -> None:
    train, test = frames
    preprocessor = build_preprocessor(numeric_imputer, scaler, encoder, outlier_rule, min_frequency=0.05)
    train_out = preprocessor.fit_transform(train)
    test_out = preprocessor.transform(test)
    assert isinstance(train_out, np.ndarray) and train_out.dtype == np.float32
    assert train_out.shape == (len(train), len(get_feature_names(preprocessor)))
    assert test_out.shape == (len(test), train_out.shape[1])
    assert not np.isnan(train_out).any() and not np.isnan(test_out).any()
    assert np.isfinite(test_out).all()


def test_options_carry_a_rationale_for_every_value() -> None:
    assert set(OPTIONS) == {"numeric_imputer", "scaler", "encoder", "outlier_rule"}
    for name, values in OPTIONS.items():
        for value, rationale in values.items():
            assert isinstance(rationale, str) and len(rationale) > 20, (name, value)
    assert None in OPTIONS["outlier_rule"]


def test_feature_names_follow_block_order(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = frames
    preprocessor = build_preprocessor(min_frequency=0.05).fit(train)
    names = get_feature_names(preprocessor)
    assert names[:N_NUMERIC] == FEATURE_SPEC["numeric"]
    assert names[-N_BINARY:] == FEATURE_SPEC["binary"]
    assert len(names) == len(set(names))
    assert all(isinstance(name, str) for name in names)


def test_get_feature_names_requires_a_fitted_preprocessor() -> None:
    with pytest.raises(NotFittedError):
        get_feature_names(build_preprocessor())


def test_onehot_infrequent_bucket_appears_for_rare_levels(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    preprocessor = build_preprocessor(min_frequency=0.05).fit(train)
    names = get_feature_names(preprocessor)
    assert "SUBUNIT_CD_infrequent_sklearn" in names  # subunits 02, 04 and 12 are each under 5 %
    assert "OCCUPATION_infrequent_sklearn" in names  # the discontinued title mapped to "missing"
    rare_levels = train["SUBUNIT_CD"].value_counts()
    rare_levels = rare_levels[rare_levels < 0.05 * len(train)].index
    assert len(rare_levels) > 0
    assert not any(f"SUBUNIT_CD_{level}" in names for level in rare_levels)
    # An unseen level at inference lands in the same bucket instead of raising.
    test_out = preprocessor.transform(test)
    assert test_out[0, names.index("SUBUNIT_CD_infrequent_sklearn")] == 1.0


def test_default_min_frequency_keeps_every_level_on_small_data(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    preprocessor = build_preprocessor().fit(train)  # 0.1 % of 320 rows is below one row
    names = get_feature_names(preprocessor)
    assert not any(name.endswith("infrequent_sklearn") for name in names)
    unseen_block = [i for i, name in enumerate(names) if name.startswith("SUBUNIT_CD_")]
    assert preprocessor.transform(test)[0, unseen_block].sum() == 0.0  # all-zero row for the unseen level


def test_standard_scaler_outputs_zero_mean_unit_variance(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = frames
    out = build_preprocessor(scaler="standard").fit_transform(train)[:, :N_NUMERIC].astype(np.float64)
    spread = out.std(axis=0)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(spread[spread > 0], 1.0, atol=1e-3)


def test_minmax_scaler_outputs_unit_interval(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = frames
    out = build_preprocessor(scaler="minmax").fit_transform(train)[:, :N_NUMERIC]
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    assert np.allclose(out.max(axis=0)[out.max(axis=0) > 0], 1.0, atol=1e-6)


def test_robust_and_no_scaler_keep_values_finite(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = frames
    robust = build_preprocessor(scaler="robust").fit_transform(train)[:, :N_NUMERIC]
    assert np.isfinite(robust).all()
    raw = build_preprocessor(scaler="none").fit_transform(train)[:, :N_NUMERIC]
    medians = train[FEATURE_SPEC["numeric"]].median()
    expected = train[FEATURE_SPEC["numeric"]].fillna(medians).to_numpy(dtype=np.float32)
    assert np.allclose(raw, expected)


def test_ordinal_encoder_codes_unseen_levels_minus_one(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    preprocessor = build_preprocessor(encoder="ordinal").fit(train)
    names = get_feature_names(preprocessor)
    assert names[N_NUMERIC:-N_BINARY] == FEATURE_SPEC["categorical"]
    out = preprocessor.transform(test)
    assert out[0, names.index("SUBUNIT_CD")] == -1.0
    assert out[1, names.index("FIPS_STATE_CD")] == -1.0
    assert (out[2:, N_NUMERIC:-N_BINARY] >= 0).all()


def test_binary_block_passes_through_unchanged(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = frames
    out = build_preprocessor().fit_transform(train)
    assert np.array_equal(out[:, -N_BINARY:], train[FEATURE_SPEC["binary"]].to_numpy(dtype=np.float32))


def test_imputer_statistics_come_from_training_rows_only(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    preprocessor = build_preprocessor(numeric_imputer="median").fit(train)
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert np.allclose(imputer.statistics_, train[FEATURE_SPEC["numeric"]].median().to_numpy(), rtol=1e-6)
    combined = pd.concat([train, test])
    assert not np.allclose(imputer.statistics_, combined[FEATURE_SPEC["numeric"]].median().to_numpy(), rtol=1e-6)


def test_outlier_clipper_iqr_fences_are_learnt_on_train_only() -> None:
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper = OutlierClipper(method="iqr").fit(train)
    assert clipper.lower_[0] == pytest.approx(-1.0) and clipper.upper_[0] == pytest.approx(7.0)
    assert np.allclose(clipper.transform(train).ravel(), [1.0, 2.0, 3.0, 4.0, 7.0])
    test = np.array([[-50.0], [np.nan], [3.5], [1000.0]])
    out = clipper.transform(test).ravel()
    assert np.allclose(out, [-1.0, np.nan, 3.5, 7.0], equal_nan=True)
    assert clipper.lower_[0] == pytest.approx(-1.0)  # transform did not refit


def test_outlier_clipper_zscore_uses_sample_standard_deviation() -> None:
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper = OutlierClipper(method="zscore", k=1.0).fit(train)
    mean, sd = train.mean(), train.std(ddof=1)
    assert clipper.upper_[0] == pytest.approx(mean + sd) and clipper.lower_[0] == pytest.approx(mean - sd)
    assert clipper.transform(train)[-1, 0] == pytest.approx(mean + sd)
    default = OutlierClipper(method="zscore").fit(train)
    assert default.upper_[0] == pytest.approx(mean + 3 * sd)


def test_outlier_clipper_handles_all_nan_and_constant_columns() -> None:
    train = np.array([[np.nan, 2.0], [np.nan, 2.0], [np.nan, 2.0]])
    clipper = OutlierClipper().fit(train)
    assert clipper.lower_[0] == -np.inf and clipper.upper_[0] == np.inf
    out = clipper.transform(np.array([[5.0, 9.0], [np.nan, -1.0]]))
    assert out[0, 0] == 5.0 and np.isnan(out[1, 0])
    # zero IQR: the fences stay open instead of collapsing the column onto its quartile
    assert clipper.lower_[1] == -np.inf and clipper.upper_[1] == np.inf
    assert np.allclose(out[:, 1], [9.0, -1.0])
    with pytest.raises(ValueError, match="method"):
        OutlierClipper(method="mad").fit(train)


def test_clipper_inside_preprocessor_bounds_extreme_test_values(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    extreme = test.copy()
    extreme.iloc[0, extreme.columns.get_loc("prev_hours")] = 1e9
    clipped = build_preprocessor(scaler="none", outlier_rule="iqr_clip").fit(train)
    unclipped = build_preprocessor(scaler="none", outlier_rule=None).fit(train)
    column = get_feature_names(clipped).index("prev_hours")
    assert clipped.transform(extreme)[0, column] < unclipped.transform(extreme)[0, column]
    assert clipped.transform(extreme)[0, column] <= train["prev_hours"].quantile(0.75) + 1.5 * (train["prev_hours"].quantile(0.75) - train["prev_hours"].quantile(0.25)) + 1.0


def test_invalid_options_raise_value_error() -> None:
    with pytest.raises(ValueError, match="numeric_imputer"):
        build_preprocessor(numeric_imputer="mode")
    with pytest.raises(ValueError, match="scaler"):
        build_preprocessor(scaler="maxabs")
    with pytest.raises(ValueError, match="encoder"):
        build_preprocessor(encoder="target")
    with pytest.raises(ValueError, match="outlier_rule"):
        build_preprocessor(outlier_rule="winsor")
    assert build_preprocessor(outlier_rule="none") is not None  # string alias for None


def test_pipeline_with_feature_engineer_scores_a_single_base_row(base: pd.DataFrame) -> None:
    pipeline = Pipeline([("features", FeatureEngineer()), ("preprocess", build_preprocessor(min_frequency=0.05))])
    pipeline.fit(base.iloc[:320])
    width = len(get_feature_names(pipeline.named_steps["preprocess"]))
    one_row = pipeline.transform(base.iloc[[350]])
    assert one_row.shape == (1, width) and one_row.dtype == np.float32
    assert not np.isnan(one_row).any()
    assert clone(pipeline).fit(base.iloc[:320]).transform(base.iloc[[350]]).shape == (1, width)


def test_iqr_clipper_leaves_a_near_constant_column_with_its_few_large_values() -> None:
    """The capped victim count is 1 on 99 % of rows, so Q1 = Q3 = 1; the fences must not collapse onto 1."""
    victims = np.ones(1000)
    victims[::100] = 7.0  # 1 % of multi-victim accidents
    train = np.column_stack([victims, np.arange(1000.0)])
    clipper = OutlierClipper(method="iqr").fit(train)
    assert clipper.lower_[0] == -np.inf and clipper.upper_[0] == np.inf
    assert clipper.lower_[1] == pytest.approx(-499.5) and clipper.upper_[1] == pytest.approx(1498.5)  # an ordinary column
    out = clipper.transform(train)
    assert out[:, 0].std() > 0 and out[:, 0].max() == 7.0
    spread = OutlierClipper(method="zscore").fit(np.full((10, 1), 4.0))
    assert spread.lower_[0] == -np.inf and spread.upper_[0] == np.inf  # zero standard deviation


def test_clippers_fit_without_numpy_deprecation_or_runtime_warnings() -> None:
    train = np.array([[1.0, np.nan, 5.0], [2.0, np.nan, 5.0], [3.0, np.nan, 5.0], [4.0, np.nan, 5.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # np.testing.suppress_warnings raised a DeprecationWarning on NumPy 2.5
        for method in ("iqr", "zscore"):
            clipper = OutlierClipper(method=method).fit(train)
            assert clipper.lower_[1] == -np.inf and clipper.upper_[1] == np.inf  # the all-NaN column


def test_knn_imputer_measures_distances_on_standardised_columns() -> None:
    rng = np.random.default_rng(0)
    n = 400
    informative = rng.normal(size=n)
    huge_noise = rng.normal(size=n) * 1e6  # a column on the scale of prev_hours, unrelated to the target
    target = 3 * informative + rng.normal(scale=0.1, size=n)
    X = np.column_stack([huge_noise, informative, target])
    missing = rng.choice(n, 40, replace=False)
    X_missing = X.copy()
    X_missing[missing, 2] = np.nan

    unscaled = KNNImputer(n_neighbors=5).fit_transform(X_missing)[missing, 2]
    scaled = ScaledKNNImputer(n_neighbors=5).fit(X_missing).transform(X_missing)
    # raw distances are decided by the million-scale column, so the neighbours ignore the informative one
    assert abs(np.corrcoef(unscaled, target[missing])[0, 1]) < 0.4
    assert np.corrcoef(scaled[missing, 2], target[missing])[0, 1] > 0.9
    # observed values come back in the original units, untouched
    observed = np.setdiff1d(np.arange(n), missing)
    np.testing.assert_allclose(scaled[:, :2], X[:, :2])
    np.testing.assert_allclose(scaled[observed, 2], target[observed])
    assert not np.isnan(scaled).any()


def test_knn_option_uses_the_scaled_imputer_and_keeps_all_columns(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = frames
    preprocessor = build_preprocessor(numeric_imputer="knn", scaler="none", min_frequency=0.05).fit(train)
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert isinstance(imputer, ScaledKNNImputer)
    names = get_feature_names(preprocessor)
    assert names[:N_NUMERIC] == FEATURE_SPEC["numeric"]
    out = preprocessor.transform(test)
    column = names.index("prev_hours")
    present = test["prev_hours"].notna().to_numpy()
    np.testing.assert_allclose(out[present, column], test["prev_hours"].to_numpy(dtype=np.float32)[present], rtol=1e-4)


def test_median_rationale_names_both_skew_ranges() -> None:
    text = OPTIONS["numeric_imputer"]["median"]
    assert "1.4 to 2.3" in text and "3.2 to 4.7" in text  # experience columns, then exposure columns
