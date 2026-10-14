"""Model zoo and tuner on 300 synthetic rows with a numeric-only preprocessor.

The feature and preprocessing modules are deliberately not imported: a small
ColumnTransformer (SimpleImputer + StandardScaler) built here stands in for them, so these
tests exercise only the contract of ``mlda.models`` and ``mlda.tuning``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlda.config import RANDOM_STATE
from mlda.models import (
    ESTIMATOR_STEP,
    MODEL_ZOO,
    PREPROCESSOR_STEP,
    ModelSpec,
    build_pipeline,
    get_estimator,
    get_preprocessor,
    grid_summary,
    quick_grid,
)
from mlda.tuning import (
    METRIC_NAMES,
    TuneResult,
    default_n_jobs,
    estimator_params,
    fold_scores,
    stratified_group_subsample,
    stratified_subsample,
    tune,
)

N_ROWS = 300
N_FEATURES = 8
N_SPLITS = 3


def make_synthetic(n_rows: int = N_ROWS) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Numeric frame, binary target and 5-row groups, with a few missing values to impute."""
    X, y = make_classification(
        n_samples=n_rows, n_features=N_FEATURES, n_informative=5, n_redundant=1,
        class_sep=1.0, random_state=RANDOM_STATE,
    )
    frame = pd.DataFrame(X, columns=[f"num_{i}" for i in range(N_FEATURES)])
    rng = np.random.default_rng(RANDOM_STATE)
    frame.iloc[rng.choice(n_rows, 10, replace=False), 0] = np.nan
    groups = np.arange(n_rows) // 5
    return frame, y.astype(int), groups


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("num", numeric, columns)])


def make_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return make_synthetic()


# ---------------------------------------------------------------------------
# MODEL_ZOO
# ---------------------------------------------------------------------------


def test_zoo_contains_the_eight_models() -> None:
    expected = {"logreg", "knn", "gaussian_nb", "decision_tree", "random_forest", "hist_gb", "svc_rbf", "mlp"}
    assert set(MODEL_ZOO) == expected
    for name, spec in MODEL_ZOO.items():
        assert spec.name == name
        assert isinstance(spec, ModelSpec)


def test_descriptions_weeks_and_search_settings() -> None:
    for spec in MODEL_ZOO.values():
        assert len(spec.description) > 40, spec.name
        assert isinstance(spec.taught_in_week, int) and 1 <= spec.taught_in_week <= 12
        assert spec.search in {"grid", "random"}
        if spec.search == "random":
            assert spec.n_iter is not None and spec.n_iter >= 1
        assert all(key.startswith(f"{ESTIMATOR_STEP}__") for key in spec.param_grid), spec.name
        assert all(isinstance(values, list) and values for values in spec.param_grid.values())


def test_sample_caps() -> None:
    assert MODEL_ZOO["svc_rbf"].sample_cap == 10_000
    assert MODEL_ZOO["knn"].sample_cap is not None
    for name in ("logreg", "gaussian_nb", "decision_tree", "random_forest", "hist_gb", "mlp"):
        assert MODEL_ZOO[name].sample_cap is None


def test_tree_models_do_not_ask_for_a_scaler() -> None:
    for name in ("decision_tree", "random_forest", "hist_gb"):
        assert MODEL_ZOO[name].preprocess_overrides.get("scaler") == "none"


def test_grid_summary_is_small_enough() -> None:
    sizes = grid_summary()
    assert set(sizes) == set(MODEL_ZOO)
    assert all(1 <= n <= 30 for n in sizes.values()), sizes
    assert MODEL_ZOO["random_forest"].n_candidates == 4


@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_param_grid_keys_address_pipeline_parameters(name: str, data) -> None:
    X, _, _ = data
    pipeline = build_pipeline(MODEL_ZOO[name], make_preprocessor(list(X.columns)))
    params = pipeline.get_params()
    for key in MODEL_ZOO[name].param_grid:
        assert key in params, f"{name}: {key} is not a pipeline parameter"


@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_each_model_fits_and_predicts_probabilities(name: str, data) -> None:
    X, y, _ = data
    spec = MODEL_ZOO[name]
    pipeline = build_pipeline(spec, make_preprocessor(list(X.columns)))
    assert [step for step, _ in pipeline.steps] == [PREPROCESSOR_STEP, ESTIMATOR_STEP]
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (N_ROWS, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(np.unique(pipeline.predict(X))) <= {0, 1}
    assert get_estimator(pipeline) is pipeline.named_steps[ESTIMATOR_STEP]
    assert get_preprocessor(pipeline) is pipeline.named_steps[PREPROCESSOR_STEP]


@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_quick_grid_has_at_most_two_in_grid_points(name: str) -> None:
    spec = MODEL_ZOO[name]
    grid = quick_grid(spec)
    assert 1 <= len(grid) <= 2
    assert len(list(ParameterGrid(grid))) <= 2
    for point in grid:
        assert set(point) == set(spec.param_grid)
        for key, values in point.items():
            assert len(values) == 1
            assert values[0] in spec.param_grid[key]


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cores", "expected"), [(4, 2), (10, 8), (20, 8), (3, 1), (1, 1), (None, 1)])
def test_default_n_jobs_leaves_two_cores(monkeypatch: pytest.MonkeyPatch, cores: int | None, expected: int) -> None:
    monkeypatch.setattr("mlda.tuning.os.cpu_count", lambda: cores)
    assert default_n_jobs() == expected  # cpu_count - 2, at least 1, at most 8


def test_stratified_subsample_keeps_prevalence_and_is_deterministic() -> None:
    y = np.array([1] * 100 + [0] * 200)
    idx = stratified_subsample(y, 90)
    assert len(idx) == 90 and len(np.unique(idx)) == 90
    assert abs(y[idx].mean() - y.mean()) < 0.02
    np.testing.assert_array_equal(idx, stratified_subsample(y, 90))
    np.testing.assert_array_equal(stratified_subsample(y, 1000), np.arange(300))


def test_stratified_group_subsample_keeps_whole_groups_and_prevalence() -> None:
    rng = np.random.default_rng(3)
    sizes = rng.integers(1, 6, 400)
    groups = np.repeat(np.arange(400), sizes)
    y = (rng.random(groups.size) < 0.45).astype(int)
    idx = stratified_group_subsample(y, groups, 600)
    rest = np.setdiff1d(np.arange(groups.size), idx)
    assert 540 <= len(idx) <= 600 and np.array_equal(idx, np.sort(idx))
    assert set(groups[idx]).isdisjoint(groups[rest])  # whole groups on each side
    assert abs(y[idx].mean() - y.mean()) < 0.03
    np.testing.assert_array_equal(idx, stratified_group_subsample(y, groups, 600))
    assert not np.array_equal(idx, stratified_group_subsample(y, groups, 600, random_state=7))
    np.testing.assert_array_equal(stratified_group_subsample(y, groups, 10_000), np.arange(groups.size))


def _check_result(result: TuneResult, spec: ModelSpec, n_rows: int) -> None:
    assert result.model == spec.name
    assert set(result.best_params) == set(spec.param_grid)
    for key, value in result.best_params.items():
        assert value in spec.param_grid[key], f"{spec.name}: {key}={value!r} not in grid"
    assert set(estimator_params(result)) == {k.rsplit("__", 1)[-1] for k in spec.param_grid}
    assert result.oof_proba.shape == (n_rows,)
    assert np.isfinite(result.oof_proba).all()
    assert np.all(result.oof_proba >= 0) and np.all(result.oof_proba <= 1)
    assert 1 <= result.n_candidates <= 2
    assert len(result.cv_results_df) == result.n_candidates
    frame = result.cv_results_df
    for metric in METRIC_NAMES:
        assert f"mean_test_{metric}" in frame.columns
        assert f"mean_train_{metric}" in frame.columns
        assert f"split0_test_{metric}" in frame.columns
    assert frame["model"].eq(spec.name).all()
    assert frame["rank_test_average_precision"].iloc[0] == 1
    assert len(fold_scores(result)) == N_SPLITS
    assert result.elapsed_seconds > 0


@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_tune_quick_returns_in_grid_params_and_full_length_oof(name: str, data) -> None:
    X, y, groups = data
    spec = MODEL_ZOO[name]
    result = tune(
        spec, X, y, groups, make_cv(), quick=True,
        preprocessor=make_preprocessor(list(X.columns)), n_jobs=1,
    )
    _check_result(result, spec, N_ROWS)
    assert len(result.fit_idx) == N_ROWS  # 300 rows sit below every cap
    proba = result.best_pipeline.predict_proba(X)
    assert proba.shape == (N_ROWS, 2)


def test_tune_with_sample_cap_scores_every_training_row(data) -> None:
    X, y, groups = data
    spec = replace(MODEL_ZOO["logreg"], sample_cap=120)
    result = tune(
        spec, X, y, groups, make_cv(), quick=True,
        preprocessor=make_preprocessor(list(X.columns)), n_jobs=1,
    )
    _check_result(result, spec, N_ROWS)
    assert len(result.fit_idx) == 120
    assert abs(y[result.fit_idx].mean() - y.mean()) < 0.05
    rest = np.setdiff1d(np.arange(N_ROWS), result.fit_idx)
    # rows outside the cap are scored by the refitted pipeline, which never saw them ...
    np.testing.assert_allclose(
        result.oof_proba[rest], result.best_pipeline.predict_proba(X.iloc[rest])[:, 1]
    )
    # ... and never saw their groups either: the cap is drawn by whole group, not by row
    assert set(groups[rest]).isdisjoint(groups[result.fit_idx])
    assert len(result.fit_idx) % 5 == 0  # the synthetic groups have five rows each


def test_oof_probabilities_come_from_models_that_never_saw_the_row(data) -> None:
    """Rebuild the folds by hand: a tuner that returned in-sample scores would fail here."""
    X, y, groups = data
    spec = MODEL_ZOO["logreg"]
    result = tune(
        spec, X, y, groups, make_cv(), quick=True,
        preprocessor=make_preprocessor(list(X.columns)), n_jobs=1,
    )
    assert len(result.fit_idx) == N_ROWS
    expected = np.full(N_ROWS, np.nan)
    for train, valid in make_cv().split(X, y, groups):
        model = clone(result.best_pipeline).fit(X.iloc[train], y[train])
        expected[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    np.testing.assert_allclose(result.oof_proba, expected)
    in_sample = result.best_pipeline.predict_proba(X)[:, 1]
    assert not np.allclose(result.oof_proba, in_sample)


def test_tidy_cv_results_strips_step_prefixes(data) -> None:
    X, y, groups = data
    result = tune(
        MODEL_ZOO["logreg"], X, y, groups, make_cv(), quick=True,
        preprocessor=make_preprocessor(list(X.columns)), n_jobs=1,
    )
    columns = list(result.cv_results_df.columns)
    assert columns[0] == "model"
    assert "param_C" in columns and "param_class_weight" in columns
    assert not any(c.startswith("param_clf__") for c in columns)
    assert columns.index("params") < columns.index("mean_test_average_precision")


def test_tune_rejects_mismatched_lengths(data) -> None:
    X, y, groups = data
    with pytest.raises(ValueError):
        tune(MODEL_ZOO["logreg"], X, y[:-1], groups, make_cv(), quick=True,
             preprocessor=make_preprocessor(list(X.columns)), n_jobs=1)
