"""Importance tables, coefficients and the SHAP fallback on a small synthetic problem.

One feature (``signal``) carries the label, three more are noise, so the expected ordering of
every table is known in advance. The pipelines are built with ``mlda.models.build_pipeline`` and a
numeric-only preprocessor, so these tests do not depend on the real feature frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlda.explain import (
    ShapSummary,
    coefficient_table,
    permutation_importance_table,
    shap_summary,
    transformed_feature_names,
)
from mlda.models import MODEL_ZOO, build_pipeline

COLUMNS = ["signal", "noise_a", "noise_b", "noise_c"]


@pytest.fixture(scope="module")
def problem() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, len(COLUMNS))), columns=COLUMNS)
    y = (X["signal"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def fitted(name: str, problem: tuple[pd.DataFrame, np.ndarray]) -> Pipeline:
    X, y = problem
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocessor = ColumnTransformer([("num", numeric, COLUMNS)], verbose_feature_names_out=False)
    return build_pipeline(MODEL_ZOO[name], preprocessor).fit(X, y)


def test_permutation_importance_ranks_the_informative_column_first(problem) -> None:
    X, y = problem
    table = permutation_importance_table(fitted("logreg", problem), X, y, n_repeats=3)
    assert list(table.columns) == ["feature", "importance_mean", "importance_std"]
    assert table["feature"].tolist()[0] == "signal"
    assert sorted(table["feature"]) == sorted(COLUMNS)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["importance_mean"].iloc[0] > 5 * abs(table["importance_mean"].iloc[1:]).max()
    assert table.attrs["scoring"] == "average_precision" and table.attrs["n_rows"] == len(X)


def test_permutation_importance_subsamples_and_is_reproducible(problem) -> None:
    X, y = problem
    pipeline = fitted("logreg", problem)
    first = permutation_importance_table(pipeline, X, y, max_rows=200, n_repeats=2)
    second = permutation_importance_table(pipeline, X, y, max_rows=200, n_repeats=2)
    assert first.attrs["n_rows"] == 200
    pd.testing.assert_frame_equal(first, second)


def test_coefficient_table_for_a_linear_model(problem) -> None:
    table = coefficient_table(fitted("logreg", problem))
    assert table is not None and table.attrs["kind"] == "coefficient"
    assert table["feature"].tolist()[0] == "signal"
    assert table["coefficient"].iloc[0] > 0
    np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))
    np.testing.assert_allclose(table["abs_coefficient"], table["coefficient"].abs())
    assert table["abs_coefficient"].is_monotonic_decreasing and len(table) == len(COLUMNS)


def test_coefficient_table_for_a_tree_uses_impurity_importance(problem) -> None:
    table = coefficient_table(fitted("decision_tree", problem))
    assert table is not None and table.attrs["kind"] == "impurity_importance"
    assert table["feature"].tolist()[0] == "signal"
    assert table["importance"].sum() == pytest.approx(1.0)


def test_coefficient_table_is_none_for_estimators_without_either(problem) -> None:
    assert coefficient_table(fitted("knn", problem)) is None


def test_transformed_feature_names_follow_the_preprocessor(problem) -> None:
    assert transformed_feature_names(fitted("logreg", problem)) == COLUMNS


def test_shap_summary_is_none_for_an_unsupported_estimator(problem) -> None:
    X, _ = problem
    assert shap_summary(fitted("knn", problem), X, max_rows=50) is None


def test_shap_summary_for_a_tree_ranks_the_informative_column_first(problem) -> None:
    pytest.importorskip("shap")
    X, _ = problem
    summary = shap_summary(fitted("random_forest", problem), X, max_rows=100)
    assert isinstance(summary, ShapSummary)
    assert summary.table["feature"].tolist()[0] == "signal"
    assert summary.values.shape == (100, len(COLUMNS)) and summary.data.shape == summary.values.shape
    assert summary.feature_names == COLUMNS and summary.explainer == "TreeExplainer"
