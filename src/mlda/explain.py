"""Model interpretation: permutation importance, coefficients and (optionally) SHAP values.

Permutation importance is computed on the raw input columns of the pipeline, so one row per
engineered feature rather than per one-hot column. Coefficients and impurity importances are
read from the fitted estimator and mapped to the transformed column names. SHAP is imported
lazily because it is slow to import and only the inference page of the app needs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from mlda.config import RANDOM_STATE
from mlda.models import get_estimator, get_preprocessor

log = logging.getLogger(__name__)


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after the preprocessor, falling back to x0..xp when unavailable."""
    preprocessor = get_preprocessor(pipeline)
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - custom transformers may not expose names
        pass
    try:
        from mlda.preprocessing import get_feature_names

        return [str(name) for name in get_feature_names(preprocessor)]
    except Exception:  # noqa: BLE001
        n_features = _n_transformed_features(pipeline)
        return [f"x{i}" for i in range(n_features)]


def _n_transformed_features(pipeline: Pipeline) -> int:
    estimator = get_estimator(pipeline)
    for attribute in ("n_features_in_", "n_features_"):
        if hasattr(estimator, attribute):
            return int(getattr(estimator, attribute))
    inner = getattr(estimator, "estimator_", None)
    if inner is not None and hasattr(inner, "n_features_in_"):
        return int(inner.n_features_in_)
    raise ValueError("cannot determine the number of transformed features")


def _subsample(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, random_state: int):
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    X_sub = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    return X_sub, y[idx]


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Any,
    scoring: str = "average_precision",
    n_repeats: int = 5,
    max_rows: int | None = 5000,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Drop in ``scoring`` when each input column is shuffled, averaged over ``n_repeats``.

    Run on held-out rows (test or validation), never on the rows the model was fitted on, so
    the importances describe generalisation. ``max_rows`` subsamples for speed.
    """
    y_arr = np.asarray(y).astype(int)
    X_sub, y_sub = _subsample(X, y_arr, max_rows, random_state)
    result = permutation_importance(
        pipeline, X_sub, y_sub, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs,
    )
    names = list(X_sub.columns) if isinstance(X_sub, pd.DataFrame) else [f"x{i}" for i in range(X_sub.shape[1])]
    table = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    table.attrs["scoring"] = scoring
    table.attrs["n_rows"] = int(len(X_sub))
    return table.sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame | None:
    """Coefficients (linear models) or impurity importances (trees) per transformed column.

    For a linear model the table has ``coefficient``, ``abs_coefficient`` and ``odds_ratio``
    (exp of the coefficient, the multiplicative change in odds for a one-unit increase of a
    standardised input). For tree ensembles it has ``importance``. Returns None when the final
    estimator exposes neither (KNN, SVC, naive Bayes, MLP).
    """
    estimator = get_estimator(pipeline)
    names = transformed_feature_names(pipeline)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float).ravel()
        table = pd.DataFrame({"feature": names[: coef.size], "coefficient": coef})
        table["abs_coefficient"] = table["coefficient"].abs()
        table["odds_ratio"] = np.exp(table["coefficient"])
        table.attrs["kind"] = "coefficient"
        return table.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        table = pd.DataFrame({"feature": names[: importances.size], "importance": importances})
        table.attrs["kind"] = "impurity_importance"
        return table.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    return None


@dataclass
class ShapSummary:
    """Mean absolute SHAP value per transformed column plus the raw values for a beeswarm."""

    table: pd.DataFrame
    values: np.ndarray
    data: np.ndarray
    feature_names: list[str]
    explainer: str


def _pick_explainer(shap: Any, estimator: Any, background: np.ndarray) -> Any | None:
    """TreeExplainer for tree models (including HistGradientBoosting, which has no
    ``feature_importances_``), LinearExplainer for models with coefficients, else None."""
    try:
        return shap.TreeExplainer(estimator)
    except Exception:  # noqa: BLE001 - "model type not supported" arrives as several types
        pass
    if hasattr(estimator, "coef_"):
        return shap.LinearExplainer(estimator, background)
    return None


def shap_summary(
    pipeline: Pipeline, X_sample: pd.DataFrame, max_rows: int = 500, random_state: int = RANDOM_STATE
) -> ShapSummary | None:
    """SHAP values of the final estimator on the preprocessed sample, or None when unsupported.

    ``shap.Explainer`` picks the tree or linear algorithm when it can; estimators it cannot
    explain quickly (KNN, SVC, naive Bayes, MLP) make it fall back to a permutation explainer
    whose cost is too high for the app, so those return None rather than hanging.
    """
    try:
        import shap
    except ImportError:
        log.info("shap is not installed; skipping SHAP summary")
        return None
    estimator = get_estimator(pipeline)
    preprocessor = get_preprocessor(pipeline)
    X_sub, _ = _subsample(X_sample, np.zeros(len(X_sample), dtype=int), max_rows, random_state)
    transformed = np.asarray(preprocessor.transform(X_sub), dtype=float)
    names = transformed_feature_names(pipeline)
    try:
        explainer = _pick_explainer(shap, estimator, transformed)
        if explainer is None:
            return None
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:  # some tree explainers return one slab per class
            values = values[..., -1]
    except Exception as error:  # noqa: BLE001 - SHAP raises many library-specific types
        log.info("SHAP cannot explain %s: %s", type(estimator).__name__, error)
        return None
    if values.shape[1] != len(names):
        names = [f"x{i}" for i in range(values.shape[1])]
    table = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    table = table.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    return ShapSummary(
        table=table,
        values=values,
        data=transformed,
        feature_names=names,
        explainer=type(explainer).__name__,
    )


__all__ = [
    "ShapSummary",
    "coefficient_table",
    "permutation_importance_table",
    "shap_summary",
    "transformed_feature_names",
]
