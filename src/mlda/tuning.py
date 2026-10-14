"""Hyper-parameter search and out-of-fold probabilities for one model specification.

``tune`` runs GridSearchCV (or RandomizedSearchCV for specs that ask for it) with several
scorers at once, refits on average precision, and then produces out-of-fold probabilities
with ``cross_val_predict`` on the same grouped folds. Those probabilities are what the
threshold analysis in ``mlda.evaluation`` uses, so no threshold is ever chosen on test rows.

Row caps. Some models (SVC, KNN) are fitted on a subsample of the training rows
(``ModelSpec.sample_cap``). The subsample is drawn by whole ``group_id`` values, stratified on
the label (``stratified_group_subsample``), so the remaining rows share no mine-year with any
row the refitted pipeline saw. The search and ``cross_val_predict`` run on the subset only and
give its rows genuine out-of-fold probabilities; the remaining rows are scored by the refitted
best pipeline and are out-of-sample by row and by group. ``TuneResult.fit_idx`` records which
rows belong to the subset.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from mlda.config import RANDOM_STATE
from mlda.models import ESTIMATOR_STEP, ModelSpec, build_pipeline, default_preprocessor, quick_grid

log = logging.getLogger(__name__)

REFIT_METRIC = "average_precision"
QUICK_SAMPLE_CAP = 10_000
SCORING: dict[str, Any] = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "accuracy": "accuracy",
    "neg_brier_score": "neg_brier_score",
}
METRIC_NAMES = tuple(SCORING)


@dataclass
class TuneResult:
    """Outcome of ``tune`` for one model.

    ``oof_proba`` has one entry per training row (positional order of ``X_train``);
    ``fit_idx`` holds the positional indices of the rows the search was fitted on, which is
    every row unless the spec has a ``sample_cap`` or ``quick=True`` capped the rows.
    """

    best_pipeline: Pipeline
    best_params: dict[str, Any]
    cv_results_df: pd.DataFrame
    oof_proba: np.ndarray
    fit_idx: np.ndarray
    best_index: int
    n_candidates: int
    elapsed_seconds: float
    model: str = ""


def default_n_jobs() -> int:
    """Parallel workers for the search: at most eight, leaving two cores for the system."""
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def stratified_subsample(y: np.ndarray, cap: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Positional indices of a class-stratified subsample of at most ``cap`` rows (sorted)."""
    y = np.asarray(y)
    if cap >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=cap, stratify=y, random_state=random_state
    )
    return np.sort(idx)


def stratified_group_subsample(
    y: np.ndarray, groups: np.ndarray, cap: int, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """Positional indices (sorted) of at most ``cap`` rows made of whole groups, class-stratified.

    Rows of one mine-year share an outcome pattern (82 % of rows have a sibling), so a row-level
    subsample would leave the rows it drops in the same groups as the rows it keeps. Here the
    groups are ordered by their positive share, then by size, with a random tie-break, and every
    k-th group is taken (systematic sampling from a random start) with k chosen so that the
    expected number of rows is ``cap``. Each stratum of the ordering is thereby represented in
    proportion, which keeps the class share of the rows close to that of the full set. Groups
    are dropped at random if the draw lands above ``cap``.
    """
    y = np.asarray(y).astype(int)
    n_rows = len(y)
    if cap >= n_rows:
        return np.arange(n_rows)
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    size = np.bincount(inverse)
    positives = np.bincount(inverse, weights=y)
    n_groups = len(size)
    rng = np.random.default_rng(random_state)
    jitter = rng.random(n_groups)
    order = np.lexsort((jitter, size, positives / size))
    spacing = n_rows / cap  # one group in `spacing` is taken, so about n_groups / spacing groups
    if n_groups < spacing:
        raise ValueError(f"cap of {cap} rows is smaller than one group in {n_rows} rows")
    positions = rng.random() * spacing + spacing * np.arange(int(np.ceil(n_groups / spacing)) + 1)
    picks = np.unique(np.floor(positions[positions < n_groups]).astype(int))
    selected = order[picks]
    total = int(size[selected].sum())
    if total > cap:  # drop random groups until the cap holds
        shuffled = selected[rng.permutation(len(selected))]
        keep_totals = np.cumsum(size[shuffled])
        selected = shuffled[keep_totals <= cap]
    chosen = np.zeros(n_groups, dtype=bool)
    chosen[selected] = True
    return np.flatnonzero(chosen[inverse])


def fit_cap(spec: ModelSpec, quick: bool) -> int | None:
    """Row cap in force for a fit: the spec's own cap, tightened in quick mode.

    Quick mode caps every model at ``QUICK_SAMPLE_CAP`` rows unless the spec names a smaller
    ``quick_sample_cap`` (the RBF SVC, whose cost is quadratic in rows).
    """
    cap = spec.sample_cap
    if not quick:
        return cap
    quick_cap = QUICK_SAMPLE_CAP if spec.quick_sample_cap is None else spec.quick_sample_cap
    return quick_cap if cap is None else min(cap, quick_cap)


def _strip_prefix(param: str) -> str:
    """``clf__estimator__C`` -> ``C``: the part of a key that names the hyper-parameter."""
    return param.rsplit("__", 1)[-1]


def tidy_cv_results(cv_results: dict[str, Any], model: str) -> pd.DataFrame:
    """One row per candidate: parameters, mean/std train and test scores, per-fold test scores.

    Columns are renamed so the table reads without knowledge of pipeline step names
    (``param_clf__C`` becomes ``param_C``) and sorted by rank on the refit metric.
    """
    frame = pd.DataFrame(cv_results)
    renamed = {
        column: f"param_{_strip_prefix(column[len('param_'):])}"
        for column in frame.columns
        if column.startswith("param_")
    }
    frame = frame.rename(columns=renamed)
    frame["params"] = frame["params"].map(
        lambda p: json.dumps({_strip_prefix(k): v for k, v in p.items()}, default=str)
    )
    frame.insert(0, "model", model)
    rank_column = f"rank_test_{REFIT_METRIC}"
    frame = frame.sort_values(rank_column, kind="stable").reset_index(drop=True)
    time_columns = [c for c in frame.columns if c.endswith("_time")]
    param_columns = [c for c in frame.columns if c.startswith("param_")]
    score_columns = [
        c for c in frame.columns if c not in time_columns + param_columns + ["model", "params"]
    ]
    return frame[["model", *param_columns, "params", *score_columns, *time_columns]]


def fold_scores(result: TuneResult, metric: str = REFIT_METRIC) -> np.ndarray:
    """Per-fold test scores of the best candidate, for paired tests between two models."""
    row = result.cv_results_df.sort_values(f"rank_test_{metric}", kind="stable").iloc[0]
    columns = [c for c in result.cv_results_df.columns if c.startswith("split") and c.endswith(f"_test_{metric}")]
    columns = sorted(columns, key=lambda c: int(c[len("split") : c.index("_")]))
    return row[columns].to_numpy(dtype=float)


def _positional(X: pd.DataFrame | np.ndarray, idx: np.ndarray) -> pd.DataFrame | np.ndarray:
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def out_of_fold_probabilities(
    template: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_idx: np.ndarray,
    cv: BaseCrossValidator,
    n_jobs: int = 1,
) -> tuple[Pipeline, np.ndarray]:
    """Refit ``template`` on the rows ``fit_idx`` and score every row without leakage.

    Rows in ``fit_idx`` get ``cross_val_predict`` probabilities on the grouped folds ``cv``;
    every other row is scored by the pipeline fitted on all of ``fit_idx``, which is
    out-of-sample for them when ``fit_idx`` was drawn by group. Returns the fitted pipeline and
    one probability per row of ``X`` (positional order).
    """
    n_rows = len(y)
    X_fit = _positional(X, fit_idx)
    y_fit, groups_fit = y[fit_idx], groups[fit_idx]
    fitted = clone(template).fit(X_fit, y_fit)
    oof = np.full(n_rows, np.nan)
    oof[fit_idx] = cross_val_predict(
        clone(template), X_fit, y_fit, groups=groups_fit, cv=cv, method="predict_proba", n_jobs=n_jobs,
    )[:, 1]
    rest = np.setdiff1d(np.arange(n_rows), fit_idx, assume_unique=True)
    if len(rest):
        oof[rest] = fitted.predict_proba(_positional(X, rest))[:, 1]
    return fitted, oof


def tune(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    groups_train: np.ndarray | pd.Series,
    cv: BaseCrossValidator,
    quick: bool = False,
    *,
    preprocessor: BaseEstimator | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
) -> TuneResult:
    """Search ``spec.param_grid`` on the training rows and return the refitted best pipeline.

    ``cv`` is the StratifiedGroupKFold from ``mlda.split.cv_splitter`` so that rows of one
    mine-year never sit on both sides of a fold. ``quick=True`` reduces the grid to two
    candidates and caps the rows at 10,000 (tests and the app). ``preprocessor`` defaults to
    ``mlda.preprocessing.build_preprocessor(**spec.preprocess_overrides)``.
    """
    started = time.perf_counter()
    y_all = np.asarray(y_train).astype(int)
    groups_all = np.asarray(groups_train)
    n_rows = len(y_all)
    if len(X_train) != n_rows or len(groups_all) != n_rows:
        raise ValueError("X_train, y_train and groups_train must have the same number of rows")

    cap = fit_cap(spec, quick)
    fit_idx = np.arange(n_rows) if cap is None else stratified_group_subsample(y_all, groups_all, cap)
    X_fit = _positional(X_train, fit_idx)
    y_fit = y_all[fit_idx]
    groups_fit = groups_all[fit_idx]
    if len(fit_idx) < n_rows:
        log.info("%s: fitting on %d of %d training rows (sample cap)", spec.name, len(fit_idx), n_rows)

    pipeline = build_pipeline(spec, preprocessor or default_preprocessor(spec))
    n_jobs = default_n_jobs() if n_jobs is None else n_jobs
    common = dict(
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=verbose,
    )
    if quick:
        search = GridSearchCV(pipeline, quick_grid(spec), **common)
    elif spec.search == "random":
        search = RandomizedSearchCV(
            pipeline,
            spec.param_grid,
            n_iter=spec.n_iter or 10,
            random_state=RANDOM_STATE,
            **common,
        )
    else:
        search = GridSearchCV(pipeline, spec.param_grid, **common)
    search.fit(X_fit, y_fit, groups=groups_fit)

    # Out-of-fold probabilities on the same grouped folds. search.best_estimator_ is already fitted
    # on all capped rows (refit), so only the cross-validated scores and the rows outside the cap
    # are computed here; the rest rows share no group with the capped rows.
    oof = np.full(n_rows, np.nan)
    oof[fit_idx] = cross_val_predict(
        search.best_estimator_, X_fit, y_fit, groups=groups_fit, cv=cv,
        method="predict_proba", n_jobs=n_jobs,
    )[:, 1]
    rest = np.setdiff1d(np.arange(n_rows), fit_idx, assume_unique=True)
    if len(rest):
        oof[rest] = search.best_estimator_.predict_proba(_positional(X_train, rest))[:, 1]

    cv_results_df = tidy_cv_results(search.cv_results_, spec.name)
    best_params = {k: v for k, v in search.best_params_.items()}
    elapsed = time.perf_counter() - started
    log.info(
        "%s: %d candidates, best %s=%.4f, %.0f s",
        spec.name, len(search.cv_results_["params"]),
        REFIT_METRIC, search.best_score_, elapsed,
    )
    return TuneResult(
        best_pipeline=search.best_estimator_,
        best_params=best_params,
        cv_results_df=cv_results_df,
        oof_proba=oof,
        fit_idx=fit_idx,
        best_index=int(search.best_index_),
        n_candidates=len(search.cv_results_["params"]),
        elapsed_seconds=elapsed,
        model=spec.name,
    )


def estimator_params(result: TuneResult) -> dict[str, Any]:
    """Best parameters without the ``clf__`` prefixes, for the manifest and the report."""
    return {_strip_prefix(k): v for k, v in result.best_params.items()}


__all__ = [
    "ESTIMATOR_STEP",
    "METRIC_NAMES",
    "REFIT_METRIC",
    "SCORING",
    "TuneResult",
    "default_n_jobs",
    "estimator_params",
    "fit_cap",
    "fold_scores",
    "out_of_fold_probabilities",
    "stratified_group_subsample",
    "stratified_subsample",
    "tidy_cv_results",
    "tune",
]
