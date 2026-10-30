"""Sensitivity runs: refit the selected pipeline on altered inputs or a different split.

The primary result (grouped split by mine-year, all features) rests on choices that a reader
can fairly question. Each function here changes one of them, re-estimates the thresholds from
out-of-fold probabilities of the new training rows, scores the new held-out rows once and
returns numbers that sit next to the primary ones in ``reports/metrics/sensitivity.json``:

* ``pure_levels`` finds categorical levels that are 100 % one class (a deterministic rule hidden
  in a feature) and ``excluding_levels`` re-scores the held-out rows without them;
* ``feature_variants`` builds the altered inputs: without ``immed_reportable`` (a near copy of
  the death code), without the dated 103(i) status, with lagged case counts restricted to cases
  already resolved at the accident, with previous-year exposure blanked for forms filed
  before the fourth-quarter employment report was due, and with the case history blanked on
  the training rows whose previous mine-year is a test mine-year (``history_without_test_outcomes``);
* ``refit_variants`` refits the selected pipeline on each variant;
* ``mine_grouped_robustness`` groups the split and the cross-validation by ``MINE_ID`` instead of
  mine-year, so that later years of a test mine cannot reach the model through its training rows.

Nothing here is fitted on held-out rows: every refit uses the training rows of its own split.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from mlda.baselines import BASELINE_ZOO, fit_baselines
from mlda.config import RANDOM_STATE
from mlda.data.join import type_from_subunit_code
from mlda.features import FEATURE_SPEC, engineer_features
from mlda.models import ModelSpec
from mlda.operating import (
    FLAG_KEY,
    LOOKUP_BASELINE,
    OPERATING_KEY,
    all_positive_floor,
    delta,
    evaluate_on_test,
    headline,
    headline_from_proba,
    model_criteria,
    oof_thresholds,
)
from mlda.split import PREVALENCE_TOLERANCE, cv_splitter, make_split
from mlda.tuning import fit_cap, out_of_fold_probabilities, stratified_group_subsample

log = logging.getLogger(__name__)

PURE_LEVEL_MIN_ROWS = 100
FALLBACK_TOLERANCE = (
    0.05  # prevalence check of the mine-grouped split when the 1 pp check cannot be met
)
# Levels that are 100 % one class by definition and known at filing, documented in
# decisions.json known_deterministic_levels: amputation (923 of 925 rows are permanent disability, code 02).
DOCUMENTED_PURE_LEVELS: frozenset[tuple[str, str]] = frozenset({("NATURE_INJURY_CD", "100")})

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "without_immed_reportable": "immed_reportable set to 0 (the flag is a near copy of the death code)",
    "without_status_103i": "status_103i_at_accident set to one constant level",
    "resolved_lag_counts": "previous-year case counts restricted to cases whose return-to-work date and "
    "closing document are earlier than the accident",
    "exposure_blank_if_q4_incomplete": "previous-year hours and employees set to missing (and the mine type "
    "taken from the accident subunit) for forms filed on or before 15 January",
    "history_without_test_outcomes": "previous-year case counts, incidence rates and lost-time share set to "
    "missing (the encoding of a mine-year without history; exposure kept) on the training rows whose "
    "previous mine-year has a test row",
}
HISTORY_VARIANT = "history_without_test_outcomes"
# The lagged counts of the base table that the case-history features are computed from. Blanking
# them and re-running engineer_features gives exactly the encoding of a row without history:
# prev_incidents and both rates NaN, prev_lost_time_share NaN and prev_share_missing 1. The
# exposure columns (prev_hours, prev_employees and what is derived from them) and
# prev_year_matched come from the production file, which holds no test outcome, so they stay.
HISTORY_CASE_COUNTS: tuple[str, ...] = ("prev_incidents", "prev_valid_cases", "prev_lost_time_cases")
HISTORY_FEATURES: tuple[str, ...] = (
    "prev_incidents",
    "prev_incident_rate_200k",
    "prev_lost_time_rate_200k",
    "prev_lost_time_share",
    "prev_share_missing",
)


# ---------------------------------------------------------------------------
# Deterministic levels
# ---------------------------------------------------------------------------


def pure_levels(
    X: pd.DataFrame, y: np.ndarray, min_rows: int = PURE_LEVEL_MIN_ROWS
) -> pd.DataFrame:
    """Levels of categorical and binary features with at least ``min_rows`` rows and one class only.

    A level that is 100 % lost-time (or 100 % not) lets a model read the label off the feature.
    Such a level can be legitimate, when it is definitional and known at filing, but it has to be
    listed and documented; ``documented`` marks the ones in ``DOCUMENTED_PURE_LEVELS``.
    """
    labels = np.asarray(y).astype(int)
    rows: list[dict[str, Any]] = []
    for column in (*FEATURE_SPEC["categorical"], *FEATURE_SPEC["binary"]):
        grouped = pd.DataFrame({"level": X[column].astype(str).to_numpy(), "y": labels}).groupby(
            "level"
        )["y"]
        summary = grouped.agg(n="size", positives="sum")
        pure = summary[
            (summary["n"] >= min_rows)
            & ((summary["positives"] == 0) | (summary["positives"] == summary["n"]))
        ]
        for level, row in pure.iterrows():
            rows.append(
                {
                    "column": column,
                    "level": str(level),
                    "n": int(row["n"]),
                    "positives": int(row["positives"]),
                    "positive_rate": float(row["positives"] / row["n"]),
                    "documented": (column, str(level)) in DOCUMENTED_PURE_LEVELS,
                }
            )
    return pd.DataFrame(
        rows, columns=["column", "level", "n", "positives", "positive_rate", "documented"]
    )


def excluding_levels(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    probas: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    levels: pd.DataFrame,
) -> dict[str, Any]:
    """Held-out headline metrics with and without the rows of the given pure levels.

    The thresholds are the ones fixed on out-of-fold training probabilities; they are applied
    unchanged to the remaining rows, so the figures show how much of each model's result the
    deterministic levels carry.
    """
    drop = np.zeros(len(X_test), dtype=bool)
    for column, level in zip(levels["column"], levels["level"], strict=True):
        drop |= X_test[column].astype(str).to_numpy() == level
    keep = ~drop
    y = np.asarray(y_test).astype(int)
    out: dict[str, Any] = {
        "levels": [f"{c}={v}" for c, v in zip(levels["column"], levels["level"], strict=True)],
        "n_test": int(len(y)),
        "n_test_excluded": int(drop.sum()),
        "prevalence_test": float(y.mean()),
        "prevalence_test_remaining": float(y[keep].mean()) if keep.any() else float("nan"),
        "models": {},
    }
    for name, proba in probas.items():
        with_rows = headline_from_proba(y, proba, thresholds[name])
        without_rows = (
            headline_from_proba(y[keep], proba[keep], thresholds[name]) if keep.any() else {}
        )
        out["models"][name] = {
            "with_levels": with_rows,
            "without_levels": without_rows,
            "delta": delta(without_rows, with_rows),
        }
    return out


# ---------------------------------------------------------------------------
# Altered inputs
# ---------------------------------------------------------------------------


def simple_variants(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two variants that only blank a column of the model inputs.

    ``immed_reportable`` becomes the integer 0 (a binary column must stay numeric), and the
    103(i) status becomes one constant level, which one-hot encodes to a single constant column.
    """
    return {
        "without_immed_reportable": X.assign(immed_reportable=np.int8(0)),
        "without_status_103i": X.assign(status_103i_at_accident="missing"),
    }


def rows_with_test_history(
    mine_ids: np.ndarray | pd.Series,
    years: np.ndarray | pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Training rows whose previous mine-year (``MINE_ID``, ``CAL_YR`` - 1) has a row in the test split.

    The lagged case counts of a row in year Y are counted from the mine's year Y - 1 cases in the
    full Accidents file. The split keeps each mine-year on one side but not the year before it,
    so a training row can carry counts, and hence a lost-time share, that include the outcomes of
    test rows. Returns a boolean mask over all rows; test rows are never marked.
    """
    mine = np.asarray(mine_ids).astype(str)
    year = pd.Series(years).astype("int64").to_numpy()
    test_mine_years = pd.MultiIndex.from_arrays([mine[test_idx], year[test_idx]])
    previous = pd.MultiIndex.from_arrays([mine, year - 1])
    is_train = np.zeros(len(mine), dtype=bool)
    is_train[np.asarray(train_idx)] = True
    return np.asarray(previous.isin(test_mine_years)) & is_train


def history_without_test_outcomes(
    clean: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> pd.DataFrame:
    """Model inputs with the case history blanked on the rows of ``rows_with_test_history``.

    The three lagged counts are set to NaN in the base table and the features are rebuilt, so the
    blanked rows get the pipeline's own encoding of a mine without a previous-year record (counts
    and rates NaN, ``prev_share_missing`` 1). Exposure and every test row are left as they are.
    """
    blank = rows_with_test_history(clean["MINE_ID"], clean["CAL_YR"], train_idx, test_idx)
    counts = {column: clean[column].mask(blank) for column in HISTORY_CASE_COUNTS}
    log.info(
        "history variant: %d of %d training rows follow a test mine-year",
        int(blank.sum()), len(train_idx),
    )
    return engineer_features(clean.assign(**counts))


def feature_variants(
    clean: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Alternative model-input frames with the same index and columns as ``X``.

    Variants that need columns the cleaned base table lacks (a base cached by an older version)
    are skipped with a log message rather than raising. ``train_idx`` and ``test_idx`` are the
    positions of the held-out split; the history variant depends on the split and is built only
    when both are given.
    """
    variants = simple_variants(X)
    resolved = [
        f"{c}_resolved" for c in ("prev_incidents", "prev_valid_cases", "prev_lost_time_cases")
    ]
    if all(c in clean.columns for c in resolved):
        counts = {
            c: clean[f"{c}_resolved"]
            for c in ("prev_incidents", "prev_valid_cases", "prev_lost_time_cases")
        }
        variants["resolved_lag_counts"] = engineer_features(clean.assign(**counts))
    else:
        log.info("base table has no *_resolved columns; skipping the resolved-before variant")
    if "prev_year_incomplete_at_filing" in clean.columns:
        flagged = clean["prev_year_incomplete_at_filing"].to_numpy(dtype=bool)
        fallback = clean["SUBUNIT_CD"].map(type_from_subunit_code)
        altered = clean.assign(
            prev_hours=clean["prev_hours"].mask(flagged),
            prev_employees=clean["prev_employees"].mask(flagged),
            type_at_time=clean["type_at_time"].mask(flagged, fallback),
        )
        variants["exposure_blank_if_q4_incomplete"] = engineer_features(altered)
    else:
        log.info("base table has no prev_year_incomplete_at_filing; skipping the exposure variant")
    if train_idx is not None and test_idx is not None:
        variants[HISTORY_VARIANT] = history_without_test_outcomes(clean, train_idx, test_idx)
    else:
        log.info("no split given; skipping the history-without-test-outcomes variant")
    return variants


def _changed_rows(variant: pd.DataFrame, X: pd.DataFrame) -> int:
    """Rows on which at least one input of the variant differs from the primary frame."""
    differs = np.zeros(len(X), dtype=bool)
    for column in X.columns:
        left, right = variant[column].to_numpy(), X[column].to_numpy()
        if X[column].dtype == object:
            differs |= left != right
        else:
            differs |= ~((left == right) | (pd.isna(left) & pd.isna(right)))
    return int(differs.sum())


def refit_variants(
    template: Pipeline,
    fit_idx: np.ndarray,
    variants: Mapping[str, pd.DataFrame],
    primary: pd.DataFrame,
    reference: Mapping[str, Any],
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sector: np.ndarray,
    cv: Any,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Refit the selected pipeline on each variant and score the held-out rows once.

    ``fit_idx`` are positions within the training rows (the capped subset of the primary fit),
    ``reference`` is the ``headline`` of the primary result. The hyper-parameters stay those of
    the primary search; the thresholds are re-estimated from out-of-fold probabilities of the
    variant, so each variant is judged at its own operating point. ``rows_changed`` counts every
    row whose inputs differ from ``primary``; ``train_rows_changed`` and its share count the
    training rows among them, the rows on which the refit itself learns from altered inputs.
    """
    y_train, g_train, y_test = y[train_idx], groups[train_idx], y[test_idx]
    out: dict[str, Any] = {
        "reference": dict(reference),
        "hyperparameters": "those selected by the primary search on the primary training rows; not re-tuned",
        "variants": {},
    }
    for name, X_variant in variants.items():
        fitted, oof = out_of_fold_probabilities(
            template, X_variant.iloc[train_idx], y_train, g_train, fit_idx, cv, n_jobs
        )
        thresholds = oof_thresholds(y_train, oof)
        proba = fitted.predict_proba(X_variant.iloc[test_idx])[:, 1]
        evaluation = evaluate_on_test(y_test, proba, sector[test_idx], thresholds)
        view = headline(evaluation)
        train_changed = _changed_rows(X_variant.iloc[train_idx], primary.iloc[train_idx])
        out["variants"][name] = {
            "description": VARIANT_DESCRIPTIONS.get(name, name),
            "rows_changed": _changed_rows(X_variant, primary),
            "train_rows_changed": train_changed,
            "train_rows_changed_share": train_changed / len(train_idx),
            "thresholds": {
                OPERATING_KEY: thresholds[OPERATING_KEY],
                FLAG_KEY: thresholds[FLAG_KEY],
            },
            "test": view,
            "delta_vs_reference": delta(view, reference),
        }
        log.info(
            "sensitivity %-34s AUC %.4f PR-AUC %.4f recall@spec0.40 %.4f",
            name,
            view["roc_auc"],
            view["pr_auc"],
            view[f"recall_at_{OPERATING_KEY}"],
        )
    return out


# ---------------------------------------------------------------------------
# Split grouped by mine
# ---------------------------------------------------------------------------


def mine_grouped_robustness(
    spec: ModelSpec,
    template: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    mine_ids: np.ndarray,
    accident_groups: np.ndarray,
    sector: np.ndarray,
    reference: Mapping[str, Any],
    *,
    cv_folds: int,
    quick: bool,
    n_jobs: int = 1,
    prevalence_tolerance: float = PREVALENCE_TOLERANCE,
) -> dict[str, Any]:
    """Repeat the split, the cross-validation and the thresholds with ``MINE_ID`` as the group.

    The primary split keeps each mine-year together but lets a test mine's other years sit in
    the training rows, and latitude and longitude identify the mine. Grouping by mine removes
    that route: no mine appears on both sides. The selected pipeline (hyper-parameters fixed)
    and the lookup baseline are refitted on the new training rows, the thresholds come from
    out-of-fold probabilities grouped by mine, and S1-S6 are evaluated as in the primary study.
    """
    groups = np.asarray(mine_ids).astype(str)
    tolerance = prevalence_tolerance
    try:
        split = make_split(
            X, y, groups, accident_groups=accident_groups, prevalence_tolerance=tolerance
        )
    except (
        AssertionError
    ) as error:  # few or very unequal mines: report the looser check instead of aborting
        log.warning(
            "mine-grouped split misses the %.2f prevalence check (%s); retrying at %.2f",
            tolerance,
            error,
            FALLBACK_TOLERANCE,
        )
        tolerance = max(tolerance, FALLBACK_TOLERANCE)
        split = make_split(
            X, y, groups, accident_groups=accident_groups, prevalence_tolerance=tolerance
        )
    train_idx, test_idx = split.train_idx, split.test_idx
    cv = cv_splitter(n_splits=cv_folds)
    y_train, g_train, y_test = y[train_idx], groups[train_idx], y[test_idx]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    cap = fit_cap(spec, quick)
    fit_idx = (
        np.arange(len(train_idx))
        if cap is None
        else stratified_group_subsample(y_train, g_train, cap)
    )
    fitted, oof = out_of_fold_probabilities(
        template, X_train, y_train, g_train, fit_idx, cv, n_jobs
    )
    thresholds = oof_thresholds(y_train, oof)
    proba = fitted.predict_proba(X_test)[:, 1]
    evaluation = evaluate_on_test(y_test, proba, sector[test_idx], thresholds)

    lookup = fit_baselines(X_train, y_train)[LOOKUP_BASELINE]
    lookup_oof = cross_val_predict(
        BASELINE_ZOO[LOOKUP_BASELINE].factory(),
        X_train,
        y_train,
        groups=g_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    lookup_thresholds = oof_thresholds(y_train, lookup_oof)
    lookup_evaluation = evaluate_on_test(
        y_test, lookup.predict_proba(X_test)[:, 1], sector[test_idx], lookup_thresholds
    )

    floor = all_positive_floor(y_test)
    lookup_cost = {k: v["cost_per_1000"] for k, v in lookup_evaluation["cost_optimal"].items()}
    view = headline(evaluation)
    return {
        "group": "MINE_ID",
        "n_groups": int(len(np.unique(groups))),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_fit_rows": int(len(fit_idx)),
        "shared_mines": int(len(np.intersect1d(groups[train_idx], groups[test_idx]))),
        "prevalence_tolerance_used": tolerance,
        "hyperparameters": (
            "selected by the primary search, whose training rows include rows that are test rows of this "
            "split; only that grid choice, not any fitted weight, imputation or threshold, has seen their labels"
        ),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "model": {
            "thresholds": {
                OPERATING_KEY: thresholds[OPERATING_KEY],
                FLAG_KEY: thresholds[FLAG_KEY],
            },
            "test": view,
            "delta_vs_primary": delta(view, reference),
            "criteria": model_criteria(evaluation, thresholds, floor, lookup_cost),
        },
        "lookup": {
            "thresholds": {
                OPERATING_KEY: lookup_thresholds[OPERATING_KEY],
                FLAG_KEY: lookup_thresholds[FLAG_KEY],
            },
            "test": headline(lookup_evaluation),
            "criteria": model_criteria(
                lookup_evaluation, lookup_thresholds, floor, lookup_cost, is_lookup=True
            ),
        },
        "random_state": RANDOM_STATE,
    }


__all__ = [
    "DOCUMENTED_PURE_LEVELS",
    "HISTORY_CASE_COUNTS",
    "HISTORY_FEATURES",
    "HISTORY_VARIANT",
    "PURE_LEVEL_MIN_ROWS",
    "VARIANT_DESCRIPTIONS",
    "excluding_levels",
    "feature_variants",
    "history_without_test_outcomes",
    "mine_grouped_robustness",
    "pure_levels",
    "refit_variants",
    "rows_with_test_history",
    "simple_variants",
]
