"""Reference rules the learned models must beat, with a scikit-learn style fit/predict API.

* ``MajorityBaseline``: always predicts the majority training class; its probability is the
  training prevalence, so ROC-AUC is 0.5 by construction.
* ``AllPositiveBaseline``: reviews every case (recall 1, specificity 0); its cost per 1,000
  cases is the floor that a model must beat at a given cost ratio.
* ``RuleNotCutOrDust``: the best single-field hand rule from the data profile, flag unless the
  nature of injury is a cut/laceration (180), dust in the eyes (320) or a scratch (310).
* ``LookupNatureBodypartCoal``: lost-time rate of the training cell
  NATURE_INJURY_CD x INJ_BODY_PART_CD x COAL_METAL_IND, falling back to nature x coal, then
  nature, then the global prevalence for cells unseen in training.

Every class exposes ``fit(X, y)``, ``predict_proba(X)`` returning an (n, 2) array and
``predict(X)``. Only the training rows passed to ``fit`` are ever used to estimate rates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

NATURE_COL = "NATURE_INJURY_CD"
BODY_PART_COL = "INJ_BODY_PART_CD"
COAL_METAL_COL = "COAL_METAL_IND"
NOT_LOST_TIME_NATURES = frozenset({"180", "320", "310"})


def _two_column(p1: np.ndarray) -> np.ndarray:
    p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
    return np.column_stack([1.0 - p1, p1])


def _codes(X: pd.DataFrame, column: str) -> pd.Series:
    """Column as stripped strings; missing stays missing so it forms its own lookup level."""
    if column not in X.columns:
        raise KeyError(f"baseline needs column {column!r}")
    values = X[column]
    return values.where(values.isna(), values.astype(str).str.strip()).fillna("missing")


class _BinaryBaseline(ClassifierMixin, BaseEstimator):
    """Shared predict/classes_ plumbing so the baselines work with scikit-learn metrics."""

    classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MajorityBaseline(_BinaryBaseline):
    """Predict the majority training class; probability equals the training prevalence."""

    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> MajorityBaseline:
        y_arr = np.asarray(y).astype(int)
        self.prevalence_ = float(y_arr.mean())
        self.majority_class_ = int(self.prevalence_ >= 0.5)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _two_column(np.full(len(X), self.prevalence_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=int)


class AllPositiveBaseline(_BinaryBaseline):
    """Flag every case for review."""

    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> AllPositiveBaseline:
        self.prevalence_ = float(np.asarray(y).astype(int).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _two_column(np.ones(len(X)))


class RuleNotCutOrDust(_BinaryBaseline):
    """Flag unless NATURE_INJURY_CD is 180 (cut), 320 (dust in eyes) or 310 (scratch).

    A hard rule, so the scores are 0 or 1. ``fit`` only records the training lost-time rate
    inside and outside the flagged group (``rate_flagged_``, ``rate_unflagged_``) for the
    report; it does not change the predictions.
    """

    def __init__(self, excluded_natures: Iterable[str] = NOT_LOST_TIME_NATURES) -> None:
        # Stored as given: scikit-learn's clone() requires __init__ to keep parameters unchanged.
        self.excluded_natures = excluded_natures

    def _flag(self, X: pd.DataFrame) -> np.ndarray:
        excluded = frozenset(str(c) for c in self.excluded_natures)
        return ~_codes(X, NATURE_COL).isin(excluded).to_numpy()

    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> RuleNotCutOrDust:
        y_arr = np.asarray(y).astype(int)
        flag = self._flag(X)
        self.rate_flagged_ = float(y_arr[flag].mean()) if flag.any() else float("nan")
        self.rate_unflagged_ = float(y_arr[~flag].mean()) if (~flag).any() else float("nan")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _two_column(self._flag(X).astype(float))


class LookupNatureBodypartCoal(_BinaryBaseline):
    """Cell lost-time rate of nature x body part x sector with coarser fall-backs.

    ``min_count`` is the smallest training cell accepted before falling back to the next
    coarser table (1 reproduces the reference figures in docs/data_profile.md).
    """

    def __init__(self, min_count: int = 1) -> None:
        self.min_count = min_count

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "nature": _codes(X, NATURE_COL).to_numpy(),
                "body": _codes(X, BODY_PART_COL).to_numpy(),
                "coal": _codes(X, COAL_METAL_COL).to_numpy(),
            }
        )

    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> LookupNatureBodypartCoal:
        keys = self._keys(X)
        keys["y"] = np.asarray(y).astype(int)
        self.prevalence_ = float(keys["y"].mean())
        self.tables_: list[tuple[list[str], pd.Series]] = []
        for level in (["nature", "body", "coal"], ["nature", "coal"], ["nature"]):
            grouped = keys.groupby(level, sort=False)["y"].agg(["mean", "size"])
            rates = grouped.loc[grouped["size"] >= self.min_count, "mean"]
            self.tables_.append((level, rates))
        self.n_cells_ = int(len(self.tables_[0][1]))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        keys = self._keys(X)
        scores = np.full(len(keys), np.nan)
        level_used = np.full(len(keys), "global", dtype=object)
        for level, rates in self.tables_:
            if rates.empty:
                continue
            index = pd.MultiIndex.from_frame(keys[level]) if len(level) > 1 else pd.Index(keys[level[0]])
            looked_up = rates.reindex(index).to_numpy(dtype=float)
            fill = np.isnan(scores) & ~np.isnan(looked_up)
            scores[fill] = looked_up[fill]
            level_used[fill] = "x".join(level)
        scores = np.where(np.isnan(scores), self.prevalence_, scores)
        self.last_level_used_ = level_used
        return _two_column(scores)


@dataclass(frozen=True)
class BaselineSpec:
    """Registry entry: how to build the baseline and what it stands for in the report."""

    name: str
    factory: Callable[[], _BinaryBaseline]
    description: str


BASELINE_ZOO: dict[str, BaselineSpec] = {
    "majority": BaselineSpec(
        "majority", MajorityBaseline,
        "Always predict the majority training class (no lost time); accuracy equals 1 - prevalence.",
    ),
    "all_positive": BaselineSpec(
        "all_positive", AllPositiveBaseline,
        "Review every case; recall 1, specificity 0; its cost per 1,000 cases is the floor to beat.",
    ),
    "rule_not_cut_or_dust": BaselineSpec(
        "rule_not_cut_or_dust", RuleNotCutOrDust,
        "Flag unless the nature of injury is a cut (180), dust in the eyes (320) or a scratch (310).",
    ),
    "lookup_nature_bodypart_coal": BaselineSpec(
        "lookup_nature_bodypart_coal", LookupNatureBodypartCoal,
        "Training lost-time rate of the nature x body part x sector cell, with coarser fall-backs.",
    ),
}


def fit_baselines(X_train: pd.DataFrame, y_train: Iterable[int]) -> dict[str, _BinaryBaseline]:
    """Fit every registered baseline on the training rows only."""
    return {name: spec.factory().fit(X_train, y_train) for name, spec in BASELINE_ZOO.items()}


__all__ = [
    "BASELINE_ZOO",
    "AllPositiveBaseline",
    "BaselineSpec",
    "LookupNatureBodypartCoal",
    "MajorityBaseline",
    "RuleNotCutOrDust",
    "fit_baselines",
]
