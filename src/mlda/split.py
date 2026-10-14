"""Group-aware train/test split, CV splitter and the temporal split.

Rows from one mine-year (``group_id``) must never straddle train and test, because
82 % of rows have a same-mine-year sibling whose outcome the model would otherwise
absorb (docs/data_profile.md section 7). ``StratifiedGroupKFold`` keeps the class
share of every fold within 1 pp of the full set while respecting the groups; fold 0
is the held-out test set and the remaining folds are rebuilt on the training rows
for GridSearchCV. Indices are row positions, saved to JSON so that every notebook
run, the CLI and the app evaluate on the same rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from mlda.config import (
    N_SPLITS,
    PROCESSED_DIR,
    RANDOM_STATE,
    STUDENT_ID,
    TEMPORAL_SPLIT_YEAR,
    TEST_FRACTION,
)

SPLIT_INDICES_PATH = PROCESSED_DIR / "split_indices.json"
PREVALENCE_TOLERANCE = 0.01  # 1 percentage point, as fixed in decisions.json split.tests


@dataclass(frozen=True, eq=False)
class SplitIndices:
    """Row positions of the training and test rows plus the seed that produced them."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    random_state: int = RANDOM_STATE
    n_splits: int = N_SPLITS
    test_fold: int = 0

    @property
    def n_rows(self) -> int:
        return int(len(self.train_idx) + len(self.test_idx))


def _check(condition: bool, message: str) -> None:
    """Raise AssertionError with a readable message (unlike ``assert``, survives -O)."""
    if not condition:
        raise AssertionError(message)


def check_split(
    split: SplitIndices,
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    accident_groups: pd.Series | np.ndarray | None = None,
    prevalence_tolerance: float = PREVALENCE_TOLERANCE,
) -> None:
    """Assert the invariants from the contract: disjoint rows and groups, nested
    accident groups, class share within ``prevalence_tolerance`` of the full set."""
    y_arr = np.asarray(y).astype(int)
    group_arr = np.asarray(groups).astype(str)
    train, test = split.train_idx, split.test_idx
    n = len(y_arr)
    _check(len(group_arr) == n, "y and groups must have the same length")
    _check(len(np.intersect1d(train, test)) == 0, "train and test row positions overlap")
    _check(len(np.union1d(train, test)) == n, "train and test do not cover every row exactly once")
    straddle = np.intersect1d(group_arr[train], group_arr[test])
    _check(len(straddle) == 0, f"{len(straddle)} group_id values appear in both train and test")
    if accident_groups is not None:
        accident_arr = np.asarray(accident_groups).astype(str)
        _check(len(accident_arr) == n, "accident_groups must have the same length as y")
        straddle = np.intersect1d(accident_arr[train], accident_arr[test])
        _check(len(straddle) == 0, f"{len(straddle)} accident_group values straddle train and test")
    prevalence = y_arr.mean()
    for name, idx in (("train", train), ("test", test)):
        gap = abs(y_arr[idx].mean() - prevalence)
        _check(
            gap <= prevalence_tolerance,
            f"{name} prevalence differs from the full set by {gap:.4f} (> {prevalence_tolerance})",
        )


def make_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
    accident_groups: pd.Series | np.ndarray | None = None,
    prevalence_tolerance: float = PREVALENCE_TOLERANCE,
) -> SplitIndices:
    """Hold out one StratifiedGroupKFold fold (fold 0) of the mine-year groups as test.

    ``test_fraction`` must be 1/k for an integer k (0.2 -> 5 folds). The split is
    checked with ``check_split`` before it is returned; pass ``accident_groups`` to
    verify as well that no same-day multi-victim accident straddles the two sets.
    """
    n_splits = round(1 / test_fraction)
    if n_splits < 2 or abs(n_splits * test_fraction - 1) > 1e-9:
        raise ValueError(f"test_fraction must be 1/k for an integer k >= 2, got {test_fraction}")
    y_arr = np.asarray(y).astype(int)
    group_arr = np.asarray(groups).astype(str)
    if len(X) != len(y_arr) or len(group_arr) != len(y_arr):
        raise ValueError("X, y and groups must have the same number of rows")
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    placeholder = np.zeros((len(y_arr), 1))
    train_idx, test_idx = next(iter(folds.split(placeholder, y_arr, group_arr)))
    split = SplitIndices(
        train_idx=np.sort(train_idx).astype(np.int64),
        test_idx=np.sort(test_idx).astype(np.int64),
        random_state=random_state,
        n_splits=n_splits,
        test_fold=0,
    )
    check_split(split, y_arr, group_arr, accident_groups, prevalence_tolerance)
    return split


def save_split_indices(split: SplitIndices, path: Path | None = None) -> Path:
    """Write the row positions and the seed to JSON (default data/processed/split_indices.json)."""
    target = Path(path) if path is not None else SPLIT_INDICES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(split)
    payload["train_idx"] = split.train_idx.tolist()
    payload["test_idx"] = split.test_idx.tolist()
    payload["n_rows"] = split.n_rows
    payload["student_id"] = STUDENT_ID
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def load_split_indices(path: Path | None = None) -> SplitIndices:
    """Read a split saved by ``save_split_indices``."""
    source = Path(path) if path is not None else SPLIT_INDICES_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    return SplitIndices(
        train_idx=np.asarray(payload["train_idx"], dtype=np.int64),
        test_idx=np.asarray(payload["test_idx"], dtype=np.int64),
        random_state=int(payload["random_state"]),
        n_splits=int(payload.get("n_splits", N_SPLITS)),
        test_fold=int(payload.get("test_fold", 0)),
    )


def cv_splitter(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedGroupKFold:
    """Validation scheme for GridSearchCV on the training rows: same grouping as the split."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def temporal_split(
    meta: pd.DataFrame, split_year: int = TEMPORAL_SPLIT_YEAR
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions for the secondary evaluation: train CAL_YR < split_year, test >= split_year."""
    years = pd.to_numeric(meta["CAL_YR"], errors="coerce").to_numpy(dtype="float64")
    if np.isnan(years).any():
        raise ValueError("meta['CAL_YR'] contains missing values")
    train_idx = np.flatnonzero(years < split_year).astype(np.int64)
    test_idx = np.flatnonzero(years >= split_year).astype(np.int64)
    return train_idx, test_idx
