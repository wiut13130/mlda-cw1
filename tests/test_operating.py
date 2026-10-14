"""Operating thresholds of constant scorers keep the decision that defines the baseline."""

from __future__ import annotations

import numpy as np

from mlda.baselines import BASELINE_ZOO
from mlda.operating import FLAG_KEY, OPERATING_KEY, evaluate_on_test, oof_thresholds


def _folds_prevalence_scores(y: np.ndarray) -> np.ndarray:
    """Out-of-fold scores of the majority rule: each fold scores its training prevalence."""
    folds = np.arange(y.size) % 5
    return np.array([y[folds != f].mean() for f in folds])


def test_constant_scorers_keep_their_decision_at_every_operating_point() -> None:
    rng = np.random.default_rng(0)
    y_train = (rng.random(2_000) < 0.486).astype(int)
    y_test = (rng.random(500) < 0.486).astype(int)
    sector = np.where(rng.random(500) < 0.4, "C", "M")
    cases = {
        "majority": (_folds_prevalence_scores(y_train), np.full(500, y_train.mean()), 0.0),
        "all_positive": (np.ones(2_000), np.ones(500), 1.0),
    }
    for name, (oof, test_proba, expected_flag_rate) in cases.items():
        assert BASELINE_ZOO[name].constant
        thresholds = oof_thresholds(y_train, oof, fixed_threshold=0.5)
        assert thresholds[OPERATING_KEY] == thresholds[FLAG_KEY] == 0.5
        evaluation = evaluate_on_test(y_test, test_proba, sector, thresholds)
        for key in ("at_0.5", f"at_{OPERATING_KEY}", f"at_{FLAG_KEY}"):
            assert evaluation[key]["flag_rate"] == expected_flag_rate, (name, key)


def test_only_constant_baselines_use_a_fixed_threshold() -> None:
    assert {name for name, spec in BASELINE_ZOO.items() if spec.constant} == {"majority", "all_positive"}
