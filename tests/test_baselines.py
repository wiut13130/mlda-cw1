"""Reference rules: valid probabilities, correct rule logic and lookup fall-backs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlda.baselines import (
    BASELINE_ZOO,
    AllPositiveBaseline,
    LookupNatureBodypartCoal,
    MajorityBaseline,
    RuleNotCutOrDust,
    fit_baselines,
)
from mlda.config import RANDOM_STATE


def make_frame(n: int = 400, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, np.ndarray]:
    """Nature/body-part/sector codes whose cell rates differ, so the lookup has signal."""
    rng = np.random.default_rng(seed)
    nature = rng.choice(["330", "180", "220", "320", "160", "310"], n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    body = rng.choice(["340", "330", "700", "420"], n)
    coal = rng.choice(["C", "M"], n, p=[0.35, 0.65])
    base_rate = np.where(np.isin(nature, ["180", "320", "310"]), 0.15, 0.6)
    base_rate = base_rate + np.where(coal == "C", 0.15, 0.0)
    y = (rng.random(n) < base_rate).astype(int)
    frame = pd.DataFrame({"NATURE_INJURY_CD": nature, "INJ_BODY_PART_CD": body, "COAL_METAL_IND": coal})
    frame.loc[frame.index[:3], "INJ_BODY_PART_CD"] = np.nan  # missing becomes its own level
    return frame, y


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    return make_frame()


def _valid(proba: np.ndarray, n: int) -> None:
    assert proba.shape == (n, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", sorted(BASELINE_ZOO))
def test_every_baseline_produces_valid_probabilities(name: str, data) -> None:
    X, y = data
    model = BASELINE_ZOO[name].factory().fit(X.iloc[:300], y[:300])
    proba = model.predict_proba(X.iloc[300:])
    _valid(proba, 100)
    pred = model.predict(X.iloc[300:])
    assert pred.shape == (100,) and set(np.unique(pred)) <= {0, 1}
    assert len(BASELINE_ZOO[name].description) > 20


def test_fit_baselines_returns_every_registered_rule(data) -> None:
    X, y = data
    fitted = fit_baselines(X, y)
    assert set(fitted) == set(BASELINE_ZOO)


def test_majority_is_constant_at_prevalence() -> None:
    X = pd.DataFrame({"NATURE_INJURY_CD": ["330"] * 10})
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    model = MajorityBaseline().fit(X, y)
    assert model.majority_class_ == 0
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], 0.3)
    assert (model.predict(X) == 0).all()
    flipped = MajorityBaseline().fit(X, 1 - y)
    assert flipped.majority_class_ == 1 and (flipped.predict(X) == 1).all()


def test_all_positive_flags_everything(data) -> None:
    X, y = data
    model = AllPositiveBaseline().fit(X, y)
    assert (model.predict(X) == 1).all()
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], 1.0)


def test_rule_not_cut_or_dust_logic() -> None:
    X = pd.DataFrame({"NATURE_INJURY_CD": ["330", "180", "320", "310", " 180 ", np.nan, "220"]})
    y = np.array([1, 0, 0, 0, 0, 1, 1])
    model = RuleNotCutOrDust().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), [1, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], [1, 0, 0, 0, 0, 1, 1])
    assert model.rate_flagged_ == pytest.approx(1.0)
    assert model.rate_unflagged_ == pytest.approx(0.0)
    with pytest.raises(KeyError):
        RuleNotCutOrDust().predict_proba(pd.DataFrame({"OTHER": [1]}))


def test_lookup_cell_rates_and_fallbacks() -> None:
    train = pd.DataFrame(
        {
            "NATURE_INJURY_CD": ["330", "330", "330", "330", "180", "180", "220", "220"],
            "INJ_BODY_PART_CD": ["340", "340", "700", "700", "340", "340", "340", "700"],
            "COAL_METAL_IND": ["C", "C", "M", "M", "M", "M", "C", "M"],
        }
    )
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    model = LookupNatureBodypartCoal().fit(train, y)
    assert model.prevalence_ == pytest.approx(0.5)
    assert model.n_cells_ == 5
    test = pd.DataFrame(
        {
            "NATURE_INJURY_CD": ["330", "330", "330", "220", "999"],
            "INJ_BODY_PART_CD": ["340", "700", "999", "999", "340"],
            "COAL_METAL_IND": ["C", "M", "C", "M", "C"],
        }
    )
    proba = model.predict_proba(test)[:, 1]
    # exact cells: (330,340,C) = 1.0 ; (330,700,M) = 0.5
    assert proba[0] == pytest.approx(1.0) and proba[1] == pytest.approx(0.5)
    # unseen body part -> nature x coal (330, C) = 1.0
    assert proba[2] == pytest.approx(1.0)
    # unseen body part -> nature x coal (220, M) = 0.0
    assert proba[3] == pytest.approx(0.0)
    # unseen nature -> global prevalence
    assert proba[4] == pytest.approx(0.5)
    assert list(model.last_level_used_) == ["naturexbodyxcoal", "naturexbodyxcoal", "naturexcoal", "naturexcoal", "global"]


def test_lookup_min_count_falls_back_sooner() -> None:
    train = pd.DataFrame(
        {
            "NATURE_INJURY_CD": ["330", "330", "330"],
            "INJ_BODY_PART_CD": ["340", "700", "700"],
            "COAL_METAL_IND": ["C", "C", "C"],
        }
    )
    y = np.array([1, 0, 0])
    strict = LookupNatureBodypartCoal(min_count=2).fit(train, y)
    proba = strict.predict_proba(train.iloc[[0]])[:, 1]
    # the (330, 340, C) cell has one row, so the nature x coal rate 1/3 is used instead
    assert proba[0] == pytest.approx(1 / 3)


def test_lookup_has_ranking_signal_on_held_out_rows(data) -> None:
    from sklearn.metrics import roc_auc_score

    X, y = data
    model = LookupNatureBodypartCoal().fit(X.iloc[:300], y[:300])
    auc = roc_auc_score(y[300:], model.predict_proba(X.iloc[300:])[:, 1])
    assert auc > 0.6
