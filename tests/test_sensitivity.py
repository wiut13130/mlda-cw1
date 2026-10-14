"""Sensitivity runs, the operating-point helpers and the temporal study on synthetic data.

Everything here uses the 2,500-row synthetic base table of ``conftest``, so no raw extract is
needed. The pipelines are real (``build_preprocessor`` plus a model from the zoo); only the data
are synthetic, which is enough to check the protocol: no held-out label reaches a threshold, a
grid choice or a model ranking, and each sensitivity variant changes exactly the inputs it names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_base
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from mlda.cleaning import apply_error_rules
from mlda.config import RANDOM_STATE
from mlda.features import FeatureFrame, feature_frame
from mlda.models import MODEL_ZOO, build_pipeline
from mlda.operating import (
    FLAG_KEY,
    OPERATING_KEY,
    all_positive_floor,
    delta,
    evaluate_on_test,
    headline,
    model_criteria,
    oof_thresholds,
)
from mlda.pipeline import ArtefactDirs, StudyData, build_study_data, temporal_evaluation, train_all
from mlda.preprocessing import build_preprocessor
from mlda.sensitivity import (
    feature_variants,
    mine_grouped_robustness,
    refit_variants,
    simple_variants,
)
from mlda.split import cv_splitter, temporal_split

N_ROWS = 2500
CV_FOLDS = 3


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return apply_error_rules(synthetic_base(N_ROWS))[0]


@pytest.fixture(scope="module")
def parts(clean: pd.DataFrame) -> FeatureFrame:
    return feature_frame(clean)


def template(name: str = "logreg"):
    spec = MODEL_ZOO[name]
    return build_pipeline(spec, build_preprocessor(min_frequency=0.02, **spec.preprocess_overrides))


def study(clean: pd.DataFrame, tmp_path: Path) -> StudyData:
    dirs = ArtefactDirs(*(tmp_path / n for n in ("processed", "models", "metrics", "cv", "predictions", "figures")))
    dirs.create()
    return build_study_data(clean, dirs)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def test_simple_variants_blank_exactly_one_input(parts: FeatureFrame) -> None:
    variants = simple_variants(parts.X)
    immed = variants["without_immed_reportable"]
    assert immed["immed_reportable"].dtype == np.int8 and immed["immed_reportable"].eq(0).all()
    status = variants["without_status_103i"]
    assert status["status_103i_at_accident"].eq("missing").all()
    for name, frame in variants.items():
        blanked = "immed_reportable" if "immed" in name else "status_103i_at_accident"
        pd.testing.assert_frame_equal(frame.drop(columns=blanked), parts.X.drop(columns=blanked))


def test_feature_variants_rebuild_lag_features_from_resolved_counts(clean: pd.DataFrame, parts: FeatureFrame) -> None:
    variants = feature_variants(clean, parts.X)
    assert set(variants) == {
        "without_immed_reportable", "without_status_103i", "resolved_lag_counts", "exposure_blank_if_q4_incomplete",
    }
    resolved = variants["resolved_lag_counts"]
    assert list(resolved.columns) == list(parts.X.columns)
    unchanged = [c for c in parts.X.columns if not c.startswith("prev_")]
    pd.testing.assert_frame_equal(resolved[unchanged], parts.X[unchanged])
    assert (resolved["prev_incidents"].dropna() <= parts.X["prev_incidents"].dropna()).all()
    pd.testing.assert_series_equal(resolved["prev_incidents"], clean["prev_incidents_resolved"].astype("float64"), check_names=False)


def test_exposure_variant_blanks_early_january_rows_only(clean: pd.DataFrame, parts: FeatureFrame) -> None:
    flagged = clean["prev_year_incomplete_at_filing"].to_numpy(dtype=bool)
    assert flagged.any(), "the synthetic base should contain forms filed by 15 January"
    variant = feature_variants(clean, parts.X)["exposure_blank_if_q4_incomplete"]
    assert variant.loc[flagged, "prev_hours"].isna().all() and variant.loc[flagged, "prev_employees"].isna().all()
    assert variant.loc[flagged, "mine_size_band"].eq("missing").all()
    pd.testing.assert_frame_equal(variant.loc[~flagged], parts.X.loc[~flagged])
    fallback = variant.loc[flagged, "type_at_time"]
    assert set(fallback) <= {"Underground", "Surface", "Facility"}


def test_variants_that_need_missing_base_columns_are_skipped(clean: pd.DataFrame, parts: FeatureFrame) -> None:
    stale = clean.drop(columns=["prev_incidents_resolved", "prev_year_incomplete_at_filing"])
    assert set(feature_variants(stale, parts.X)) == {"without_immed_reportable", "without_status_103i"}


# ---------------------------------------------------------------------------
# Refits
# ---------------------------------------------------------------------------


def test_refit_variants_use_the_training_rows_only_and_report_deltas(clean: pd.DataFrame, parts: FeatureFrame, tmp_path: Path) -> None:
    data = study(clean, tmp_path)
    variants = feature_variants(clean, data.X)
    train_idx, test_idx = data.train_idx, data.test_idx
    pipeline = template()
    fit_idx = np.arange(len(train_idx))
    reference = headline(
        evaluate_on_test(
            data.y[test_idx],
            clone(pipeline).fit(data.X.iloc[train_idx], data.y[train_idx]).predict_proba(data.X.iloc[test_idx])[:, 1],
            data.sector[test_idx],
            oof_thresholds(
                data.y[train_idx],
                cross_val_predict(
                    clone(pipeline), data.X.iloc[train_idx], data.y[train_idx], groups=data.groups[train_idx],
                    cv=cv_splitter(CV_FOLDS), method="predict_proba",
                )[:, 1],
            ),
        )
    )
    out = refit_variants(
        pipeline, fit_idx, variants, data.X, reference, data.y, data.groups, train_idx, test_idx, data.sector,
        cv_splitter(CV_FOLDS),
    )
    assert out["reference"] == reference and set(out["variants"]) == set(variants)
    for name, entry in out["variants"].items():
        assert 0 < entry["thresholds"][OPERATING_KEY] < 1 and 0 < entry["thresholds"][FLAG_KEY] < 1, name
        assert entry["delta_vs_reference"]["roc_auc"] == pytest.approx(entry["test"]["roc_auc"] - reference["roc_auc"])
        assert entry["description"]
    assert out["variants"]["without_immed_reportable"]["rows_changed"] == int(data.X["immed_reportable"].sum())
    assert out["variants"]["without_status_103i"]["rows_changed"] == len(data.X)


def test_delta_compares_matching_numeric_entries_only() -> None:
    assert delta({"a": 0.5, "b": 1.0, "c": "text", "d": float("nan")}, {"a": 0.25, "b": 2.0, "c": "x", "d": 1.0, "e": 3.0}) == {
        "a": 0.25, "b": -1.0,
    }


def test_mine_grouped_robustness_separates_mines_and_reports_the_criteria(clean: pd.DataFrame, parts: FeatureFrame, tmp_path: Path) -> None:
    data = study(clean, tmp_path)
    spec = MODEL_ZOO["logreg"]
    out = mine_grouped_robustness(
        spec, template(), data.X, data.y, data.meta["MINE_ID"].to_numpy(), data.meta["accident_group"].to_numpy(),
        data.sector, {"roc_auc": 0.8}, cv_folds=CV_FOLDS, quick=True, prevalence_tolerance=0.1,
    )
    assert out["group"] == "MINE_ID" and out["shared_mines"] == 0
    assert out["n_train"] + out["n_test"] == len(data.y) and out["n_groups"] == data.meta["MINE_ID"].nunique()
    assert {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "all_pass"} <= set(out["model"]["criteria"])
    assert out["lookup"]["criteria"]["S4"]["pass"] is None  # the baseline is not judged against itself
    assert 0.5 <= out["model"]["test"]["roc_auc"] <= 1.0
    assert out["model"]["delta_vs_primary"]["roc_auc"] == pytest.approx(out["model"]["test"]["roc_auc"] - 0.8)


def test_mine_grouped_split_falls_back_to_a_looser_prevalence_check_and_says_so(clean: pd.DataFrame, tmp_path: Path) -> None:
    """Forty mines cannot be split within 1 pp of prevalence; the run must not abort, and must record it."""
    data = study(clean, tmp_path)
    out = mine_grouped_robustness(
        MODEL_ZOO["logreg"], template(), data.X, data.y, data.meta["MINE_ID"].to_numpy(),
        data.meta["accident_group"].to_numpy(), data.sector, {"roc_auc": 0.8}, cv_folds=CV_FOLDS, quick=True,
    )
    assert out["prevalence_tolerance_used"] == 0.05
    strict = mine_grouped_robustness(
        MODEL_ZOO["logreg"], template(), data.X, data.y, data.meta["MINE_ID"].to_numpy(),
        data.meta["accident_group"].to_numpy(), data.sector, {"roc_auc": 0.8}, cv_folds=CV_FOLDS, quick=True,
        prevalence_tolerance=0.1,
    )
    assert strict["prevalence_tolerance_used"] == 0.1


def test_model_criteria_follow_the_documented_thresholds() -> None:
    y = np.array([1] * 60 + [0] * 40)
    rng = np.random.default_rng(0)
    good = np.where(y == 1, rng.uniform(0.6, 1.0, y.size), rng.uniform(0.0, 0.5, y.size))  # separable scores
    thresholds = oof_thresholds(y, good)
    evaluation = evaluate_on_test(y, good, np.array(["C"] * 100), thresholds)
    floor = all_positive_floor(y)
    lookup_cost = {k: 0.5 * v for k, v in floor.items()}  # a baseline that beats the floor but not this model
    checks = model_criteria(evaluation, thresholds, floor, lookup_cost)
    assert checks["S1"]["pass"] and checks["S2"]["pass"] and checks["S3"]["pass"] and checks["S5"]["pass"]
    assert checks["S4"]["pass"] is True and checks["all_pass"] is True
    worse = model_criteria(evaluation, thresholds, floor, lookup_cost, temporal_brier={"C": 0.3})
    assert worse["S6"]["pass"] is False and worse["all_pass"] is False  # a sector Brier above 0.20 fails S6
    assert model_criteria(evaluation, thresholds, floor, lookup_cost, is_lookup=True)["S4"]["pass"] is None
    beaten = model_criteria(evaluation, thresholds, floor, {k: 0.0 for k in floor})  # lookup as cheap as this model
    assert beaten["S4"]["pass"] is False and beaten["all_pass"] is False


# ---------------------------------------------------------------------------
# Temporal study: nothing from the later period reaches a choice
# ---------------------------------------------------------------------------


def test_temporal_thresholds_and_ranking_come_from_pre_2020_rows(clean: pd.DataFrame, tmp_path: Path) -> None:
    data = study(clean, tmp_path)
    dirs = ArtefactDirs(*(tmp_path / n for n in ("processed", "models", "metrics", "cv", "predictions", "figures")))
    specs = [MODEL_ZOO["logreg"], MODEL_ZOO["decision_tree"]]
    payload = temporal_evaluation(specs, data, cv_splitter(CV_FOLDS), True, dirs, 1)
    train_idx, test_idx = temporal_split(data.meta)
    assert payload["n_train"] == len(train_idx) and payload["n_test"] == len(test_idx)
    ranking = [row["model"] for row in payload["selection"]["ranking"]]
    assert set(ranking) == {"logreg", "decision_tree"} and list(payload["models"]) == ranking
    saved = json.loads((dirs.metrics / "temporal_split.json").read_text(encoding="utf-8"))
    assert saved["models"].keys() == payload["models"].keys()

    # Recompute the "with" variant by hand from the 2010-2019 rows only and compare the threshold.
    X, y, groups = data.X.iloc[train_idx], data.y[train_idx], data.groups[train_idx]
    entry = payload["models"][ranking[0]]
    spec = MODEL_ZOO[ranking[0]]
    from mlda.tuning import tune

    result = tune(spec, X, y, groups, cv_splitter(CV_FOLDS), quick=True, n_jobs=1)
    expected = oof_thresholds(y, result.oof_proba)
    assert entry["with_status_103i"]["thresholds"][OPERATING_KEY] == pytest.approx(expected[OPERATING_KEY])
    # the later period never entered the fit: every fitted row is before 2020
    assert data.years[train_idx].max() < 2020 <= data.years[test_idx].min()
    for variant in ("without_status_103i", "without_immed_reportable"):
        assert variant in entry and f"delta_{variant}" in entry
    assert payload["level_drift"]["e28_retired_occupation_column_all_zero"] is False  # no retired title in this base


def test_temporal_level_drift_reports_the_retired_occupation_column(clean: pd.DataFrame, tmp_path: Path) -> None:
    """E28 maps the retired scoop-car title to "missing", a one-hot column that is all zero after 2014."""
    retired = "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator"
    altered = clean.copy()
    early = (altered["CAL_YR"] <= 2014).to_numpy() & (np.arange(len(altered)) % 5 == 0)
    altered.loc[early, "OCCUPATION"] = retired
    data = study(altered, tmp_path)
    assert int((data.X["OCCUPATION"] == "missing").sum()) == int(early.sum())
    dirs = ArtefactDirs(*(tmp_path / n for n in ("processed", "models", "metrics", "cv", "predictions", "figures")))
    payload = temporal_evaluation([MODEL_ZOO["decision_tree"]], data, cv_splitter(CV_FOLDS), True, dirs, 1)
    assert payload["level_drift"]["e28_retired_occupation_column_all_zero"] is True
    assert "OCCUPATION_missing" in payload["level_drift"]["columns"]


# ---------------------------------------------------------------------------
# Whole pipeline on synthetic data (no raw extracts needed)
# ---------------------------------------------------------------------------


@pytest.mark.slow  # about 25 s on an idle machine, over a minute when the cores are shared
def test_train_all_quick_runs_end_to_end_on_the_synthetic_base(tmp_path: Path) -> None:
    dirs = ArtefactDirs(*(tmp_path / n for n in ("processed", "models", "metrics", "cv", "predictions", "figures")))
    base = synthetic_base(N_ROWS, seed=RANDOM_STATE)
    models = ["decision_tree", "gaussian_nb"]
    manifest = train_all(quick=True, models=models, base=base, dirs=dirs)
    assert manifest["quick"] is True and manifest["models"] == models and manifest["n_rows"] == N_ROWS
    assert set(manifest["temporal_models"]) <= set(models)
    sensitivity = json.loads((dirs.metrics / "sensitivity.json").read_text(encoding="utf-8"))
    assert {"pure_levels", "excluding_pure_levels", "primary_split_variants", "mine_grouped_split"} <= set(sensitivity)
    predictions = pd.read_parquet(dirs.predictions / "test_predictions.parquet")
    assert all(predictions[c].dtype == np.float64 for c in predictions if c.startswith("proba_"))
    assert (dirs.metrics / "success_criteria.json").is_file() and (dirs.metrics / "statistical_tests.json").is_file()
