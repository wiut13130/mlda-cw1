"""Quick end-to-end run of the study (two models) into a temporary artefact tree.

The run needs the real base table, so the module is skipped when the raw extracts are
absent, and it is marked slow because two models are tuned, scored and explained.
The config directories are monkeypatched so nothing is written under the repository.

To keep the module-scoped fixture well under the 90 s budget of a real-data test, the run uses
a seeded 20 % sample of the mine-years (whole ``group_id`` values, so the grouping structure the
split and the cross-validation rely on is intact). Stage costs scale with the number of rows;
the tuner caps every model at 10,000 rows in quick mode either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlda import config
from mlda.config import RANDOM_STATE
from mlda.data.join import persistable
from mlda.evaluation import COMPARISON_COLUMNS, threshold_for_flag_rate, threshold_for_specificity
from mlda.pipeline import FIGURES, train_all
from mlda.split import load_split_indices

pytestmark = pytest.mark.slow

MODELS = ["logreg", "decision_tree"]
MINE_YEAR_FRACTION = 0.2
DIRECTORIES = {
    "PROCESSED_DIR": "data/processed",
    "MODELS_DIR": "models",
    "METRICS_DIR": "reports/metrics",
    "CV_DIR": "reports/cv",
    "PREDICTIONS_DIR": "reports/predictions",
    "FIGURES_DIR": "reports/figures",
}


@pytest.fixture(scope="module")
def sample_base(real_base: pd.DataFrame) -> pd.DataFrame:
    """A seeded sample of whole mine-years of the real base table, without free text."""
    groups = real_base["group_id"].unique()
    chosen = np.random.default_rng(RANDOM_STATE).choice(groups, size=int(MINE_YEAR_FRACTION * len(groups)), replace=False)
    return persistable(real_base[real_base["group_id"].isin(chosen)].reset_index(drop=True))


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, sample_base: pd.DataFrame) -> tuple[Path, dict]:
    """Run train_all(quick=True) once for the module with every output directory redirected."""
    root = tmp_path_factory.mktemp("artefacts")
    with pytest.MonkeyPatch.context() as patch:
        for name, relative in DIRECTORIES.items():
            patch.setattr(config, name, root / relative)
        manifest = train_all(quick=True, models=MODELS, base=sample_base)
    return root, manifest


def test_key_artefacts_exist(run: tuple[Path, dict]) -> None:
    root, manifest = run
    expected = [
        "data/processed/features.parquet",
        "data/processed/split_indices.json",
        "models/logreg.joblib",
        "models/decision_tree.joblib",
        "models/manifest.json",
        "reports/metrics/error_rules.csv",
        "reports/metrics/baselines.json",
        "reports/metrics/comparison.csv",
        "reports/metrics/comparison_operating_point.csv",
        "reports/metrics/thresholds.json",
        "reports/metrics/logreg.json",
        "reports/metrics/decision_tree.json",
        "reports/metrics/statistical_tests.json",
        "reports/metrics/temporal_split.json",
        "reports/metrics/sensitivity.json",
        "reports/metrics/explanations.json",
        "reports/metrics/success_criteria.json",
        "reports/metrics/run_manifest.json",
        "reports/metrics/eda_describe_numeric.csv",
        "reports/metrics/eda_describe_categorical.csv",
        "reports/metrics/eda_correlations_pearson.csv",
        "reports/metrics/eda_point_biserial.csv",
        "reports/metrics/eda_cramers_v.csv",
        "reports/metrics/eda_vif.csv",
        "reports/metrics/eda_missingness.csv",
        "reports/cv/logreg_cv_results.csv",
        "reports/cv/decision_tree_cv_results.csv",
        "reports/predictions/test_predictions.parquet",
        "reports/predictions/oof_predictions.parquet",
    ]
    missing = [name for name in expected if not (root / name).is_file()]
    assert missing == []
    figures = {p.name for p in (root / "reports/figures").glob("*.png")}
    for key in ("target", "histograms", "correlations", "roc", "pr", "confusion", "calibration", "validation"):
        assert FIGURES[key] in figures, key
    assert any(name.startswith("F15_learning_curve") for name in figures)
    assert any(name.startswith("F17_feature_importance") for name in figures)
    assert any(name.startswith("F18_threshold_cost_curve") for name in figures)
    assert manifest["quick"] is True and manifest["models"] == MODELS
    assert manifest["best_model"] in MODELS
    assert set(manifest["stage_seconds"]) >= {
        "load_and_clean", "features_and_split", "eda", "baselines", "model_logreg", "temporal_split", "sensitivity",
    }


def test_test_predictions_are_probabilities_on_held_out_rows(run: tuple[Path, dict]) -> None:
    root, manifest = run
    test = pd.read_parquet(root / "reports/predictions/test_predictions.parquet")
    oof = pd.read_parquet(root / "reports/predictions/oof_predictions.parquet")
    split = load_split_indices(root / "data/processed/split_indices.json")
    assert len(test) == len(split.test_idx) == manifest["n_test"]
    assert len(oof) == len(split.train_idx) == manifest["n_train"]
    assert set(test["y_true"].unique()) <= {0, 1}
    for column in ("y_true", "group_id", "sector", "CAL_YR"):
        assert column in test.columns
    proba_columns = [c for c in test.columns if c.startswith("proba_")]
    assert {f"proba_{m}" for m in MODELS} <= set(proba_columns)
    assert "proba_lookup_nature_bodypart_coal" in proba_columns
    for column in proba_columns:
        assert test[column].dtype == np.float64, column  # float32 moved rows across stored thresholds
        values = test[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0, column
    assert set(test["group_id"]).isdisjoint(oof["group_id"])


def test_stored_thresholds_are_reproduced_exactly_from_the_stored_oof_scores(run: tuple[Path, dict]) -> None:
    """thresholds.json and oof_predictions.parquet agree to the last bit (both are float64)."""
    root, _ = run
    thresholds = json.loads((root / "reports/metrics/thresholds.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(root / "reports/predictions/oof_predictions.parquet")
    for model in [*MODELS, "lookup_nature_bodypart_coal"]:
        y, p = oof["y_true"].to_numpy(), oof[f"proba_{model}"].to_numpy()
        assert threshold_for_specificity(y, p, 0.40) == thresholds[model]["specificity_0.40"], model
        assert threshold_for_flag_rate(y, p, 0.5) == thresholds[model]["flag_rate_0.50"], model


def test_comparison_and_metrics_are_consistent(run: tuple[Path, dict]) -> None:
    root, _ = run
    comparison = pd.read_csv(root / "reports/metrics/comparison.csv", index_col="model")
    assert list(comparison.columns) == COMPARISON_COLUMNS
    assert set(MODELS) <= set(comparison.index)
    assert comparison["PR-AUC"].is_monotonic_decreasing
    metrics = json.loads((root / "reports/metrics/logreg.json").read_text(encoding="utf-8"))
    assert metrics["test"]["at_0.5"]["pr_auc"] == pytest.approx(comparison.loc["logreg", "PR-AUC"])
    assert 0.0 <= metrics["thresholds"]["specificity_0.40"] <= 1.0
    assert metrics["n_fit_rows"] <= 10_000 and metrics["cv_folds"] == 3
    criteria = json.loads((root / "reports/metrics/success_criteria.json").read_text(encoding="utf-8"))
    assert set(MODELS) <= set(criteria["models"])
    assert {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "all_pass"} <= set(criteria["models"]["logreg"])
    assert set(criteria["robustness_mine_grouped_split"]["criteria"]) == {criteria["robustness_mine_grouped_split"]["model"], "lookup_nature_bodypart_coal"}


def test_temporal_study_uses_only_pre_2020_labels_for_its_choices(run: tuple[Path, dict]) -> None:
    root, _ = run
    temporal = json.loads((root / "reports/metrics/temporal_split.json").read_text(encoding="utf-8"))
    primary = json.loads((root / "reports/metrics/logreg.json").read_text(encoding="utf-8"))
    assert [row["model"] for row in temporal["selection"]["ranking"]][:2] == list(temporal["models"])
    assert "2010-2019" in temporal["thresholds"]
    entry = temporal["models"]["logreg"]
    for variant in ("with_status_103i", "without_status_103i", "without_immed_reportable"):
        assert variant in entry and "thresholds" in entry[variant]
    # the threshold comes from the 2010-2019 rows, so it differs from the one the primary split chose
    assert entry["with_status_103i"]["thresholds"]["specificity_0.40"] != primary["thresholds"]["specificity_0.40"]
    assert temporal["level_drift"]["e28_retired_occupation_column_all_zero"] is True
    achieved = entry["with_status_103i"]["test"]["at_specificity_0.40"]["specificity"]
    assert 0.30 < achieved < 0.50


def test_sensitivity_report_lists_the_documented_pure_level_and_every_variant(run: tuple[Path, dict]) -> None:
    root, manifest = run
    sensitivity = json.loads((root / "reports/metrics/sensitivity.json").read_text(encoding="utf-8"))
    levels = {(r["column"], r["level"]) for r in sensitivity["pure_levels"]["found"]}
    assert ("NATURE_INJURY_CD", "100") in levels
    assert sensitivity["pure_levels"]["undocumented"] == [] and manifest["undocumented_pure_levels"] == []
    excluded = sensitivity["excluding_pure_levels"]
    assert excluded["n_test_excluded"] > 0 and "logreg" in excluded["models"]
    assert set(sensitivity["primary_split_variants"]["variants"]) == {
        "without_immed_reportable", "without_status_103i", "resolved_lag_counts", "exposure_blank_if_q4_incomplete",
        "history_without_test_outcomes",
    }
    grouped = sensitivity["mine_grouped_split"]
    assert grouped["group"] == "MINE_ID" and grouped["shared_mines"] == 0
    stats = json.loads((root / "reports/metrics/statistical_tests.json").read_text(encoding="utf-8"))
    assert "clustered_p_value" in stats["mcnemar_best_vs_lookup"]["at_0.5"]
    assert stats["bootstrap_best_model"]["metrics"]["pr_auc"]["mine_years"]["resampling"].endswith("clusters")
