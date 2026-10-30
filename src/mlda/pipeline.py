"""End-to-end study runner: from the raw extracts to every committed artefact.

``train_all`` chains the modules of this package in the order the report follows: load the
base table, apply the error rules, engineer features, hold out the grouped test fold, run
the exploratory tables and figures on the training rows, fit the baselines, tune each model
of the zoo on the training rows with grouped cross-validation, choose every threshold on
out-of-fold probabilities, score the held-out rows once, compare, test, draw and explain.
Every artefact the notebook and the app read is written
under the directories named in ``mlda.config`` (read at call time, so tests can redirect
them). Test rows are only ever passed to ``predict_proba``; nothing is fitted on them.

Two studies go beyond the primary split. The temporal study (``temporal_evaluation``) repeats
the whole protocol on CAL_YR 2010-2019 alone, so that its thresholds, hyper-parameters and
model choice carry no information from 2020-2025. The sensitivity study
(``run_sensitivity``) refits the selected pipeline on altered inputs and on a split grouped by
mine, and re-scores the held-out rows without deterministic levels.

``quick=True`` keeps the same stages but shrinks every search to two candidates, 10,000
stratified training rows and three folds, so the whole run finishes in a few minutes; it is
the mode the tests and the app's live-training page use.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, cross_val_predict, learning_curve

from mlda import config, plots
from mlda.baselines import BASELINE_ZOO, fit_baselines
from mlda.cleaning import apply_error_rules
from mlda.config import (
    COST_RATIOS,
    GROUP_KEY,
    N_SPLITS,
    RANDOM_STATE,
    SECTOR_COL,
    STUDENT_ID,
    SUCCESS_CRITERIA,
    TARGET,
    TEMPORAL_SPLIT_YEAR,
)
from mlda.data.join import load_base_table
from mlda.eda import (
    correlations,
    cramers_v_table,
    describe_categorical,
    describe_numeric,
    missingness_table,
    point_biserial,
    vif_table,
)
from mlda.evaluation import (
    bootstrap_ci,
    by_sector,
    classification_metrics,
    compare_models,
    cost_from_counts,
    curves,
    mcnemar_test,
    paired_fold_test,
    summarise_cv,
    threshold_sweep,
)
from mlda.explain import coefficient_table, permutation_importance_table, shap_summary
from mlda.features import FEATURE_SPEC, META_COLUMNS, feature_frame
from mlda.models import MODEL_ZOO, ModelSpec
from mlda.operating import (
    FLAG_KEY,
    FLAG_RATE_TARGET,
    LOOKUP_BASELINE,
    OPERATING_KEY,
    all_positive_floor,
    delta,
    evaluate_on_test,
    headline,
    model_criteria,
    oof_thresholds,
)
from mlda.persistence import environment_record, save_model
from mlda.preprocessing import build_preprocessor, get_feature_names
from mlda.sensitivity import (
    PURE_LEVEL_MIN_ROWS,
    excluding_levels,
    feature_variants,
    mine_grouped_robustness,
    pure_levels,
    refit_variants,
    simple_variants,
)
from mlda.split import SplitIndices, cv_splitter, make_split, save_split_indices, temporal_split
from mlda.tuning import (
    METRIC_NAMES,
    QUICK_SAMPLE_CAP,
    REFIT_METRIC,
    TuneResult,
    default_n_jobs,
    estimator_params,
    fold_scores,
    out_of_fold_probabilities,
    stratified_subsample,
    tune,
)

log = logging.getLogger(__name__)

QUICK_CV_FOLDS = 3
QUICK_FIGURE_ROWS = 10_000
BOOTSTRAP_DRAWS = 1000
QUICK_BOOTSTRAP_DRAWS = 200
TEMPORAL_VARIANTS: dict[str, str | None] = {
    "with_status_103i": None,
    "without_status_103i": "without_status_103i",
    "without_immed_reportable": "without_immed_reportable",
}
PERMUTATION_ROWS = 5_000
SHAP_ROWS = 500
QUICK_EXPLANATION_ROWS = 1_500  # permutation importance and SHAP rows in quick mode
LEARNING_CURVE_SIZES = np.linspace(0.2, 1.0, 5)
QQ_COLUMNS = (
    "TOT_EXPER", "log1p_tot_exper", "prev_hours", "log1p_prev_hours", "hours_into_shift", "prev_lost_time_share",
)
OUTLIER_COLUMNS = {"TOT_EXPER": "log1p_tot_exper", "prev_hours": "log1p_prev_hours"}
# Figure file names shared with the report and the app (F01 is the data-flow diagram).
FIGURES: dict[str, str] = {
    "target": "F02_target_distribution.png",
    "histograms": "F03_histograms.png",
    "boxplots": "F04_class_boxplots.png",
    "pairplot": "F05_pairplot_top.png",
    "correlations": "F06_correlation_heatmaps.png",
    "categorical": "F07_categorical_vs_target.png",
    "missingness": "F08_missingness.png",
    "qq": "F09_qq_plots.png",
    "outliers": "F10_outlier_fences.png",
    "pca": "F11_pca_projection.png",
    "roc": "F12_roc_curves.png",
    "pr": "F13_pr_curves.png",
    "validation": "F14_validation_curves.png",
    "learning": "F15_learning_curve.png",
    "confusion": "F16_confusion_matrices.png",
    "importance": "F17_feature_importance.png",
    "coefficients": "F17b_coefficients_logreg.png",
    "cost": "F18_threshold_cost_curve.png",
    "calibration": "F19_calibration.png",
}


# ---------------------------------------------------------------------------
# Small helpers: directories, timing, JSON
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtefactDirs:
    """Where the run writes; built from ``mlda.config`` at call time so tests can redirect it."""

    processed: Path
    models: Path
    metrics: Path
    cv: Path
    predictions: Path
    figures: Path

    @classmethod
    def from_config(cls) -> ArtefactDirs:
        return cls(
            processed=Path(config.PROCESSED_DIR),
            models=Path(config.MODELS_DIR),
            metrics=Path(config.METRICS_DIR),
            cv=Path(config.CV_DIR),
            predictions=Path(config.PREDICTIONS_DIR),
            figures=Path(config.FIGURES_DIR),
        )

    def create(self) -> None:
        for directory in (self.processed, self.models, self.metrics, self.cv, self.predictions, self.figures):
            directory.mkdir(parents=True, exist_ok=True)


class StageTimer:
    """Collects wall-clock seconds per stage for the run manifest and the log."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        log.info("stage %s: start", name)
        yield
        elapsed = time.perf_counter() - started
        self.seconds[name] = round(elapsed, 2)
        log.info("stage %s: done in %.1f s", name, elapsed)


def _jsonable(value: Any) -> Any:
    """Plain JSON types only: numpy scalars unwrapped, NaN -> null, timestamps as ISO text."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
        handle.write("\n")
    return path


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8")
    return path


def _save_figure(fig: Any, dirs: ArtefactDirs, key: str, suffix: str = "") -> Path:
    """Write a figure at the report resolution; ``suffix`` inserts a model name before .png."""
    name = FIGURES[key]
    if suffix:
        name = name.replace(".png", f"_{suffix}.png")
    path = dirs.figures / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plots.DPI, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class StudyData:
    """Model inputs, labels, groups, reporting columns and the held-out split, all positional.

    ``variants`` holds the altered input frames of the sensitivity runs (same index and columns
    as ``X``), built from the cleaned base table before it is released.
    """

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    years: np.ndarray
    sector: np.ndarray
    split: SplitIndices
    variants: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def train_idx(self) -> np.ndarray:
        return self.split.train_idx

    @property
    def test_idx(self) -> np.ndarray:
        return self.split.test_idx


@dataclass
class ScoredModel:
    """One model or baseline after tuning: probabilities, thresholds and the metrics payload."""

    name: str
    oof_proba: np.ndarray
    test_proba: np.ndarray
    thresholds: dict[str, Any]
    evaluation: dict[str, Any]
    curves: dict[str, pd.DataFrame]
    spec: ModelSpec | None = None
    result: TuneResult | None = None
    cv_mean_ap: float = float("nan")
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stages 1-3: load, clean, features, split
# ---------------------------------------------------------------------------


def load_and_clean(dirs: ArtefactDirs, base: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base table (cached parquet unless ``base`` is given) with the error rules applied.

    The rule log is written to reports/metrics/error_rules.csv (Table 4 of the report).
    """
    if base is None:
        base = load_base_table()
    clean, rule_log = apply_error_rules(base)
    _write_csv(rule_log, dirs.metrics / "error_rules.csv", index=False)
    log.info("error rules applied to %d rows; %d rules logged", len(clean), len(rule_log))
    return clean, rule_log


def build_study_data(clean: pd.DataFrame, dirs: ArtefactDirs) -> StudyData:
    """Feature frame, persisted features.parquet, grouped split and persisted split indices."""
    parts = feature_frame(clean)
    y = parts.y.to_numpy(dtype=int)
    groups = parts.groups.to_numpy(dtype=str)
    years = pd.to_numeric(parts.meta["CAL_YR"]).to_numpy(dtype=np.int64)
    sector = parts.meta[SECTOR_COL].astype(str).to_numpy()

    table = parts.X.copy()
    table[TARGET] = parts.y.to_numpy()
    table[GROUP_KEY] = groups
    for column in META_COLUMNS:
        if column not in table.columns:  # COAL_METAL_IND is already a feature column
            table[column] = parts.meta[column].to_numpy()
    path = dirs.processed / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, compression="zstd", index=False)
    log.info("wrote %s: %d rows x %d columns", path.name, *table.shape)

    split = make_split(parts.X, y, groups, accident_groups=parts.meta["accident_group"])
    save_split_indices(split, dirs.processed / "split_indices.json")
    log.info(
        "split: %d train / %d test rows, prevalence %.4f / %.4f",
        len(split.train_idx), len(split.test_idx), y[split.train_idx].mean(), y[split.test_idx].mean(),
    )
    # The split is passed so that run_sensitivity also refits the variant without test outcomes in
    # the lagged case history; the cleaned base is released before run_sensitivity runs.
    variants = feature_variants(clean, parts.X, split.train_idx, split.test_idx)
    return StudyData(parts.X, y, groups, parts.meta.reset_index(drop=True), years, sector, split, variants)


# ---------------------------------------------------------------------------
# Stage 4: exploratory analysis on the training rows
# ---------------------------------------------------------------------------


def run_eda(data: StudyData, clean: pd.DataFrame, dirs: ArtefactDirs, quick: bool = False) -> dict[str, Any]:
    """EDA tables (reports/metrics/eda_*.csv) and figures F02-F11 from the training rows.

    Everything here describes the rows the models learn from; the held-out fold stays
    untouched until the final scoring, which is why the tables use the training rows and
    not the whole window. The target distribution figure alone uses all rows, because it
    reports the N and prevalence of the study. With ``quick=True`` the two slowest figures
    (kernel density histograms and box plots, about 9 s on 80,000 rows) are drawn from a
    random sample of ``QUICK_FIGURE_ROWS`` training rows; every table still uses all of them.
    """
    numeric = FEATURE_SPEC["numeric"]
    categorical = FEATURE_SPEC["categorical"]
    train_idx = data.train_idx
    X_train = data.X.iloc[train_idx]
    y_train = data.y[train_idx]
    frame = X_train.assign(**{TARGET: y_train})
    base_train = clean.iloc[train_idx]

    _write_csv(describe_numeric(frame, numeric), dirs.metrics / "eda_describe_numeric.csv")
    # Raw source columns where the base has them, so blank codes count as missing rather
    # than as the "missing" level the feature frame fills in.
    sources = pd.DataFrame(
        {c: (base_train[c] if c in base_train.columns else X_train[c]).to_numpy() for c in categorical}
    )
    _write_csv(describe_categorical(sources, categorical), dirs.metrics / "eda_describe_categorical.csv")
    corr = correlations(frame, numeric)
    _write_csv(corr.pearson, dirs.metrics / "eda_correlations_pearson.csv")
    _write_csv(corr.spearman, dirs.metrics / "eda_correlations_spearman.csv")
    _write_csv(corr.disagreements, dirs.metrics / "eda_correlations_disagreements.csv", index=False)
    biserial = point_biserial(frame, numeric, TARGET)
    _write_csv(biserial, dirs.metrics / "eda_point_biserial.csv", index=False)
    cramers = cramers_v_table(frame, categorical, TARGET)
    _write_csv(cramers, dirs.metrics / "eda_cramers_v.csv", index=False)
    _write_csv(vif_table(X_train[numeric]), dirs.metrics / "eda_vif.csv", index=False)
    _write_csv(missingness_table(base_train), dirs.metrics / "eda_missingness.csv", index=False)

    drawn = frame
    if quick and len(frame) > QUICK_FIGURE_ROWS:
        drawn = frame.sample(n=QUICK_FIGURE_ROWS, random_state=RANDOM_STATE)
    top_numeric = biserial["feature"].head(5).tolist()
    top_categorical = cramers["feature"].head(8).tolist()
    _save_figure(plots.target_distribution(data.y), dirs, "target")
    # four columns and short panels so the 20-panel grids print at full text width on one page
    _save_figure(
        plots.histograms_grid(drawn, numeric, hue=TARGET, ncols=4, panel_height=1.6, label_width=22),
        dirs, "histograms",
    )
    _save_figure(
        plots.class_boxplots(drawn, numeric, TARGET, showfliers=False, ncols=4, panel_height=1.6, label_width=22),
        dirs, "boxplots",
    )
    _save_figure(plots.pairplot_top(frame, top_numeric, TARGET), dirs, "pairplot")
    _save_figure(plots.correlation_heatmaps(corr.pearson, corr.spearman), dirs, "correlations")
    _save_figure(plots.categorical_vs_target(frame, top_categorical, TARGET), dirs, "categorical")
    _save_figure(plots.missingness(base_train), dirs, "missingness")
    _save_figure(plots.qq_plots(frame, list(QQ_COLUMNS)), dirs, "qq")
    after = pd.DataFrame({raw: frame[logged].to_numpy() for raw, logged in OUTLIER_COLUMNS.items()})
    _save_figure(
        plots.outlier_fences(frame, list(OUTLIER_COLUMNS), after=after, after_labels=OUTLIER_COLUMNS),
        dirs, "outliers",
    )
    _save_figure(plots.pca_projection(X_train[numeric], y_train), dirs, "pca")
    return {
        "rows_described": int(len(train_idx)),
        "scope": "training rows of the grouped split (target distribution: all rows)",
        "figure_rows_histograms_boxplots": int(len(drawn)),
        "top_numeric_by_point_biserial": top_numeric,
        "top_categorical_by_cramers_v": top_categorical,
    }


# ---------------------------------------------------------------------------
# Thresholds and test evaluation shared by models and baselines
# (oof_thresholds and evaluate_on_test live in mlda.operating and are re-exported here)
# ---------------------------------------------------------------------------


def _calibration_records(model_curves: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    return model_curves["calibration"].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Stage 5: baselines
# ---------------------------------------------------------------------------


def run_baselines(data: StudyData, cv: BaseCrossValidator, dirs: ArtefactDirs) -> dict[str, ScoredModel]:
    """Fit the reference rules on the training rows, score the test rows, write baselines.json.

    Out-of-fold scores come from ``cross_val_predict`` on the same grouped folds as the
    models, so the lookup's thresholds are chosen the way the models' thresholds are.
    """
    train_idx, test_idx = data.train_idx, data.test_idx
    X_train, y_train, g_train = data.X.iloc[train_idx], data.y[train_idx], data.groups[train_idx]
    X_test, y_test = data.X.iloc[test_idx], data.y[test_idx]
    fitted = fit_baselines(X_train, y_train)
    scored: dict[str, ScoredModel] = {}
    payload: dict[str, Any] = {}
    for name, model in fitted.items():
        # The rules are cheap to fit, so one process is faster than spawning workers.
        oof = cross_val_predict(
            BASELINE_ZOO[name].factory(), X_train, y_train, groups=g_train, cv=cv, method="predict_proba", n_jobs=1,
        )[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        thresholds = oof_thresholds(y_train, oof, fixed_threshold=0.5 if BASELINE_ZOO[name].constant else None)
        evaluation = evaluate_on_test(y_test, test_proba, data.sector[test_idx], thresholds)
        model_curves = curves(y_test, test_proba)
        scored[name] = ScoredModel(name, oof, test_proba, thresholds, evaluation, model_curves)
        payload[name] = {
            "description": BASELINE_ZOO[name].description,
            "thresholds": {k: v for k, v in thresholds.items() if k != "oof_cost_table"},
            "oof_cost_table": thresholds["oof_cost_table"],
            "test": evaluation,
            "calibration_bins": _calibration_records(model_curves),
        }
        log.info(
            "baseline %-28s test AUC %.4f PR-AUC %.4f recall@0.5 %.4f",
            name, evaluation["at_0.5"]["roc_auc"], evaluation["at_0.5"]["pr_auc"], evaluation["at_0.5"]["recall"],
        )
    _write_json(dirs.metrics / "baselines.json", payload)
    return scored


# ---------------------------------------------------------------------------
# Stage 6: one model of the zoo
# ---------------------------------------------------------------------------


def _cv_summary(result: TuneResult) -> dict[str, Any]:
    """Mean and std of every scorer for the selected candidate (rank 1 on average precision)."""
    best = result.cv_results_df.iloc[0]
    summary: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        summary[metric] = {
            "mean_test": float(best[f"mean_test_{metric}"]),
            "std_test": float(best[f"std_test_{metric}"]),
            "mean_train": float(best[f"mean_train_{metric}"]),
        }
    return summary


def run_model(
    spec: ModelSpec, data: StudyData, cv: BaseCrossValidator, quick: bool, dirs: ArtefactDirs, n_jobs: int
) -> ScoredModel:
    """Tune, choose thresholds out of fold, score the test rows once, persist model and tables."""
    train_idx, test_idx = data.train_idx, data.test_idx
    X_train, y_train, g_train = data.X.iloc[train_idx], data.y[train_idx], data.groups[train_idx]
    X_test, y_test = data.X.iloc[test_idx], data.y[test_idx]

    result = tune(spec, X_train, y_train, g_train, cv, quick=quick, n_jobs=n_jobs)
    thresholds = oof_thresholds(y_train, result.oof_proba)
    started = time.perf_counter()
    test_proba = result.best_pipeline.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - started
    evaluation = evaluate_on_test(y_test, test_proba, data.sector[test_idx], thresholds)
    model_curves = curves(y_test, test_proba)
    cv_summary = _cv_summary(result)

    metrics: dict[str, Any] = {
        "model": spec.name,
        "description": spec.description,
        "taught_in_week": spec.taught_in_week,
        "quick": quick,
        "search": "grid (two candidates)" if quick else spec.search,
        "n_candidates": result.n_candidates,
        "best_params": estimator_params(result),
        "preprocess_overrides": dict(spec.preprocess_overrides),
        "n_train_rows": int(len(train_idx)),
        "n_fit_rows": int(len(result.fit_idx)),
        "n_test_rows": int(len(test_idx)),
        "cv_folds": int(cv.get_n_splits()),
        "cv": cv_summary,
        f"fold_scores_{REFIT_METRIC}": fold_scores(result).tolist(),
        "tuning_seconds": round(result.elapsed_seconds, 2),
        "predict_seconds": round(predict_seconds, 3),
        "thresholds": {k: v for k, v in thresholds.items() if k != "oof_cost_table"},
        "oof_cost_table": thresholds["oof_cost_table"],
        "test": evaluation,
        "calibration_bins": _calibration_records(model_curves),
    }
    save_model(
        result.best_pipeline, spec.name, best_params=result.best_params,
        extra={
            "quick": quick, "n_train_rows": int(len(train_idx)), "n_fit_rows": int(len(result.fit_idx)),
            "cv_folds": int(cv.get_n_splits()), "test_roc_auc": evaluation["at_0.5"]["roc_auc"],
            "test_pr_auc": evaluation["at_0.5"]["pr_auc"],
            "thresholds": {OPERATING_KEY: thresholds[OPERATING_KEY], FLAG_KEY: thresholds[FLAG_KEY], "cost_optimal": thresholds["cost_optimal"]},
        },
        models_dir=dirs.models,
    )
    _write_csv(result.cv_results_df, dirs.cv / f"{spec.name}_cv_results.csv", index=False)
    _write_json(dirs.metrics / f"{spec.name}.json", metrics)
    log.info(
        "%-14s CV AP %.4f | test AUC %.4f PR-AUC %.4f | recall@spec0.40 %.4f | %.0f s",
        spec.name, cv_summary[REFIT_METRIC]["mean_test"], evaluation["at_0.5"]["roc_auc"],
        evaluation["at_0.5"]["pr_auc"], evaluation[f"at_{OPERATING_KEY}"]["recall"], result.elapsed_seconds,
    )
    return ScoredModel(
        spec.name, result.oof_proba, test_proba, thresholds, evaluation, model_curves,
        spec=spec, result=result, cv_mean_ap=cv_summary[REFIT_METRIC]["mean_test"], metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Stage 7: predictions, comparison tables, statistical tests
# ---------------------------------------------------------------------------


def write_predictions(data: StudyData, models: Sequence[ScoredModel], baselines: dict[str, ScoredModel], dirs: ArtefactDirs) -> None:
    """test_predictions.parquet (held-out rows) and oof_predictions.parquet (training rows).

    Probabilities are stored as float64, the precision the thresholds in thresholds.json were
    computed from, so re-applying a stored threshold to a stored score flags the same rows as the
    run did. (float32 moved rows across the boundary: differences of up to 1e-8 in the scores.)
    """
    for which, idx, attribute in (("test", data.test_idx, "test_proba"), ("oof", data.train_idx, "oof_proba")):
        frame = pd.DataFrame(
            {
                "row": idx,
                "y_true": data.y[idx].astype("int8"),
                GROUP_KEY: data.groups[idx],
                "sector": data.sector[idx],
                "CAL_YR": data.years[idx],
                "MINE_ID": data.meta["MINE_ID"].to_numpy()[idx],
                "ACCIDENT_DT": data.meta["ACCIDENT_DT"].to_numpy()[idx],
            }
        )
        for model in models:
            frame[f"proba_{model.name}"] = np.asarray(getattr(model, attribute), dtype="float64")
        for name, baseline in baselines.items():
            frame[f"proba_{name}"] = np.asarray(getattr(baseline, attribute), dtype="float64")
        path = dirs.predictions / f"{which}_predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, compression="zstd", index=False)
        log.info("wrote %s (%d rows, %d columns)", path.name, *frame.shape)


def write_comparison(models: Sequence[ScoredModel], baselines: dict[str, ScoredModel], dirs: ArtefactDirs) -> pd.DataFrame:
    """comparison.csv (threshold 0.5), the operating-point variant, thresholds.json and cv_summary.csv."""
    everything = list(models) + list(baselines.values())
    at_half = {m.name: m.evaluation["at_0.5"] for m in everything}
    table = compare_models(at_half)
    _write_csv(table, dirs.metrics / "comparison.csv")
    operating = {m.name: m.evaluation[f"at_{OPERATING_KEY}"] for m in everything}
    _write_csv(compare_models(operating, extra_columns=True), dirs.metrics / "comparison_operating_point.csv")
    thresholds = {
        m.name: {OPERATING_KEY: m.thresholds[OPERATING_KEY], FLAG_KEY: m.thresholds[FLAG_KEY], "cost_optimal": m.thresholds["cost_optimal"]}
        for m in everything
    }
    _write_json(dirs.metrics / "thresholds.json", thresholds)
    if models:
        cv_frame = pd.concat([m.result.cv_results_df for m in models if m.result is not None], ignore_index=True)
        _write_csv(summarise_cv(cv_frame), dirs.metrics / "cv_summary.csv")
    return table


def rank_models(models: Sequence[ScoredModel]) -> list[ScoredModel]:
    """Best first by cross-validated average precision (chosen on training folds, not on test)."""
    return sorted(models, key=lambda m: (-m.cv_mean_ap, m.name))


def statistical_tests(
    ranked: Sequence[ScoredModel],
    lookup: ScoredModel,
    y_test: np.ndarray,
    dirs: ArtefactDirs,
    groups_test: np.ndarray | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """McNemar best model vs the lookup baseline, bootstrap intervals and the paired fold test.

    Test rows cluster in mine-years, so every test that can use ``groups_test`` (the held-out
    ``group_id`` values) reports a clustered version beside the row-wise one, and the clustered
    one is the one to quote. The paired fold test is run only when the two models were fitted on
    the same rows; otherwise their folds are not comparable and the p-values are null.
    """
    best = ranked[0]
    payload: dict[str, Any] = {
        "ranking_rule": f"cross-validated mean {REFIT_METRIC} of the selected candidate",
        "ranking": [{"model": m.name, "cv_mean_average_precision": m.cv_mean_ap} for m in ranked],
        "mcnemar_best_vs_lookup": {},
        "clustering": "group_id (mine-year) of the held-out rows" if groups_test is not None else None,
    }
    for key, threshold_best, threshold_lookup in (
        ("at_0.5", 0.5, 0.5),
        (f"at_{OPERATING_KEY}", best.thresholds[OPERATING_KEY], lookup.thresholds[OPERATING_KEY]),
    ):
        outcome = mcnemar_test(
            y_test, best.test_proba >= threshold_best, lookup.test_proba >= threshold_lookup, groups=groups_test
        )
        outcome.update({"model_a": best.name, "model_b": lookup.name, "threshold_a": threshold_best, "threshold_b": threshold_lookup})
        payload["mcnemar_best_vs_lookup"][key] = outcome

    n_boot = QUICK_BOOTSTRAP_DRAWS if quick else BOOTSTRAP_DRAWS
    intervals: dict[str, Any] = {}
    for metric, threshold in (("roc_auc", 0.5), ("pr_auc", 0.5), ("recall", float(best.thresholds[OPERATING_KEY]))):
        label = metric if metric != "recall" else f"recall_at_{OPERATING_KEY}"
        intervals[label] = {"rows": bootstrap_ci(y_test, best.test_proba, metric, threshold, n_boot=n_boot)}
        if groups_test is not None:
            intervals[label]["mine_years"] = bootstrap_ci(
                y_test, best.test_proba, metric, threshold, n_boot=n_boot, groups=groups_test
            )
    payload["bootstrap_best_model"] = {"model": best.name, "n_boot": n_boot, "metrics": intervals}

    if len(ranked) >= 2 and ranked[0].result is not None and ranked[1].result is not None:
        a, b = ranked[0], ranked[1]
        scores_a, scores_b = fold_scores(a.result), fold_scores(b.result)
        same_rows = bool(np.array_equal(a.result.fit_idx, b.result.fit_idx))
        entry: dict[str, Any] = {
            "models": [a.name, b.name],
            "metric": REFIT_METRIC,
            "same_fit_rows": same_rows,
            "note": None if same_rows else "the two models were fitted on different row subsets (sample cap), so their folds are not comparable and no p-value is reported",
            "scores_a": scores_a.tolist(),
            "scores_b": scores_b.tolist(),
        }
        if same_rows:
            entry.update(paired_fold_test(scores_a, scores_b))
        else:
            entry.update(
                {"mean_difference": float(np.mean(scores_a - scores_b)), "t_p_value": None, "corrected_t_p_value": None, "wilcoxon_p_value": None}
            )
        payload["paired_fold_test_top_two"] = entry
    _write_json(dirs.metrics / "statistical_tests.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 8: evaluation figures
# ---------------------------------------------------------------------------


def validation_panel(result: TuneResult) -> dict[str, Any] | None:
    """Training and validation average precision against the grid parameter with most values.

    The other parameters are held at the selected candidate's values when that leaves at
    least two points; otherwise (randomised search, quick grids) every candidate is used
    and the best candidate per value is kept.
    """
    frame = result.cv_results_df
    best = estimator_params(result)
    param_columns = [c for c in frame.columns if c.startswith("param_")]
    counts = {c: frame[c].astype(str).nunique() for c in param_columns}
    if not counts or max(counts.values()) < 2:
        return None
    column = max(param_columns, key=lambda c: counts[c])
    param = column[len("param_"):]
    mask = pd.Series(True, index=frame.index)
    for other in param_columns:
        if other != column:
            mask &= frame[other].astype(str) == str(best[other[len("param_"):]])
    chosen = frame[mask] if mask.sum() >= 2 else frame
    chosen = chosen.drop_duplicates(subset=column)  # sorted by rank, so the best candidate per value stays
    values = chosen[column].tolist()

    def is_number(v: Any) -> bool:
        return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)

    numeric = all(is_number(v) for v in values)
    if numeric:
        x = np.asarray(values, dtype=float)
        order = np.argsort(x)
        integral = bool(np.all(np.mod(x, 1) == 0))
        x = x.astype(int) if integral else x
        best_value: Any = int(best[param]) if integral else float(best[param])
    else:
        # Numbers with None (unlimited max_depth): ascending, None last, so the axis reads in order.
        numbers_or_none = all(v is None or is_number(v) for v in values)
        x = np.asarray([str(int(v)) if is_number(v) and float(v).is_integer() else str(v) for v in values], dtype=object)
        if numbers_or_none:
            order = np.array(sorted(range(len(values)), key=lambda i: (values[i] is None, values[i] or 0)))
        else:
            order = np.arange(len(values))
        best_value = str(best[param])
    chosen = chosen.iloc[order]
    return {
        "x": x[order],
        "train_mean": chosen[f"mean_train_{REFIT_METRIC}"].to_numpy(dtype=float),
        "train_std": chosen[f"std_train_{REFIT_METRIC}"].to_numpy(dtype=float),
        "val_mean": chosen[f"mean_test_{REFIT_METRIC}"].to_numpy(dtype=float),
        "val_std": chosen[f"std_test_{REFIT_METRIC}"].to_numpy(dtype=float),
        "param_name": param,
        "best_value": best_value if best_value in list(x) else None,
        "log_x": None,
    }


def _confusion(metrics: dict[str, Any]) -> np.ndarray:
    return np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])


def evaluation_figures(
    ranked: Sequence[ScoredModel], baselines: dict[str, ScoredModel], y_test: np.ndarray, dirs: ArtefactDirs
) -> list[str]:
    """F12-F14 and F16-F19 from the stored test probabilities and CV tables (nothing refitted)."""
    best = ranked[0]
    lookup = baselines[LOOKUP_BASELINE]
    shown = list(ranked) + [lookup]
    written: list[str] = []

    roc = {m.name: m.curves["roc"] for m in shown}
    auc = {m.name: m.evaluation["at_0.5"]["roc_auc"] for m in shown}
    written.append(_save_figure(plots.roc_curves(roc, auc), dirs, "roc").name)
    pr = {m.name: m.curves["pr"] for m in shown}
    ap = {m.name: m.evaluation["at_0.5"]["pr_auc"] for m in shown}
    written.append(_save_figure(plots.pr_curves(pr, float(y_test.mean()), ap), dirs, "pr").name)

    panels = {m.name: p for m in ranked if m.result is not None and (p := validation_panel(m.result)) is not None}
    if panels:
        written.append(_save_figure(plots.curves_grid(panels, kind="validation"), dirs, "validation").name)

    matrices = {
        f"{plots.display_name(best.name)}\nthreshold 0.50": _confusion(best.evaluation["at_0.5"]),
        f"{plots.display_name(best.name)}\nspecificity 0.40 threshold {best.thresholds[OPERATING_KEY]:.3f}": _confusion(best.evaluation[f"at_{OPERATING_KEY}"]),
        "Lookup baseline\nthreshold 0.50": _confusion(lookup.evaluation["at_0.5"]),
        f"Lookup baseline\nspecificity 0.40 threshold {lookup.thresholds[OPERATING_KEY]:.3f}": _confusion(lookup.evaluation[f"at_{OPERATING_KEY}"]),
    }
    written.append(_save_figure(plots.confusion_matrices(matrices), dirs, "confusion").name)

    sweep = threshold_sweep(y_test, best.test_proba)
    chosen = {
        "Specificity 0.40": best.thresholds[OPERATING_KEY],
        "Flag rate 50 %": best.thresholds[FLAG_KEY],
        "Cost-optimal 3:1": best.thresholds["cost_optimal"]["3"],
    }
    floor = cost_from_counts(0, int((y_test == 0).sum()), int(y_test.size), COST_RATIOS[0])
    written.append(_save_figure(plots.threshold_cost_curve(sweep, chosen=chosen, cost_floor=floor), dirs, "cost", best.name).name)
    _write_csv(sweep, dirs.metrics / f"threshold_sweep_test_{best.name}.csv", index=False)

    bins = {m.name: m.curves["calibration"] for m in shown}
    brier = {m.name: m.evaluation["at_0.5"]["brier"] for m in shown}
    written.append(_save_figure(plots.calibration(bins, brier), dirs, "calibration").name)
    return written


def learning_curve_figure(best: ScoredModel, data: StudyData, cv: BaseCrossValidator, dirs: ArtefactDirs, n_jobs: int) -> dict[str, Any]:
    """F15: ``sklearn.model_selection.learning_curve`` for the best model on the 10,000-row quick subsample."""
    assert best.result is not None
    train_idx = data.train_idx
    y_train = data.y[train_idx]
    subset = train_idx[stratified_subsample(y_train, QUICK_SAMPLE_CAP)]
    sizes, train_scores, val_scores = learning_curve(
        clone(best.result.best_pipeline), data.X.iloc[subset], data.y[subset], groups=data.groups[subset],
        train_sizes=LEARNING_CURVE_SIZES, cv=cv, scoring=REFIT_METRIC, n_jobs=n_jobs, random_state=RANDOM_STATE,
    )
    table = pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1, ddof=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1, ddof=1),
        }
    )
    _write_csv(table, dirs.cv / f"{best.name}_learning_curve.csv", index=False)
    fig = plots.learning_curve(table["train_size"], table["train_mean"], table["train_std"], table["val_mean"], table["val_std"])
    _save_figure(fig, dirs, "learning", best.name)
    return {"model": best.name, "n_rows": int(len(subset)), "cv_folds": int(cv.get_n_splits()), "table": table.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Stage 9: temporal split, explanations, success criteria
# ---------------------------------------------------------------------------


def _variant_frame(data: StudyData, name: str | None) -> pd.DataFrame:
    """The primary inputs (``None``) or a named altered frame, rebuilt on the fly when absent."""
    if name is None:
        return data.X
    return data.variants[name] if name in data.variants else simple_variants(data.X)[name]


def _cv_average_precision(result: TuneResult) -> float:
    """Cross-validated mean average precision of the selected candidate (rank 1 in the table)."""
    return float(result.cv_results_df.iloc[0][f"mean_test_{REFIT_METRIC}"])


def temporal_evaluation(
    specs: Sequence[ModelSpec], data: StudyData, cv: BaseCrossValidator, quick: bool, dirs: ArtefactDirs, n_jobs: int
) -> dict[str, Any]:
    """The primary protocol repeated on CAL_YR < 2020, scored once on CAL_YR >= 2020.

    Everything that the primary run learns from labels is learnt again here from the 2010-2019
    rows alone: every model is re-tuned with the same grid and grouped cross-validation, the two
    best by cross-validated average precision are kept, and each one's thresholds come from
    out-of-fold probabilities on those rows. (Reusing the primary run's thresholds, grid choice
    or ranking would not be forward-in-time: 79.5 % of the 2020-2025 rows are training rows of
    the primary split, so their labels would already have set them.) The selected pipelines are
    scored on 2020-2025 with and without the dated 103(i) status and with and without
    ``immed_reportable``. Level drift is measured as one-hot columns that are all zero in the
    later period, which includes the retired scoop-car occupation level of rule E28.
    """
    train_idx, test_idx = temporal_split(data.meta)
    y_train, y_test = data.y[train_idx], data.y[test_idx]
    g_train = data.groups[train_idx]
    X_train = data.X.iloc[train_idx]
    sector_test = data.sector[test_idx]
    payload: dict[str, Any] = {
        "split_year": TEMPORAL_SPLIT_YEAR,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "protocol": (
            "the primary protocol repeated on the 2010-2019 rows only: every model re-tuned with the same "
            "grid and grouped cross-validation, the two best by cross-validated average precision kept, "
            "thresholds from out-of-fold probabilities on the 2010-2019 rows, 2020-2025 labels used only "
            "for the final scores"
        ),
        "thresholds": "out-of-fold thresholds of the 2010-2019 fit rows, applied unchanged to 2020-2025",
        "models": {},
    }
    preprocessor = build_preprocessor().fit(X_train)
    transformed = preprocessor.transform(data.X.iloc[test_idx])
    names = get_feature_names(preprocessor)
    onehot = [i for i, name in enumerate(names) if name not in FEATURE_SPEC["numeric"] and name not in FEATURE_SPEC["binary"]]
    all_zero = [names[i] for i in onehot if transformed[:, i].sum() == 0]
    payload["level_drift"] = {
        "n_onehot_columns": len(onehot),
        "n_all_zero_in_test_period": len(all_zero),
        "columns": all_zero,
        "e28_retired_occupation_column_all_zero": "OCCUPATION_missing" in all_zero,
    }

    tuned: dict[str, TuneResult] = {}
    for spec in specs:
        tuned[spec.name] = tune(spec, X_train, y_train, g_train, cv, quick=quick, n_jobs=n_jobs)
    selected = sorted(tuned.values(), key=lambda r: (-_cv_average_precision(r), r.model))
    payload["selection"] = {
        "rule": f"cross-validated mean {REFIT_METRIC} on the 2010-2019 rows",
        "ranking": [
            {"model": r.model, "cv_mean_average_precision": _cv_average_precision(r), "best_params": estimator_params(r)}
            for r in selected
        ],
    }

    for result in selected[:2]:
        entry: dict[str, Any] = {"n_fit_rows": int(len(result.fit_idx)), "best_params": estimator_params(result)}
        for variant, variant_name in TEMPORAL_VARIANTS.items():
            X_variant = _variant_frame(data, variant_name)
            started = time.perf_counter()
            if variant_name is None:
                pipeline, oof = result.best_pipeline, result.oof_proba
            else:
                pipeline, oof = out_of_fold_probabilities(
                    result.best_pipeline, X_variant.iloc[train_idx], y_train, g_train, result.fit_idx, cv, n_jobs
                )
            thresholds = oof_thresholds(y_train, oof)
            proba = pipeline.predict_proba(X_variant.iloc[test_idx])[:, 1]
            operating = float(thresholds[OPERATING_KEY])
            sectors = by_sector(y_test, proba, sector_test, operating)
            entry[variant] = {
                "fit_seconds": round(time.perf_counter() - started, 1),
                "thresholds": {OPERATING_KEY: thresholds[OPERATING_KEY], FLAG_KEY: thresholds[FLAG_KEY]},
                "test": {
                    "at_0.5": classification_metrics(y_test, proba, 0.5),
                    f"at_{OPERATING_KEY}": classification_metrics(y_test, proba, operating),
                    f"at_{FLAG_KEY}": classification_metrics(y_test, proba, float(thresholds[FLAG_KEY])),
                },
                "by_sector_at_operating": sectors.to_dict(orient="index"),
                "brier_by_sector": sectors["brier"].to_dict(),
            }
            log.info(
                "temporal %-14s %-26s AUC %.4f PR-AUC %.4f Brier %.4f spec@op %.4f",
                result.model, variant, entry[variant]["test"]["at_0.5"]["roc_auc"],
                entry[variant]["test"]["at_0.5"]["pr_auc"], entry[variant]["test"]["at_0.5"]["brier"],
                entry[variant]["test"][f"at_{OPERATING_KEY}"]["specificity"],
            )
        reference = _temporal_headline(entry["with_status_103i"])
        for variant in ("without_status_103i", "without_immed_reportable"):
            entry[f"delta_{variant}"] = delta(_temporal_headline(entry[variant]), reference)
        payload["models"][result.model] = entry
    _write_json(dirs.metrics / "temporal_split.json", payload)
    return payload


def _temporal_headline(entry: dict[str, Any]) -> dict[str, Any]:
    """``headline`` of one temporal variant entry."""
    return headline(entry["test"])


def run_sensitivity(
    ranked: Sequence[ScoredModel],
    baselines: dict[str, ScoredModel],
    data: StudyData,
    cv: BaseCrossValidator,
    quick: bool,
    dirs: ArtefactDirs,
    n_jobs: int,
) -> dict[str, Any]:
    """Deterministic levels, altered inputs and a mine-grouped split for the selected model.

    Writes reports/metrics/sensitivity.json with four blocks: ``pure_levels`` (levels that are
    100 % one class, documented or not), ``excluding_pure_levels`` (held-out metrics of every
    model and baseline with and without those rows), ``primary_split_variants`` (the selected
    pipeline refitted without ``immed_reportable``, without the 103(i) status, with resolved-case
    lag counts, with blanked early-January exposure and with the case history blanked on the
    training rows whose previous mine-year is a test mine-year; each variant records how many
    training rows it altered) and ``mine_grouped_split`` (S1-S6 when the split and the
    cross-validation are grouped by ``MINE_ID``). The variants come from ``data.variants``, which
    ``build_study_data`` builds with the same ``data.split`` that is used here.
    """
    best = ranked[0]
    assert best.result is not None and best.spec is not None
    train_idx, test_idx = data.train_idx, data.test_idx
    y_test = data.y[test_idx]
    levels = pure_levels(data.X, data.y)
    everyone = [*ranked, *baselines.values()]
    payload: dict[str, Any] = {
        "model": best.name,
        "pure_levels": {
            "rule": f"levels of categorical and binary features with at least {PURE_LEVEL_MIN_ROWS} rows that are 100 % one class",
            "found": levels.to_dict(orient="records"),
            "undocumented": levels.loc[~levels["documented"], ["column", "level"]].to_dict(orient="records"),
        },
        "excluding_pure_levels": excluding_levels(
            data.X.iloc[test_idx], y_test,
            {m.name: m.test_proba for m in everyone}, {m.name: m.thresholds for m in everyone}, levels,
        ),
    }
    reference = headline(best.evaluation)
    payload["primary_split_variants"] = refit_variants(
        best.result.best_pipeline, best.result.fit_idx, data.variants, data.X, reference,
        data.y, data.groups, train_idx, test_idx, data.sector, cv, n_jobs,
    )
    payload["mine_grouped_split"] = mine_grouped_robustness(
        best.spec, best.result.best_pipeline, data.X, data.y,
        data.meta["MINE_ID"].to_numpy(), data.meta["accident_group"].to_numpy(), data.sector, reference,
        cv_folds=cv.get_n_splits(), quick=quick, n_jobs=n_jobs,
    )
    _write_json(dirs.metrics / "sensitivity.json", payload)
    return payload


def explanations(
    ranked: Sequence[ScoredModel],
    data: StudyData,
    dirs: ArtefactDirs,
    n_jobs: int,
    quick: bool = False,
    codebook: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Permutation importance (best model, 5,000 test rows), coefficient tables and a SHAP attempt.

    Quick mode uses ``QUICK_EXPLANATION_ROWS`` rows for the permutation importance and the SHAP sample.
    ``codebook`` (``plots.code_descriptions`` of the cleaned base) decodes one-hot levels in F17/F17b.
    """
    best = ranked[0]
    assert best.result is not None
    X_test, y_test = data.X.iloc[data.test_idx], data.y[data.test_idx]
    payload: dict[str, Any] = {"best_model": best.name}

    permutation_rows = min(PERMUTATION_ROWS, QUICK_EXPLANATION_ROWS) if quick else PERMUTATION_ROWS
    permutation = permutation_importance_table(best.result.best_pipeline, X_test, y_test, max_rows=permutation_rows, n_jobs=n_jobs)
    _write_csv(permutation, dirs.metrics / f"permutation_importance_{best.name}.csv", index=False)
    importance = permutation.rename(columns={"importance_mean": "importance", "importance_std": "std"})
    _save_figure(plots.feature_importance(importance, codebook=codebook), dirs, "importance", best.name)
    payload["permutation_importance"] = {"rows": int(permutation.attrs["n_rows"]), "top_10": permutation.head(10).to_dict(orient="records")}

    by_name = {m.name: m for m in ranked}
    for name in dict.fromkeys(["logreg", best.name]):
        model = by_name.get(name)
        if model is None or model.result is None:
            continue
        table = coefficient_table(model.result.best_pipeline)
        if table is None:
            payload[f"estimator_table_{name}"] = None
            continue
        kind = table.attrs.get("kind", "coefficient")
        _write_csv(table, dirs.metrics / f"{kind}s_{name}.csv", index=False)
        payload[f"estimator_table_{name}"] = {"kind": kind, "top_10": table.head(10).to_dict(orient="records")}
        if name == "logreg":
            fig = plots.feature_importance(
                table.rename(columns={"coefficient": "importance"}),
                value_label="Logistic regression coefficient (standardised inputs)",
                codebook=codebook,
            )
            _save_figure(fig, dirs, "coefficients")

    summary = shap_summary(best.result.best_pipeline, X_test, max_rows=min(SHAP_ROWS, QUICK_EXPLANATION_ROWS) if quick else SHAP_ROWS)
    if summary is None:
        payload["shap"] = {"available": False, "reason": "no fast SHAP explainer for this estimator"}
    else:
        _write_csv(summary.table, dirs.metrics / f"shap_summary_{best.name}.csv", index=False)
        payload["shap"] = {"available": True, "explainer": summary.explainer, "rows": int(summary.values.shape[0]), "top_10": summary.table.head(10).to_dict(orient="records")}
    _write_json(dirs.metrics / "explanations.json", payload)
    return payload


def success_criteria(
    ranked: Sequence[ScoredModel],
    baselines: dict[str, ScoredModel],
    temporal: dict[str, Any],
    y_test: np.ndarray,
    dirs: ArtefactDirs,
    sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """S1-S7 of decisions.json (success_criteria) for every model, evaluated on the held-out rows.

    ``temporal`` supplies the per-sector Brier scores that S6 also checks for the models that
    were evaluated on the temporal split. ``sensitivity`` adds the same criteria for the split
    grouped by mine, next to the primary ones.
    """
    lookup = baselines[LOOKUP_BASELINE]
    pass_ratios = [int(r) for r in SUCCESS_CRITERIA["cost_pass_ratios"]]
    floor = all_positive_floor(y_test)
    lookup_cost = {k: v["cost_per_1000"] for k, v in lookup.evaluation["cost_optimal"].items()}
    payload: dict[str, Any] = {
        "criteria": {
            "S1": f"test ROC-AUC >= {SUCCESS_CRITERIA['roc_auc_min']}",
            "S2": f"test PR-AUC >= {SUCCESS_CRITERIA['pr_auc_min']}",
            "S3": f"test recall >= {SUCCESS_CRITERIA['recall_min_at_operating_point']} at the out-of-fold threshold for specificity {config.OPERATING_SPECIFICITY}",
            "S4": f"test cost per 1,000 at the out-of-fold cost-optimal threshold below the all-positive floor and below the lookup baseline at ratios {pass_ratios}",
            "S5": f"test recall >= {SUCCESS_CRITERIA['recall_at_50pct_flag_rate_min']} at the out-of-fold threshold for a 50 % flag rate",
            "S6": f"test Brier <= {SUCCESS_CRITERIA['brier_max']} (and per sector on the temporal split where evaluated)",
            "S7": "McNemar vs the lookup baseline and paired fold test between the top two (statistical_tests.json)",
        },
        "all_positive_floor_test": floor,
        "lookup_cost_optimal_test": lookup_cost,
        "models": {},
    }
    for model in list(ranked) + [lookup]:
        temporal_brier: dict[str, float] = {}
        if model.name in temporal.get("models", {}):
            temporal_brier = temporal["models"][model.name]["with_status_103i"]["brier_by_sector"]
        payload["models"][model.name] = model_criteria(
            model.evaluation, model.thresholds, floor, lookup_cost,
            is_lookup=model.name == lookup.name, temporal_brier=temporal_brier,
        )
    if sensitivity is not None:
        grouped = sensitivity["mine_grouped_split"]
        payload["robustness_mine_grouped_split"] = {
            "note": "split and cross-validation grouped by MINE_ID; the selected model and the lookup baseline refitted",
            "model": sensitivity["model"],
            "n_train": grouped["n_train"],
            "n_test": grouped["n_test"],
            "criteria": {sensitivity["model"]: grouped["model"]["criteria"], lookup.name: grouped["lookup"]["criteria"]},
        }
    _write_json(dirs.metrics / "success_criteria.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_models(models: Sequence[str] | None) -> list[ModelSpec]:
    names = list(MODEL_ZOO) if models is None else list(models)
    unknown = [n for n in names if n not in MODEL_ZOO]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {list(MODEL_ZOO)}")
    if not names:
        raise ValueError("at least one model is required")
    return [MODEL_ZOO[n] for n in names]


def _artefact_listing(dirs: ArtefactDirs) -> list[dict[str, Any]]:
    """Every file under the run's directories with its size, for the manifest."""
    listing: list[dict[str, Any]] = []
    for directory in dict.fromkeys((dirs.processed, dirs.models, dirs.metrics, dirs.cv, dirs.predictions, dirs.figures)):
        for path in sorted(directory.glob("*")):
            if path.is_file():
                listing.append({"directory": directory.name, "file": path.name, "bytes": path.stat().st_size})
    return listing


def train_all(
    quick: bool = False,
    models: list[str] | None = None,
    *,
    base: pd.DataFrame | None = None,
    dirs: ArtefactDirs | None = None,
) -> dict[str, Any]:
    """Run the whole study and write every artefact; returns the run manifest.

    ``quick`` shrinks every search to two candidates on 10,000 stratified training rows with
    three grouped folds. ``models`` restricts the zoo (names from ``MODEL_ZOO``). ``base``
    bypasses ``load_base_table`` (tests pass the frame they built); ``dirs`` overrides the
    output directories, which otherwise come from ``mlda.config`` at call time.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    specs = _resolve_models(models)
    dirs = dirs or ArtefactDirs.from_config()
    dirs.create()
    timer = StageTimer()
    n_jobs = default_n_jobs()
    cv = cv_splitter(n_splits=QUICK_CV_FOLDS if quick else N_SPLITS)
    log.info("train_all: quick=%s, models=%s, cv folds=%d, n_jobs=%d", quick, [s.name for s in specs], cv.get_n_splits(), n_jobs)

    with timer.stage("load_and_clean"):
        clean, rule_log = load_and_clean(dirs, base)
    with timer.stage("features_and_split"):
        data = build_study_data(clean, dirs)
        # Width of the default one-hot design matrix on the training rows (reported, not reused).
        preprocessor_width = len(get_feature_names(build_preprocessor().fit(data.X.iloc[data.train_idx])))
        log.info("default preprocessor on the training rows: %d transformed columns", preprocessor_width)
    with timer.stage("eda"):
        eda = run_eda(data, clean, dirs, quick=quick)
    codebook = plots.code_descriptions(clean)
    _write_json(dirs.metrics / "code_descriptions.json", codebook)
    del clean
    with timer.stage("baselines"):
        baselines = run_baselines(data, cv, dirs)

    scored: list[ScoredModel] = []
    for spec in specs:
        with timer.stage(f"model_{spec.name}"):
            scored.append(run_model(spec, data, cv, quick, dirs, n_jobs))
    ranked = rank_models(scored)
    y_test = data.y[data.test_idx]

    with timer.stage("predictions_and_comparison"):
        write_predictions(data, ranked, baselines, dirs)
        comparison = write_comparison(ranked, baselines, dirs)
        tests = statistical_tests(
            ranked, baselines[LOOKUP_BASELINE], y_test, dirs, groups_test=data.groups[data.test_idx], quick=quick
        )
    with timer.stage("evaluation_figures"):
        figures = evaluation_figures(ranked, baselines, y_test, dirs)
    with timer.stage("learning_curve"):
        learning = learning_curve_figure(ranked[0], data, cv, dirs, n_jobs)
    with timer.stage("temporal_split"):
        temporal = temporal_evaluation(specs, data, cv, quick, dirs, n_jobs)
    with timer.stage("sensitivity"):
        sensitivity = run_sensitivity(ranked, baselines, data, cv, quick, dirs, n_jobs)
    with timer.stage("explanations"):
        explained = explanations(ranked, data, dirs, n_jobs, quick=quick, codebook=codebook)
    with timer.stage("success_criteria"):
        criteria = success_criteria(ranked, baselines, temporal, y_test, dirs, sensitivity)

    manifest: dict[str, Any] = {
        "student_id": STUDENT_ID,
        "quick": quick,
        "models": [s.name for s in specs],
        "best_model": ranked[0].name,
        "ranking_by_cv_average_precision": [m.name for m in ranked],
        "n_rows": int(len(data.y)),
        "prevalence": float(data.y.mean()),
        "n_train": int(len(data.train_idx)),
        "n_test": int(len(data.test_idx)),
        "prevalence_train": float(data.y[data.train_idx].mean()),
        "prevalence_test": float(y_test.mean()),
        "n_input_features": int(data.X.shape[1]),
        "n_transformed_features_default_preprocessor": int(preprocessor_width),
        "cv_folds": int(cv.get_n_splits()),
        "quick_sample_cap": QUICK_SAMPLE_CAP if quick else None,
        "n_jobs": n_jobs,
        "error_rules_logged": int(len(rule_log)),
        "eda": eda,
        "learning_curve": {k: v for k, v in learning.items() if k != "table"},
        "statistical_tests": {k: v for k, v in tests.items() if k != "ranking"},
        "temporal_models": list(temporal["models"]),
        "undocumented_pure_levels": sensitivity["pure_levels"]["undocumented"],
        "success_all_pass": {name: checks["all_pass"] for name, checks in criteria["models"].items()},
        "comparison_at_0.5": comparison.round(4).to_dict(orient="index"),
        "evaluation_figures": figures,
        "explanations": {"best_model": explained["best_model"], "shap": explained["shap"]},
        "stage_seconds": timer.seconds,
        "total_seconds": round(time.perf_counter() - started, 1),
        "environment": environment_record(),
        "artefacts": _artefact_listing(dirs),
    }
    _write_json(dirs.metrics / "run_manifest.json", manifest)
    log.info("train_all finished in %.1f s; best model %s", manifest["total_seconds"], ranked[0].name)
    return manifest


__all__ = [
    "FIGURES",
    "FLAG_KEY",
    "LOOKUP_BASELINE",
    "FLAG_RATE_TARGET",
    "OPERATING_KEY",
    "ArtefactDirs",
    "ScoredModel",
    "StudyData",
    "build_study_data",
    "evaluate_on_test",
    "evaluation_figures",
    "explanations",
    "learning_curve_figure",
    "load_and_clean",
    "oof_thresholds",
    "rank_models",
    "run_baselines",
    "run_eda",
    "run_model",
    "run_sensitivity",
    "statistical_tests",
    "success_criteria",
    "temporal_evaluation",
    "train_all",
    "validation_panel",
    "write_comparison",
    "write_predictions",
]
