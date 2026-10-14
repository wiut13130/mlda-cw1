"""Every figure builder returns a Figure and, when asked, writes a 200 dpi PNG into FIGURES_DIR."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image

import mlda.plots as plots
from mlda.config import COST_RATIOS, RANDOM_STATE
from mlda.eda import correlations

NUMERIC = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "prev_hours", "hours_into_shift"]
CONTRACT_BUILDERS = [
    "target_distribution", "histograms_grid", "class_boxplots", "pairplot_top", "correlation_heatmaps",
    "categorical_vs_target", "missingness", "qq_plots", "outlier_fences", "pca_projection", "roc_curves",
    "pr_curves", "validation_curve", "learning_curve", "confusion_matrices", "feature_importance",
    "threshold_cost_curve", "calibration",
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 500
    nature = rng.choice(["330", "180", "220", "320", "160"], size=n, p=[0.35, 0.25, 0.15, 0.1, 0.15])
    sector = rng.choice(["C", "M"], size=n)
    rate = pd.Series(nature).map({"330": 0.7, "180": 0.2, "220": 0.6, "320": 0.1, "160": 0.5}).to_numpy()
    df = pd.DataFrame(
        {
            "TOT_EXPER": rng.exponential(8, n),
            "MINE_EXPER": rng.exponential(4, n),
            "JOB_EXPER": rng.exponential(3, n),
            "prev_hours": rng.lognormal(11, 1.2, n),
            "hours_into_shift": rng.uniform(0, 12, n),
            "NATURE_INJURY_CD": nature,
            "COAL_METAL_IND": sector,
            "lost_time": (rng.random(n) < rate).astype("int8"),
        }
    )
    df.loc[rng.choice(n, 25, replace=False), "TOT_EXPER"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "NATURE_INJURY_CD"] = np.nan
    return df


@pytest.fixture
def figures_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "figures"  # deliberately absent: save_figure must create it
    monkeypatch.setattr(plots, "FIGURES_DIR", target)
    return target


def _sweep() -> pd.DataFrame:
    thr = np.linspace(0, 1, 51)
    sweep = pd.DataFrame(
        {
            "threshold": thr, "recall": 1 - thr, "specificity": thr, "precision": 0.5 + 0.4 * thr,
            "f1": 0.6 - 0.3 * (thr - 0.4) ** 2, "flag_rate": 1 - thr,
        }
    )
    for ratio in COST_RATIOS:
        sweep[f"cost_per_1000_at_{ratio}"] = 100_000 * (1 + ratio * thr)
    return sweep


def _builders(df: pd.DataFrame) -> dict[str, Callable[..., Figure]]:
    """Every contract builder bound to synthetic inputs; only `save` is left open."""
    rng = np.random.default_rng(RANDOM_STATE)
    corr = correlations(df, NUMERIC)
    fpr = np.linspace(0, 1, 40)
    bins = pd.DataFrame({"mean_predicted": np.linspace(0.05, 0.95, 10), "fraction_positive": np.linspace(0.02, 0.98, 10), "n": rng.integers(10, 400, 10)})
    sklearn_bins = pd.DataFrame({"prob_pred": np.linspace(0.1, 0.9, 5), "prob_true": [0.2, 0.3, 0.55, 0.8, 0.85]})
    return {
        "target_distribution": lambda **kw: plots.target_distribution(df["lost_time"], **kw),
        "histograms_grid": lambda **kw: plots.histograms_grid(df, NUMERIC, hue="lost_time", **kw),
        "class_boxplots": lambda **kw: plots.class_boxplots(df, NUMERIC, "lost_time", **kw),
        "pairplot_top": lambda **kw: plots.pairplot_top(df, NUMERIC[:3], "lost_time", sample_n=200, **kw),
        "correlation_heatmaps": lambda **kw: plots.correlation_heatmaps(corr.pearson, corr.spearman, **kw),
        "categorical_vs_target": lambda **kw: plots.categorical_vs_target(df, ["NATURE_INJURY_CD", "COAL_METAL_IND"], "lost_time", min_n=30, **kw),
        "missingness": lambda **kw: plots.missingness(df, **kw),
        "qq_plots": lambda **kw: plots.qq_plots(df, NUMERIC[:3], sample_n=200, **kw),
        "outlier_fences": lambda **kw: plots.outlier_fences(df, NUMERIC[:2], after=df[NUMERIC[:2]].apply(np.log1p), **kw),
        "pca_projection": lambda **kw: plots.pca_projection(df[NUMERIC], df["lost_time"], sample_n=300, **kw),
        "roc_curves": lambda **kw: plots.roc_curves(
            {"logreg": (fpr, np.sqrt(fpr)), "majority": pd.DataFrame({"fpr": fpr, "tpr": fpr})}, auc={"logreg": 0.8}, **kw
        ),
        "pr_curves": lambda **kw: plots.pr_curves({"hist_gb": (fpr, 1 - 0.5 * fpr)}, prevalence=0.486, average_precision={"hist_gb": 0.77}, **kw),
        "validation_curve": lambda **kw: plots.validation_curve(
            [0.01, 0.1, 1, 10], [0.7, 0.75, 0.8, 0.85], [0.01] * 4, [0.69, 0.74, 0.76, 0.75], [0.02] * 4,
            "C (inverse regularisation strength)", best_value=1, **kw
        ),
        "learning_curve": lambda **kw: plots.learning_curve([1000, 5000, 20000], [0.9, 0.85, 0.82], [0.01] * 3, [0.7, 0.74, 0.77], [0.02] * 3, **kw),
        "confusion_matrices": lambda **kw: plots.confusion_matrices(
            {"Threshold 0.50": np.array([[400, 100], [80, 420]]), "Tuned threshold 0.31": np.array([[300, 200], [30, 470]])}, **kw
        ),
        "feature_importance": lambda **kw: plots.feature_importance(
            pd.DataFrame({"feature": [f"f{i}" for i in range(25)], "importance": rng.normal(size=25), "std": rng.random(25) * 0.1}), **kw
        ),
        "threshold_cost_curve": lambda **kw: plots.threshold_cost_curve(_sweep(), chosen={"Specificity 0.40": 0.4}, cost_floor=102_829, **kw),
        "calibration": lambda **kw: plots.calibration({"logreg": bins, "lookup_nature_bodypart_coal": sklearn_bins}, brier={"logreg": 0.18}, **kw),
    }


def test_every_contract_builder_exists() -> None:
    for name in CONTRACT_BUILDERS:
        assert callable(getattr(plots, name)), name


@pytest.mark.parametrize("name", CONTRACT_BUILDERS)
def test_builder_returns_figure_and_writes_200_dpi_png(name: str, frame: pd.DataFrame, figures_dir: Path) -> None:
    build = _builders(frame)[name]
    fig = build()
    assert isinstance(fig, Figure)
    assert not figures_dir.exists()  # no save requested, nothing written

    saved = build(save=f"{name}.png")
    assert isinstance(saved, Figure)
    path = figures_dir / f"{name}.png"
    assert path.is_file() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        assert dpi is not None and dpi[0] == pytest.approx(200, abs=1)
        assert image.width <= 6.3 * 200 * 1.15  # print width: A4 text width plus tight-bbox padding


def test_save_appends_png_suffix_and_creates_nested_folders(frame: pd.DataFrame, figures_dir: Path) -> None:
    plots.target_distribution(frame["lost_time"], save="eda/target")
    assert (figures_dir / "eda" / "target.png").is_file()


def test_save_rejects_absolute_paths(frame: pd.DataFrame, figures_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plots.target_distribution(frame["lost_time"], save=str(tmp_path / "elsewhere.png"))


def test_backend_is_non_interactive() -> None:
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def test_axis_labels_are_british_with_units() -> None:
    assert plots.axis_label("TOT_EXPER") == "Total mining experience (years)"
    assert plots.axis_label("prev_hours").endswith("(hours)")
    assert plots.axis_label("unknown_column") == "unknown column"
    assert plots.display_name("logreg") == "Logistic regression"
    for label in plots.LABELS.values():
        assert "color" not in label and "normalize" not in label


def test_figure_axis_labels_use_units(frame: pd.DataFrame) -> None:
    fig = plots.histograms_grid(frame, ["TOT_EXPER"], hue="lost_time")
    assert "years" in fig.axes[0].get_xlabel()
    fig = plots.roc_curves({"logreg": (np.linspace(0, 1, 5), np.linspace(0, 1, 5))})
    assert fig.axes[0].get_xlabel() == "False positive rate (1 - specificity)"


def test_class_colours_are_consistent_across_figures(frame: pd.DataFrame) -> None:
    assert plots.CLASS_COLOURS[0] != plots.CLASS_COLOURS[1]
    assert plots.MODEL_COLOURS["logreg"] != plots.MODEL_COLOURS["knn"]
    fig = plots.target_distribution(frame["lost_time"])
    bars = [patch for patch in fig.axes[0].patches]
    assert tuple(bars[0].get_facecolor()[:3]) == pytest.approx(plots.CLASS_COLOURS[0])
    assert tuple(bars[1].get_facecolor()[:3]) == pytest.approx(plots.CLASS_COLOURS[1])


def test_threshold_cost_curve_finds_cost_columns_by_ratio() -> None:
    sweep = _sweep().rename(columns={f"cost_per_1000_at_{r}": f"cost_{r}" for r in COST_RATIOS})
    fig = plots.threshold_cost_curve(sweep, cost_ratios=(2, 5))
    bottom = fig.axes[1]
    labels = [line.get_label() for line in bottom.get_lines()]
    assert labels == ["FN:FP = 2:1", "FN:FP = 5:1"]
    with pytest.raises(KeyError):
        plots.threshold_cost_curve(sweep.drop(columns=["cost_2"]), cost_ratios=(2,))


def test_validation_curve_handles_categorical_parameter_values() -> None:
    fig = plots.validation_curve(["gini", "entropy"], [0.7, 0.75], [0.01, 0.01], [0.69, 0.74], [0.02, 0.02], "criterion", best_value="entropy")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["gini", "entropy"]
    assert ax.get_xscale() == "linear"


def test_validation_curve_uses_log_axis_for_wide_numeric_grids() -> None:
    fig = plots.validation_curve([0.001, 0.01, 0.1, 1, 10], [0.7] * 5, [0.01] * 5, [0.7] * 5, [0.01] * 5, "C")
    assert fig.axes[0].get_xscale() == "log"


def test_confusion_matrices_reject_wrong_shape() -> None:
    with pytest.raises(ValueError):
        plots.confusion_matrices({"bad": np.zeros((3, 3))})


def test_curves_grid_one_panel_per_model() -> None:
    panels = {
        "logreg": {"x": [0.1, 1, 10], "train_mean": [0.7, 0.8, 0.85], "train_std": [0.01] * 3, "val_mean": [0.7, 0.76, 0.75], "val_std": [0.02] * 3, "param_name": "C", "best_value": 1},
        "knn": {"x": [5, 15, 51], "train_mean": [0.9, 0.8, 0.78], "train_std": [0.01] * 3, "val_mean": [0.7, 0.74, 0.76], "val_std": [0.02] * 3, "param_name": "n_neighbors"},
    }
    fig = plots.curves_grid(panels, kind="validation")
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    assert visible[0].get_title() == "Logistic regression"
    with pytest.raises(ValueError):
        plots.curves_grid(panels, kind="other")
