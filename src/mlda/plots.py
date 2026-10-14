"""Figure builders shared by the notebook, the report and the Streamlit app.

Every builder returns a matplotlib Figure and leaves the caller to decide what to
do with it (display in the notebook, `st.pyplot` in the app, embed in the report).
Passing `save="F02_target.png"` also writes a 200 dpi PNG into FIGURES_DIR, which
is created on demand. Figures are sized for the A4 text width (about 6.3 in), use
the seaborn colour-blind palette with a fixed colour per class and per model, and
carry British-English axis labels with units. The non-interactive Agg backend is
selected because the figures are produced on headless machines (CI, the Streamlit
server) and are never shown in a window; a notebook keeps its inline backend and
displays the returned Figure objects directly.

The evaluation builders (`roc_curves`, `validation_curve`, `confusion_matrices`,
`threshold_cost_curve`, `calibration` and friends) take precomputed arrays and
tables so that drawing never retrains a model or touches test rows.
"""

from __future__ import annotations

import functools
import re
import textwrap
import warnings
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mlda.config import COST_RATIOS, FIGURES_DIR, RANDOM_STATE
from mlda.eda import class_rate_by_level, missingness_table

_INTERACTIVE_NOTEBOOK_BACKENDS = ("inline", "ipympl", "nbagg", "widget")
if not any(tag in matplotlib.get_backend().lower() for tag in _INTERACTIVE_NOTEBOOK_BACKENDS):
    matplotlib.use("Agg")

WIDTH = 6.3  # A4 text width in inches with 2.5 cm margins
DPI = 200
PALETTE = sns.color_palette("colorblind")
GREY = PALETTE[7]
CLASS_LABELS = {0: "No lost time", 1: "Lost time"}
CLASS_COLOURS = {0: PALETTE[0], 1: PALETTE[3]}
# one colour per model everywhere, so a model keeps its identity across ROC, PR and
# calibration figures; baselines are grey and distinguished by line style
MODEL_LABELS = {
    "logreg": "Logistic regression",
    "knn": "K-nearest neighbours",
    "gaussian_nb": "Gaussian Naive Bayes",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gb": "Histogram gradient boosting",
    "svc_rbf": "SVM (RBF kernel)",
    "mlp": "Multilayer perceptron",
    "majority": "Baseline: majority class",
    "all_positive": "Baseline: flag every case",
    "rule_not_cut_or_dust": "Baseline: not cut, scratch or dust",
    "lookup_nature_bodypart_coal": "Baseline: nature x body part x sector lookup",
}
MODEL_COLOURS = {
    name: PALETTE[i]
    for i, name in enumerate(
        [
            "logreg",
            "knn",
            "gaussian_nb",
            "decision_tree",
            "random_forest",
            "hist_gb",
            "svc_rbf",
            "mlp",
        ]
    )
}
BASELINE_STYLES = {
    "majority": ":",
    "all_positive": "-.",
    "rule_not_cut_or_dust": "--",
    "lookup_nature_bodypart_coal": (0, (5, 1, 1, 1)),
}
LABELS: dict[str, str] = {
    "TOT_EXPER": "Total mining experience (years)",
    "MINE_EXPER": "Experience at this mine (years)",
    "JOB_EXPER": "Experience in this job (years)",
    "log1p_tot_exper": "log(1 + total mining experience in years)",
    "exper_ratio_mine": "Share of experience gained at this mine (ratio)",
    "exper_ratio_job": "Share of experience gained in this job (ratio)",
    "NO_INJURIES": "Persons injured in the accident (count)",
    "no_injuries_capped": "Persons injured in the accident (count, capped at 10)",
    "accident_hour": "Hour of the accident (0-23)",
    "shift_begin_hour": "Hour the shift began (0-23)",
    "hours_into_shift": "Time elapsed in the shift (hours)",
    "latitude": "Latitude (degrees north)",
    "longitude": "Longitude (degrees east)",
    "LATITUDE": "Latitude (degrees north)",
    "LONGITUDE": "Longitude (degrees east)",
    "prev_hours": "Employee-hours at the mine in the previous year (hours)",
    "log1p_prev_hours": "log(1 + employee-hours in the previous year)",
    "prev_employees": "Average employees at the mine in the previous year (persons)",
    "prev_hours_per_employee": "Hours per employee in the previous year (hours)",
    "prev_incidents": "Incidents reported at the mine in the previous year (count)",
    "prev_incident_rate_200k": "log(1 + incidents per 200,000 employee-hours, previous year)",
    "prev_lost_time_rate_200k": "log(1 + lost-time cases per 200,000 employee-hours, previous year)",
    "prev_lost_time_share": "Lost-time share of the mine's previous-year cases (ratio)",
    "same_year_hours": "Employee-hours at the mine in the accident year (hours)",
    "same_year_employees": "Average employees at the mine in the accident year (persons)",
    "DAYS_LOST": "Days away from work (days)",
    "DAYS_RESTRICT": "Days of restricted activity (days)",
    "CAL_YR": "Calendar year",
    "lost_time": "Lost-time injury",
    "SUBUNIT_CD": "Mine subunit",
    "FIPS_STATE_CD": "State (FIPS code)",
    "UG_LOCATION_CD": "Underground location",
    "UG_MINING_METHOD_CD": "Underground mining method",
    "MINING_EQUIP_CD": "Mining equipment involved",
    "EQUIP_MFR_NAME": "Equipment manufacturer",
    "CLASSIFICATION_CD": "Accident classification",
    "ACCIDENT_TYPE_CD": "Accident type",
    "OCCUPATION": "Occupation of the injured worker",
    "ACTIVITY_CD": "Activity at the time of the accident",
    "INJURY_SOURCE_CD": "Source of the injury",
    "NATURE_INJURY_CD": "Nature of the injury",
    "INJ_BODY_PART_CD": "Body part injured",
    "COAL_METAL_IND": "Sector (coal or metal/non-metal)",
    "PRIMARY_CANVASS": "Primary canvass (commodity group)",
    "type_at_time": "Mine type at the time of the accident",
    "status_103i_at_accident": "Section 103(i) status at the accident",
    "mine_size_band": "Mine size band (average employees, previous year)",
    "SAFETY_COMMITTEE_IND": "Safety committee at the mine",
    "PART48_TRAINING": "Part 48 training programme",
    "PORTABLE_OPERATION": "Portable operation",
    "accident_weekday": "Weekday of the accident (0 = Monday)",
    "accident_month": "Month of the accident (1-12)",
}
_STYLE_RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "axes.edgecolor": "0.3",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def axis_label(col: str) -> str:
    """British-English label with units for a known column; otherwise the name, spaced."""
    return LABELS.get(col, col.replace("_", " "))


def display_name(name: str) -> str:
    """Report-ready name of a model or baseline key (falls back to the key itself)."""
    return MODEL_LABELS.get(name, name)


def _wrap(label: str, width: int = 34) -> str:
    """Break a long axis label over several lines so small grid panels do not overlap."""
    return textwrap.fill(label, width)


def _compact_ticks(ax: Axes, values: np.ndarray, axis: str = "x") -> None:
    """Use 1k / 1M style tick labels for large magnitudes instead of the 1e6 offset text."""
    if values.size and np.nanmax(np.abs(values)) >= 1e4:
        formatter = EngFormatter(sep="")
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def _styled[**P](func: Callable[P, Figure]) -> Callable[P, Figure]:
    """Apply the shared seaborn style while a builder draws, without changing global rcParams."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        with sns.axes_style("whitegrid", _STYLE_RC), sns.plotting_context("paper", font_scale=0.95):
            return func(*args, **kwargs)

    return wrapper


def save_figure(fig: Figure, name: str, dpi: int = DPI) -> Path:
    """Write `fig` as a PNG under FIGURES_DIR (created on demand) and return the path.

    `name` is a file name relative to FIGURES_DIR; ".png" is appended when missing.
    """
    if Path(name).is_absolute():
        raise ValueError("save takes a file name relative to FIGURES_DIR, not an absolute path")
    path = FIGURES_DIR / (name if name.lower().endswith(".png") else f"{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _finish(fig: Figure, save: str | None) -> Figure:
    if save:
        save_figure(fig, save)
    return fig


def _figure(
    width: float, height: float, nrows: int = 1, ncols: int = 1, **kwargs: Any
) -> tuple[Figure, Any]:
    """Figure with constrained layout built without pyplot, so nothing leaks into the global state."""
    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(nrows, ncols, **kwargs)
    return fig, axes


def _grid(
    n_panels: int, ncols: int, panel_height: float, **kwargs: Any
) -> tuple[Figure, list[Axes]]:
    """Grid of small panels; surplus axes are hidden so the last row can be partial."""
    ncols = max(1, min(ncols, n_panels))
    nrows = ceil(n_panels / ncols)
    fig, axes = _figure(WIDTH, panel_height * nrows + 0.2, nrows, ncols, squeeze=False, **kwargs)
    flat = list(axes.flat)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


def _heatmap(**kwargs: Any) -> Axes:
    """sns.heatmap without the PendingDeprecationWarning seaborn 0.13 raises on matplotlib 3.11."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return sns.heatmap(**kwargs)


def _model_colour(name: str, position: int) -> tuple[float, float, float]:
    if name in BASELINE_STYLES:
        return GREY
    return MODEL_COLOURS.get(name, PALETTE[position % len(PALETTE)])


def _model_style(name: str) -> Any:
    return BASELINE_STYLES.get(name, "-")


def _xy(curve: Any, x_keys: Sequence[str], y_keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Accept a (x, y) pair or a mapping/DataFrame holding the named columns."""
    if isinstance(curve, (pd.DataFrame, Mapping)):
        x_key = next((k for k in x_keys if k in curve), None)
        y_key = next((k for k in y_keys if k in curve), None)
        if x_key is None or y_key is None:
            raise KeyError(f"curve needs columns {x_keys[0]} and {y_keys[0]}; got {list(curve)}")
        return np.asarray(curve[x_key], dtype=float), np.asarray(curve[y_key], dtype=float)
    x, y = curve[0], curve[1]
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _class_frame(df: pd.DataFrame, cols: Sequence[str], target: str) -> pd.DataFrame:
    """Copy of the needed columns with the target replaced by its display label."""
    frame = df[[*cols, target]].copy()
    frame[target] = frame[target].map(CLASS_LABELS).astype("object")
    return frame


_CLASS_PALETTE = {CLASS_LABELS[k]: v for k, v in CLASS_COLOURS.items()}
_CLASS_ORDER = [CLASS_LABELS[0], CLASS_LABELS[1]]


# ---------------------------------------------------------------------------
# EDA figures (F2-F11)
# ---------------------------------------------------------------------------


@_styled
def target_distribution(y: pd.Series | np.ndarray, save: str | None = None) -> Figure:
    """F2: bar chart of the two outcome classes with counts and percentages."""
    counts = pd.Series(np.asarray(y)).value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    fig, ax = _figure(WIDTH * 0.62, 2.9)
    bars = ax.bar(
        _CLASS_ORDER, counts.to_numpy(), color=[CLASS_COLOURS[0], CLASS_COLOURS[1]], width=0.6
    )
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        share = 100.0 * value / total if total else 0.0
        ax.annotate(
            f"{value:,}\n({share:.1f} %)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_ylim(0, counts.max() * 1.18 if total else 1)
    ax.set_xlabel("Injury outcome")
    ax.set_ylabel("Cases (count)")
    ax.grid(axis="x", visible=False)
    return _finish(fig, save)


@_styled
def histograms_grid(
    df: pd.DataFrame,
    cols: Sequence[str],
    bins: int | Sequence[float] = 30,
    kde: bool = True,
    hue: str | None = None,
    ncols: int = 3,
    save: str | None = None,
) -> Figure:
    """F3: density histograms (with KDE) of numeric columns, optionally split by the target.

    Density rather than frequency keeps the two classes comparable when their
    sizes differ, and the KDE smooths the bin-count sensitivity of the histogram.
    """
    fig, axes = _grid(len(cols), ncols, 2.2)
    frame = df
    palette = None
    if hue is not None and set(df[hue].dropna().unique()) <= {0, 1}:
        frame = _class_frame(df, cols, hue)
        palette = _CLASS_PALETTE
    for i, (ax, col) in enumerate(zip(axes, cols, strict=True)):
        sns.histplot(
            data=frame,
            x=col,
            hue=hue,
            bins=bins,
            stat="density",
            common_norm=False,
            kde=kde,
            element="step",
            alpha=0.35,
            palette=palette,
            hue_order=_CLASS_ORDER if palette else None,
            color=PALETTE[0],
            legend=(i == 0 and hue is not None),
            ax=ax,
            line_kws={"linewidth": 1.2},
        )
        _compact_ticks(ax, pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        ax.set_xlabel(_wrap(axis_label(col)))
        ax.set_ylabel("Density")
    if hue is not None and axes[0].get_legend() is not None:
        axes[0].get_legend().set_title(axis_label(hue))
    return _finish(fig, save)


@_styled
def class_boxplots(
    df: pd.DataFrame,
    cols: Sequence[str],
    target: str,
    ncols: int = 3,
    showfliers: bool = True,
    save: str | None = None,
) -> Figure:
    """F4: box plots of each numeric column by outcome class.

    The box shows median and quartiles directly, which is the comparison the
    skewed experience and exposure variables need; fliers are drawn small so the
    IQR fences stay readable.
    """
    frame = _class_frame(df, cols, target)
    fig, axes = _grid(len(cols), ncols, 2.3)
    for ax, col in zip(axes, cols, strict=True):
        sns.boxplot(
            data=frame,
            x=target,
            y=col,
            hue=target,
            order=_CLASS_ORDER,
            hue_order=_CLASS_ORDER,
            palette=_CLASS_PALETTE,
            legend=False,
            width=0.55,
            linewidth=0.8,
            fliersize=1.2,
            showfliers=showfliers,
            ax=ax,
        )
        _compact_ticks(ax, pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), axis="y")
        ax.set_xlabel("")
        ax.set_ylabel(_wrap(axis_label(col), 30))
        ax.grid(axis="x", visible=False)
    return _finish(fig, save)


def pairplot_top(
    df: pd.DataFrame,
    cols: Sequence[str],
    target: str,
    sample_n: int = 5000,
    save: str | None = None,
) -> Figure:
    """F5: corner pair plot of the strongest numeric features, coloured by outcome.

    Rows are subsampled (seeded) because a scatter of 100,000 points is a solid
    blob; the KDE diagonal shows each class's marginal distribution.
    """
    frame = _class_frame(df, cols, target).dropna()
    if len(frame) > sample_n:
        frame = frame.sample(sample_n, random_state=RANDOM_STATE)
    with sns.axes_style("whitegrid", _STYLE_RC), sns.plotting_context("paper", font_scale=0.85):
        grid = sns.pairplot(
            frame,
            vars=list(cols),
            hue=target,
            hue_order=_CLASS_ORDER,
            palette=_CLASS_PALETTE,
            corner=True,
            diag_kind="kde",
            height=WIDTH / max(len(cols), 2),
            plot_kws={"s": 7, "alpha": 0.45, "linewidth": 0},
            diag_kws={"common_norm": False, "fill": True, "alpha": 0.4},
        )
    for ax in grid.axes.flat:
        if ax is None:
            continue
        if ax.get_xlabel():
            ax.set_xlabel(_wrap(axis_label(ax.get_xlabel()), 28))
        if ax.get_ylabel():
            ax.set_ylabel(_wrap(axis_label(ax.get_ylabel()), 28))
    fig = grid.figure
    if grid.legend is not None:
        # seaborn hangs the legend outside the grid and widens the figure to fit it; the
        # corner layout leaves the top-right empty, so the legend moves there and the
        # figure stays at the text width
        handles = grid.legend.legend_handles
        labels = [text.get_text() for text in grid.legend.get_texts()]
        grid.legend.remove()
        fig.set_size_inches(WIDTH, WIDTH)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1)
        fig.legend(
            handles,
            labels,
            title="Injury outcome",
            loc="upper right",
            bbox_to_anchor=(0.96, 0.96),
            markerscale=2.5,
            fontsize=7,
            title_fontsize=8,
        )
    plt.close(fig)  # pairplot registers its figure with pyplot; the returned object stays usable
    return _finish(fig, save)


@_styled
def correlation_heatmaps(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    annot: bool | None = None,
    save: str | None = None,
) -> Figure:
    """F6: Pearson and Spearman matrices side by side on one diverging scale centred at zero.

    Cells are annotated only for small matrices (12 columns or fewer) so the
    numbers stay legible at print size.
    """
    k = len(pearson)
    annot = (k <= 12) if annot is None else annot
    fig, axes = _figure(WIDTH, WIDTH / 2 + 0.9, 1, 2, sharey=True)
    for ax, matrix, title, last in (
        (axes[0], pearson, "Pearson r", False),
        (axes[1], spearman, "Spearman rho", True),
    ):
        _heatmap(
            data=matrix,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="RdBu_r",
            square=True,
            annot=annot,
            fmt=".2f",
            annot_kws={"size": 5.5},
            linewidths=0.4,
            linecolor="white",
            cbar=last,
            cbar_kws={"shrink": 0.75, "label": "Correlation coefficient"} if last else None,
        )
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6.5)
        ax.tick_params(axis="y", labelrotation=0, labelsize=6.5)
    return _finish(fig, save)


def _shorten(text: Any, limit: int = 30) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@_styled
def categorical_vs_target(
    df: pd.DataFrame,
    cols: Sequence[str],
    target: str,
    min_n: int = 100,
    max_levels: int = 12,
    ncols: int = 2,
    save: str | None = None,
) -> Figure:
    """F7: positive rate per level of each categorical column, with the overall rate marked.

    Levels are ordered by size and rare ones pooled (see eda.class_rate_by_level),
    so the bar length is the evidence that justifies keeping the column, and the
    annotated n shows how much data each rate rests on.
    """
    overall = 100.0 * pd.to_numeric(df[target], errors="coerce").mean()
    fig, axes = _grid(len(cols), ncols, 2.6)
    for i, (ax, col) in enumerate(zip(axes, cols, strict=True)):
        table = class_rate_by_level(df, col, target, min_n)
        pooled = table["level"].astype(str).str.startswith("other (n <")
        table = pd.concat([table[~pooled].head(max_levels), table[pooled]]).iloc[::-1]
        colours = [
            GREY if (p or lvl == "missing") else PALETTE[0]
            for p, lvl in zip(pooled.loc[table.index], table["level"], strict=True)
        ]
        y_pos = np.arange(len(table))
        ax.barh(y_pos, table["positive_rate_pct"], color=colours, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_shorten(v) for v in table["level"]], fontsize=6.5)
        ax.axvline(
            overall,
            color=PALETTE[3],
            linestyle="--",
            linewidth=1,
            label=f"Overall rate {overall:.1f} %",
        )
        for y, (rate, n) in enumerate(zip(table["positive_rate_pct"], table["n"], strict=True)):
            ax.annotate(
                f"n = {n:,}",
                (rate, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=6,
            )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Lost-time rate (% of cases)")
        ax.set_title(axis_label(col), fontsize=8.5)
        ax.grid(axis="y", visible=False)
        if i == 0:
            ax.legend(loc="lower right", fontsize=6.5)
    return _finish(fig, save)


@_styled
def missingness(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    top_n: int = 30,
    include_complete: bool = False,
    save: str | None = None,
) -> Figure:
    """F8: percentage of missing values per column, most incomplete first.

    Complete columns are omitted by default because the point of the figure is to
    show where imputation and missing indicators are needed.
    """
    table = missingness_table(df, cols)
    if not include_complete:
        table = table[table["n_missing"] > 0]
    table = table.head(top_n).iloc[::-1]
    fig, ax = _figure(WIDTH, max(1.6, 0.24 * len(table) + 0.9))
    if table.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _finish(fig, save)
    y_pos = np.arange(len(table))
    ax.barh(y_pos, table["missing_pct"], color=PALETTE[0], height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(table["column"], fontsize=7)
    for y, (pct, n) in enumerate(zip(table["missing_pct"], table["n_missing"], strict=True)):
        ax.annotate(
            f"{pct:.1f} % ({n:,})",
            (pct, y),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    ax.set_xlim(0, min(100, table["missing_pct"].max() * 1.3 + 1))
    ax.set_xlabel("Missing values (% of rows)")
    ax.grid(axis="y", visible=False)
    return _finish(fig, save)


@_styled
def qq_plots(
    df: pd.DataFrame,
    cols: Sequence[str],
    sample_n: int = 5000,
    ncols: int = 3,
    save: str | None = None,
) -> Figure:
    """F9: normal Q-Q plots; departure from the line is the visual normality check.

    Points are a seeded subsample, which keeps the file small without changing the
    quantile pattern; skewness is quoted in each panel to tie the plot to Table 2.
    """
    fig, axes = _grid(len(cols), ncols, 2.3)
    for ax, col in zip(axes, cols, strict=True):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        skew = stats.skew(values.to_numpy(dtype=float), bias=False) if len(values) > 2 else np.nan
        if len(values) > sample_n:
            values = values.sample(sample_n, random_state=RANDOM_STATE)
        if len(values) < 3:
            ax.text(0.5, 0.5, "Too few values", ha="center", va="center", transform=ax.transAxes)
            continue
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(
            values.to_numpy(dtype=float), dist="norm"
        )
        ax.scatter(theoretical, ordered, s=4, alpha=0.6, color=PALETTE[0], linewidths=0)
        ax.plot(
            theoretical,
            slope * theoretical + intercept,
            color=PALETTE[3],
            linewidth=1,
            linestyle="--",
        )
        _compact_ticks(ax, ordered, axis="y")
        ax.set_xlabel(_wrap("Theoretical normal quantiles (standard deviations)", 30))
        ax.set_ylabel(_wrap(axis_label(col), 30), fontsize=7)
        ax.set_title(f"skewness = {skew:.2f}", fontsize=8)
    return _finish(fig, save)


def _fences(values: np.ndarray, k: float, z: float) -> dict[str, tuple[float, float, int]]:
    """IQR and Z-score fences with the number of observations outside each pair."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low_iqr, high_iqr = q1 - k * iqr, q3 + k * iqr
    mean, sd = values.mean(), values.std(ddof=1) if values.size > 1 else 0.0
    low_z, high_z = mean - z * sd, mean + z * sd
    return {
        "iqr": (low_iqr, high_iqr, int(((values < low_iqr) | (values > high_iqr)).sum())),
        "zscore": (low_z, high_z, int(((values < low_z) | (values > high_z)).sum())),
    }


@_styled
def outlier_fences(
    df: pd.DataFrame,
    cols: Sequence[str],
    k: float = 1.5,
    z: float = 3.0,
    after: pd.DataFrame | None = None,
    bins: int = 60,
    save: str | None = None,
) -> Figure:
    """F10: histograms with IQR (Q1 - k IQR, Q3 + k IQR) and Z-score (mean +/- z sd) fences.

    When `after` is given (the same columns after cleaning or a log transform) each
    variable gets a before/after pair, which shows why the IQR rule is preferred on
    skewed variables: the Z-score fences move with the tail they are meant to detect.
    """
    frames = [("Before", df)] + ([("After", after)] if after is not None else [])
    ncols = len(frames)
    nrows = len(cols)
    fig, axes = _figure(WIDTH, 2.0 * nrows + 0.3, nrows, ncols, squeeze=False)
    for r, col in enumerate(cols):
        for c, (stage, frame) in enumerate(frames):
            ax = axes[r, c]
            values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size < 2:
                ax.text(
                    0.5, 0.5, "Too few values", ha="center", va="center", transform=ax.transAxes
                )
                continue
            fences = _fences(values, k, z)
            ax.hist(values, bins=bins, color=PALETTE[0], alpha=0.75)
            n = values.size
            for which, colour, style, label in (
                ("iqr", PALETTE[3], "--", f"IQR fences (k = {k:g})"),
                ("zscore", PALETTE[2], ":", f"Z-score fences (|z| > {z:g})"),
            ):
                low, high, flagged = fences[which]
                for x in (low, high):
                    ax.axvline(x, color=colour, linestyle=style, linewidth=1.1)
                ax.plot(
                    [],
                    [],
                    color=colour,
                    linestyle=style,
                    label=f"{label}: {flagged:,} flagged ({100 * flagged / n:.1f} %)",
                )
            _compact_ticks(ax, values)
            ax.set_xlabel(_wrap(axis_label(col), 48))
            ax.set_ylabel("Cases (count)")
            ax.legend(fontsize=6, loc="upper right")
            if r == 0 and ncols > 1:
                ax.set_title(stage, fontsize=9)
    return _finish(fig, save)


@_styled
def pca_projection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    sample_n: int = 5000,
    save: str | None = None,
) -> Figure:
    """F11: first two principal components of the standardised numeric features, by class.

    Median imputation, StandardScaler and PCA are fitted on the rows passed in, so
    the caller should pass training rows (or the EDA sample) only. The explained
    variance in the axis labels says how much of the spread the picture shows.
    """
    matrix = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    filled = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(filled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(components), size=min(sample_n, len(components)), replace=False)
    fig, ax = _figure(WIDTH * 0.72, WIDTH * 0.62)
    for cls in (0, 1):
        pick = idx[labels[idx] == cls]
        ax.scatter(
            components[pick, 0],
            components[pick, 1],
            s=6,
            alpha=0.45,
            linewidths=0,
            color=CLASS_COLOURS[cls],
            label=CLASS_LABELS[cls],
        )
    ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Principal component 1 ({ratio[0]:.1f} % of variance)")
    ax.set_ylabel(f"Principal component 2 ({ratio[1]:.1f} % of variance)")
    ax.legend(title="Injury outcome", markerscale=2.5, fontsize=7)
    return _finish(fig, save)


# ---------------------------------------------------------------------------
# Evaluation figures (F12-F19): precomputed inputs only, nothing is refitted here
# ---------------------------------------------------------------------------


@_styled
def roc_curves(
    curves: Mapping[str, Any], auc: Mapping[str, float] | None = None, save: str | None = None
) -> Figure:
    """F12: ROC curves, one per model, with the chance diagonal.

    Each value in `curves` is a (fpr, tpr) pair or a table with columns fpr/tpr;
    `auc` adds the area to the legend label.
    """
    fig, ax = _figure(WIDTH * 0.68, WIDTH * 0.62)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="Chance (AUC = 0.500)")
    for i, (name, curve) in enumerate(curves.items()):
        fpr, tpr = _xy(
            curve, ("fpr", "false_positive_rate"), ("tpr", "true_positive_rate", "recall")
        )
        label = display_name(name)
        if auc and name in auc:
            label += f" (AUC = {auc[name]:.3f})"
        ax.plot(
            fpr,
            tpr,
            color=_model_colour(name, i),
            linestyle=_model_style(name),
            linewidth=1.4,
            label=label,
        )
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6.5)
    return _finish(fig, save)


@_styled
def pr_curves(
    curves: Mapping[str, Any],
    prevalence: float,
    average_precision: Mapping[str, float] | None = None,
    save: str | None = None,
) -> Figure:
    """F13: precision-recall curves with the prevalence line as the no-skill baseline.

    Each value in `curves` is a (recall, precision) pair or a table with those columns.
    """
    fig, ax = _figure(WIDTH * 0.68, WIDTH * 0.62)
    ax.axhline(
        prevalence,
        color=GREY,
        linestyle=":",
        linewidth=1,
        label=f"No skill (prevalence = {prevalence:.3f})",
    )
    for i, (name, curve) in enumerate(curves.items()):
        recall, precision = _xy(curve, ("recall", "tpr"), ("precision",))
        label = display_name(name)
        if average_precision and name in average_precision:
            label += f" (AP = {average_precision[name]:.3f})"
        ax.plot(
            recall,
            precision,
            color=_model_colour(name, i),
            linestyle=_model_style(name),
            linewidth=1.4,
            label=label,
        )
    ax.set_xlabel("Recall (true positive rate)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=6.5)
    return _finish(fig, save)


def _draw_mean_std(
    ax: Axes,
    x: Sequence[Any],
    train_mean: Sequence[float],
    train_std: Sequence[float],
    val_mean: Sequence[float],
    val_std: Sequence[float],
    metric_name: str,
    x_label: str,
    log_x: bool | None,
    best: Any | None,
) -> None:
    """Shared drawing for validation and learning curves: mean lines with +/- 1 sd bands."""
    x_arr = np.asarray(x)
    numeric_x = np.issubdtype(x_arr.dtype, np.number)
    positions = x_arr.astype(float) if numeric_x else np.arange(len(x_arr))
    if log_x is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_x = bool(
                numeric_x and positions.min() > 0 and positions.max() / positions.min() >= 100
            )
    for mean, std, colour, label in (
        (train_mean, train_std, PALETTE[1], "Training folds"),
        (val_mean, val_std, PALETTE[0], "Validation folds"),
    ):
        mean_arr, std_arr = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        ax.plot(
            positions,
            mean_arr,
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=colour,
            label=label,
        )
        ax.fill_between(
            positions, mean_arr - std_arr, mean_arr + std_arr, color=colour, alpha=0.18, linewidth=0
        )
    if best is not None:
        best_pos = float(best) if numeric_x else list(x_arr).index(best)
        ax.axvline(
            best_pos, color=PALETTE[3], linestyle="--", linewidth=1, label=f"Selected: {best}"
        )
    if log_x:
        ax.set_xscale("log")
    if not numeric_x:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(v) for v in x_arr], fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=6.5)


@_styled
def validation_curve(
    param_values: Sequence[Any],
    train_mean: Sequence[float],
    train_std: Sequence[float],
    val_mean: Sequence[float],
    val_std: Sequence[float],
    param_name: str,
    metric_name: str = "Average precision (PR-AUC)",
    log_x: bool | None = None,
    best_value: Any | None = None,
    save: str | None = None,
) -> Figure:
    """F14: training and cross-validation score against one hyperparameter.

    Takes the mean and standard deviation across folds that GridSearchCV already
    produced, so drawing never refits. The gap between the two lines is the
    overfitting diagnosis; `best_value` marks the value the search selected.
    """
    fig, ax = _figure(WIDTH * 0.7, 2.9)
    _draw_mean_std(
        ax,
        param_values,
        train_mean,
        train_std,
        val_mean,
        val_std,
        metric_name,
        param_name,
        log_x,
        best_value,
    )
    return _finish(fig, save)


@_styled
def learning_curve(
    train_sizes: Sequence[float],
    train_mean: Sequence[float],
    train_std: Sequence[float],
    val_mean: Sequence[float],
    val_std: Sequence[float],
    metric_name: str = "Average precision (PR-AUC)",
    save: str | None = None,
) -> Figure:
    """F15: score against training-set size from precomputed fold means and standard deviations.

    Converging lines at the right edge mean more data would not help (bias dominates);
    a persistent gap means variance dominates.
    """
    fig, ax = _figure(WIDTH * 0.7, 2.9)
    _draw_mean_std(
        ax,
        train_sizes,
        train_mean,
        train_std,
        val_mean,
        val_std,
        metric_name,
        "Training set size (rows)",
        False,
        None,
    )
    return _finish(fig, save)


@_styled
def curves_grid(
    panels: Mapping[str, Mapping[str, Any]],
    kind: str = "validation",
    metric_name: str = "Average precision (PR-AUC)",
    ncols: int = 2,
    save: str | None = None,
) -> Figure:
    """One validation or learning curve per model in a single grid (F14/F15 as a page).

    Each panel mapping holds `x`, `train_mean`, `train_std`, `val_mean`, `val_std` and,
    for validation curves, `param_name` plus optional `best_value` and `log_x`.
    """
    if kind not in ("validation", "learning"):
        raise ValueError("kind must be 'validation' or 'learning'")
    fig, axes = _grid(len(panels), ncols, 2.5)
    for ax, (name, panel) in zip(axes, panels.items(), strict=True):
        x_label = (
            panel.get("param_name", "") if kind == "validation" else "Training set size (rows)"
        )
        _draw_mean_std(
            ax,
            panel["x"],
            panel["train_mean"],
            panel["train_std"],
            panel["val_mean"],
            panel["val_std"],
            metric_name,
            x_label,
            panel.get("log_x") if kind == "validation" else False,
            panel.get("best_value") if kind == "validation" else None,
        )
        ax.set_title(display_name(name), fontsize=8.5)
    return _finish(fig, save)


@_styled
def confusion_matrices(
    matrices: Mapping[str, np.ndarray],
    labels: Sequence[str] = (CLASS_LABELS[0], CLASS_LABELS[1]),
    ncols: int = 2,
    save: str | None = None,
) -> Figure:
    """F16: 2x2 confusion matrices (rows = actual, columns = predicted) with row percentages.

    Each value is a scikit-learn style array [[TN, FP], [FN, TP]]; the key names
    the panel, for example "Threshold 0.50" or "Cost-optimal threshold 0.31".
    """
    fig, axes = _grid(len(matrices), ncols, 2.6)
    for ax, (name, matrix) in zip(axes, matrices.items(), strict=True):
        counts = np.asarray(matrix, dtype=float)
        if counts.shape != (2, 2):
            raise ValueError(f"{name}: expected a 2x2 matrix, got shape {counts.shape}")
        row_totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(row_totals > 0, 100 * counts / row_totals, 0.0)
        annotations = np.array(
            [
                [f"{int(c):,}\n({s:.1f} %)" for c, s in zip(row, srow, strict=True)]
                for row, srow in zip(counts, shares, strict=True)
            ]
        )
        _heatmap(
            data=shares,
            ax=ax,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            xticklabels=list(labels),
            yticklabels=list(labels),
            annot_kws={"size": 7.5},
            square=True,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(name, fontsize=8.5)
        ax.tick_params(labelsize=7)
    return _finish(fig, save)


@_styled
def feature_importance(
    importances: pd.DataFrame | pd.Series,
    top_n: int = 20,
    value_label: str = "Permutation importance (drop in average precision)",
    save: str | None = None,
) -> Figure:
    """F17: horizontal bars of the `top_n` largest |importance| values, with error bars if given.

    Accepts a Series indexed by feature or a DataFrame with columns `feature`,
    `importance` and optionally `std`. Signed values (coefficients) are coloured by
    sign so direction is visible at a glance.
    """
    if isinstance(importances, pd.Series):
        table = importances.rename("importance").rename_axis("feature").reset_index()
    else:
        table = importances.copy()
    if "std" not in table:
        table["std"] = np.nan
    table = (
        table.loc[table["importance"].abs().sort_values(ascending=False).index]
        .head(top_n)
        .iloc[::-1]
    )
    fig, ax = _figure(WIDTH, max(1.8, 0.26 * len(table) + 0.9))
    colours = [PALETTE[0] if v >= 0 else PALETTE[3] for v in table["importance"]]
    y_pos = np.arange(len(table))
    errors = table["std"].to_numpy(dtype=float)
    ax.barh(
        y_pos,
        table["importance"],
        color=colours,
        height=0.7,
        xerr=None if np.isnan(errors).all() else np.nan_to_num(errors),
        error_kw={"elinewidth": 0.8, "ecolor": "0.3"},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(f, 45) for f in table["feature"]], fontsize=7)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel(value_label)
    ax.grid(axis="y", visible=False)
    return _finish(fig, save)


def _cost_column(sweep: pd.DataFrame, ratio: int) -> str:
    """Find the sweep column holding the cost at `ratio` (any name like cost_per_1000_at_5 or cost_5)."""
    for column in sweep.columns:
        match = re.fullmatch(r"(?i).*cost.*?(\d+)", str(column))
        if match and int(match.group(1)) == ratio:
            return str(column)
    raise KeyError(f"no cost column for ratio {ratio} in {list(sweep.columns)}")


@_styled
def threshold_cost_curve(
    sweep: pd.DataFrame,
    cost_ratios: Sequence[int] = COST_RATIOS,
    chosen: Mapping[str, float] | None = None,
    cost_floor: float | None = None,
    cost_columns: Mapping[int, str] | None = None,
    save: str | None = None,
) -> Figure:
    """F18: classification rates (top) and expected cost per 1,000 cases (bottom) against the threshold.

    `sweep` is evaluation.threshold_sweep's output: a `threshold` column, the rate
    columns recall / specificity / precision / f1 / flag_rate, and one cost column
    per ratio (found by name, or given explicitly in `cost_columns`). `chosen` marks
    operating points as vertical lines and `cost_floor` the all-positive cost. The
    cost axis is logarithmic because the ratios span more than an order of magnitude.
    """
    thresholds = sweep["threshold"].to_numpy(dtype=float)
    fig, (top, bottom) = _figure(WIDTH, 5.4, 2, 1, sharex=True)
    rate_colours = {
        "recall": PALETTE[0],
        "specificity": PALETTE[1],
        "precision": PALETTE[2],
        "f1": PALETTE[4],
        "flag_rate": PALETTE[5],
    }
    rate_labels = {
        "recall": "Recall (TPR)",
        "specificity": "Specificity (TNR)",
        "precision": "Precision",
        "f1": "F1",
        "flag_rate": "Flag rate",
    }
    for key, colour in rate_colours.items():
        if key in sweep:
            top.plot(thresholds, sweep[key], color=colour, linewidth=1.3, label=rate_labels[key])
    top.set_ylabel("Rate (proportion)")
    top.set_ylim(0, 1.02)
    top.legend(fontsize=6.5, ncol=3, loc="lower center")
    shades = sns.color_palette("Blues_d", n_colors=len(cost_ratios))
    for colour, ratio in zip(shades, cost_ratios, strict=True):
        column = cost_columns[ratio] if cost_columns else _cost_column(sweep, ratio)
        bottom.plot(
            thresholds, sweep[column], color=colour, linewidth=1.3, label=f"FN:FP = {ratio}:1"
        )
    if cost_floor is not None:
        bottom.axhline(
            cost_floor,
            color=GREY,
            linestyle=":",
            linewidth=1,
            label=f"Flag every case ({cost_floor:,.0f})",
        )
    bottom.set_yscale("log")
    bottom.set_ylabel("Expected cost per 1,000 cases (USD, log scale)")
    bottom.set_xlabel("Decision threshold on the predicted probability")
    bottom.set_xlim(0, 1)
    bottom.legend(fontsize=6.5, ncol=2, loc="upper left")
    for i, (label, value) in enumerate((chosen or {}).items()):
        for ax in (top, bottom):
            ax.axvline(value, color=PALETTE[3], linestyle="--", linewidth=1, alpha=0.9)
        top.annotate(
            f"{label} ({value:.2f})",
            (value, 1.0),
            xytext=(3, -10 - 9 * i),
            textcoords="offset points",
            fontsize=6.5,
            color=PALETTE[3],
        )
    return _finish(fig, save)


def _pick_column(frame: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    if required:
        raise KeyError(f"calibration table needs one of {aliases}; got {list(frame.columns)}")
    return None


@_styled
def calibration(
    bins: Mapping[str, pd.DataFrame],
    brier: Mapping[str, float] | None = None,
    save: str | None = None,
) -> Figure:
    """F19: reliability diagram from binned tables (mean predicted probability vs observed rate).

    Each table needs a predicted-mean column (`mean_predicted` or `prob_pred`), an
    observed-rate column (`fraction_positive` or `prob_true`) and optionally a count
    column (`n` or `count`), which scales the markers so sparse bins look tentative.
    `brier` adds the Brier score to each legend label.
    """
    fig, ax = _figure(WIDTH * 0.68, WIDTH * 0.62)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="Perfectly calibrated")
    for i, (name, table) in enumerate(bins.items()):
        pred = table[
            _pick_column(table, ("mean_predicted", "prob_pred", "mean_proba", "predicted"))
        ].to_numpy(dtype=float)
        obs = table[
            _pick_column(table, ("fraction_positive", "prob_true", "observed_rate", "observed"))
        ].to_numpy(dtype=float)
        count_col = _pick_column(table, ("n", "count", "n_cases"), required=False)
        sizes = 18.0
        if count_col is not None:
            counts = table[count_col].to_numpy(dtype=float)
            sizes = 10 + 60 * counts / counts.max() if counts.max() > 0 else 18.0
        label = display_name(name)
        if brier and name in brier:
            label += f" (Brier = {brier[name]:.3f})"
        colour = _model_colour(name, i)
        ax.plot(pred, obs, color=colour, linestyle=_model_style(name), linewidth=1.2)
        ax.scatter(pred, obs, s=sizes, color=colour, alpha=0.85, linewidths=0, label=label)
    ax.set_xlabel("Mean predicted probability of a lost-time injury")
    ax.set_ylabel("Observed lost-time rate (proportion)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(
        fontsize=6.5,
        loc="upper left",
        title="Marker size: cases in bin"
        if any(_pick_column(t, ("n", "count", "n_cases"), required=False) for t in bins.values())
        else None,
        title_fontsize=6.5,
    )
    return _finish(fig, save)
