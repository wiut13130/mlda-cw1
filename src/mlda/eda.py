"""Descriptive statistics and association measures for the exploratory analysis.

The functions here produce the tables behind the EDA section of the report
(descriptive statistics, Pearson against Spearman, point-biserial and Cramer's V
against the target, VIF, missingness, class rate per level). They are pure
pandas-in, pandas-out helpers so the notebook and the Streamlit app share one
definition of every statistic. Sample statistics use ddof=1 throughout because
the rows are a sample of mining accidents, not the population of them.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

NUMERIC_STATS = [
    "n",
    "missing_pct",
    "mean",
    "median",
    "trimmed_mean_10",
    "variance",
    "std",
    "min",
    "q1",
    "q3",
    "max",
    "iqr",
    "skewness",
    "kurtosis",
]
MISSING_LEVEL = "missing"


def _numeric_columns(df: pd.DataFrame, cols: Sequence[str] | None) -> list[str]:
    """Resolve `cols`, defaulting to every numeric column, and fail loudly on typos."""
    if cols is None:
        return list(df.select_dtypes("number").columns)
    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        raise KeyError(f"columns not in frame: {unknown}")
    return list(cols)


def describe_numeric(df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Table 2 of the report: one row per numeric column with location, spread and shape.

    Columns, in order: n (non-missing), missing_pct, mean, median, trimmed_mean_10
    (10 % cut from each tail, scipy.stats.trim_mean), variance and std (ddof=1),
    min, q1, q3, max, iqr, skewness (bias-corrected, as pandas reports it) and
    kurtosis (excess kurtosis, bias-corrected). The trimmed mean sits beside the
    mean and median so the reader can see how far a few extreme values pull the mean.
    """
    cols = _numeric_columns(df, cols)
    rows: list[dict[str, float]] = []
    total = len(df)
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        n = values.size
        row: dict[str, float] = dict.fromkeys(NUMERIC_STATS, np.nan)
        row["n"] = n
        row["missing_pct"] = 100.0 * (total - n) / total if total else np.nan
        if n:
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            row.update(
                mean=values.mean(),
                median=median,
                trimmed_mean_10=stats.trim_mean(values, 0.1),
                min=values.min(),
                q1=q1,
                q3=q3,
                max=values.max(),
                iqr=q3 - q1,
            )
        if n > 1:
            row["variance"] = values.var(ddof=1)
            row["std"] = np.sqrt(row["variance"])
        # the unbiased estimators need three (skewness) and four (kurtosis) observations
        if n > 2:
            row["skewness"] = stats.skew(values, bias=False)
        if n > 3:
            row["kurtosis"] = stats.kurtosis(values, fisher=True, bias=False)
        rows.append(row)
    table = pd.DataFrame(rows, index=pd.Index(cols, name="variable"), columns=NUMERIC_STATS)
    table["n"] = table["n"].astype(int)
    return table


def describe_categorical(df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Cardinality and mode of each categorical column, with the mode's share of present values.

    A high top share (over about 50 %) warns that one-hot encoding will produce a
    near-constant column, and the level count predicts the one-hot width.
    """
    if cols is None:
        cols = list(df.select_dtypes(exclude="number").columns)
    rows = []
    total = len(df)
    for col in cols:
        series = df[col]
        counts = series.value_counts(dropna=True)
        n = int(series.notna().sum())
        rows.append(
            {
                "n": n,
                "missing_pct": 100.0 * (total - n) / total if total else np.nan,
                "n_levels": int(len(counts)),
                "top_level": counts.index[0] if len(counts) else None,
                "top_n": int(counts.iloc[0]) if len(counts) else 0,
                "top_share_pct": 100.0 * counts.iloc[0] / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(cols, name="variable"))


class CorrelationResult(NamedTuple):
    """Pearson and Spearman matrices plus the pairs on which the two coefficients disagree."""

    pearson: pd.DataFrame
    spearman: pd.DataFrame
    disagreements: pd.DataFrame


def correlations(
    df: pd.DataFrame, cols: Sequence[str] | None = None, threshold: float = 0.1
) -> CorrelationResult:
    """Pearson and Spearman correlation matrices on pairwise-complete observations.

    `disagreements` lists each unordered pair whose |pearson - spearman| exceeds
    `threshold`, largest gap first. A Spearman coefficient well above Pearson points
    to a monotone but non-linear relation (or outliers pulling Pearson), which is
    the signal that a log transform or a rank-based model may help.
    """
    cols = _numeric_columns(df, cols)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    upper_i, upper_j = np.triu_indices(len(cols), k=1)
    pairs = pd.DataFrame(
        {
            "var_a": [cols[i] for i in upper_i],
            "var_b": [cols[j] for j in upper_j],
            "pearson": pearson.to_numpy()[upper_i, upper_j],
            "spearman": spearman.to_numpy()[upper_i, upper_j],
        }
    )
    pairs["abs_diff"] = (pairs["pearson"] - pairs["spearman"]).abs()
    disagreements = (
        pairs[pairs["abs_diff"] > threshold]
        .sort_values("abs_diff", ascending=False)
        .reset_index(drop=True)
    )
    return CorrelationResult(pearson, spearman, disagreements)


def _binary_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Return the target as float 0/1 (NaN kept) or raise when it is not binary."""
    y = pd.to_numeric(df[target], errors="coerce")
    levels = set(y.dropna().unique())
    if not levels <= {0, 1}:
        raise ValueError(f"{target} must be coded 0/1, found levels {sorted(levels)}")
    return y


def point_biserial(df: pd.DataFrame, cols: Sequence[str], target: str) -> pd.DataFrame:
    """Point-biserial correlation of each numeric column with the 0/1 target.

    The point-biserial coefficient is Pearson's r with one variable dichotomous, so
    it is the correct numeric-versus-binary analogue of the Pearson matrix. Rows are
    sorted by |r|; p-values come from scipy.stats.pointbiserialr and are only a
    rough guide at this sample size, where tiny effects are significant.
    """
    y = _binary_target(df, target)
    rows = []
    for col in _numeric_columns(df, cols):
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        r, p = np.nan, np.nan
        if n >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
            result = stats.pointbiserialr(
                y[mask].to_numpy(dtype=int), x[mask].to_numpy(dtype=float)
            )
            r, p = float(result.statistic), float(result.pvalue)
        rows.append({"feature": col, "n": n, "r": r, "p_value": p})
    table = pd.DataFrame(rows, columns=["feature", "n", "r", "p_value"])
    order = table["r"].abs().sort_values(ascending=False, na_position="last").index
    return table.loc[order].reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013) between two categorical series.

    The plain V is inflated for columns with many levels (OCCUPATION, INJURY_SOURCE_CD),
    so the correction subtracts the expected chi-square under independence from phi^2
    and shrinks the level counts; a column that is independent of the target lands
    at 0 instead of a small positive number. NaN is returned when a column has a
    single level, because association is then undefined.
    """
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else np.nan


def cramers_v_table(
    df: pd.DataFrame, cols: Sequence[str], target: str, missing_as_level: bool = True
) -> pd.DataFrame:
    """Bias-corrected Cramer's V of each categorical column with the target, strongest first.

    Missing values become the level "missing" by default because in this data a
    blank code is informative (the structural "?" placeholders, see decisions.json
    E13-E17); pass missing_as_level=False to measure association on complete rows only.
    """
    y = df[target]
    rows = []
    for col in cols:
        x = df[col].astype("object")
        if missing_as_level:
            x = x.where(x.notna(), MISSING_LEVEL)
        mask = x.notna() & y.notna()
        table = pd.crosstab(x[mask], y[mask])
        chi2, p = np.nan, np.nan
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
        rows.append(
            {
                "feature": col,
                "n": int(mask.sum()),
                "n_levels": int(x[mask].nunique()),
                "cramers_v": cramers_v(x[mask], y[mask]),
                "chi2": chi2,
                "p_value": p,
            }
        )
    table_out = pd.DataFrame(
        rows, columns=["feature", "n", "n_levels", "cramers_v", "chi2", "p_value"]
    )
    return table_out.sort_values("cramers_v", ascending=False, na_position="last").reset_index(
        drop=True
    )


def vif_table(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of every numeric column, largest first.

    Missing values are filled with the column median (SimpleImputer) because VIF
    needs a complete matrix, and a constant is added so each auxiliary regression
    has an intercept; the constant itself is not reported. Columns with zero
    variance get NaN (their VIF is undefined). Pass training rows only: the table
    describes the design matrix the models will see. VIF above 5-10 flags the
    collinearity that destabilises logistic-regression coefficients.
    """
    numeric = X_numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    features = list(numeric.columns)
    result = pd.Series(np.nan, index=pd.Index(features, name="feature"), dtype=float)
    if features:
        filled = SimpleImputer(strategy="median").fit_transform(numeric.to_numpy(dtype=float))
        varying = filled.std(axis=0) > 0
        kept = [f for f, keep in zip(features, varying, strict=True) if keep]
        if kept:
            exog = add_constant(filled[:, varying], has_constant="add")
            with warnings.catch_warnings():
                # statsmodels warns about a poorly conditioned matrix exactly when a
                # column is (near) collinear, which is what the table is meant to expose
                warnings.simplefilter("ignore", UserWarning)
                vifs = [variance_inflation_factor(exog, i) for i in range(1, exog.shape[1])]
            result.loc[kept] = vifs
    table = result.rename("vif").reset_index()
    return table.sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


def missingness_table(df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Count and percentage of missing values per column, most incomplete first."""
    cols = list(df.columns) if cols is None else list(cols)
    n_missing = df[cols].isna().sum()
    total = len(df)
    table = pd.DataFrame(
        {
            "column": cols,
            "dtype": [str(df[c].dtype) for c in cols],
            "n_missing": n_missing.to_numpy(dtype=int),
            "n_present": total - n_missing.to_numpy(dtype=int),
            "missing_pct": 100.0 * n_missing.to_numpy() / total if total else np.nan,
        }
    )
    return table.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def class_rate_by_level(df: pd.DataFrame, col: str, target: str, min_n: int = 100) -> pd.DataFrame:
    """Positive rate of the 0/1 target within each level of a categorical column.

    Levels with fewer than `min_n` rows are pooled into one "other (n < min_n)" row,
    because a rate estimated on a handful of cases is noise rather than evidence.
    Missing values form their own level. `lift` is the level's rate divided by the
    overall rate, so 1.0 means no association. Rows are sorted by size, pooled row last.
    """
    y = _binary_target(df, target)
    level = df[col].astype("object").where(df[col].notna(), MISSING_LEVEL)
    mask = y.notna()
    level, y = level[mask], y[mask]
    total = int(mask.sum())
    overall = float(y.mean()) if total else np.nan
    grouped = pd.DataFrame({"level": level, "y": y}).groupby("level", sort=False)["y"]
    summary = grouped.agg(n="size", positives="sum").reset_index()
    frequent = summary[summary["n"] >= min_n].sort_values("n", ascending=False)
    rare = summary[summary["n"] < min_n]
    if len(rare):
        pooled = pd.DataFrame(
            {
                "level": [f"other (n < {min_n})"],
                "n": [rare["n"].sum()],
                "positives": [rare["positives"].sum()],
            }
        )
        frequent = pd.concat([frequent, pooled], ignore_index=True)
    frequent = frequent.reset_index(drop=True)
    frequent["n"] = frequent["n"].astype(int)
    frequent["share_pct"] = 100.0 * frequent["n"] / total if total else np.nan
    frequent["positive_rate_pct"] = 100.0 * frequent["positives"] / frequent["n"]
    frequent["lift"] = (frequent["positives"] / frequent["n"]) / overall if overall else np.nan
    return frequent[["level", "n", "share_pct", "positive_rate_pct", "lift"]]
