"""Classification metrics, threshold analysis, cost model and statistical comparisons.

Definitions used throughout (``pred = proba >= threshold``):

* recall = TPR = TP / (TP + FN); specificity = TNR = TN / (TN + FP); flag_rate = (TP + FP) / N
* pr_auc is scikit-learn's ``average_precision_score`` (the step-wise area, not the trapezoid)
* expected_cost_per_1000 = 1000 * (FN * ratio * C_FP + FP * C_FP) / N, in cost units where
  C_FP = ``COST_FP_USD`` and the cost of a missed lost-time case is ``ratio`` times C_FP

Thresholds are always chosen on out-of-fold or validation probabilities by the caller; nothing
here fits anything, so every function can be applied to the held-out test set once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from mlda.config import COST_FP_USD, COST_RATIOS, OPERATING_SPECIFICITY, RANDOM_STATE

COMPARISON_COLUMNS = ["TPR", "TNR", "AUC", "Recall", "Precision", "F1", "PR-AUC", "Accuracy"]
COMPARISON_FOOTNOTE = (
    "TPR (true positive rate) and Recall are the same quantity, TP / (TP + FN); TNR is "
    "specificity, TN / (TN + FP). Both names are kept because the brief lists both."
)
_COLUMN_SOURCE = {
    "TPR": "recall",
    "TNR": "specificity",
    "AUC": "roc_auc",
    "Recall": "recall",
    "Precision": "precision",
    "F1": "f1",
    "PR-AUC": "pr_auc",
    "Accuracy": "accuracy",
}
EXTRA_COMPARISON_COLUMNS = {
    "Balanced accuracy": "balanced_accuracy",
    "MCC": "mcc",
    "Brier": "brier",
    "Log loss": "log_loss",
    "Flag rate": "flag_rate",
    "Threshold": "threshold",
}
_EPS = 1e-15


def _arrays(y_true: Iterable[int], proba: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true has {y.shape[0]} rows but proba has {p.shape[0]}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1")
    return y, p


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(y_true: Iterable[int], pred: Iterable[int]) -> dict[str, int]:
    """TP, FP, TN, FN for binary labels and binary predictions."""
    y = np.asarray(y_true).astype(int).ravel()
    yhat = np.asarray(pred).astype(int).ravel()
    if y.shape != yhat.shape:
        raise ValueError("y_true and pred must have the same length")
    return {
        "tp": int(np.sum((y == 1) & (yhat == 1))),
        "fp": int(np.sum((y == 0) & (yhat == 1))),
        "tn": int(np.sum((y == 0) & (yhat == 0))),
        "fn": int(np.sum((y == 1) & (yhat == 0))),
    }


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Threshold-dependent metrics derived from a 2x2 confusion matrix."""
    n = tp + fp + tn + fn
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return {
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "mcc": mcc,
        "flag_rate": _safe_div(tp + fp, n),
    }


def ranking_metrics(y_true: Iterable[int], proba: Iterable[float]) -> dict[str, float]:
    """Threshold-free metrics; AUCs are NaN when only one class is present."""
    y, p = _arrays(y_true, proba)
    clipped = np.clip(p, _EPS, 1 - _EPS)
    both_classes = 0 < y.sum() < y.size
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def classification_metrics(
    y_true: Iterable[int], proba: Iterable[float], threshold: float = 0.5
) -> dict[str, float]:
    """All reported metrics at one threshold plus the confusion-matrix counts.

    Returns accuracy, balanced_accuracy, precision, recall (TPR), specificity (TNR), f1, mcc,
    roc_auc, pr_auc (average precision), brier, log_loss, tp/fp/tn/fn, flag_rate, threshold,
    n and prevalence.
    """
    y, p = _arrays(y_true, proba)
    counts = confusion_counts(y, p >= threshold)
    out: dict[str, float] = {}
    out.update(metrics_from_counts(**counts))
    out.update(ranking_metrics(y, p))
    out.update(counts)
    out["threshold"] = float(threshold)
    out["n"] = int(y.size)
    out["prevalence"] = float(y.mean())
    return out


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def _candidate_thresholds(p: np.ndarray) -> np.ndarray:
    """Every distinct probability plus one value above the maximum (flag nothing)."""
    unique = np.unique(p)
    return np.append(unique, np.nextafter(unique[-1], np.inf))


def _counts_at_thresholds(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised TP/FP/TN/FN for ``pred = p >= t`` at every ``t`` (sorted-prefix trick)."""
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    y_sorted = y[order]
    pos_below = np.concatenate([[0], np.cumsum(y_sorted)])  # positives with p < t
    all_below = np.searchsorted(p_sorted, thresholds, side="left")
    n_pos = int(y.sum())
    n = y.size
    fn = pos_below[all_below]
    tn = all_below - fn
    tp = n_pos - fn
    fp = (n - n_pos) - tn
    return tp, fp, tn, fn


def threshold_for_specificity(
    y_true: Iterable[int], proba: Iterable[float], target: float = OPERATING_SPECIFICITY
) -> float:
    """Lowest threshold whose specificity reaches ``target`` (maximum recall at that TNR).

    This is the capacity operating point of the study: specificity 0.40 means roughly 75 %
    of cases are flagged for review.
    """
    y, p = _arrays(y_true, proba)
    if not 0 <= target <= 1:
        raise ValueError("target specificity must lie in [0, 1]")
    thresholds = _candidate_thresholds(p)
    _, fp, tn, _ = _counts_at_thresholds(y, p, thresholds)
    n_neg = int((y == 0).sum())
    if n_neg == 0:
        return float(thresholds[0])
    specificity = tn / n_neg
    ok = np.nonzero(specificity >= target - 1e-12)[0]
    return float(thresholds[ok[0]])


def threshold_for_flag_rate(
    y_true: Iterable[int], proba: Iterable[float], flag_rate: float = 0.5
) -> float:
    """Lowest threshold that flags at most ``flag_rate`` of the rows (capacity constraint).

    ``y_true`` is accepted for a uniform signature; only the probabilities matter.
    """
    y, p = _arrays(y_true, proba)
    if not 0 <= flag_rate <= 1:
        raise ValueError("flag_rate must lie in [0, 1]")
    thresholds = _candidate_thresholds(p)
    tp, fp, _, _ = _counts_at_thresholds(y, p, thresholds)
    flagged = (tp + fp) / y.size
    ok = np.nonzero(flagged <= flag_rate + 1e-12)[0]
    return float(thresholds[ok[0]])


def cost_from_counts(fn: float, fp: float, n: float, ratio: float, c_fp: float = COST_FP_USD) -> float:
    """1000 * (FN * ratio * C_FP + FP * C_FP) / N."""
    if n <= 0:
        raise ValueError("n must be positive")
    return float(1000.0 * (fn * ratio * c_fp + fp * c_fp) / n)


def expected_cost_per_1000(
    y_true: Iterable[int], pred: Iterable[int], ratio: float, c_fp: float = COST_FP_USD
) -> float:
    """Expected cost of a decision rule per 1,000 cases for a given FN:FP cost ratio."""
    counts = confusion_counts(y_true, pred)
    n = sum(counts.values())
    return cost_from_counts(counts["fn"], counts["fp"], n, ratio, c_fp)


def threshold_sweep(
    y_true: Iterable[int],
    proba: Iterable[float],
    thresholds: Iterable[float] | None = None,
    ratios: Iterable[float] = COST_RATIOS,
) -> pd.DataFrame:
    """Confusion counts, rates and expected cost per 1,000 cases at every threshold.

    The default grid is 0.000, 0.001, ..., 1.000 plus every distinct probability, so the
    "flag everything" rule (threshold 0) and the exact values of a discrete baseline are
    both candidates. Recall and flag rate are non-increasing in the threshold, specificity is
    non-decreasing.
    """
    y, p = _arrays(y_true, proba)
    if thresholds is None:
        grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), np.unique(p)]))
    else:
        grid = np.unique(np.asarray(list(thresholds), dtype=float))
    tp, fp, tn, fn = _counts_at_thresholds(y, p, grid)
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(n_pos, tp / max(n_pos, 1), 0.0)
        specificity = np.where(n_neg, tn / max(n_neg, 1), 0.0)
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, _EPS), 0.0)
    frame = pd.DataFrame(
        {
            "threshold": grid,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "accuracy": (tp + tn) / n,
            "balanced_accuracy": (recall + specificity) / 2,
            "flag_rate": (tp + fp) / n,
        }
    )
    for ratio in ratios:
        frame[cost_column(ratio)] = 1000.0 * (fn * ratio * COST_FP_USD + fp * COST_FP_USD) / n
    return frame


def cost_column(ratio: float) -> str:
    """Name of the sweep column holding the cost per 1,000 cases at ``ratio``:1."""
    label = int(ratio) if float(ratio).is_integer() else ratio
    return f"cost_per_1000_at_{label}"


def cost_optimal_threshold(
    y_true: Iterable[int],
    proba: Iterable[float],
    ratio: float,
    thresholds: Iterable[float] | None = None,
) -> float:
    """Threshold with the lowest expected cost at ``ratio``:1 on the threshold sweep.

    Ties are broken towards the lowest threshold, so when flagging every case is optimal
    (the situation at 10:1 and above in this study) the answer is 0.0.
    """
    sweep = threshold_sweep(y_true, proba, thresholds=thresholds, ratios=[ratio])
    column = cost_column(ratio)
    best = int(np.argmin(sweep[column].to_numpy()))
    return float(sweep["threshold"].iloc[best])


def cost_table(
    y_true: Iterable[int], proba: Iterable[float], ratios: Iterable[float] = COST_RATIOS
) -> pd.DataFrame:
    """Cost-optimal threshold and its operating characteristics for every ratio.

    Also reports the all-positive floor (review every case) so the report can mark it.
    """
    y, p = _arrays(y_true, proba)
    sweep = threshold_sweep(y, p, ratios=ratios)
    floor_fp = int((y == 0).sum())
    rows = []
    for ratio in ratios:
        column = cost_column(ratio)
        best = sweep.iloc[int(np.argmin(sweep[column].to_numpy()))]
        rows.append(
            {
                "ratio": ratio,
                "threshold": float(best["threshold"]),
                "cost_per_1000": float(best[column]),
                "recall": float(best["recall"]),
                "specificity": float(best["specificity"]),
                "precision": float(best["precision"]),
                "flag_rate": float(best["flag_rate"]),
                "all_positive_cost_per_1000": cost_from_counts(0, floor_fp, y.size, ratio),
            }
        )
    frame = pd.DataFrame(rows)
    frame["beats_floor"] = frame["cost_per_1000"] < frame["all_positive_cost_per_1000"]
    return frame


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def calibration_table(y_true: Iterable[int], proba: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Observed positive share against mean predicted probability in equal-width bins."""
    y, p = _arrays(y_true, proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else float("nan"),
                "fraction_positive": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def curves(y_true: Iterable[int], proba: Iterable[float], n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """ROC points, precision-recall points and calibration bins as three DataFrames."""
    y, p = _arrays(y_true, proba)
    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    return {
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}),
        "pr": pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.append(pr_thresholds, np.nan),
            }
        ),
        "calibration": calibration_table(y, p, n_bins=n_bins),
    }


# ---------------------------------------------------------------------------
# Statistical comparisons
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: Iterable[int],
    pred_a: Iterable[int],
    pred_b: Iterable[int],
    exact: bool | None = None,
    groups: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """McNemar's test on the discordant pairs of two classifiers scored on the same rows.

    Uses the exact binomial version when fewer than 25 rows are discordant (or when
    ``exact=True``), otherwise the chi-square statistic with continuity correction, both from
    ``statsmodels``. ``n_a_only`` counts rows only A got right; ``n_b_only`` rows only B got right.

    The test treats every row as independent. Rows of one mine-year are not (they share a mine,
    a year and often an accident), so the p-value is too small. When ``groups`` is given, the
    clustered statistic of Durkalski et al. (2003) is added: with b_k and c_k the discordant
    counts of cluster k, X2 = (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2 on one degree of
    freedom. Both versions are returned; report the clustered one.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(pred_a).astype(int).ravel() == y
    b = np.asarray(pred_b).astype(int).ravel() == y
    if a.shape != y.shape or b.shape != y.shape:
        raise ValueError("predictions must have the same length as y_true")
    both = int(np.sum(a & b))
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    use_exact = (a_only + b_only < 25) if exact is None else exact
    table = np.array([[both, a_only], [b_only, neither]])
    result = mcnemar(table, exact=use_exact, correction=True)
    out: dict[str, Any] = {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_both_correct": both,
        "n_a_only": a_only,
        "n_b_only": b_only,
        "n_neither": neither,
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "method": "exact binomial" if use_exact else "chi-square with continuity correction",
    }
    if groups is not None:
        out.update(_clustered_mcnemar(a, b, np.asarray(list(groups)).ravel()))
    return out


def _clustered_mcnemar(a_correct: np.ndarray, b_correct: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Durkalski clustered McNemar statistic from per-row correctness and cluster labels."""
    if groups.shape != a_correct.shape:
        raise ValueError("groups must have the same length as y_true")
    _, cluster = np.unique(groups.astype(str), return_inverse=True)
    sign = a_correct.astype(int) - b_correct.astype(int)  # +1: only A right, -1: only B right
    per_cluster = np.bincount(cluster, weights=sign)
    numerator = float(per_cluster.sum()) ** 2
    denominator = float((per_cluster**2).sum())
    statistic = numerator / denominator if denominator > 0 else 0.0
    return {
        "clustered_statistic": statistic,
        "clustered_p_value": float(stats.chi2.sf(statistic, df=1)),
        "n_clusters": int(len(per_cluster)),
        "n_discordant_clusters": int(np.count_nonzero(per_cluster)),
        "clustered_method": "Durkalski et al. (2003) clustered McNemar, chi-square on 1 df",
    }


def paired_fold_test(
    scores_a: Iterable[float], scores_b: Iterable[float], test_train_ratio: float | None = None
) -> dict[str, Any]:
    """Paired t-test (plain and variance-corrected) and Wilcoxon test on per-fold scores.

    The folds of k-fold cross-validation share most of their training rows, so the fold
    differences are positively correlated and the plain paired t-test understates their variance.
    The corrected resampled t-test of Nadeau and Bengio (2003) inflates the variance by
    ``1/k + n_test/n_train`` (``test_train_ratio``, default ``1/(k-1)`` for k-fold) and uses k - 1
    degrees of freedom; report that one. When every fold difference is the same number the sample
    variance is zero, no t statistic exists and both t p-values are NaN.

    With five folds the smallest two-sided Wilcoxon p-value is 0.0625, so that test can only
    hint at a difference; the mean difference and its folds are reported alongside.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("scores must be paired arrays with at least two folds")
    diff = a - b
    k = int(a.size)
    ratio = 1.0 / (k - 1) if test_train_ratio is None else float(test_train_ratio)
    nan = float("nan")
    if np.allclose(diff, 0):
        t_stat, t_p, c_stat, c_p, w_stat, w_p = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    else:
        variance = float(diff.var(ddof=1))
        if not np.allclose(diff, diff[0], rtol=0.0, atol=1e-12):  # a constant difference has no spread
            t_res = stats.ttest_rel(a, b)
            t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
            c_stat = float(diff.mean() / np.sqrt((1.0 / k + ratio) * variance))
            c_p = float(2 * stats.t.sf(abs(c_stat), df=k - 1))
        else:
            t_stat, t_p, c_stat, c_p = nan, nan, nan, nan
        w_res = stats.wilcoxon(a, b, zero_method="wilcox")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    return {
        "n_folds": k,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "test_train_ratio": ratio,
        "corrected_t_statistic": c_stat,
        "corrected_t_p_value": c_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
    }


def _metric_function(metric: str, threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Fast path for the metrics the bootstrap most often needs."""
    if metric == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    if metric == "pr_auc":
        return lambda y, p: float(average_precision_score(y, p))
    if metric == "brier":
        return lambda y, p: float(brier_score_loss(y, p))
    if metric in {"accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "flag_rate"}:
        return lambda y, p: metrics_from_counts(**confusion_counts(y, p >= threshold))[metric]
    return lambda y, p: classification_metrics(y, p, threshold)[metric]


def bootstrap_ci(
    y_true: Iterable[int],
    proba: Iterable[float],
    metric: str = "pr_auc",
    threshold: float = 0.5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
    groups: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval for one metric from ``classification_metrics``.

    Without ``groups`` rows are resampled with replacement, which treats them as independent.
    With ``groups`` (the mine-year ``group_id``) whole groups are resampled instead, so the
    interval reflects that rows of one mine-year move together; on the held-out rows this
    interval is more than twice as wide as the row bootstrap, and it is the one to report.
    Replicates with a single class are skipped for the ranking metrics, where they would be
    undefined.
    """
    y, p = _arrays(y_true, proba)
    fn = _metric_function(metric, threshold)
    rng = np.random.default_rng(random_state)
    estimate = fn(y, p)
    values = []
    n = y.size
    if groups is None:
        cluster = None
        n_clusters = n
    else:
        labels = np.asarray(list(groups)).ravel()
        if labels.shape != y.shape:
            raise ValueError("groups must have the same length as y_true")
        _, cluster = np.unique(labels.astype(str), return_inverse=True)
        n_clusters = int(cluster.max()) + 1
    rows = np.arange(n)
    for _ in range(n_boot):
        if cluster is None:
            idx = rng.integers(0, n, n)
        else:  # each drawn cluster contributes all of its rows, once per draw
            draws = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
            idx = np.repeat(rows, draws[cluster])
        ys = y[idx]
        if metric in {"roc_auc", "pr_auc"} and not 0 < ys.sum() < ys.size:
            continue
        values.append(fn(ys, p[idx]))
    arr = np.asarray(values, dtype=float)
    lower, upper = np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if arr.size else (np.nan, np.nan)
    return {
        "metric": metric,
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "n_boot": int(arr.size),
        "alpha": float(alpha),
        "resampling": "rows" if cluster is None else f"{n_clusters} clusters",
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def compare_models(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    extra_columns: bool = False,
    sort_by: str | None = "PR-AUC",
) -> pd.DataFrame:
    """Comparison table in the brief's column order: TPR, TNR, AUC, Recall, Precision, F1,
    PR-AUC, Accuracy (one row per model, index = model name).

    The returned frame carries ``attrs["footnote"]`` explaining that TPR equals Recall.
    ``extra_columns=True`` appends balanced accuracy, MCC, Brier, log loss, flag rate and the
    threshold after the eight required columns.
    """
    mapping = dict(_COLUMN_SOURCE)
    if extra_columns:
        mapping.update(EXTRA_COMPARISON_COLUMNS)
    rows = {
        model: {column: metrics.get(source, np.nan) for column, source in mapping.items()}
        for model, metrics in metrics_by_model.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")[list(mapping)]
    table.index.name = "model"
    if sort_by is not None and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, kind="stable")
    table.attrs["footnote"] = COMPARISON_FOOTNOTE
    return table


def by_sector(
    y_true: Iterable[int],
    proba: Iterable[float],
    sector: Iterable[Any],
    threshold: float = 0.5,
    include_all: bool = True,
) -> pd.DataFrame:
    """``classification_metrics`` per sector value (COAL_METAL_IND) plus an "all" row."""
    y, p = _arrays(y_true, proba)
    s = pd.Series(np.asarray(sector), dtype="object").reset_index(drop=True)
    if len(s) != y.size:
        raise ValueError("sector must have the same length as y_true")
    rows: dict[str, dict[str, float]] = {}
    for level in sorted(s.dropna().unique(), key=str):
        mask = (s == level).to_numpy()
        rows[str(level)] = classification_metrics(y[mask], p[mask], threshold)
    if include_all:
        rows["all"] = classification_metrics(y, p, threshold)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "sector"
    return table


def summarise_cv(cv_results_df: pd.DataFrame, metrics: Iterable[str] | None = None) -> pd.DataFrame:
    """Mean +/- std of the best candidate's test scores, one row per model.

    Expects the tidy frame produced by ``mlda.tuning.tidy_cv_results`` (possibly several
    models concatenated); the best candidate is rank 1 on average precision.
    """
    if metrics is None:
        metrics = [
            c[len("mean_test_"):] for c in cv_results_df.columns if c.startswith("mean_test_")
        ]
    best = (
        cv_results_df.sort_values("rank_test_average_precision", kind="stable")
        .groupby("model", sort=False)
        .head(1)
        .set_index("model")
    )
    out = pd.DataFrame(index=best.index)
    for metric in metrics:
        out[f"{metric}_mean"] = best[f"mean_test_{metric}"]
        out[f"{metric}_std"] = best[f"std_test_{metric}"]
    return out


__all__ = [
    "COMPARISON_COLUMNS",
    "COMPARISON_FOOTNOTE",
    "bootstrap_ci",
    "by_sector",
    "calibration_table",
    "classification_metrics",
    "compare_models",
    "confusion_counts",
    "cost_column",
    "cost_from_counts",
    "cost_optimal_threshold",
    "cost_table",
    "curves",
    "expected_cost_per_1000",
    "mcnemar_test",
    "metrics_from_counts",
    "paired_fold_test",
    "ranking_metrics",
    "summarise_cv",
    "threshold_for_flag_rate",
    "threshold_for_specificity",
    "threshold_sweep",
]
