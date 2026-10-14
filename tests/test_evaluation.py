"""Metrics, thresholds, cost model and statistical tests against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mlda.config import COST_FP_USD, COST_RATIOS
from mlda.evaluation import (
    COMPARISON_COLUMNS,
    COMPARISON_FOOTNOTE,
    bootstrap_ci,
    by_sector,
    classification_metrics,
    compare_models,
    confusion_counts,
    cost_column,
    cost_optimal_threshold,
    cost_table,
    curves,
    expected_cost_per_1000,
    mcnemar_test,
    paired_fold_test,
    summarise_cv,
    threshold_for_flag_rate,
    threshold_for_specificity,
    threshold_sweep,
)

# TP=30, FP=10, TN=50, FN=10 at threshold 0.5
Y_HAND = np.array([1] * 30 + [1] * 10 + [0] * 10 + [0] * 50)
P_HAND = np.array([0.9] * 30 + [0.1] * 10 + [0.9] * 10 + [0.1] * 50)


def random_scores(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.25), 0, 1)
    return y, p


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


def test_confusion_counts_match_hand_matrix() -> None:
    assert confusion_counts(Y_HAND, P_HAND >= 0.5) == {"tp": 30, "fp": 10, "tn": 50, "fn": 10}


def test_metrics_match_hand_computed_values() -> None:
    m = classification_metrics(Y_HAND, P_HAND, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (30, 10, 50, 10)
    assert m["recall"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(50 / 60, abs=1e-4)  # 0.8333
    assert m["precision"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.80)
    assert m["f1"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 50 / 60) / 2)
    assert m["mcc"] == pytest.approx(1400 / 2400)  # (30*50 - 10*10) / sqrt(40*40*60*60)
    assert m["flag_rate"] == pytest.approx(0.4)
    # two distinct scores: AUC = P(pos > neg) + 0.5 P(tie)
    assert m["roc_auc"] == pytest.approx(0.75 * (50 / 60) + 0.5 * (0.75 * (10 / 60) + 0.25 * (50 / 60)))
    assert m["brier"] == pytest.approx((30 * 0.01 + 10 * 0.81 + 10 * 0.81 + 50 * 0.01) / 100)
    assert m["n"] == 100 and m["prevalence"] == pytest.approx(0.4)
    assert m["threshold"] == 0.5
    for key in ("pr_auc", "log_loss"):
        assert np.isfinite(m[key])


def test_metrics_handle_degenerate_predictions() -> None:
    m = classification_metrics([1, 0, 1, 0], [0.2, 0.1, 0.3, 0.05], threshold=0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["flag_rate"] == 0.0
    single = classification_metrics([1, 1, 1], [0.2, 0.7, 0.9])
    assert np.isnan(single["roc_auc"]) and np.isnan(single["pr_auc"])
    with pytest.raises(ValueError):
        classification_metrics([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.1])


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


def test_threshold_for_specificity_hits_target() -> None:
    y, p = random_scores()
    for target in (0.40, 0.60, 0.90):
        t = threshold_for_specificity(y, p, target)
        spec = classification_metrics(y, p, t)["specificity"]
        assert target <= spec <= target + 0.01, (target, spec)
        # one step down would break the constraint: the threshold is the lowest admissible
        lower = np.max(p[p < t]) if (p < t).any() else t
        assert classification_metrics(y, p, lower)["specificity"] < target or lower == t


def test_threshold_for_specificity_on_hand_example() -> None:
    assert threshold_for_specificity(Y_HAND, P_HAND, 0.80) == pytest.approx(0.9)
    assert threshold_for_specificity(Y_HAND, P_HAND, 0.0) == pytest.approx(0.1)
    assert threshold_for_specificity(Y_HAND, P_HAND, 1.0) > 0.9


def test_threshold_for_flag_rate_respects_capacity() -> None:
    y, p = random_scores()
    t = threshold_for_flag_rate(y, p, 0.5)
    rate = np.mean(p >= t)
    assert 0.49 <= rate <= 0.5
    assert threshold_for_flag_rate(Y_HAND, P_HAND, 0.4) == pytest.approx(0.9)


def test_expected_cost_tiny_example() -> None:
    # FN = 1, FP = 1, N = 4, ratio 3: 1000 * (1*3*200 + 1*200) / 4 = 200,000
    cost = expected_cost_per_1000([1, 1, 0, 0], [1, 0, 1, 0], ratio=3)
    assert cost == pytest.approx(200_000.0)
    assert expected_cost_per_1000([1, 1, 0, 0], [1, 1, 0, 0], ratio=20) == 0.0
    # all-positive floor: 1000 * (1 - prevalence) * C_FP, independent of the ratio
    y = np.array([1] * 486 + [0] * 514)
    for ratio in COST_RATIOS:
        assert expected_cost_per_1000(y, np.ones_like(y), ratio) == pytest.approx(1000 * 0.514 * COST_FP_USD)


def test_threshold_sweep_is_monotone_and_complete() -> None:
    y, p = random_scores()
    sweep = threshold_sweep(y, p)
    assert sweep["threshold"].is_monotonic_increasing
    assert sweep["threshold"].iloc[0] == 0.0 and sweep["threshold"].iloc[-1] == 1.0
    assert (np.diff(sweep["recall"]) <= 1e-12).all()
    assert (np.diff(sweep["flag_rate"]) <= 1e-12).all()
    assert (np.diff(sweep["specificity"]) >= -1e-12).all()
    assert sweep["recall"].iloc[0] == 1.0 and sweep["flag_rate"].iloc[0] == 1.0
    assert (sweep[["tp", "fp", "tn", "fn"]].sum(axis=1) == len(y)).all()
    for ratio in COST_RATIOS:
        assert cost_column(ratio) in sweep.columns
    # every row of the sweep agrees with classification_metrics at that threshold
    row = sweep.iloc[len(sweep) // 2]
    m = classification_metrics(y, p, row["threshold"])
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (m["tp"], m["fp"], m["tn"], m["fn"])
    assert row[cost_column(5)] == pytest.approx(expected_cost_per_1000(y, p >= row["threshold"], 5))


def test_cost_optimal_threshold_separable_and_floor() -> None:
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.6, 1.0, 50)])
    t = cost_optimal_threshold(y, p, ratio=5)
    assert 0.3 < t <= 0.6
    assert expected_cost_per_1000(y, p >= t, 5) == 0.0
    # a weak scorer at a very high ratio: flag everything, threshold 0
    y2, p2 = random_scores(seed=3)
    assert cost_optimal_threshold(y2, p2, ratio=1000) == 0.0
    table = cost_table(y2, p2)
    assert list(table["ratio"]) == list(COST_RATIOS)
    assert (table["cost_per_1000"] <= table["all_positive_cost_per_1000"] + 1e-9).all()


# ---------------------------------------------------------------------------
# curves, tables, tests
# ---------------------------------------------------------------------------


def test_curves_return_three_frames() -> None:
    y, p = random_scores(500)
    c = curves(y, p, n_bins=5)
    assert set(c) == {"roc", "pr", "calibration"}
    assert list(c["roc"].columns) == ["fpr", "tpr", "threshold"]
    assert c["roc"]["tpr"].iloc[-1] == 1.0
    assert len(c["pr"]) == len(c["pr"]["threshold"]) and np.isnan(c["pr"]["threshold"].iloc[-1])
    assert len(c["calibration"]) == 5 and c["calibration"]["count"].sum() == 500


def test_compare_models_column_order_and_footnote() -> None:
    metrics = {
        "a": classification_metrics(Y_HAND, P_HAND),
        "b": classification_metrics(Y_HAND, 1 - P_HAND),
    }
    table = compare_models(metrics)
    assert list(table.columns) == COMPARISON_COLUMNS
    assert list(table.columns) == ["TPR", "TNR", "AUC", "Recall", "Precision", "F1", "PR-AUC", "Accuracy"]
    assert table.attrs["footnote"] == COMPARISON_FOOTNOTE and "TPR" in table.attrs["footnote"]
    assert (table["TPR"] == table["Recall"]).all()
    assert table.index[0] == "a"  # sorted by PR-AUC descending
    assert table.loc["a", "TNR"] == pytest.approx(50 / 60, abs=1e-4)
    wide = compare_models(metrics, extra_columns=True, sort_by=None)
    assert wide.columns[:8].tolist() == COMPARISON_COLUMNS and "Brier" in wide.columns
    assert list(wide.index) == ["a", "b"]


def test_by_sector_rows() -> None:
    sector = np.array(["C"] * 50 + ["M"] * 50)
    table = by_sector(Y_HAND, P_HAND, sector, threshold=0.5)
    assert list(table.index) == ["C", "M", "all"]
    assert table.loc["all", "tp"] == 30
    assert table.loc["C", "n"] == 50 and table.loc["M", "n"] == 50
    assert table.loc["C", "tp"] + table.loc["M", "tp"] == 30


def test_mcnemar_counts_and_pvalue() -> None:
    y = np.zeros(60, dtype=int)
    pred_a = np.zeros(60, dtype=int)
    pred_b = np.zeros(60, dtype=int)
    pred_a[:4] = 1  # A wrong on 4 rows that B gets right
    pred_b[4:24] = 1  # B wrong on 20 rows that A gets right
    out = mcnemar_test(y, pred_a, pred_b)
    assert out["n_a_only"] == 20 and out["n_b_only"] == 4 and out["n_both_correct"] == 36
    assert out["n_neither"] == 0
    assert 0 <= out["p_value"] <= 1
    assert out["p_value"] < 0.05
    assert out["method"] == "exact binomial"  # 24 discordant rows: below the exact/chi-square cut
    assert out["accuracy_a"] == pytest.approx(56 / 60) and out["accuracy_b"] == pytest.approx(40 / 60)
    assert mcnemar_test(y, pred_a, pred_b, exact=False)["method"].startswith("chi-square")
    # 25 or more discordant rows switch the automatic choice to the chi-square statistic
    pred_b[24:30] = 1
    assert mcnemar_test(y, pred_a, pred_b)["method"].startswith("chi-square")
    same = mcnemar_test(y, pred_a, pred_a)
    assert same["p_value"] == pytest.approx(1.0)


FOLD_A = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
FOLD_DELTA = np.array([0.02, 0.0, 0.03, -0.01, 0.02])  # differences that vary from fold to fold


def test_paired_fold_test_matches_hand_computed_t_statistics() -> None:
    out = paired_fold_test(FOLD_A, FOLD_A - FOLD_DELTA)
    assert out["n_folds"] == 5
    assert out["mean_difference"] == pytest.approx(0.012)
    # d = 0.012 +/- 0.01643 (sample sd), so t = 0.012 / (0.01643 / sqrt(5)) = 1.633 on 4 df
    assert out["t_statistic"] == pytest.approx(1.633, abs=1e-3)
    assert out["t_p_value"] == pytest.approx(0.1778, abs=1e-3)
    # Nadeau and Bengio: the variance is multiplied by 1/k + n_test/n_train = 1/5 + 1/4
    expected = 0.012 / np.sqrt((1 / 5 + 1 / 4) * FOLD_DELTA.var(ddof=1))
    assert out["test_train_ratio"] == pytest.approx(0.25)
    assert out["corrected_t_statistic"] == pytest.approx(expected)
    assert out["corrected_t_statistic"] == pytest.approx(1.0887, abs=1e-3)
    assert out["corrected_t_p_value"] == pytest.approx(2 * stats.t.sf(expected, df=4))
    assert out["corrected_t_p_value"] > out["t_p_value"]  # the plain test is too optimistic
    assert 0 < out["wilcoxon_p_value"] <= 1
    explicit = paired_fold_test(FOLD_A, FOLD_A - FOLD_DELTA, test_train_ratio=1.0)
    assert explicit["corrected_t_statistic"] < out["corrected_t_statistic"]


def test_paired_fold_test_constant_difference_has_no_t_statistic() -> None:
    out = paired_fold_test(FOLD_A, FOLD_A - 0.02)  # zero spread: significance cannot be read from t
    assert out["mean_difference"] == pytest.approx(0.02)
    assert np.isnan(out["t_statistic"]) and np.isnan(out["t_p_value"])
    assert np.isnan(out["corrected_t_statistic"]) and np.isnan(out["corrected_t_p_value"])
    assert out["wilcoxon_p_value"] == pytest.approx(0.0625)  # smallest possible with five folds


def test_paired_fold_test_ties_and_bad_input() -> None:
    tie = paired_fold_test(FOLD_A, FOLD_A)
    assert tie["t_p_value"] == 1.0 and tie["corrected_t_p_value"] == 1.0 and tie["wilcoxon_p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_fold_test(FOLD_A, FOLD_A[:-1])


def test_clustered_mcnemar_matches_the_durkalski_formula() -> None:
    # Per cluster b_k - c_k = 3, -1, 2, 0 (A-only minus B-only rows): X2 = (3 - 1 + 2)^2 / (9 + 1 + 4) = 16 / 14
    y = np.zeros(11, dtype=int)
    pred_a = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # wrong only where A predicts 1
    pred_b = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    groups = np.array(["g0", "g0", "g0", "g1", "g2", "g2", "g3", "g3", "g3", "g3", "g3"])
    out = mcnemar_test(y, pred_a, pred_b, groups=groups)
    assert (out["n_a_only"], out["n_b_only"]) == (5, 1)
    assert out["n_clusters"] == 4 and out["n_discordant_clusters"] == 3
    assert out["clustered_statistic"] == pytest.approx(16 / 14)
    assert out["clustered_p_value"] == pytest.approx(stats.chi2.sf(16 / 14, df=1))
    assert "p_value" in out and out["method"] == "exact binomial"  # the row-wise result is still reported
    assert "clustered_statistic" not in mcnemar_test(y, pred_a, pred_b)


def test_clustered_mcnemar_with_one_row_per_cluster_is_the_uncorrected_chi_square() -> None:
    y = np.zeros(60, dtype=int)
    pred_a = np.zeros(60, dtype=int)
    pred_b = np.zeros(60, dtype=int)
    pred_a[:4] = 1
    pred_b[4:24] = 1
    out = mcnemar_test(y, pred_a, pred_b, groups=np.arange(60))
    b, c = out["n_a_only"], out["n_b_only"]
    assert out["clustered_statistic"] == pytest.approx((b - c) ** 2 / (b + c))
    clustered_tie = mcnemar_test(y, pred_a, pred_a, groups=np.arange(60))
    assert clustered_tie["clustered_statistic"] == 0.0 and clustered_tie["clustered_p_value"] == 1.0
    with pytest.raises(ValueError, match="groups"):
        mcnemar_test(y, pred_a, pred_b, groups=np.arange(59))


@pytest.mark.parametrize("metric", ["pr_auc", "roc_auc", "recall", "brier", "log_loss"])
def test_bootstrap_ci_brackets_the_estimate(metric: str) -> None:
    y, p = random_scores(400)
    out = bootstrap_ci(y, p, metric=metric, n_boot=200)
    assert out["metric"] == metric and out["n_boot"] == 200
    assert out["lower"] <= out["estimate"] <= out["upper"]
    assert out["estimate"] == pytest.approx(classification_metrics(y, p)[metric])
    again = bootstrap_ci(y, p, metric=metric, n_boot=200)
    assert again["lower"] == out["lower"] and again["upper"] == out["upper"]


def test_cluster_bootstrap_interval_is_wider_when_rows_move_together() -> None:
    rng = np.random.default_rng(1)
    n_clusters, size = 80, 12
    groups = np.repeat(np.arange(n_clusters), size)
    y = (rng.random(n_clusters * size) < 0.5).astype(int)
    shift = np.repeat(rng.normal(0, 1.5, n_clusters), size)  # a score offset shared inside a cluster
    p = 1 / (1 + np.exp(-(1.2 * (2 * y - 1) + shift + rng.normal(0, 0.5, y.size))))
    rows = bootstrap_ci(y, p, metric="roc_auc", n_boot=300)
    clustered = bootstrap_ci(y, p, metric="roc_auc", n_boot=300, groups=groups)
    assert rows["estimate"] == clustered["estimate"]
    assert rows["resampling"] == "rows" and clustered["resampling"] == f"{n_clusters} clusters"
    assert clustered["lower"] <= clustered["estimate"] <= clustered["upper"]
    assert clustered["upper"] - clustered["lower"] > 1.3 * (rows["upper"] - rows["lower"])
    again = bootstrap_ci(y, p, metric="roc_auc", n_boot=300, groups=groups)
    assert again["lower"] == clustered["lower"] and again["upper"] == clustered["upper"]
    with pytest.raises(ValueError, match="groups"):
        bootstrap_ci(y, p, groups=groups[:-1])


def test_cluster_bootstrap_with_singleton_clusters_matches_the_row_bootstrap() -> None:
    y, p = random_scores(400)
    rows = bootstrap_ci(y, p, metric="pr_auc", n_boot=300)
    singles = bootstrap_ci(y, p, metric="pr_auc", n_boot=300, groups=np.arange(400))
    width = rows["upper"] - rows["lower"]
    assert abs((singles["upper"] - singles["lower"]) - width) < 0.35 * width


def test_summarise_cv_picks_rank_one_per_model() -> None:
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "b"],
            "rank_test_average_precision": [2, 1, 1],
            "mean_test_average_precision": [0.7, 0.8, 0.75],
            "std_test_average_precision": [0.01, 0.02, 0.03],
            "mean_test_roc_auc": [0.8, 0.85, 0.82],
            "std_test_roc_auc": [0.01, 0.01, 0.01],
        }
    )
    out = summarise_cv(frame)
    assert out.loc["a", "average_precision_mean"] == 0.8
    assert out.loc["b", "roc_auc_std"] == 0.01
    assert set(out.columns) == {
        "average_precision_mean", "average_precision_std", "roc_auc_mean", "roc_auc_std",
    }
