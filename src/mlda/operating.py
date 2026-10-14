"""Operating thresholds, held-out scoring and the success criteria S1-S6.

These helpers sit below ``mlda.pipeline`` and ``mlda.sensitivity`` so that the primary study,
the temporal split and every sensitivity run choose thresholds and judge results in exactly
the same way: thresholds come from out-of-fold probabilities of the rows a model was fitted
on, and the held-out rows are scored once with those thresholds fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mlda.config import COST_RATIOS, OPERATING_SPECIFICITY, SUCCESS_CRITERIA
from mlda.evaluation import (
    by_sector,
    classification_metrics,
    cost_from_counts,
    cost_table,
    threshold_for_flag_rate,
    threshold_for_specificity,
)

FLAG_RATE_TARGET = 0.5
OPERATING_KEY = f"specificity_{OPERATING_SPECIFICITY:.2f}"
FLAG_KEY = f"flag_rate_{FLAG_RATE_TARGET:.2f}"
LOOKUP_BASELINE = "lookup_nature_bodypart_coal"


def oof_thresholds(y_train: np.ndarray, oof: np.ndarray) -> dict[str, Any]:
    """Operating thresholds chosen on out-of-fold probabilities only.

    ``specificity_0.40`` is the capacity operating point (S3), ``flag_rate_0.50`` the
    half-capacity point (S5) and ``cost_optimal`` the cheapest threshold per FN:FP ratio
    (S4) with the full out-of-fold cost table kept for the report.
    """
    table = cost_table(y_train, oof)
    operating = threshold_for_specificity(y_train, oof, OPERATING_SPECIFICITY)
    return {
        OPERATING_KEY: operating,
        FLAG_KEY: threshold_for_flag_rate(y_train, oof, FLAG_RATE_TARGET),
        "cost_optimal": {
            str(int(r)): float(t) for r, t in zip(table["ratio"], table["threshold"], strict=True)
        },
        "oof_cost_table": table.to_dict(orient="records"),
        "oof_metrics_at_0.5": classification_metrics(y_train, oof, 0.5),
        "oof_metrics_at_operating": classification_metrics(y_train, oof, operating),
    }


def evaluate_on_test(
    y_test: np.ndarray, proba: np.ndarray, sector: np.ndarray, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Held-out metrics at 0.5 and at every threshold fixed on the training folds."""
    n_neg = int((y_test == 0).sum())
    operating = float(thresholds[OPERATING_KEY])
    out: dict[str, Any] = {
        "at_0.5": classification_metrics(y_test, proba, 0.5),
        f"at_{OPERATING_KEY}": classification_metrics(y_test, proba, operating),
        f"at_{FLAG_KEY}": classification_metrics(y_test, proba, float(thresholds[FLAG_KEY])),
    }
    cost: dict[str, Any] = {}
    for ratio in COST_RATIOS:
        threshold = float(thresholds["cost_optimal"][str(int(ratio))])
        m = classification_metrics(y_test, proba, threshold)
        cost[str(int(ratio))] = {
            "threshold": threshold,
            "cost_per_1000": cost_from_counts(m["fn"], m["fp"], m["n"], ratio),
            "all_positive_cost_per_1000": cost_from_counts(0, n_neg, int(y_test.size), ratio),
            "recall": m["recall"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "flag_rate": m["flag_rate"],
        }
    out["cost_optimal"] = cost
    out["by_sector"] = {
        "at_0.5": by_sector(y_test, proba, sector, 0.5).to_dict(orient="index"),
        f"at_{OPERATING_KEY}": by_sector(y_test, proba, sector, operating).to_dict(orient="index"),
    }
    return out


def all_positive_floor(y_test: np.ndarray) -> dict[str, float]:
    """Cost per 1,000 of flagging every case, per FN:FP ratio (the no-model floor of S4)."""
    n_neg = int((y_test == 0).sum())
    return {str(r): cost_from_counts(0, n_neg, int(y_test.size), r) for r in COST_RATIOS}


def model_criteria(
    evaluation: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    floor: Mapping[str, float],
    lookup_cost: Mapping[str, float],
    *,
    is_lookup: bool = False,
    temporal_brier: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """S1-S7 for one scored model, from its held-out evaluation and out-of-fold thresholds.

    ``floor`` is the all-positive cost per ratio, ``lookup_cost`` the lookup baseline's
    cost-optimal cost per ratio, and ``temporal_brier`` the per-sector Brier scores of the
    temporal split when the model was evaluated there (S6 then needs those below the limit too).
    """
    temporal_brier = dict(temporal_brier or {})
    pass_ratios = [int(r) for r in SUCCESS_CRITERIA["cost_pass_ratios"]]
    s4: dict[str, Any] = {}
    for ratio in COST_RATIOS:
        key = str(int(ratio))
        cost = evaluation["cost_optimal"][key]["cost_per_1000"]
        s4[key] = {
            "cost_per_1000": cost,
            "floor": floor[key],
            "lookup": lookup_cost[key],
            "below_floor": bool(cost < floor[key]),
            "below_lookup": bool(cost < lookup_cost[key]),
            "distance_to_floor": cost - floor[key],
            "counts_for_pass": int(ratio) in pass_ratios,
        }
    brier_ok = evaluation["at_0.5"]["brier"] <= SUCCESS_CRITERIA["brier_max"] and all(
        v <= SUCCESS_CRITERIA["brier_max"] for v in temporal_brier.values()
    )
    half, operating, flag = (
        evaluation["at_0.5"],
        evaluation[f"at_{OPERATING_KEY}"],
        evaluation[f"at_{FLAG_KEY}"],
    )
    checks: dict[str, Any] = {
        "S1": {
            "value": half["roc_auc"],
            "pass": bool(half["roc_auc"] >= SUCCESS_CRITERIA["roc_auc_min"]),
        },
        "S2": {
            "value": half["pr_auc"],
            "pass": bool(half["pr_auc"] >= SUCCESS_CRITERIA["pr_auc_min"]),
        },
        "S3": {
            "value": operating["recall"],
            "specificity_achieved": operating["specificity"],
            "threshold": thresholds[OPERATING_KEY],
            "pass": bool(operating["recall"] >= SUCCESS_CRITERIA["recall_min_at_operating_point"]),
        },
        "S4": {
            "by_ratio": s4,
            "pass": None
            if is_lookup
            else all(s4[str(r)]["below_floor"] and s4[str(r)]["below_lookup"] for r in pass_ratios),
        },
        "S5": {
            "value": flag["recall"],
            "flag_rate_achieved": flag["flag_rate"],
            "pass": bool(flag["recall"] >= SUCCESS_CRITERIA["recall_at_50pct_flag_rate_min"]),
        },
        "S6": {
            "value": half["brier"],
            "temporal_brier_by_sector": temporal_brier,
            "pass": bool(brier_ok),
        },
        "S7": {"pass": True, "see": "statistical_tests.json"},
    }
    checks["all_pass"] = all(
        bool(checks[key]["pass"]) for key in ("S1", "S2", "S3", "S4", "S5", "S6")
    )
    return checks


def headline(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Compact view of an evaluation for the sensitivity tables: the figures the report quotes."""
    half, operating, flag = (
        evaluation["at_0.5"],
        evaluation[f"at_{OPERATING_KEY}"],
        evaluation[f"at_{FLAG_KEY}"],
    )
    return {
        "roc_auc": half["roc_auc"],
        "pr_auc": half["pr_auc"],
        "brier": half["brier"],
        "recall_at_0.5": half["recall"],
        "specificity_at_0.5": half["specificity"],
        f"recall_at_{OPERATING_KEY}": operating["recall"],
        f"specificity_at_{OPERATING_KEY}": operating["specificity"],
        f"recall_at_{FLAG_KEY}": flag["recall"],
    }


def headline_from_proba(
    y: np.ndarray, proba: np.ndarray, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """``headline`` for a subset of rows scored with thresholds that were fixed beforehand."""
    half = classification_metrics(y, proba, 0.5)
    operating = classification_metrics(y, proba, float(thresholds[OPERATING_KEY]))
    flag = classification_metrics(y, proba, float(thresholds[FLAG_KEY]))
    return headline({"at_0.5": half, f"at_{OPERATING_KEY}": operating, f"at_{FLAG_KEY}": flag})


def delta(variant: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Variant minus reference for every numeric entry of two ``headline`` dictionaries."""
    return {
        key: float(variant[key] - reference[key])
        for key in variant
        if key in reference
        and isinstance(variant[key], (int, float, np.floating))
        and not pd.isna(variant[key])
    }


__all__ = [
    "FLAG_KEY",
    "FLAG_RATE_TARGET",
    "LOOKUP_BASELINE",
    "OPERATING_KEY",
    "all_positive_floor",
    "delta",
    "evaluate_on_test",
    "headline",
    "headline_from_proba",
    "model_criteria",
    "oof_thresholds",
]
