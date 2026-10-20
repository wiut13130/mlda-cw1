"""Follow-up checks raised in review, computed from the committed artefacts of the full run.

    uv run python -m mlda.followup                 # every check (about 40 minutes on 8 cores)
    uv run python -m mlda.followup --skip-grids    # the checks that need no new search (a few minutes)

The test rows had already been scored when these checks were written, so nothing here changes
the selected model, its hyper-parameters, its thresholds or any other file in reports/metrics.
Every check reads ``data/processed`` (features, split indices, base table) and
``reports/predictions`` and writes ``reports/metrics/followup_checks.json``, with the flat
table ``reports/metrics/followup_capacity_policy.csv`` beside it (one row per test group x
model x flagging policy, the numbers of the ``return_to_work`` block); the wider-grid searches
also write ``reports/cv/followup_<model>_cv_results.csv``.

* ``capacity``: recall of the selected model when a given share of the test reports with the
  highest scores is flagged, so the recommendation can be read at review capacities other than
  three cases in four (the cut-off uses the scores only; labels only count the hits);
* ``s5_top_half``: criterion S5 in its planned wording, recall among the 50 % of test reports
  with the highest scores, beside the scored version at the out-of-fold threshold;
* ``equal_workload``: recall of every model at one common flag rate, the median of the flag
  rates that the out-of-fold capacity thresholds produce on the test rows;
* ``return_to_work``: how many reports were filed after the worker was back at work, and the
  scores on the test reports whose outcome was still open at filing; for the selected model,
  the multilayer perceptron, the lookup and the hand rule, on all test rows and on the open
  ones, each model's capacity threshold is set beside two equal workloads (exactly the top 50 %
  and the top 75 % of the group by score), so that no model gains recall by flagging more;
* ``logreg_interaction``: logistic regression with the selected settings, refitted with one
  extra one-hot block for the joint nature x body part x sector cell, under the same grouped
  folds, to test whether those cells explain the lead of gradient boosting;
* ``pure_levels_training_rows``: the deterministic-level search repeated on training rows only;
* ``wider_grids``: KNN with more neighbours, the full random-forest grid and an HGB grid with
  leaf size in place of the inert boosting-round cap, each anchored by a candidate of the
  original search so that the reproduction can be checked.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlda.baselines import BODY_PART_COL, COAL_METAL_COL, NATURE_COL
from mlda.config import (
    CV_DIR,
    GROUP_KEY,
    METRICS_DIR,
    MODELS_DIR,
    ONE_HOT_MIN_FREQUENCY,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    TARGET,
)
from mlda.evaluation import (
    classification_metrics,
    confusion_counts,
    cost_from_counts,
    metrics_from_counts,
    threshold_for_flag_rate,
)
from mlda.features import FEATURE_COLUMNS
from mlda.models import ESTIMATOR_STEP, MODEL_ZOO, build_pipeline, default_preprocessor
from mlda.operating import FLAG_KEY, LOOKUP_BASELINE, OPERATING_KEY
from mlda.sensitivity import pure_levels
from mlda.split import cv_splitter, load_split_indices
from mlda.tuning import REFIT_METRIC, default_n_jobs, tune

log = logging.getLogger(__name__)

OUTPUT_NAME = "followup_checks.json"
LF = chr(10)  # files are written with LF line ends on every platform, as .gitattributes stores them
CELL_COLUMN = "nature_bodypart_sector_cell"
SELECTED_MODEL = "hist_gb"
CAPACITY_SHARES = (0.50, 7 / 12, 2 / 3, 0.75, 5 / 6)
# Equal-workload comparison on the return-to-work groups: the selected model, the runner-up
# network, the lookup and the hand rule. The capacity threshold lets each model flag a different
# share of a group (the hand rule flags every report that is not a cut or dust), so recall is
# also read at the same exact share of top-scored reports for all four.
POLICY_MODELS: tuple[str, ...] = (SELECTED_MODEL, "mlp", LOOKUP_BASELINE, "rule_not_cut_or_dust")
POLICY_GROUPS: tuple[str, ...] = ("all_test_rows", "open_at_filing")
CAPACITY_POLICY = "capacity_threshold"
TOP_SHARE_POLICIES: dict[str, float] = {"top_50pct": 0.50, "top_75pct": 0.75}
POLICY_COLUMNS: tuple[str, ...] = (
    "n", "prevalence", "flag_rate", "recall", "specificity", "precision", "cutoff", "missed_per_1000",
)
POLICY_TABLE_NAME = "followup_capacity_policy.csv"
INTERACTION_SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc"}

# Wider grids run after the test rows were scored. Each contains one candidate of the original
# search (KNN K = 51 with distance weights, the four sampled forests, HGB at its selected values)
# so that the reproduction of the stored score can be read off the result.
WIDER_GRIDS: dict[str, dict[str, Any]] = {
    "knn": {
        "param_grid": {
            "clf__n_neighbors": [51, 101, 151],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [2],
        },
        "search": "grid",
        "note": "n_neighbors extended beyond the edge of 51 of the original grid",
    },
    "random_forest": {
        "param_grid": dict(MODEL_ZOO["random_forest"].param_grid),
        "search": "grid",
        "note": "all 24 points of the grid that RandomizedSearchCV sampled four of",
    },
    "hist_gb": {
        "param_grid": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [400],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__min_samples_leaf": [20, 100],
            "clf__l2_regularization": [0.0, 1.0],
            "clf__early_stopping": [True],
        },
        "search": "grid",
        "note": "min_samples_leaf replaces the inert max_iter dimension and max_leaf_nodes adds 63",
    },
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_inputs(processed_dir: Path = PROCESSED_DIR, predictions_dir: Path = PREDICTIONS_DIR) -> dict[str, Any]:
    """Features, labels, groups, the persisted split, the base table fields and the test scores."""
    table = pd.read_parquet(processed_dir / "features.parquet")
    split = load_split_indices(processed_dir / "split_indices.json")
    base = pd.read_parquet(
        processed_dir / "base.parquet",
        columns=["MINE_ID", "ACCIDENT_DT", TARGET, "RETURN_TO_WORK_DT", "filing_date"],
    )
    aligned = (
        len(base) == len(table)
        and (base["MINE_ID"].astype(str).to_numpy() == table["MINE_ID"].astype(str).to_numpy()).all()
        and (base[TARGET].to_numpy() == table[TARGET].to_numpy()).all()
    )
    if not aligned:
        raise ValueError("base.parquet and features.parquet are not row-aligned; rerun mlda-train")
    test_scores = pd.read_parquet(predictions_dir / "test_predictions.parquet")
    if not np.array_equal(test_scores["row"].to_numpy(), split.test_idx):
        raise ValueError("test_predictions.parquet does not follow the persisted test indices")
    return {
        "X": table[list(FEATURE_COLUMNS)],
        "y": table[TARGET].to_numpy(dtype=int),
        "groups": table[GROUP_KEY].astype(str).to_numpy(),
        "split": split,
        "base": base,
        "test_scores": test_scores,
    }


# ---------------------------------------------------------------------------
# Checks on the stored test scores (no fitting)
# ---------------------------------------------------------------------------


def recall_at_share(y: np.ndarray, proba: np.ndarray, share: float) -> dict[str, float]:
    """Flag the ``share`` of rows with the highest scores and report what that catches.

    Counts are also given per 1,000 cases, with the expected cost at the 2:1 and 5:1 ratios
    (the rows that the per-1,000 table of the report adds for other review capacities).
    """
    threshold = threshold_for_flag_rate(y, proba, share)
    metrics = classification_metrics(y, proba, threshold)
    n = float(metrics["n"])
    tp, fp, fn = (float(metrics[k]) for k in ("tp", "fp", "fn"))
    return {
        "target_share": float(share),
        "threshold": float(threshold),
        "flag_rate": float(metrics["flag_rate"]),
        "recall": float(metrics["recall"]),
        "precision": float(metrics["precision"]),
        "specificity": float(metrics["specificity"]),
        "per_1000": {
            "flagged": 1000 * (tp + fp) / n,
            "caught": 1000 * tp / n,
            "missed": 1000 * fn / n,
            "false_alarms": 1000 * fp / n,
            "cost_at_2": cost_from_counts(fn, fp, n, 2),
            "cost_at_5": cost_from_counts(fn, fp, n, 5),
        },
    }


def capacity_curve(y: np.ndarray, proba: np.ndarray, shares: tuple[float, ...] = CAPACITY_SHARES) -> list[dict[str, float]]:
    """``recall_at_share`` for each review capacity in ``shares``."""
    return [recall_at_share(y, proba, share) for share in shares]


def model_columns(test_scores: pd.DataFrame) -> list[str]:
    """Names of the scored models and baselines in the prediction table (``proba_<name>``)."""
    return [c[len("proba_"):] for c in test_scores.columns if c.startswith("proba_")]


def s5_top_half(test_scores: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    """S5 two ways: recall among the top half of test scores, and at the out-of-fold 50 % threshold."""
    y = test_scores["y_true"].to_numpy(dtype=int)
    out: dict[str, Any] = {}
    for name in model_columns(test_scores):
        if name in ("majority", "all_positive") or name not in thresholds:
            continue
        proba = test_scores[f"proba_{name}"].to_numpy(dtype=float)
        oof = classification_metrics(y, proba, float(thresholds[name][FLAG_KEY]))
        out[name] = {
            "top_half_of_test_scores": recall_at_share(y, proba, 0.5),
            "out_of_fold_threshold": {
                "threshold": float(thresholds[name][FLAG_KEY]),
                "flag_rate": float(oof["flag_rate"]),
                "recall": float(oof["recall"]),
            },
        }
    return out


def equal_workload(test_scores: pd.DataFrame, thresholds: dict[str, Any], models: list[str]) -> dict[str, Any]:
    """Recall of each model at one common flag rate: the median test flag rate at the capacity thresholds."""
    y = test_scores["y_true"].to_numpy(dtype=int)
    flag_rates = {
        name: float(
            classification_metrics(y, test_scores[f"proba_{name}"].to_numpy(dtype=float), float(thresholds[name][OPERATING_KEY]))["flag_rate"]
        )
        for name in models
    }
    common = float(np.median(list(flag_rates.values())))
    return {
        "rule": "median over the tuned models of the test flag rate at each model's out-of-fold capacity threshold",
        "common_flag_rate": common,
        "flag_rate_at_capacity_threshold": flag_rates,
        "models": {name: recall_at_share(y, test_scores[f"proba_{name}"].to_numpy(dtype=float), common) for name in models},
    }


def top_share_flags(proba: np.ndarray, share: float) -> np.ndarray:
    """Flag exactly ``share`` of the rows, those with the highest scores (round(share x n), halves up).

    A threshold cannot do this when scores tie (the hand rule has two values and the lookup a few
    hundred), so the rows are ordered by a stable sort on descending score and the first k are
    flagged: among tied rows the earlier row is taken. Every model then reviews the same number
    of reports, and differences in recall are differences in ranking alone.
    """
    if not 0 <= share <= 1:
        raise ValueError("share must lie in [0, 1]")
    scores = np.asarray(proba, dtype=float)
    k = int(np.floor(share * scores.size + 0.5))
    order = np.argsort(-scores, kind="stable")
    flagged = np.zeros(scores.size, dtype=bool)
    flagged[order[:k]] = True
    return flagged


def policy_metrics(y: np.ndarray, flagged: np.ndarray, cutoff: float) -> dict[str, float]:
    """What one set of flagged reports catches: the columns of the capacity-policy table.

    ``missed_per_1000`` is the lost-time cases left unflagged per 1,000 reports of the group, the
    cost that a reviewer with a fixed workload weighs against the reports read.
    """
    labels = np.asarray(y, dtype=int)
    counts = confusion_counts(labels, flagged)
    rates = metrics_from_counts(**counts)
    return {
        "n": int(labels.size),
        "prevalence": float(labels.mean()),
        "flag_rate": float(rates["flag_rate"]),
        "recall": float(rates["recall"]),
        "specificity": float(rates["specificity"]),
        "precision": float(rates["precision"]),
        "cutoff": float(cutoff),
        "missed_per_1000": 1000 * counts["fn"] / labels.size,
    }


def flag_policies(
    y: np.ndarray, proba: np.ndarray, threshold: float, shares: dict[str, float] = TOP_SHARE_POLICIES,
) -> dict[str, dict[str, float]]:
    """The stored capacity threshold beside each equal-workload policy, for one model on one group.

    The cut-off of a top-share policy is the lowest flagged score; rows at that score can remain
    unflagged when they tie, which is why the flags and not the cut-off define the policy.
    """
    scores = np.asarray(proba, dtype=float)
    out = {CAPACITY_POLICY: policy_metrics(y, scores >= threshold, threshold)}
    for name, share in shares.items():
        flagged = top_share_flags(scores, share)
        cutoff = float(scores[flagged].min()) if flagged.any() else float("nan")
        out[name] = policy_metrics(y, flagged, cutoff)
    return out


def capacity_policy_table(test_rows: dict[str, Any]) -> pd.DataFrame:
    """One row per group x model x policy from the ``policies`` blocks of ``return_to_work_groups``."""
    records = [
        {"group": group, "model": model, "policy": policy, **{c: metrics[c] for c in POLICY_COLUMNS}}
        for group, entry in test_rows.items()
        for model, values in entry.items()
        if isinstance(values, dict) and "policies" in values
        for policy, metrics in values["policies"].items()
    ]
    return pd.DataFrame(records, columns=["group", "model", "policy", *POLICY_COLUMNS])


def return_to_work_groups(
    base: pd.DataFrame, test_idx: np.ndarray, test_scores: pd.DataFrame, thresholds: dict[str, Any],
    models: tuple[str, ...] = POLICY_MODELS,
) -> dict[str, Any]:
    """Reports filed after the worker was back at work, and scores on the reports still open at filing.

    ``back_at_work`` is a return-to-work date on or before the filing date (the date in
    DOCUMENT_NO). Every other report, including a blank return-to-work date, is ``open_at_filing``:
    the filer could not yet know whether days would be lost, which is the case the score is for.
    On the groups in ``POLICY_GROUPS`` each model entry also holds ``policies``: its capacity
    threshold and the equal workloads of ``TOP_SHARE_POLICIES`` (see ``flag_policies``).
    """
    back = (base["RETURN_TO_WORK_DT"] <= base["filing_date"]).to_numpy()
    labels = base[TARGET].to_numpy(dtype=int)
    summary = {
        "n_rows": int(len(base)),
        "back_at_work_rows": int(back.sum()),
        "back_at_work_share": float(back.mean()),
        "lost_time_rate_back_at_work": float(labels[back].mean()),
        "lost_time_rate_open_at_filing": float(labels[~back].mean()),
        "return_to_work_blank_rows": int(base["RETURN_TO_WORK_DT"].isna().sum()),
    }
    y_test = test_scores["y_true"].to_numpy(dtype=int)
    back_test = back[test_idx]
    groups_out: dict[str, Any] = {}
    for label, mask in (("all_test_rows", np.ones_like(back_test)), ("open_at_filing", ~back_test), ("back_at_work", back_test)):
        entry: dict[str, Any] = {"n": int(mask.sum()), "prevalence": float(y_test[mask].mean())}
        for name in models:
            proba = test_scores[f"proba_{name}"].to_numpy(dtype=float)[mask]
            at_capacity = classification_metrics(y_test[mask], proba, float(thresholds[name][OPERATING_KEY]))
            entry[name] = {
                "pr_auc": float(at_capacity["pr_auc"]),
                "roc_auc": float(at_capacity["roc_auc"]),
                "capacity_threshold": float(thresholds[name][OPERATING_KEY]),
                "recall_at_capacity_threshold": float(at_capacity["recall"]),
                "specificity_at_capacity_threshold": float(at_capacity["specificity"]),
                "precision_at_capacity_threshold": float(at_capacity["precision"]),
                "flag_rate_at_capacity_threshold": float(at_capacity["flag_rate"]),
            }
            if label in POLICY_GROUPS:
                entry[name]["policies"] = flag_policies(y_test[mask], proba, float(thresholds[name][OPERATING_KEY]))
        groups_out[label] = entry
    return {"all_in_scope_rows": summary, "test_rows": groups_out}


def pure_levels_on_training_rows(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, stored: list[dict[str, Any]]) -> dict[str, Any]:
    """Repeat the deterministic-level search on training rows only and compare the level set."""
    found = pure_levels(X.iloc[train_idx], y[train_idx])
    training_set = sorted((r["column"], r["level"]) for r in found.to_dict(orient="records"))
    stored_set = sorted((r["column"], r["level"]) for r in stored)
    return {
        "found_on_training_rows": found.to_dict(orient="records"),
        "same_levels_as_all_rows": training_set == stored_set,
    }


# ---------------------------------------------------------------------------
# Logistic regression with the joint nature x body part x sector cell
# ---------------------------------------------------------------------------


def joint_cell(X: pd.DataFrame) -> pd.Series:
    """The lookup baseline's cell as one categorical code: nature | body part | sector."""
    parts = [X[c].astype("string").fillna("missing") for c in (NATURE_COL, BODY_PART_COL, COAL_METAL_COL)]
    return (parts[0] + "|" + parts[1] + "|" + parts[2]).astype(str).rename(CELL_COLUMN)


def with_joint_cell(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` plus the joint cell column."""
    out = X.copy()
    out[CELL_COLUMN] = joint_cell(X).to_numpy()
    return out


def interaction_preprocessor(min_frequency: float | int | None = ONE_HOT_MIN_FREQUENCY) -> ColumnTransformer:
    """The logistic-regression preprocessor plus a one-hot block for the joint cell (rare cells pooled)."""
    cell_encoder = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                    sparse_output=False, dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("base", default_preprocessor(MODEL_ZOO["logreg"]), list(FEATURE_COLUMNS)),
            ("cell", cell_encoder, [CELL_COLUMN]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def logreg_interaction(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    selected_params: dict[str, Any], n_jobs: int,
) -> dict[str, Any]:
    """Grouped-CV and test PR-AUC of logistic regression with and without the joint cell block.

    Both variants use the hyper-parameters selected for logistic regression in the main search
    (no re-tuning) and the same five grouped folds, so the reference reproduces the stored score.
    """
    cv = cv_splitter()
    X_cells = with_joint_cell(X)
    spec = MODEL_ZOO["logreg"]
    params = {k: v for k, v in selected_params.items() if k.startswith(f"{ESTIMATOR_STEP}__")}
    variants = {
        "logreg_selected": build_pipeline(spec, default_preprocessor(spec)).set_params(**params),
        "logreg_with_joint_cell": build_pipeline(spec, interaction_preprocessor()).set_params(**params),
    }
    out: dict[str, Any] = {"selected_params": {k.split("__")[-1]: v for k, v in params.items()}}
    for name, pipeline in variants.items():
        started = time.perf_counter()
        X_train, X_test = X_cells.iloc[train_idx], X_cells.iloc[test_idx]
        scores = cross_validate(
            clone(pipeline), X_train, y[train_idx], groups=groups[train_idx], cv=cv,
            scoring=INTERACTION_SCORING, n_jobs=n_jobs,
        )
        fitted = clone(pipeline).fit(X_train, y[train_idx])
        proba = fitted.predict_proba(X_test)[:, 1]
        test = classification_metrics(y[test_idx], proba, 0.5)
        width = len(fitted.named_steps["pre"].get_feature_names_out())
        out[name] = {
            "cv_average_precision_mean": float(np.mean(scores["test_average_precision"])),
            "cv_average_precision_sd": float(np.std(scores["test_average_precision"], ddof=1)),
            "cv_average_precision_folds": [float(v) for v in scores["test_average_precision"]],
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "test_pr_auc": float(test["pr_auc"]),
            "test_roc_auc": float(test["roc_auc"]),
            "encoded_columns": int(width),
            "seconds": round(time.perf_counter() - started, 1),
        }
        log.info("%s: CV AP %.4f, test PR-AUC %.4f (%d columns)", name, out[name]["cv_average_precision_mean"],
                 out[name]["test_pr_auc"], width)
    return out


# ---------------------------------------------------------------------------
# Wider grids
# ---------------------------------------------------------------------------


def wider_grid_search(
    name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray,
    stored_best: float, cv_dir: Path, n_jobs: int,
) -> dict[str, Any]:
    """Run the wider grid for one model on the training rows and summarise the best candidate."""
    config = WIDER_GRIDS[name]
    spec = replace(MODEL_ZOO[name], param_grid=config["param_grid"], search=config["search"], n_iter=None)
    result = tune(spec, X.iloc[train_idx], y[train_idx], groups[train_idx], cv_splitter(), n_jobs=n_jobs)
    table = result.cv_results_df
    cv_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(cv_dir / f"followup_{name}_cv_results.csv", index=False, lineterminator=LF)
    best = table.iloc[0]
    candidates = [
        {
            "params": json.loads(row["params"]),
            "mean_validation_pr_auc": float(row[f"mean_test_{REFIT_METRIC}"]),
            "sd_validation_pr_auc": float(row[f"std_test_{REFIT_METRIC}"]),
            "mean_training_pr_auc": float(row[f"mean_train_{REFIT_METRIC}"]),
            "mean_fit_seconds": float(row["mean_fit_time"]),
        }
        for _, row in table.iterrows()
    ]
    return {
        "note": config["note"],
        "n_candidates": int(len(table)),
        "rows_searched": int(len(result.fit_idx)),
        "seconds": round(result.elapsed_seconds, 1),
        "best_params": json.loads(best["params"]),
        "best_mean_validation_pr_auc": float(best[f"mean_test_{REFIT_METRIC}"]),
        "best_sd_validation_pr_auc": float(best[f"std_test_{REFIT_METRIC}"]),
        "stored_selected_validation_pr_auc": float(stored_best),
        "gain_over_stored": float(best[f"mean_test_{REFIT_METRIC}"] - stored_best),
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(skip_grids: bool = False, grids: tuple[str, ...] = tuple(WIDER_GRIDS), output: Path | None = None) -> dict[str, Any]:
    """Run the checks, write ``followup_checks.json`` and the capacity-policy CSV; returns the payload.

    The CSV is written beside the JSON (``reports/metrics`` unless ``output`` says otherwise), so a
    run into another directory never overwrites the committed table.
    """
    started = time.perf_counter()
    n_jobs = default_n_jobs()
    inputs = load_inputs()
    X, y, groups, split = inputs["X"], inputs["y"], inputs["groups"], inputs["split"]
    test_scores = inputs["test_scores"]
    thresholds = _read_json(METRICS_DIR / "thresholds.json")
    manifest = _read_json(MODELS_DIR / "manifest.json")
    sensitivity = _read_json(METRICS_DIR / "sensitivity.json")
    models = [name for name in MODEL_ZOO if name in thresholds]
    y_test = test_scores["y_true"].to_numpy(dtype=int)
    selected = test_scores[f"proba_{SELECTED_MODEL}"].to_numpy(dtype=float)

    output = output or METRICS_DIR / OUTPUT_NAME
    previous = _read_json(output) if output.exists() else {}
    payload: dict[str, Any] = {
        "description": "checks added after the test rows were scored; none of them changes the selected model or its thresholds",
        "selected_model": SELECTED_MODEL,
        "capacity": {"model": SELECTED_MODEL, "points": capacity_curve(y_test, selected)},
        "s5_top_half": s5_top_half(test_scores, thresholds),
        "equal_workload": equal_workload(test_scores, thresholds, models),
        "return_to_work": return_to_work_groups(inputs["base"], split.test_idx, test_scores, thresholds),
        "pure_levels_training_rows": pure_levels_on_training_rows(
            X, y, split.train_idx, sensitivity["pure_levels"]["found"]
        ),
    }
    payload["logreg_interaction"] = logreg_interaction(
        X, y, groups, split.train_idx, split.test_idx, manifest["logreg"]["best_params"], n_jobs,
    )
    if skip_grids:
        payload["wider_grids"] = previous.get("wider_grids", {})
    else:
        wider = dict(previous.get("wider_grids", {}))
        for name in grids:
            stored = _read_json(METRICS_DIR / f"{name}.json")
            stored_best = stored["cv"][REFIT_METRIC]["mean_test"]
            log.info("wider grid for %s", name)
            wider[name] = wider_grid_search(name, X, y, groups, split.train_idx, stored_best, CV_DIR, n_jobs)
        payload["wider_grids"] = wider
    payload["seconds"] = round(time.perf_counter() - started, 1)
    output.write_text(json.dumps(payload, indent=2, default=str) + LF, encoding="utf-8", newline=LF)
    policy_path = output.parent / POLICY_TABLE_NAME
    capacity_policy_table(payload["return_to_work"]["test_rows"]).to_csv(policy_path, index=False, lineterminator=LF)
    log.info("wrote %s and %s in %.0f s", output, policy_path.name, payload["seconds"])
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Follow-up checks on the committed full-run artefacts.")
    parser.add_argument("--skip-grids", action="store_true", help="keep any stored wider-grid results and run no new search")
    parser.add_argument("--grids", nargs="+", choices=list(WIDER_GRIDS), default=list(WIDER_GRIDS),
                        help="which wider grids to search (default: all)")
    args = parser.parse_args()
    run(skip_grids=args.skip_grids, grids=tuple(args.grids))


if __name__ == "__main__":
    main()
