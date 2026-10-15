"""Page 4: tune one model live on a sample of training rows and compare it with the full run."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _common import (
    FEATURES_PATH,
    LIVE_CV_FOLDS,
    LIVE_N_JOBS,
    LIVE_ROWS_DEFAULT,
    LIVE_ROWS_MAX,
    LIVE_ROWS_MIN,
    LOW_MEMORY,
    MAX_LIVE_CANDIDATES,
    SPLIT_PATH,
    bar_chart,
    ensure_entrypoint,
    live_grid,
    live_tuning,
    metrics_csv,
    model_label,
    preprocessing_choice,
    require_files,
    run_manifest,
    stamp,
)

from mlda.config import ONE_HOT_MIN_FREQUENCY, RANDOM_STATE
from mlda.models import MODEL_ZOO
from mlda.preprocessing import DEFAULTS

ensure_entrypoint(__file__)

SCORES = {
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC (average precision)",
    "f1": "F1",
    "recall": "Recall",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "neg_brier_score": "Brier score (lower is better)",
}
TABLE_COLUMNS = [
    "params",
    "rank_test_average_precision",
    "mean_test_average_precision",
    "std_test_average_precision",
    "mean_train_average_precision",
    "mean_test_roc_auc",
    "mean_test_recall",
    "mean_fit_time",
]

st.title("Model training")
st.write(
    "Pick a model and a sample size, then tune it here and now. The live run searches a **reduced grid** "
    f"of at most {MAX_LIVE_CANDIDATES} combinations spread evenly through the full grid (first and last "
    f"included; two for the slow RBF SVM), with {LIVE_CV_FOLDS}-fold `StratifiedGroupKFold` on whole mine-years, "
    "on training rows only. "
    f"It runs on {LIVE_N_JOBS} worker so that the hosted app stays within its memory limit. It shows how the "
    "search works; the committed models come from the full run."
)
if LOW_MEMORY:
    st.caption(
        f"This host runs in low-memory mode: samples of up to {LIVE_ROWS_MAX:,} rows, "
        f"{LIVE_CV_FOLDS}-fold cross-validation and {LIVE_N_JOBS} worker."
    )
require_files(FEATURES_PATH, SPLIT_PATH)

model_col, rows_col, seed_col = st.columns([2, 2, 1])
model_name = model_col.selectbox("Model", list(MODEL_ZOO), format_func=model_label, key="mt_model")
n_rows = rows_col.slider(
    "Training rows in the sample",
    LIVE_ROWS_MIN,
    LIVE_ROWS_MAX,
    LIVE_ROWS_DEFAULT,
    step=1_000,
    key="mt_rows",
)
seed = int(seed_col.number_input("Seed", min_value=0, value=RANDOM_STATE, step=1, key="mt_seed"))
spec = MODEL_ZOO[model_name]
st.markdown(
    f"**{model_label(model_name)}** (taught in week {spec.taught_in_week}). {spec.description}"
)
grid = [
    {key.rsplit("__", 1)[-1]: values[0] for key, values in point.items()}
    for point in live_grid(spec)
]
st.caption(f"Full search: {spec.n_candidates} candidates ({spec.search}). Reduced grid used here:")
st.dataframe(pd.DataFrame(grid).astype(str), hide_index=True)
caps = [cap for cap in (spec.sample_cap, spec.quick_sample_cap) if cap]
if caps and min(caps) < n_rows:
    st.caption(
        f"This model is fitted on at most {min(caps):,} of the sampled rows, because its cost grows "
        f"faster than linearly with rows."
    )

use_choice = st.toggle(
    "Use the preprocessing chosen on the Preprocessing page",
    key="mt_use_pp",
    help="Off: the model's own preprocessing, as in the full run. On: the imputer, scaler, encoder, outlier rule "
    "and min_frequency last chosen on the Preprocessing page, and the result is compared with the model's own.",
)
choice = preprocessing_choice(st.session_state)
if use_choice:
    if choice is None:
        st.info("The Preprocessing page has not been opened in this session, so its default options are used.")
        choice = {**DEFAULTS, "min_frequency": ONE_HOT_MIN_FREQUENCY}
    st.caption(
        "Preprocessing for this run: "
        + ", ".join(f"{name} = {'none' if value is None else value}" for name, value in choice.items())
    )
preprocessing = tuple(sorted(choice.items())) if use_choice and choice else None

if st.button("Run tuning", type="primary", key="mt_run"):
    st.session_state["mt_request"] = (model_name, n_rows, seed, preprocessing)
request = st.session_state.get("mt_request")
if request is None:
    st.info("Press **Run tuning** to start. Results are cached per (model, sample size, seed, preprocessing).")
    st.stop()

version = stamp(FEATURES_PATH, SPLIT_PATH)
with st.spinner(
    f"Tuning {model_label(request[0])} on {request[1]:,} training rows ({LIVE_CV_FOLDS} folds)"
):
    result = live_tuning(request[0], request[1], request[2], version, request[3])
    own = None if request[3] is None else live_tuning(request[0], request[1], request[2], version)
cv_results = result["cv_results"]
best = cv_results.sort_values("rank_test_average_precision").iloc[0]
st.subheader(f"Result for {model_label(request[0])}, {request[1]:,} rows, seed {request[2]}")
if own is not None:
    own_best = own["cv_results"].sort_values("rank_test_average_precision").iloc[0]
    change = best["mean_test_average_precision"] - own_best["mean_test_average_precision"]
    st.metric(
        "PR-AUC change from the Preprocessing page choice",
        f"{change:+.3f}",
        help=f"Best CV PR-AUC with the chosen preprocessing ({best['mean_test_average_precision']:.3f}) minus the best "
        f"with the model's own preprocessing ({own_best['mean_test_average_precision']:.3f}), same rows, folds and grid.",
        border=True,
    )
best_col, fit_col, time_col = st.columns(3)
best_col.metric(
    "Best PR-AUC (CV mean)",
    f"{best['mean_test_average_precision']:.3f}",
    delta=f"+/- {best['std_test_average_precision']:.3f}",
    delta_color="off",
    delta_arrow="off",
    border=True,
)
fit_col.metric(
    "Rows fitted",
    f"{result['n_fit']:,}",
    help=(
        f"{result['n_candidates']} candidates x {LIVE_CV_FOLDS} folds in the search. The run also refits the best "
        f"candidate and scores out-of-fold probabilities ({LIVE_CV_FOLDS} more fits), which this page does not show."
    ),
    border=True,
)
time_col.metric("Tuning time", f"{result['seconds']:.1f} s", border=True)

st.markdown("**Cross-validation results, best candidate first**")
st.dataframe(
    cv_results[[c for c in TABLE_COLUMNS if c in cv_results.columns]].round(4), hide_index=True
)
params_col, folds_col = st.columns(2)
with params_col:
    st.markdown("**Best parameters**")
    st.json(result["best_params"])
with folds_col:
    st.markdown("**PR-AUC of the best candidate in each fold**")
    folds = pd.DataFrame(
        {
            "fold": [f"fold {i + 1}" for i in range(len(result["fold_scores"]))],
            "average precision": result["fold_scores"],
        }
    )
    st.altair_chart(
        bar_chart(folds, "fold", "average precision", title="Average precision", height=140),
        width="stretch",
    )

st.subheader("Live run against the full run")
summary = metrics_csv("cv_summary.csv", index_col=0)
if summary is None or request[0] not in summary.index:
    st.info("The full-run cross-validation summary for this model is not available.")
else:
    full = summary.loc[request[0]]
    sign = {"neg_brier_score": -1.0}
    rows = [
        {
            "metric": label,
            "live mean": sign.get(key, 1.0) * best[f"mean_test_{key}"],
            "live std": best[f"std_test_{key}"],
            "full-run mean": sign.get(key, 1.0) * full[f"{key}_mean"],
            "full-run std": full[f"{key}_std"],
        }
        for key, label in SCORES.items()
    ]
    comparison = pd.DataFrame(rows).assign(difference=lambda t: t["live mean"] - t["full-run mean"])
    st.dataframe(comparison.round(4), hide_index=True)
    st.caption(
        f"The live run uses {request[1]:,} rows and {LIVE_CV_FOLDS} folds; the full run uses every training row and "
        f"{run_manifest().get('cv_folds', 'more')} folds plus the full grid. Fewer rows and folds "
        f"change both the level "
        "and the spread of the scores, so the order of the candidates matters more than the absolute value."
    )
