"""Page 6: evaluation dashboard built from the stored held-out scores and metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _common import (
    TEST_PREDICTIONS_PATH,
    bar_chart,
    best_model,
    criteria_rows,
    curve_frame,
    ensure_entrypoint,
    fmt_p,
    held_out_scores,
    index_of,
    line_chart,
    metrics_csv,
    metrics_json,
    model_label,
    model_thresholds,
    require,
    score_models,
    sensitivity_table,
    significance_tables,
    stamp,
    temporal_table,
)

from mlda.config import COST_RATIOS, OPERATING_SPECIFICITY, SUCCESS_CRITERIA
from mlda.evaluation import COMPARISON_FOOTNOTE, by_sector, classification_metrics, cost_from_counts
from mlda.models import MODEL_ZOO
from mlda.operating import FLAG_KEY, LOOKUP_BASELINE, OPERATING_KEY

ensure_entrypoint(__file__)

SECTOR_NAMES = {"C": "Coal", "M": "Metal and non-metal", "all": "All"}
BADGES = {
    "Pass": ("green", ":material/check:"),
    "Fail": ("red", ":material/close:"),
    "n/a": ("gray", None),
    "Reported": ("blue", None),
}
IMPORTANCE = {  # label -> (file, value column, signed, axis title)
    "Permutation importance": (
        "permutation_importance_{best}.csv",
        "importance_mean",
        False,
        "Drop in PR-AUC when the column is shuffled",
    ),
    "SHAP (mean absolute)": (
        "shap_summary_{best}.csv",
        "mean_abs_shap",
        False,
        "Mean absolute SHAP value",
    ),
    "Logistic regression coefficients": (
        "coefficients_logreg.csv",
        "coefficient",
        True,
        "Coefficient on standardised inputs",
    ),
}
TOP_N = 15
OPERATING_HELP = (
    f"The threshold at which {OPERATING_SPECIFICITY:.0%} of non-lost-time cases stay unflagged, "
    "chosen on out-of-fold scores."
)

scores = require(held_out_scores(), "The held-out predictions")
best = best_model()
detail = require(metrics_json(f"{best}.json"), "The best model's metrics")
version = stamp(TEST_PREDICTIONS_PATH)
half, operating = detail["test"]["at_0.5"], detail["test"][f"at_{OPERATING_KEY}"]
available = score_models(scores)

st.title("Evaluation")
st.write(
    f"All figures come from the **{detail['n_test_rows']:,} held-out test rows** (whole mine-years "
    f"never seen in training). "
    "Thresholds were fixed beforehand on out-of-fold scores, so no number here was tuned on the test rows."
)
targets = SUCCESS_CRITERIA
card_specs = (
    ("ROC-AUC", half["roc_auc"], f"target {targets['roc_auc_min']}", None),
    ("PR-AUC", half["pr_auc"], f"target {targets['pr_auc_min']}", None),
    (
        "Recall (op. point)",
        operating["recall"],
        f"target {targets['recall_min_at_operating_point']}",
        OPERATING_HELP,
    ),
    ("Specificity (op. point)", operating["specificity"], None, OPERATING_HELP),
    ("Brier score", half["brier"], f"limit {targets['brier_max']}", None),
)
for slot, (label, value, note, help_text) in zip(st.columns(5), card_specs, strict=True):
    slot.metric(
        label,
        f"{value:.3f}",
        note,
        delta_color="off",
        delta_arrow="off",
        help=help_text,
        border=True,
    )
st.caption(
    f"Best model by cross-validated PR-AUC: {model_label(best)}. Operating point: specificity "
    f"{OPERATING_SPECIFICITY:.2f} on out-of-fold scores, "
    f"threshold {detail['thresholds'][OPERATING_KEY]:.3f}."
)

comparison_tab, curves_tab, threshold_tab, criteria_tab, sensitivity_tab, importance_tab = st.tabs(
    [
        "Comparison",
        "Curves",
        "Threshold explorer",
        "Criteria and tests",
        "Sensitivity",
        "Importance",
    ]
)

with comparison_tab:
    comparison = require(metrics_csv("comparison.csv", index_col=0), "The comparison table")
    st.markdown("**Held-out metrics at threshold 0.50, best PR-AUC first**")
    st.dataframe(comparison.rename(index=model_label).style.format("{:.3f}"))
    st.caption(COMPARISON_FOOTNOTE)
    at_operating = metrics_csv("comparison_operating_point.csv", index_col=0)
    if at_operating is not None:
        st.markdown(
            f"**The same models at their operating point (specificity {OPERATING_SPECIFICITY:.2f} "
            f"on out-of-fold scores)**"
        )
        st.dataframe(at_operating.rename(index=model_label).style.format("{:.3f}"))

with curves_tab:
    st.caption(
        "ROC and precision-recall curves are drawn from the stored scores. The calibration curve "
        "compares the predicted "
        "probability with the observed share of lost-time cases in ten bins; points on the dashed "
        "line are perfectly calibrated."
    )
    default = [m for m in [*MODEL_ZOO, LOOKUP_BASELINE] if m in available]
    chosen = st.multiselect(
        "Models and baselines", available, default=default, format_func=model_label
    )
    panels = (
        (
            "roc",
            "fpr",
            "tpr",
            "False positive rate (1 - specificity)",
            "True positive rate (recall)",
            True,
        ),
        ("pr", "recall", "precision", "Recall", "Precision", False),
        (
            "calibration",
            "mean_predicted",
            "fraction_positive",
            "Mean predicted probability",
            "Observed share of lost-time cases",
            True,
        ),
    )
    for slot, (kind, x, y, x_title, y_title, diagonal) in zip(
        [*st.columns(2), st.container()], panels, strict=True
    ):
        frame = curve_frame(kind, version)
        frame = frame[frame["model"].isin(chosen)].assign(
            model=lambda t: t["model"].map(model_label)
        )
        slot.altair_chart(
            line_chart(frame, x, y, "model", x_title=x_title, y_title=y_title, diagonal=diagonal),
            width="stretch",
        )

with threshold_tab:
    st.caption(
        "Move the threshold to see how many cases are flagged, what is missed and what it costs. "
        "The confusion matrix, rates "
        "and cost are recomputed from the stored test scores."
    )
    model = st.selectbox(
        "Model", available, index=index_of(available, best), format_func=model_label, key="ev_model"
    )
    stored = model_thresholds(model)
    ratio = st.select_slider(
        "Cost of a missed case relative to an unnecessary review",
        COST_RATIOS,
        3,
        lambda r: f"{r}:1",
        key="ev_ratio",
    )
    start = float(stored.get(OPERATING_KEY, 0.5))
    threshold = st.slider(
        "Decision threshold", 0.0, 1.0, start, 0.005, format="%.3f", key=f"ev_threshold_{model}"
    )
    if stored:
        st.caption(
            f"Stored thresholds: operating point {stored[OPERATING_KEY]:.3f}, 50 % flag rate {stored[FLAG_KEY]:.3f}, "
            f"cost-optimal at {ratio}:1 {stored['cost_optimal'][str(ratio)]:.3f}."
        )
    y_true, proba = scores["y_true"].to_numpy(), scores[f"proba_{model}"].to_numpy()
    m = classification_metrics(y_true, proba, threshold)
    cost = cost_from_counts(m["fn"], m["fp"], m["n"], ratio)
    floor = cost_from_counts(0, m["tn"] + m["fp"], m["n"], ratio)
    matrix = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Actual: no lost time", "Actual: lost time"],
        columns=["Not flagged", "Flagged"],
    )
    left, right = st.columns([2, 3])
    left.dataframe(matrix)
    first, second = right.columns(3), right.columns(3)
    first[0].metric("Precision", f"{m['precision']:.3f}")
    first[1].metric("Recall", f"{m['recall']:.3f}")
    first[2].metric("Specificity", f"{m['specificity']:.3f}")
    second[0].metric("Flag rate", f"{m['flag_rate']:.1%}")
    second[1].metric(f"Cost per 1,000 at {ratio}:1", f"USD {cost:,.0f}")
    second[2].metric(
        "Flag every case", f"USD {floor:,.0f}", help="The no-model floor: review everything."
    )
    st.markdown("**By sector at this threshold**")
    sectors = by_sector(y_true, proba, scores["sector"].to_numpy(), threshold)
    columns = [
        "n",
        "prevalence",
        "recall",
        "specificity",
        "precision",
        "flag_rate",
        "roc_auc",
        "brier",
    ]
    st.dataframe(sectors[columns].rename(index=SECTOR_NAMES).round(3))

with criteria_tab:
    success = require(metrics_json("success_criteria.json"), "The success criteria")
    pick = st.selectbox(
        "Model",
        list(success["models"]),
        index=index_of(list(success["models"]), best),
        format_func=model_label,
        key="ev_criteria",
    )
    st.caption(
        "S1 to S6 were fixed before modelling and are evaluated on the held-out rows; S7 is the "
        "statistical comparison below."
    )
    for row in criteria_rows(success, pick):
        text, observed, verdict = st.columns([5, 4, 1])
        text.write(f"**{row['code']}** {row['criterion']}")
        observed.write(row["observed"])
        colour, icon = BADGES[row["verdict"]]
        with verdict:
            st.badge(row["verdict"], color=colour, icon=icon)
    tests = require(metrics_json("statistical_tests.json"), "The statistical tests")
    mcnemar, bootstrap = significance_tables(tests)
    st.markdown(
        f"**McNemar test, {model_label(tests['bootstrap_best_model']['model'])} against the lookup baseline**"
    )
    st.dataframe(mcnemar, hide_index=True)
    st.markdown(f"**Bootstrap intervals ({tests['bootstrap_best_model']['n_boot']} draws)**")
    st.dataframe(bootstrap, hide_index=True)
    paired = tests.get("paired_fold_test_top_two")
    if paired:
        default_note = (
            f"corrected t-test p = {fmt_p(paired.get('corrected_t_p_value'))}, "
            f"Wilcoxon p = {fmt_p(paired.get('wilcoxon_p_value'))}"
        )
        names = " vs ".join(model_label(name) for name in paired["models"])
        st.caption(
            f"Paired fold test, {names}: mean difference {paired['mean_difference']:.4f}; "
            f"{paired.get('note') or default_note}."
        )

with sensitivity_tab:
    temporal = require(metrics_json("temporal_split.json"), "The temporal split results")
    st.markdown(
        f"**Temporal split: fit on years before {temporal['split_year']}, score {temporal['split_year']} onwards**"
    )
    st.caption(
        f"{temporal['n_train']:,} training and {temporal['n_test']:,} test rows. Thresholds, grids "
        f"and model choice use the "
        f"earlier years only. {temporal['level_drift']['n_all_zero_in_test_period']} one-hot levels "
        f"are unseen in the test period."
    )
    st.dataframe(temporal_table(temporal).round(3), hide_index=True)
    sensitivity = require(metrics_json("sensitivity.json"), "The sensitivity results")
    st.markdown(
        f"**Sensitivity of {model_label(sensitivity['model'])}: change against the primary result**"
    )
    st.dataframe(sensitivity_table(sensitivity).round(4), hide_index=True)
    st.caption(
        "A negative change in ROC-AUC, PR-AUC or recall means the variant is worse than the primary "
        "result; for Brier, a positive change is worse."
    )

with importance_tab:
    source = st.radio("Importance measure", list(IMPORTANCE), horizontal=True)
    file, value, signed, title = IMPORTANCE[source]
    table = metrics_csv(file.format(best=best))
    if table is None:
        st.info(f"{file.format(best=best)} is not available for this run.")
    else:
        top = table.reindex(table[value].abs().sort_values(ascending=False).index).head(TOP_N)
        st.altair_chart(
            bar_chart(top, "feature", value, title=title, signed=signed), width="stretch"
        )
        st.caption(
            f"Model: {model_label('logreg' if signed else best)}. Importance describes what the "
            f"model uses, not what causes lost time."
        )
