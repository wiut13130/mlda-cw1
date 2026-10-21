"""Page 6: evaluation dashboard built from the stored held-out scores and metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from _common import (
    TEST_PREDICTIONS_PATH,
    bar_chart,
    best_model,
    capacity_policy,
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
    open_calibration,
    require,
    score_models,
    sensitivity_table,
    significance_tables,
    stamp,
    temporal_table,
    workload_checks,
    workload_rule,
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
FOLLOWUP_COMMAND = "uv run python -m mlda.followup"
CAPACITY_MODEL = "hist_gb"  # the selected model; the policy files hold it beside the comparators
CAPACITY_SHORT = "HGB"  # its short name, as used in the report text
POLICY_GROUPS = {"Reports still open at filing": "open_at_filing", "All test reports": "all_test_rows"}
POLICY_NAMES = {
    "capacity_threshold": "Capacity threshold",
    "top_50pct": "Top 50 % by score",
    "top_75pct": "Top 75 % by score",
}
RULE_NAMES = {"top_75pct": "Workload rule (top 75 %)", "top_50pct": "Top 50 % (comparison only)"}
CALIBRATION_GROUPS = {
    "open_at_filing": "Open at filing",
    "back_at_work": "Return-to-work date at filing",
    "all_test_rows": "All test reports",
}
DIFFERENCE_NAMES = {
    "hist_gb_minus_lookup_nature_bodypart_coal": "HGB minus lookup baseline",
    "hist_gb_minus_mlp": "HGB minus MLP",
}
OPERATING_HELP = (
    f"The threshold at which {OPERATING_SPECIFICITY:.0%} of non-lost-time cases stay unflagged, "
    "chosen on out-of-fold scores."
)

def followup_missing(what: str) -> None:
    """Point to the command that writes the follow-up artefacts instead of failing the page."""
    st.info(f"{what} is not available. Create it with `{FOLLOWUP_COMMAND}`.")


def policy_row(
    policy: pd.DataFrame | None, group: str, name: str, model: str = CAPACITY_MODEL
) -> pd.Series | None:
    """One row of the capacity-policy table, or None when the table or the row is missing."""
    if policy is None:
        return None
    rows = policy[(policy["group"] == group) & (policy["model"] == model) & (policy["policy"] == name)]
    return rows.iloc[0] if len(rows) else None


def capacity_intro(policy: pd.DataFrame | None) -> str:
    """The two operating points and the reason the capacity threshold over-flags open reports."""
    everything = policy_row(policy, "all_test_rows", "capacity_threshold")
    still_open = policy_row(policy, "open_at_filing", "capacity_threshold")
    threshold = (
        "The capacity threshold is the probability at which out-of-fold specificity on all training "
        f"rows is {OPERATING_SPECIFICITY:.2f}"
    )
    if everything is None or still_open is None:
        return (
            f"{threshold}. The workload rule instead reviews the top three in four of the reports "
            "still open at filing, ranked by score. A specificity target flags more of the open "
            "reports, because their lost-time prevalence is higher than on all filings."
        )
    return (
        f"{threshold} ({CAPACITY_SHORT} cut-off {everything['cutoff']:.3f}); on all "
        f"test reports it flags {everything['flag_rate']:.1%} and finds {everything['recall']:.1%} of "
        "the lost-time cases. The workload rule instead reviews the top three in four of the reports "
        "still open at filing, ranked by score. The threshold over-flags open reports because the "
        "flag rate equals prevalence x recall + (1 - prevalence) x (1 - specificity): the open "
        f"reports have a lost-time prevalence of {still_open['prevalence']:.3f} against "
        f"{everything['prevalence']:.3f} on all reports, so the same threshold flags "
        f"{still_open['flag_rate']:.1%} of them."
    )


def policy_table(policy: pd.DataFrame, group: str) -> pd.DataFrame:
    """Models and policies of one report group with readable labels, in file order."""
    rows = policy[policy["group"] == group]
    return pd.DataFrame(
        {
            "Model": rows["model"].map(model_label),
            "Policy": rows["policy"].map(lambda name: POLICY_NAMES.get(name, name)),
            "Flag rate": rows["flag_rate"],
            "Recall": rows["recall"],
            "Precision": rows["precision"],
            "Specificity": rows["specificity"],
            "Cut-off": rows["cutoff"],
            "Missed per 1,000": rows["missed_per_1000"],
        }
    ).reset_index(drop=True)


def rule_table(rule: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold cut-offs with the flag rate and recall they realise on open and on all test reports."""
    keys = ["model", "policy"]
    on_open = rule[rule["test_group"] == "open_at_filing"].set_index(keys)
    on_all = rule[rule["test_group"] == "all_test_rows"].set_index(keys)
    table = pd.DataFrame(
        {
            "Cut-off (out-of-fold)": on_open["cutoff_oof"],
            "Flag rate, open reports": on_open["flag_rate"],
            "Recall, open reports": on_open["recall"],
            "Flag rate, all reports": on_all["flag_rate"],
            "Recall, all reports": on_all["recall"],
        }
    ).reset_index()
    table = table.sort_values("policy", ascending=False, kind="stable")  # the workload rule first
    table["model"] = table["model"].map(model_label)
    table["policy"] = table["policy"].map(lambda name: RULE_NAMES.get(name, name))
    return table.rename(columns={"model": "Model", "policy": "Policy"}).reset_index(drop=True)


def calibration_summary(block: dict[str, Any]) -> pd.DataFrame:
    """Mean predicted against observed rate and Brier score for each group and model."""
    summary = block.get("open_calibration", {})
    return pd.DataFrame(
        [
            {
                "Reports": label,
                "Model": model_label(model),
                "n": values["n"],
                "Mean predicted": values["mean_predicted"],
                "Observed rate": values["observed_rate"],
                "Predicted minus observed": values["predicted_minus_observed"],
                "Brier": values["brier"],
            }
            for group, label in CALIBRATION_GROUPS.items()
            for model, values in summary.get(group, {}).items()
        ]
    )


def checks_caption(block: dict[str, Any]) -> str:
    """Paired-bootstrap differences in recall under the rule and the open-only refit, as one sentence."""
    boot = block.get("paired_bootstrap_open_at_filing", {})
    metric = f"recall_top_{round(boot.get('share', 0) * 100)}pct"
    parts = []
    for key, entry in boot.get("differences", {}).items():
        if metric in entry:
            d = entry[metric]
            parts.append(
                f"{DIFFERENCE_NAMES.get(key, key)} {d['estimate']:+.3f} "
                f"({d['lower']:+.3f} to {d['upper']:+.3f})"
            )
    text = ""
    if parts:
        text = (
            f"Paired mine-year bootstrap on the {boot['n']:,} reports still open at filing "
            f"({boot['n_boot']:,} draws, {1 - boot['alpha']:.0%} intervals), difference in recall when "
            f"the top {boot['share']:.0%} are reviewed: {'; '.join(parts)}. "
        )
    refit = block.get("open_only_refit")
    if refit:
        text += (
            f"A refit of {CAPACITY_SHORT} on the {refit['training_rows_open']:,} open "
            f"training reports alone gives recall {refit['open_only_refit']['top_75pct']['recall']:.3f} "
            f"under the rule, against {refit['all_rows_model']['top_75pct']['recall']:.3f} for the model "
            "fitted on all filings. These checks were run after the test rows were scored and change "
            "no stored model or threshold."
        )
    return text.strip()


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

(
    comparison_tab,
    curves_tab,
    threshold_tab,
    capacity_tab,
    criteria_tab,
    sensitivity_tab,
    importance_tab,
) = st.tabs(
    [
        "Comparison",
        "Curves",
        "Threshold explorer",
        "Capacity policy",
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
    still_open = policy_row(capacity_policy(), "open_at_filing", "capacity_threshold")
    if still_open is not None:
        st.caption(
            f"The stored capacity threshold of {CAPACITY_SHORT} flags "
            f"{still_open['flag_rate']:.1%} of the reports still open at filing; see the Capacity "
            "policy tab."
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

with capacity_tab:
    policy = capacity_policy()
    st.write(capacity_intro(policy))
    if policy is None:
        followup_missing("The capacity-policy table")
    else:
        chosen_group = st.radio(
            "Reports scored", list(POLICY_GROUPS), horizontal=True, key="ev_policy_group"
        )
        group = POLICY_GROUPS[chosen_group]
        rates = ["Flag rate", "Recall", "Precision", "Specificity", "Cut-off"]
        st.dataframe(
            policy_table(policy, group).style.format(
                {**dict.fromkeys(rates, "{:.3f}"), "Missed per 1,000": "{:.1f}"}
            ),
            hide_index=True,
        )
        in_group = policy[policy["group"] == group]
        st.caption(
            f"{int(in_group['n'].iloc[0]):,} test reports, lost-time prevalence "
            f"{in_group['prevalence'].iloc[0]:.3f}. The capacity threshold is fixed on out-of-fold "
            "scores. For the top-share rows the cut-off is the score that flags exactly that share of "
            "these reports, so they compare models at an equal workload. The hand rule has two score "
            "values only, so its top-share rows break ties by row order and do not rank reports."
        )
    st.markdown("**Deployed cut-off of the workload rule**")
    rule = workload_rule()
    if rule is None:
        followup_missing("The workload-rule table")
    else:
        rule_columns = [
            "Cut-off (out-of-fold)",
            "Flag rate, open reports",
            "Recall, open reports",
            "Flag rate, all reports",
            "Recall, all reports",
        ]
        st.dataframe(
            rule_table(rule).style.format(dict.fromkeys(rule_columns, "{:.3f}")), hide_index=True
        )
        st.caption(
            "The cut-off is the score that the top three in four of the "
            f"{int(rule['train_open_rows'].iloc[0]):,} open-at-filing training reports reach on their "
            "out-of-fold scores. It was fixed before the test rows were scored, so the flag rates "
            "and recalls are realised, not tuned."
        )
    checks = workload_checks()
    with st.expander("Calibration on open reports"):
        if checks is None or not checks.get("open_calibration"):
            followup_missing("The calibration summary")
        else:
            st.dataframe(
                calibration_summary(checks).style.format(
                    {
                        "n": "{:,}",
                        "Mean predicted": "{:.3f}",
                        "Observed rate": "{:.3f}",
                        "Predicted minus observed": "{:+.3f}",
                        "Brier": "{:.3f}",
                    }
                ),
                hide_index=True,
            )
        calibration = open_calibration()
        if calibration is None:
            followup_missing("The calibration bins")
        else:
            st.markdown("**Reliability of the reports still open at filing**")
            bins = calibration[calibration["group"] == "open_at_filing"].assign(
                model=lambda t: t["model"].map(model_label)
            )
            st.altair_chart(
                line_chart(
                    bins,
                    "mean_predicted",
                    "observed_rate",
                    "model",
                    x_title="Mean predicted probability",
                    y_title="Observed share of lost-time cases",
                    diagonal=True,
                ),
                width="stretch",
            )
            st.caption(
                "Ten equal-frequency bins of the predicted probability. Points above the dashed "
                "line mean the observed rate exceeds the prediction, so for these reports the "
                "probability is a ranking score, not a rate."
            )
    if checks is not None:
        st.caption(checks_caption(checks))

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
