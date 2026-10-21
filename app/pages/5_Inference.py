"""Page 5: score one accident or a CSV of accidents with a saved model."""

from __future__ import annotations

from datetime import date, time
from itertools import cycle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from _common import (
    BASE_PATH,
    BATCH_REQUIRED,
    CASE_HISTORY_PATH,
    CODE_FIELDS,
    FOLLOWUP_COMMAND,
    MAX_BATCH_ROWS,
    MAX_TABLE_ROWS,
    MINE_ID_RULE,
    MINE_YEAR_PATH,
    NONE_LEVEL,
    STATUS_103I_LEVELS,
    axis_label,
    bar_chart,
    best_model,
    coordinate,
    cost_note,
    ensure_entrypoint,
    explain_row,
    hhmm,
    index_of,
    level_select,
    load_pipeline,
    mine_history,
    mine_profiles,
    mine_status_103i,
    model_label,
    model_thresholds,
    normalise_mine_id,
    prepare_batch,
    ranked_models,
    raw_case_frame,
    require,
    require_files,
    row_shap,
    score_frame,
    shap_supported,
    template_csv,
    workload_cutoff,
)

from mlda.config import COST_RATIOS, OPERATING_SPECIFICITY, YEAR_MAX, YEAR_MIN
from mlda.features import FeatureEngineer
from mlda.operating import OPERATING_KEY

ensure_entrypoint(__file__)

MINE_FIELDS = {
    "COAL_METAL_IND": "Sector",
    "PRIMARY_CANVASS": "Commodity group",
    "FIPS_STATE_CD": "State",
}
CASE_FIELDS = [
    c for c in CODE_FIELDS if c not in (*MINE_FIELDS, "IMMED_NOTIFY_CD")
]  # the circumstances of the accident
WORKER_FIELDS = (
    ("TOT_EXPER", "Total experience (years)", 8.0),
    ("MINE_EXPER", "Experience at this mine (years)", 3.0),
    ("JOB_EXPER", "Experience in this job (years)", 2.0),
)
HISTORY_FIELDS = (
    ("prev_hours", "Employee-hours", 1000.0),
    ("prev_employees", "Employees", 1.0),
    ("prev_incidents", "Cases filed", 1.0),
    ("prev_valid_cases", "Valid cases (01 to 06)", 1.0),
    ("prev_lost_time_cases", "Lost-time cases", 1.0),
)
TOP_N = 12
RULE_HELP = "top three in four of reports still open at filing; cut-off from out-of-fold training scores"
RULE_CAPTION = (
    "For a report with no return-to-work date yet, use the workload rule; the probability is calibrated to all "
    "filings and runs about 0.12 below the observed rate on open reports, so read it as a ranking score there. "
    "The cost note below applies to all filings."
)


def mine_block() -> tuple[str, date, dict[str, Any] | None, dict[str, Any]] | None:
    """Mine id and accident date (outside the form so the lookup updates at once), with the lookup.

    Three cases are told apart, because they need different action from the reader. A malformed id is an
    error and returns None: padding "abc" to "0000abc" used to score it as an unmatched mine, so nothing is
    scored until the id is a number. A well-formed id that the reference data does not know is a warning (the
    documented fallback for a new mine: the reader enters the context by hand). A known mine without
    previous-year production is a notice only, since its profile is looked up and just the exposure is missing.
    """
    profiles = mine_profiles()
    examples = profiles["n_cases"].nlargest(3).index.tolist()
    id_col, date_col = st.columns(2)
    entered = id_col.text_input("MINE_ID (7 digits)", examples[0], max_chars=7, key="mine_id")
    mine_id = normalise_mine_id(entered)
    accident_date = date_col.date_input(
        "Accident date",
        date(YEAR_MAX, 6, 2),
        date(YEAR_MIN, 1, 1),
        date(YEAR_MAX + 1, 12, 31),
        key="accident_date",
    )
    if mine_id is None:
        st.error(f"{MINE_ID_RULE}. Enter 1 to 7 digits; a shorter number is padded with leading zeros.")
        return None
    profile = profiles.loc[mine_id].to_dict() if mine_id in profiles.index else None
    history = mine_history(mine_id, accident_date)
    previous = accident_date.year - 1
    if history["prev_year_matched"]:
        st.success(
            f"Mine {mine_id} is in the previous-year table: {history['prev_hours']:,.0f} employee-hours and "
            f"{history['prev_employees']:,.0f} employees in {previous}; "
            f"{history['prev_incidents']:.0f} cases filed before "
            f"{accident_date}, {history['prev_lost_time_cases']:.0f} of them lost-time. Mines with "
            f"many cases: {', '.join(examples)}."
        )
        if profile is None:
            st.info(
                f"Mine {mine_id} has no accident in the training data, so its profile is not on file: sector, "
                "commodity group, state and location start at defaults. Check them under 'Mine context'."
            )
    elif profile is None:
        st.warning(
            f"Mine {mine_id} is unknown to the reference data: it has no row in the mine profile table and no "
            f"production record for {previous}. Its sector, commodity group, state and location are blank or "
            "at defaults, and the exposure is 0, so enter them by hand under 'Mine context' (a new mine "
            "is scored this way). Scoring continues."
        )
    else:
        st.info(
            f"No previous-year production record for mine {mine_id} and {previous}. Enter the exposure by hand under "
            "'Mine context', or leave the hours at 0 to score the case as unmatched."
        )
    return mine_id, accident_date, profile, history


def case_form(
    mine_id: str, accident_date: date, profile: dict[str, Any] | None, history: dict[str, Any]
) -> dict | None:
    """The accident form; returns the raw field values when it is submitted, otherwise None."""
    suffix = f"{mine_id}_{accident_date.year}"  # a new mine or year resets the looked-up defaults
    known, matched = profile or {}, bool(history["prev_year_matched"])
    fields: dict[str, Any] = {
        "MINE_ID": mine_id,
        "ACCIDENT_DT": pd.Timestamp(accident_date),
        "STATE": known.get("STATE"),
    }
    exposure: dict[str, float] = {}
    with st.form("case_form"):
        st.markdown("**What happened**")
        for slot, column in zip(cycle(st.columns(3)), CASE_FIELDS):
            with slot:
                fields[column] = level_select(axis_label(column), column, f"f_{column}")
        st.markdown("**Who and when**")
        row = st.columns(4)
        for slot, (column, label, default) in zip(row[:3], WORKER_FIELDS, strict=True):
            fields[column] = slot.number_input(label, 0.0, 60.0, default, 0.5, key=f"f_{column}")
        fields["NO_INJURIES"] = row[3].number_input("People injured", 1, 36, 1, key="f_injured")
        row = st.columns(4)
        fields["ACCIDENT_TIME"] = hhmm(
            row[0].time_input("Accident time", time(10, 30), step=60, key="f_time")
        )
        fields["SHIFT_BEGIN_TIME"] = hhmm(
            row[1].time_input("Shift start", time(7, 0), step=60, key="f_shift")
        )
        fields["CONTRACTOR_ID"] = (
            "contractor" if row[2].checkbox("Contractor employee", key="f_contractor") else np.nan
        )
        with row[3]:
            fields["IMMED_NOTIFY_CD"] = level_select(
                "Immediate notification", "IMMED_NOTIFY_CD", "f_notify", NONE_LEVEL
            )
        with st.expander("Mine context: looked up from the mine, edit to override"):
            left, middle, right = st.columns(3)
            with left:
                for column, label in MINE_FIELDS.items():
                    fields[column] = level_select(
                        label, column, f"f_{column}_{suffix}", known.get(column)
                    )
            with middle:
                status = mine_status_103i(mine_id, accident_date)
                fields["status_103i_at_accident"] = st.selectbox(
                    "Section 103(i) status",
                    STATUS_103I_LEVELS,
                    index=index_of(list(STATUS_103I_LEVELS), status),
                    key=f"f_103i_{mine_id}_{accident_date.isoformat()}",
                    help="Looked up from the mine's recorded status and its date, as in the training data.",
                )
            with right:
                for column, bound in (("LATITUDE", 90.0), ("LONGITUDE", 180.0)):
                    default = coordinate(known.get(column), bound)
                    label = f"{column.capitalize()} (0 if unknown)"
                    fields[column] = st.number_input(
                        label,
                        -bound,
                        bound,
                        default,
                        0.0001,
                        format="%.4f",
                        key=f"f_{column}_{suffix}",
                    )
            st.markdown("Previous-year exposure and cases (hours of 0 mean the mine is unmatched)")
            for slot, (column, label, step) in zip(st.columns(5), HISTORY_FIELDS, strict=True):
                default = float(history[column]) if matched else 0.0
                exposure[column] = slot.number_input(
                    label, 0.0, 1e9, default, step, format="%.0f", key=f"f_{column}_{suffix}"
                )
        submitted = st.form_submit_button("Score this case", type="primary")
    if not submitted:
        return None
    use = exposure["prev_hours"] > 0
    fields.update({column: (value if use else np.nan) for column, value in exposure.items()})
    fields.update(
        prev_year_matched=use, type_at_time=history["type_at_time"] if use and matched else np.nan
    )
    return fields


def show_result(raw: pd.DataFrame, model_name: str) -> None:
    """Probability, decisions at the stored thresholds, an expected-cost note and an explanation."""
    pipeline = load_pipeline(model_name)
    features = FeatureEngineer().transform(raw)
    probability = float(pipeline.predict_proba(features)[0, 1])
    thresholds = model_thresholds(model_name)
    operating = thresholds.get(OPERATING_KEY, 0.5)
    cutoff = workload_cutoff(model_name)
    st.subheader(
        f"Result for mine {raw['MINE_ID'].iloc[0]} on {raw['ACCIDENT_DT'].iloc[0]:%Y-%m-%d}"
    )
    dated = st.selectbox(
        "Return-to-work date already on the form?",
        ("No", "Yes"),
        key="inf_dated",
        help="The workload rule is meant for reports that are still open at filing.",
    ) == "Yes"
    if dated:
        st.info("The outcome is recorded on the form, so the score is for information only.")
    first, second, third, fourth = st.columns(4)
    first.metric("Probability of a lost-time injury", f"{probability:.1%}", border=True)
    if cutoff is None:
        second.metric(
            "Workload rule (open reports)",
            "not available",
            border=True,
            help=f"No cut-off for this model in followup_workload_rule.csv. Create it with `{FOLLOWUP_COMMAND}`.",
        )
    else:
        second.metric(
            "Workload rule (open reports)",
            "Review" if probability >= cutoff else "Routine handling",
            None if dated else "Applies to this report",
            delta_color="green",
            delta_arrow="off",
            border=True,
            help=f"{RULE_HELP} ({cutoff:.3f}).",
        )
    decision = "Flag for early review" if probability >= operating else "Routine handling"
    help_text = (
        f"Threshold {operating:.3f}, chosen on out-of-fold scores so that {OPERATING_SPECIFICITY:.0%} "
        "of non-lost-time cases stay unflagged."
    )
    third.metric("Capacity threshold (all filings)", decision, border=True, help=help_text)
    fourth.metric(
        "At threshold 0.50", "Lost time" if probability >= 0.5 else "No lost time", border=True
    )
    st.caption(RULE_CAPTION)
    if cutoff is None:
        st.caption(
            f"The workload-rule cut-off of this model is not available. Create it with `{FOLLOWUP_COMMAND}`."
        )

    ratio = st.select_slider(
        "Cost of a missed case relative to an unnecessary review",
        COST_RATIOS,
        3,
        lambda r: f"{r}:1",
        key="inf_ratio",
        help="Computed at the model's probability on all filings, not on reports still open at filing.",
    )
    st.info(cost_note(probability, ratio, thresholds.get("cost_optimal", {}).get(str(ratio))))

    st.subheader("Why this score")
    explanation = explain_row(model_name, pipeline, features)
    if explanation is not None:
        title, table, value, row_specific = explanation
        signed = bool((table[value] < 0).any())
        st.altair_chart(
            bar_chart(table.head(TOP_N), "feature", value, title=title, signed=signed),
            width="stretch",
        )
    else:
        row_specific = False
    if row_specific:
        return
    if not shap_supported(pipeline):
        st.caption(
            "No row-level explanation exists for this model type: it has no coefficients, no tree importances and "
            "no fast SHAP explainer. "
            + (
                "The chart above is a global importance."
                if explanation is not None
                else "Only the probability is shown."
            )
            + " The Evaluation page lists the held-out permutation importance of the best model."
        )
    elif st.button("Explain this case with SHAP", key="inf_shap"):
        with st.spinner("Computing SHAP values for this row"):
            shap_table = row_shap(pipeline, features)
        if shap_table is None:
            st.info("SHAP could not explain this estimator within the time limit of the app.")
        else:
            chart = bar_chart(
                shap_table.head(TOP_N),
                "feature",
                "shap",
                title="SHAP value for this case (log-odds)",
                signed=True,
            )
            st.altair_chart(chart, width="stretch")


def batch_section(model_name: str) -> None:
    """Score an uploaded CSV of raw accidents and offer the predictions for download."""
    st.write("Upload a CSV with one accident per row. Required columns:")
    st.code(", ".join(BATCH_REQUIRED), language="text")
    st.caption(
        "Optional: CONTRACTOR_ID, IMMED_NOTIFY_CD, and any mine or history column (otherwise they "
        "are looked up from MINE_ID and "
        f"ACCIDENT_DT). At most {MAX_BATCH_ROWS:,} rows."
    )
    st.download_button(
        "Download a template (five real rows)",
        template_csv(),
        "batch_template.csv",
        "text/csv",
        key="inf_template",
    )
    upload = st.file_uploader("CSV file", type="csv", key="inf_upload")
    if upload is None:
        return
    try:
        uploaded = pd.read_csv(upload, dtype=str)
        raw = prepare_batch(uploaded)
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Could not read the file: {error}")
        return
    probability = score_frame(load_pipeline(model_name), raw)
    operating = model_thresholds(model_name).get(OPERATING_KEY, 0.5)
    cutoff = workload_cutoff(model_name)
    flags: dict[str, Any] = {"flag_operating_point": probability >= operating}
    if cutoff is not None:  # without a cut-off the column is left out, so that no blank can read as "routine"
        flags["flag_workload_rule"] = (probability >= cutoff).astype(int)
    result = uploaded.assign(probability=probability, **flags, flag_at_0_5=probability >= 0.5)
    first, second, third, fourth = st.columns(4)
    first.metric("Rows scored", f"{len(result):,}", border=True)
    second.metric(
        "Flagged at the capacity threshold",
        f"{result['flag_operating_point'].mean():.1%}",
        border=True,
    )
    if cutoff is None:
        third.metric(
            "Flagged under the workload rule",
            "not available",
            border=True,
            help=f"No cut-off for this model in followup_workload_rule.csv. Create it with `{FOLLOWUP_COMMAND}`.",
        )
    else:
        flagged = int(result["flag_workload_rule"].sum())
        third.metric(
            "Flagged under the workload rule",
            f"{flagged:,}",
            border=True,
            help=f"{flagged / len(result):.1%} of the rows at the cut-off {cutoff:.3f}.",
        )
    fourth.metric(
        "Mines without previous-year data", f"{(~raw['prev_year_matched']).sum():,}", border=True
    )
    st.caption(
        "flag_workload_rule is for reports with no return-to-work date yet; flag_operating_point and "
        "flag_at_0_5 are for all filings."
        if cutoff is not None
        else f"The workload-rule cut-off of this model is not available. Create it with `{FOLLOWUP_COMMAND}`."
    )
    st.dataframe(result.head(MAX_TABLE_ROWS), hide_index=True)
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions (CSV)", csv, "predictions.csv", "text/csv", key="inf_download"
    )


st.title("Inference")
st.write(
    "Describe an accident, or upload a CSV, and a saved model returns the probability that the case "
    "becomes a lost-time injury. "
    "Each row goes through `FeatureEngineer`, the same code that built the training features, then "
    "through the model's fitted "
    "preprocessor. The score supports a case reviewer; it does not replace one."
)
require_files(BASE_PATH, MINE_YEAR_PATH, CASE_HISTORY_PATH)
names = require(ranked_models(), "A saved model")
model_name = st.selectbox(
    "Model", names, index=index_of(names, best_model()), format_func=model_label, key="inf_model"
)

single_tab, batch_tab = st.tabs(["One accident", "Batch (CSV)"])
with single_tab:
    # A malformed id skips the form and any earlier result without st.stop(), which would also blank the Batch tab.
    if (block := mine_block()) is not None:
        submitted = case_form(*block)
        if submitted is not None:
            st.session_state["inf_case"] = raw_case_frame(submitted)
        if "inf_case" in st.session_state:
            show_result(st.session_state["inf_case"], model_name)
        else:
            st.info("Fill in the form and press **Score this case**.")
with batch_tab:
    batch_section(model_name)
