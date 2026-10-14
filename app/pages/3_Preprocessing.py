"""Page 3: choose preprocessing options and watch a training sample change."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _common import (
    FEATURES_PATH,
    SPLIT_PATH,
    axis_label,
    ensure_entrypoint,
    histogram,
    preprocess_sample,
    require_files,
    stamp,
)

from mlda.config import ONE_HOT_MIN_FREQUENCY, RANDOM_STATE
from mlda.features import FEATURE_SPEC
from mlda.preprocessing import OPTIONS

ensure_entrypoint(__file__)

MAX_SAMPLE = 10_000

st.title("Preprocessing choices")
st.write(
    "Each control is an argument of `mlda.preprocessing.build_preprocessor`. The transformer is fitted on a "
    "stratified sample of **training rows only**, so nothing is learnt from the held-out test rows. Change "
    "an option and the matrix below is rebuilt."
)
require_files(FEATURES_PATH, SPLIT_PATH)


def show_none(value: str | None) -> str:
    """Label for an option whose value is None (the outlier rule 'no clipping'), which a selectbox cannot show."""
    return "none" if value is None else value


imputer_col, scaler_col, encoder_col, outlier_col = st.columns(4)
imputer = imputer_col.selectbox(
    "Numeric imputer", list(OPTIONS["numeric_imputer"]), key="pp_imputer"
)
scaler = scaler_col.selectbox("Scaler", list(OPTIONS["scaler"]), key="pp_scaler")
encoder = encoder_col.selectbox("Categorical encoder", list(OPTIONS["encoder"]), key="pp_encoder")
outlier_rule = outlier_col.selectbox(
    "Outlier rule", list(OPTIONS["outlier_rule"]), format_func=show_none, key="pp_outlier"
)
freq_col, size_col = st.columns(2)
min_frequency = freq_col.slider(
    "min_frequency of a one-hot level (share of rows)",
    0.0005,
    0.05,
    ONE_HOT_MIN_FREQUENCY,
    step=0.0005,
    format="%.4f",
    key="pp_min_frequency",
    help="Levels rarer than this share of the sample share one 'infrequent' column. Used by the one-hot encoder only.",
)
n_rows = size_col.slider(
    "Training rows in the sample", 1_000, MAX_SAMPLE, 5_000, step=1_000, key="pp_rows"
)

with st.expander("Why each option exists", expanded=True):
    for name, choice in (
        ("numeric_imputer", imputer),
        ("scaler", scaler),
        ("encoder", encoder),
        ("outlier_rule", outlier_rule),
    ):
        st.markdown(
            f"**{name.replace('_', ' ').capitalize()} = {show_none(choice)}.** {OPTIONS[name][choice]}"
        )
if imputer == "knn":
    st.caption(
        "The KNN imputer searches neighbours among all sample rows, so it is much slower than the median."
    )

result = preprocess_sample(
    (imputer, scaler, encoder, outlier_rule, float(min_frequency)),
    n_rows,
    RANDOM_STATE,
    stamp(FEATURES_PATH, SPLIT_PATH),
)
n_before = result.n_inputs
n_missing_before = int(result.numeric_raw.isna().sum().sum())
n_missing_after = int(pd.isna(result.matrix).sum())

shape_col, width_col, nan_col, time_col = st.columns(4)
shape_col.metric(
    "Transformed shape", f"{result.matrix.shape[0]:,} x {result.matrix.shape[1]:,}", border=True
)
width_col.metric(
    "Columns before and after", f"{n_before} to {result.matrix.shape[1]:,}", border=True
)
nan_col.metric(
    "Missing values after",
    f"{n_missing_after:,}",
    delta=f"{n_missing_before:,} before",
    delta_color="off",
    delta_arrow="off",
    border=True,
)
time_col.metric("Fit and transform time", f"{result.seconds:.2f} s", border=True)

st.subheader("First rows of the transformed matrix")
default_columns = result.names[: min(12, len(result.names))]
shown = st.multiselect("Columns to show", result.names, default=default_columns)
if shown:
    positions = [result.names.index(name) for name in shown]
    st.dataframe(pd.DataFrame(result.matrix[:10, positions], columns=shown), hide_index=True)

st.subheader("One numeric column before and after")
column = st.selectbox(
    "Numeric column", FEATURE_SPEC["numeric"], format_func=axis_label, key="pp_column"
)
before = pd.DataFrame({"value": result.numeric_raw[column].astype(float)})
after = pd.DataFrame({"value": result.matrix[:, result.names.index(column)].astype(float)})
left, right = st.columns(2)
left.altair_chart(histogram(before, "value", title="Before (raw feature)"), width="stretch")
right.altair_chart(
    histogram(after, "value", title="After (imputed, clipped and scaled)"), width="stretch"
)
st.caption(
    f"Before: {int(before['value'].isna().sum()):,} missing values in this column. "
    "After: none, because the imputer fills them with a statistic learnt on the training sample."
)
