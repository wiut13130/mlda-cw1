"""Page 2: exploratory analysis of the training rows, read from the precomputed EDA tables."""

from __future__ import annotations

import streamlit as st
from _common import (
    CLASS_NAMES,
    FEATURES_PATH,
    MAX_CHART_ROWS,
    axis_label,
    bar_chart,
    boxplot,
    ensure_entrypoint,
    heatmap,
    histogram,
    metrics_csv,
    require,
    require_files,
    sample_for,
)

from mlda.config import RANDOM_STATE
from mlda.features import FEATURE_SPEC

ensure_entrypoint(__file__)

TOP_N = 15

st.title("Exploratory analysis")
st.write(
    "Every table and chart describes the **training rows** of the grouped split. The held-out test "
    "rows are not used for any choice made here."
)
require_files(FEATURES_PATH)
summary_tab, distribution_tab, correlation_tab, association_tab, missing_tab = st.tabs(
    [
        "Summary statistics",
        "Distributions",
        "Correlations",
        "Associations with the target",
        "Missingness",
    ]
)

with summary_tab:
    st.caption(
        "Each row summarises one feature of the training rows: location, spread and shape. A mean far "
        "from the median and a large skewness mark the right-skewed experience and exposure columns "
        "that the imputer and scaler choices on the Preprocessing page respond to."
    )
    kind = st.radio("Features", ["Numeric", "Categorical"], horizontal=True)
    file = "eda_describe_numeric.csv" if kind == "Numeric" else "eda_describe_categorical.csv"
    table = require(metrics_csv(file, index_col=0), "The summary statistics")
    st.dataframe(table.round(3))

with distribution_tab:
    st.caption(
        "The histogram overlays the two classes for one numeric feature and the box plot compares their "
        "quartiles and tails. Where the shapes overlap, the feature alone separates lost-time injuries "
        "from the rest only weakly."
    )
    column = st.selectbox("Numeric feature", FEATURE_SPEC["numeric"], format_func=axis_label)
    sample, labels, _ = sample_for(
        MAX_CHART_ROWS, RANDOM_STATE, False, tuple(FEATURE_SPEC["numeric"])
    )
    frame = sample[[column]].assign(outcome=[CLASS_NAMES[int(v)] for v in labels])
    st.caption(
        f"Random stratified sample of {len(frame):,} training rows; missing values are left out of the charts."
    )
    left, right = st.columns(2)
    left.altair_chart(histogram(frame, column, "outcome"), width="stretch")
    right.altair_chart(boxplot(frame, column, "outcome"), width="stretch")

with correlation_tab:
    st.caption(
        "The heatmap shows pairwise correlation among the numeric features: Pearson measures linear "
        "association and Spearman rank association. Pairs where the two disagree are monotone but not "
        "linear, or are driven by outliers, which is why hours and employees are log-transformed."
    )
    method = st.radio("Coefficient", ["Pearson", "Spearman"], horizontal=True)
    matrix = require(
        metrics_csv(f"eda_correlations_{method.lower()}.csv", index_col=0), "The correlation matrix"
    )
    st.altair_chart(heatmap(matrix, method), width="stretch")
    gaps = metrics_csv("eda_correlations_disagreements.csv")
    if gaps is not None:
        st.markdown("**Largest gaps between Pearson and Spearman**")
        st.dataframe(gaps.head(TOP_N).round(3), hide_index=True)
    vif = metrics_csv("eda_vif.csv")
    if vif is not None:
        with st.expander("Variance inflation factors (collinearity)"):
            st.dataframe(vif.round(2), hide_index=True)

with association_tab:
    st.caption(
        "Bars show how strongly each feature is associated with the lost-time label: point-biserial r for "
        "numeric features and bias-corrected Cramer's V for categorical ones. Association is not causation, "
        "and a mine-history feature can rank high because it also reflects how the operator uses restricted duty."
    )
    biserial = require(metrics_csv("eda_point_biserial.csv"), "The point-biserial table")
    cramers = require(metrics_csv("eda_cramers_v.csv"), "The Cramer's V table")
    top_r = biserial.reindex(biserial["r"].abs().sort_values(ascending=False).index).head(TOP_N)
    top_v = cramers.sort_values("cramers_v", ascending=False).head(TOP_N)
    st.write(
        f"Strongest numeric association: **{top_r['feature'].iloc[0]}** (r = {top_r['r'].iloc[0]:.2f}). "
        f"Strongest categorical association: **{top_v['feature'].iloc[0]}** (V = {top_v['cramers_v'].iloc[0]:.2f})."
    )
    left, right = st.columns(2)
    left.altair_chart(
        bar_chart(top_r, "feature", "r", title="Point-biserial r", signed=True), width="stretch"
    )
    right.altair_chart(
        bar_chart(top_v, "feature", "cramers_v", title="Cramer's V"), width="stretch"
    )

with missing_tab:
    st.caption(
        "Each bar is the share of training rows with a missing value in a source column after the error "
        "rules. Structural gaps, such as underground-only fields at surface mines, are kept as their own "
        "level instead of being imputed."
    )
    missing = require(metrics_csv("eda_missingness.csv"), "The missingness table")
    floor = st.slider("Show columns with at least this share missing (%)", 0.0, 50.0, 1.0, step=0.5)
    shown = (
        missing[missing["missing_pct"] >= floor]
        .sort_values("missing_pct", ascending=False)
        .head(25)
    )
    if shown.empty:
        st.info("No column reaches this share.")
    else:
        st.altair_chart(
            bar_chart(shown, "column", "missing_pct", title="Missing (%)"), width="stretch"
        )
        st.dataframe(shown.round(2), hide_index=True)
