"""Page 1: the three MSHA sources, how they are joined, the target and a filterable sample."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _common import (
    BASE_PATH,
    base_filter_options,
    base_sample,
    base_summary,
    data_dictionary,
    ensure_entrypoint,
    require,
    require_files,
    snapshot_label,
    source_table,
    stamp,
)

from mlda.config import EXCLUDED_CODES, POSITIVE_CODES, TARGET_SOURCE_COL, YEAR_MAX, YEAR_MIN

ensure_entrypoint(__file__)

SECTORS = {"All": "All", "C": "Coal", "M": "Metal and non-metal"}

st.title("Data explorer")
st.write(
    "Three open files from the US Mine Safety and Health Administration (MSHA) are joined into one "
    "table with one row per injured person. This page documents the sources, the joins and the "
    "target, and lets you browse the result."
)
require_files(BASE_PATH)
summary = require(base_summary(), "The base table")
sources_tab, joins_tab, target_tab, sample_tab, dictionary_tab = st.tabs(
    ["Sources", "Joins", "Target", "Filterable sample", "Data dictionary"]
)

with sources_tab:
    st.caption(
        f"Files pulled {snapshot_label()}. Licence: works of the US federal government, public domain "
        "(17 U.S.C. 105), so they may be copied, redistributed and analysed freely."
    )
    st.dataframe(
        source_table(),
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
            "rows": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.metric(
        "Analytic base table",
        f"{summary['n_rows']:,} rows x {summary['n_columns']} columns",
        help=f"Accident years {YEAR_MIN} to {YEAR_MAX}, degree-of-injury codes 01 to 06.",
        border=True,
    )

with joins_tab:
    n = summary["n_rows"]
    matches = summary["matches"]
    joins = [
        (
            "J1",
            "Accidents to Mines",
            "MINE_ID",
            "_m_mines",
            "Mine attributes: commodity, size flags, coordinates, 103(i) status.",
        ),
        (
            "J2",
            "Accidents to Production, previous year",
            "MINE_ID, CAL_YR - 1",
            "_m_prev",
            "Hours, employees, mine type and lagged case counts (model inputs).",
        ),
        (
            "J3",
            "Accidents to Production, same year",
            "MINE_ID, CAL_YR",
            "_m_same",
            "Exploratory rates only; never a model input, the year is unfinished at filing.",
        ),
    ]
    table = pd.DataFrame(
        {
            "join": [j[0] for j in joins],
            "tables": [j[1] for j in joins],
            "keys": [j[2] for j in joins],
            "matched rows": [matches[j[3]] for j in joins],
            "match rate": [matches[j[3]] / n for j in joins],
            "unmatched rows": [n - matches[j[3]] for j in joins],
            "use": [j[4] for j in joins],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={"match rate": st.column_config.NumberColumn(format="percent")},
    )
    st.markdown(
        'All joins are **left joins from the accident rows** with `validate="many_to_one"` and an '
        "indicator column, and the row count is asserted unchanged after each one. Unmatched rows are "
        "kept; their exposure features are missing and the flag `prev_year_matched` records it.\n\n"
        "The model uses the **previous** year's production because Form 7000-2 is filed quarterly: for an "
        "accident in the first quarter, no part of the current year has been reported when the injury "
        "form is filed, while the last quarter of the previous year is due by 15 January. Case counts "
        "from the previous year include only cases filed strictly before the accident date."
    )

with target_tab:
    target = summary["target"].copy()
    target["class"] = [
        "lost time (positive)" if c in POSITIVE_CODES else "no lost time (negative)"
        for c in target["code"]
    ]
    target["share"] = target["rows"] / summary["n_rows"]
    st.markdown(
        f"`lost_time = 1` when `{TARGET_SOURCE_COL}` is 01 to 04 (fatality, permanent disability, days "
        "away from work with or without restricted activity); `0` for 05 and 06. Codes 01 to 04 are the "
        "trigger for wage-replacement (indemnity) payments in workers' compensation."
    )
    st.dataframe(
        target,
        hide_index=True,
        column_config={
            "rows": st.column_config.NumberColumn(format="%d"),
            "share": st.column_config.NumberColumn(format="percent"),
        },
    )
    left, right = st.columns(2)
    left.metric("Prevalence of lost time", f"{summary['prevalence']:.1%}", border=True)
    right.metric(
        "Excluded degree codes",
        ", ".join(EXCLUDED_CODES),
        help="No injured person, illness, natural causes, non-employee, other.",
        border=True,
    )

with sample_tab:
    options = base_filter_options(stamp(BASE_PATH))
    sector_col, year_col, state_col = st.columns(3)
    sector = sector_col.radio("Sector", list(SECTORS), format_func=SECTORS.get, horizontal=True)
    years = year_col.slider(
        "Accident year", options["years"][0], options["years"][1], options["years"]
    )
    states = state_col.multiselect("State (empty means all)", options["states"])
    sample, n_match = base_sample(sector, tuple(years), tuple(states), stamp(BASE_PATH))
    if sample.empty:
        st.info("No rows match these filters.")
    else:
        st.caption(f"{n_match:,} rows match; the table shows a random sample of {len(sample):,}.")
        st.dataframe(sample, hide_index=True)
        st.download_button(
            "Download the filtered sample (CSV)",
            sample.to_csv(index=False).encode("utf-8"),
            file_name="base_sample.csv",
            mime="text/csv",
        )

with dictionary_tab:
    dictionary = data_dictionary()
    kinds = st.multiselect(
        "Feature type",
        sorted(dictionary["type"].unique()),
        default=sorted(dictionary["type"].unique()),
    )
    st.dataframe(dictionary[dictionary["type"].isin(kinds)], hide_index=True)
    st.caption(
        "Origin tells which source a feature comes from; 'Derived' features are computed from other columns."
    )
