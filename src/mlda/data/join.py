"""Analytic base table: filtered accidents joined to mine attributes, lagged exposure and history.

One row is one injured person on one Form 7000-1 in 2010-2025 with a valid degree-of-injury
code. Each row receives the mine's stable attributes, the previous calendar year's production
(hours, employees, subunit set) and the previous year's case history restricted to cases filed
before the accident date. The previous year is used because at filing time the current year's
quarterly employment reports are incomplete and the current year's cases would include the row
itself. The filter sequence and the expected counts are in
docs/profile/decisions.json (row_filter_sequence, join).

Four columns go beyond the contract and exist only for the sensitivity runs (they are never model
inputs): the lagged case counts rebuilt from cases that were already resolved at the accident
(``*_resolved``), and ``prev_year_incomplete_at_filing`` for forms filed before the previous
year's fourth-quarter employment report was due.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd

from mlda.config import (
    EXCLUDED_CODES,
    MINES_KEEP_COLUMNS,
    NEGATIVE_CODES,
    POSITIVE_CODES,
    PROCESSED_DIR,
    TARGET,
    TARGET_SOURCE_COL,
    YEAR_MAX,
    YEAR_MIN,
)
from mlda.data.load import load_accidents, load_mines, load_production
from mlda.data.schema import PERSIST_DROP_COLUMNS, SchemaError, validate_base

log = logging.getLogger(__name__)

BASE_PARQUET = PROCESSED_DIR / "base.parquet"
MINE_YEAR_PARQUET = PROCESSED_DIR / "mine_year_features.parquet"
CASE_HISTORY_PARQUET = PROCESSED_DIR / "mine_case_history.parquet"

UNDERGROUND_SUBUNITS = frozenset({"01", "02"})
FACILITY_SUBUNITS = frozenset({"30", "99", "17"})
VALID_CODES: tuple[str, ...] = (*POSITIVE_CODES, *NEGATIVE_CODES)
# Case history starts one year before the window so that 2010 rows have lagged counts.
HISTORY_YEAR_MIN = YEAR_MIN - 1
# 30 CFR 50.30: the quarterly employment report is due 15 days after the quarter ends, so a
# form filed on or before 15 January cannot rely on the previous year's fourth-quarter hours.
Q4_REPORT_DUE = (1, 15)
RESOLVED_SUFFIX = "_resolved"


# ---------------------------------------------------------------------------
# Step 1: row filters
# ---------------------------------------------------------------------------


def filter_rows(accidents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the row-filter sequence of decisions.json and return (rows kept, step log).

    The steps are applied one at a time, rather than as a single mask, so that the count
    removed by each rule can be reported (Table 2 of the report and data/README.md).
    """
    code = accidents[TARGET_SOURCE_COL]
    steps: list[tuple[str, pd.Series]] = [
        (f"CAL_YR >= {YEAR_MIN}", accidents["CAL_YR"] >= YEAR_MIN),
        (f"CAL_YR <= {YEAR_MAX}", accidents["CAL_YR"] <= YEAR_MAX),
    ]
    steps += [(f"drop {TARGET_SOURCE_COL} {c}", code != c) for c in EXCLUDED_CODES]
    steps.append((f"drop {TARGET_SOURCE_COL} missing", code.notna()))

    keep = pd.Series(True, index=accidents.index)
    records: list[dict[str, object]] = [{"step": "all rows", "removed": 0, "kept": int(keep.sum())}]
    for name, condition in steps:
        before = int(keep.sum())
        keep &= condition.fillna(False).astype(bool)
        after = int(keep.sum())
        records.append({"step": name, "removed": before - after, "kept": after})
        log.info("filter %-32s removed %7d kept %7d", name, before - after, after)

    kept = accidents.loc[keep].copy()
    # Belt and braces: the remaining codes must be exactly the positive and negative sets.
    unexpected = kept[TARGET_SOURCE_COL][~kept[TARGET_SOURCE_COL].isin(VALID_CODES)].unique()
    if len(unexpected):
        raise ValueError(f"unexpected degree codes after filtering: {sorted(unexpected)}")
    return kept, pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Steps 3, 6, 7, 8: helpers that are also used by the app
# ---------------------------------------------------------------------------


def filing_date(document_no: pd.Series) -> pd.Series:
    """Decode the filing date embedded in DOCUMENT_NO (positions 2-8, Julian YYYYDDD).

    The accident date is when the injury happened; the filing date is when MSHA received the
    form. Only the latter tells whether a prior case was knowable when a later form was filed.
    Unparsable values become NaT and are never counted as "filed before".
    """
    return pd.to_datetime(document_no.astype("string").str[1:8], format="%Y%j", errors="coerce")


def type_from_subunits(subunits: Iterable[str]) -> str:
    """Mine type implied by the subunits with a production row in a year (contract step 7).

    A subunit counts when MinesProdYearly has a row for it, even if the row reports zero hours.
    """
    codes = frozenset(subunits)
    if codes & UNDERGROUND_SUBUNITS:
        return "Underground"
    if codes <= FACILITY_SUBUNITS:
        return "Facility"
    return "Surface"


def type_from_subunit_code(code: object) -> str:
    """Fallback mine type from the accident's own SUBUNIT_CD when no t-1 production exists."""
    if code in UNDERGROUND_SUBUNITS:
        return "Underground"
    if code in FACILITY_SUBUNITS:
        return "Facility"
    return "Surface"


def mine_year_table(production: pd.DataFrame) -> pd.DataFrame:
    """Aggregate production to one row per (MINE_ID, CALENDAR_YR): hours, employees, subunits, type.

    Hours and employees are summed over subunits; the subunit set is kept as a frozenset so
    the type rule can be applied once per mine-year instead of once per accident row.
    """
    table = (
        production.groupby(["MINE_ID", "CALENDAR_YR"], sort=False)
        .agg(hours=("ANNUAL_HRS", "sum"), employees=("AVG_ANNUAL_EMPL", "sum"), subunits=("SUBUNIT_CD", frozenset))
        .reset_index()
    )
    table["type"] = table["subunits"].map(type_from_subunits)
    log.info("production aggregated to %d mine-years", len(table))
    return table


def case_history_table(accidents: pd.DataFrame) -> pd.DataFrame:
    """Every Accidents case (any degree code) from HISTORY_YEAR_MIN on, with its dates.

    This is the lookup the lagged case counts are built from, and the table persisted for the
    inference page so that the same counts can be rebuilt for any accident date. Besides the
    filing date it keeps the return-to-work date and the filing date of the closing document
    (``closed_date``), which show when a case was resolved; both are NaT when absent or unparsable.
    """
    history = accidents.loc[
        accidents["CAL_YR"] >= HISTORY_YEAR_MIN,
        ["MINE_ID", "CAL_YR", "DOCUMENT_NO", TARGET_SOURCE_COL, "RETURN_TO_WORK_DT", "CLOSED_DOC_NO"],
    ].copy()
    history["filing_date"] = filing_date(history.pop("DOCUMENT_NO"))
    history["closed_date"] = filing_date(history.pop("CLOSED_DOC_NO"))
    history = history.rename(columns={"RETURN_TO_WORK_DT": "return_to_work_dt"})
    columns = ["MINE_ID", "CAL_YR", "filing_date", TARGET_SOURCE_COL, "return_to_work_dt", "closed_date"]
    return history[columns].reset_index(drop=True)


def lagged_case_counts(base: pd.DataFrame, cases: pd.DataFrame, resolved_before: bool = False) -> pd.DataFrame:
    """Counts of previous-year cases at the same mine filed strictly before each row's accident.

    Vectorised: every base row is paired with every case at its mine in CAL_YR - 1 (about
    1.6 million pairs on the full data), the pairs are filtered on filing_date < ACCIDENT_DT and
    counted per row. Rows with t-1 production but no qualifying case get 0; rows without t-1
    production get NaN, because exposure and history are missing on the same rows.

    ``resolved_before=True`` is the sensitivity variant: a case also has to be resolved by the
    accident date, meaning that its return-to-work date and its closing document, when present,
    are both earlier than ACCIDENT_DT. The default counts cases by their final degree code even
    when the case was still open at the accident (68,036 pairs have a later return-to-work date),
    which is what the filing-date rule of the contract allows.
    """
    keys = pd.DataFrame(
        {
            "_row": np.arange(len(base)),
            "MINE_ID": base["MINE_ID"].to_numpy(),
            "_prev_yr": (base["CAL_YR"] - 1).to_numpy(),
            "ACCIDENT_DT": base["ACCIDENT_DT"].to_numpy(),
        }
    )
    history = cases.rename(columns={"CAL_YR": "_case_yr"})
    pairs = keys.merge(history, left_on=["MINE_ID", "_prev_yr"], right_on=["MINE_ID", "_case_yr"], how="inner")
    n_pairs = len(pairs)
    keep = pairs["filing_date"] < pairs["ACCIDENT_DT"]
    if resolved_before:
        still_open = (pairs["return_to_work_dt"] >= pairs["ACCIDENT_DT"]) | (pairs["closed_date"] >= pairs["ACCIDENT_DT"])
        keep &= ~still_open.fillna(False)
    pairs = pairs.loc[keep]
    log.info(
        "lagged cases%s: %d (row, t-1 case) pairs, %d counted",
        " (resolved before the accident)" if resolved_before else "", n_pairs, len(pairs),
    )

    code = pairs[TARGET_SOURCE_COL]
    pairs = pairs.assign(_valid=code.isin(VALID_CODES), _lost=code.isin(POSITIVE_CODES))
    counts = pairs.groupby("_row").agg(
        prev_incidents=("_row", "size"),
        prev_valid_cases=("_valid", "sum"),
        prev_lost_time_cases=("_lost", "sum"),
    )
    counts = counts.reindex(range(len(base))).fillna(0).astype("float64")
    counts.loc[~base["prev_year_matched"].to_numpy(), :] = np.nan
    counts.index = base.index
    return counts


def previous_year_incomplete(filing: pd.Series, cal_yr: pd.Series) -> pd.Series:
    """True when the form was filed on or before 15 January of CAL_YR (unparsable dates: False).

    For those rows the fourth-quarter employment report of the previous year was not yet due,
    so the previous-year hours and employees joined to them may be incomplete.
    """
    month, day = Q4_REPORT_DUE
    cutoff = pd.to_datetime({"year": cal_yr.astype("int64"), "month": month, "day": day})
    return (filing <= cutoff).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Steps 5 and 6: joins
# ---------------------------------------------------------------------------


def _join_mines(base: pd.DataFrame, mines: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Left-join the allowed mine columns; return the frame and the COAL_METAL_IND agreement rate."""
    keep = [*MINES_KEEP_COLUMNS, "COAL_METAL_IND"]
    right = mines[["MINE_ID", *keep]].rename(columns={"COAL_METAL_IND": "mine_coal_metal_ind"})
    n_before = len(base)
    merged = base.merge(right, on="MINE_ID", how="left", validate="many_to_one", indicator="_m_mines")
    if len(merged) != n_before:
        raise RuntimeError(f"mines join changed the row count: {n_before} -> {len(merged)}")
    merged["_m_mines"] = merged["_m_mines"].astype(str)
    unmatched = int((merged["_m_mines"] != "both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} accident rows have no record in Mines.txt (contract requires 100 % match)")
    agreement = float((merged["COAL_METAL_IND"] == merged["mine_coal_metal_ind"]).mean())
    log.info("mines join: %d rows, match 100.00 %%, COAL_METAL_IND agreement %.2f %%", len(merged), 100 * agreement)
    return merged, agreement


def _join_production(base: pd.DataFrame, mine_years: pd.DataFrame) -> pd.DataFrame:
    n_before = len(base)
    prev = mine_years.rename(
        columns={"hours": "prev_hours", "employees": "prev_employees", "subunits": "prev_subunits", "type": "_prev_type"}
    )
    base = base.assign(_prev_yr=base["CAL_YR"] - 1)
    merged = base.merge(
        prev, left_on=["MINE_ID", "_prev_yr"], right_on=["MINE_ID", "CALENDAR_YR"],
        how="left", validate="many_to_one", indicator="_m_prev",
    )
    if len(merged) != n_before:
        raise RuntimeError(f"previous-year production join changed the row count: {n_before} -> {len(merged)}")
    merged["_m_prev"] = merged["_m_prev"].astype(str)
    merged["prev_year_matched"] = merged["_m_prev"].eq("both")
    merged = merged.drop(columns=["_prev_yr", "CALENDAR_YR"])
    matched = int(merged["prev_year_matched"].sum())
    log.info(
        "previous-year production join: matched %d of %d (%.2f %%), %d unmatched kept with NaN",
        matched, n_before, 100 * matched / n_before, n_before - matched,
    )

    same = mine_years[["MINE_ID", "CALENDAR_YR", "hours", "employees"]].rename(
        columns={"hours": "same_year_hours", "employees": "same_year_employees"}
    )
    merged = merged.merge(
        same, left_on=["MINE_ID", "CAL_YR"], right_on=["MINE_ID", "CALENDAR_YR"],
        how="left", validate="many_to_one", indicator="_m_same",
    )
    if len(merged) != n_before:
        raise RuntimeError(f"same-year production join changed the row count: {n_before} -> {len(merged)}")
    merged["_m_same"] = merged["_m_same"].astype(str)
    merged = merged.drop(columns=["CALENDAR_YR"])
    log.info("same-year production join (EDA only): matched %.2f %%", 100 * merged["_m_same"].eq("both").mean())
    return merged


def _subunits_to_text(subunits: pd.Series) -> pd.Series:
    """Serialise frozensets as sorted '|'-joined strings so they survive parquet."""
    return subunits.map(lambda s: "|".join(sorted(s)) if isinstance(s, frozenset) else np.nan)


# ---------------------------------------------------------------------------
# Public build and load
# ---------------------------------------------------------------------------


def build_base_table(accidents: pd.DataFrame, mines: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Build the analytic base table from the three loaded extracts (pure; nothing is written).

    Follows the row_filter_sequence and join sections of decisions.json step by step. Every raw accident column is kept
    (the leakage columns are needed for EDA and label-validity checks and are dropped in
    `mlda.features`), plus the mine attributes and the derived columns listed in
    `mlda.data.schema.DERIVED_COLUMNS`.
    """
    started = time.perf_counter()
    base, filter_log = filter_rows(accidents)
    base[TARGET] = base[TARGET_SOURCE_COL].isin(POSITIVE_CODES).astype("int8")
    log.info("N = %d, positives = %d, prevalence = %.4f", len(base), int(base[TARGET].sum()), base[TARGET].mean())

    base["filing_date"] = filing_date(base["DOCUMENT_NO"])
    log.info("filing_date unparsable: %d rows", int(base["filing_date"].isna().sum()))
    base["group_id"] = base["MINE_ID"] + "_" + base["CAL_YR"].astype(str)
    base["accident_group"] = base["MINE_ID"] + "_" + base["ACCIDENT_DT"].dt.strftime("%Y-%m-%d")

    base, coal_metal_agreement = _join_mines(base, mines)

    mine_years = mine_year_table(production)
    base = _join_production(base, mine_years)

    fallback = base["SUBUNIT_CD"].map(type_from_subunit_code)
    base["type_at_time"] = base["_prev_type"].where(base["prev_year_matched"], fallback)
    base = base.drop(columns=["_prev_type"])
    base["prev_subunits"] = _subunits_to_text(base["prev_subunits"])
    log.info("type_at_time: %s", base["type_at_time"].value_counts().to_dict())

    cases = case_history_table(accidents)
    counts = lagged_case_counts(base, cases)
    resolved = lagged_case_counts(base, cases, resolved_before=True).add_suffix(RESOLVED_SUFFIX)
    base = pd.concat([base, counts, resolved], axis=1)
    base["prev_year_incomplete_at_filing"] = previous_year_incomplete(base["filing_date"], base["CAL_YR"])
    log.info(
        "previous-year hours possibly incomplete at filing (filed by 15 January): %d rows",
        int(base["prev_year_incomplete_at_filing"].sum()),
    )
    log.info(
        "prev_incidents available for %d rows (%d true zeros)",
        int(base["prev_incidents"].notna().sum()), int(base["prev_incidents"].eq(0).sum()),
    )

    # attrs are set last because pandas drops them on merge and concat.
    base = base.reset_index(drop=True)
    base.attrs["row_filter_log"] = filter_log.to_dict(orient="records")
    base.attrs["coal_metal_ind_agreement"] = coal_metal_agreement
    base.attrs["build_seconds"] = round(time.perf_counter() - started, 1)
    log.info("base table built in %.1f s: %d rows x %d columns", base.attrs["build_seconds"], *base.shape)
    return base


def persistable(base: pd.DataFrame) -> pd.DataFrame:
    """The base table without free text and names (see schema.PERSIST_DROP_COLUMNS)."""
    return base.drop(columns=[c for c in PERSIST_DROP_COLUMNS if c in base.columns])


def write_helper_tables(accidents: pd.DataFrame, production: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist the two lookup tables the inference page needs (contract step 9) and return them."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mine_years = mine_year_table(production)
    in_range = mine_years["CALENDAR_YR"].between(HISTORY_YEAR_MIN, YEAR_MAX)
    features = mine_years.loc[in_range, ["MINE_ID", "CALENDAR_YR", "hours", "employees", "subunits", "type"]].copy()
    features["subunits"] = _subunits_to_text(features["subunits"])
    features = features.reset_index(drop=True)
    features.to_parquet(MINE_YEAR_PARQUET, compression="zstd", index=False)

    history = case_history_table(accidents)
    history = history.loc[history["CAL_YR"].between(HISTORY_YEAR_MIN, YEAR_MAX)].reset_index(drop=True)
    history.to_parquet(CASE_HISTORY_PARQUET, compression="zstd", index=False)
    log.info(
        "wrote %s (%d mine-years) and %s (%d cases)",
        MINE_YEAR_PARQUET.name, len(features), CASE_HISTORY_PARQUET.name, len(history),
    )
    return features, history


def load_base_table(force: bool = False) -> pd.DataFrame:
    """Return the base table, building it from the raw extracts and caching it as parquet.

    The cached frame omits NARRATIVE and the name columns (schema.PERSIST_DROP_COLUMNS), so the
    same column set is returned whether or not the cache existed. `force=True` rebuilds.
    """
    if BASE_PARQUET.exists() and not force:
        started = time.perf_counter()
        base = pd.read_parquet(BASE_PARQUET)
        try:
            validate_base(base)
        except SchemaError as error:  # a cache written by an older version lacks newer columns
            log.warning("cached %s fails the schema (%s); rebuilding from the raw extracts", BASE_PARQUET.name, error)
        else:
            log.info("loaded cached %s in %.1f s (%d rows)", BASE_PARQUET.name, time.perf_counter() - started, len(base))
            return base

    started = time.perf_counter()
    accidents = load_accidents()
    mines = load_mines()
    production = load_production()
    log.info("raw extracts loaded in %.1f s", time.perf_counter() - started)

    base = persistable(build_base_table(accidents, mines, production))
    validate_base(base)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base.to_parquet(BASE_PARQUET, compression="zstd", index=False)
    write_helper_tables(accidents, production)
    log.info(
        "wrote %s (%.1f MB) in %.1f s total",
        BASE_PARQUET.name, BASE_PARQUET.stat().st_size / 1e6, time.perf_counter() - started,
    )
    return base
