"""Expected shape of the analytic base table and a validator for it.

The base table is rebuilt from the raw extracts by `mlda.data.join` and cached as parquet.
Every later stage (cleaning, features, the app) reads it by column name, so the names are
fixed here and checked in one place. The reference counts come from
`docs/profile/decisions.json` (target section) and are pinned to the 2026-10-09 snapshot.
"""

from __future__ import annotations

import pandas as pd

from mlda.config import (
    MINES_KEEP_COLUMNS,
    NEGATIVE_CODES,
    POSITIVE_CODES,
    TARGET,
    TARGET_SOURCE_COL,
    YEAR_MAX,
    YEAR_MIN,
)

# Column order of data/raw/Accidents.txt (57 columns, matches the codebook).
RAW_ACCIDENT_COLUMNS: tuple[str, ...] = (
    "MINE_ID", "CONTROLLER_ID", "CONTROLLER_NAME", "OPERATOR_ID", "OPERATOR_NAME",
    "CONTRACTOR_ID", "DOCUMENT_NO", "SUBUNIT_CD", "SUBUNIT", "ACCIDENT_DT", "CAL_YR", "CAL_QTR",
    "FISCAL_YR", "FISCAL_QTR", "ACCIDENT_TIME", "DEGREE_INJURY_CD", "DEGREE_INJURY",
    "FIPS_STATE_CD", "UG_LOCATION_CD", "UG_LOCATION", "UG_MINING_METHOD_CD", "UG_MINING_METHOD",
    "MINING_EQUIP_CD", "MINING_EQUIP", "EQUIP_MFR_CD", "EQUIP_MFR_NAME", "EQUIP_MODEL_NO",
    "SHIFT_BEGIN_TIME", "CLASSIFICATION_CD", "CLASSIFICATION", "ACCIDENT_TYPE_CD",
    "ACCIDENT_TYPE", "NO_INJURIES", "TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "OCCUPATION_CD",
    "OCCUPATION", "ACTIVITY_CD", "ACTIVITY", "INJURY_SOURCE_CD", "INJURY_SOURCE",
    "NATURE_INJURY_CD", "NATURE_INJURY", "INJ_BODY_PART_CD", "INJ_BODY_PART", "SCHEDULE_CHARGE",
    "DAYS_RESTRICT", "DAYS_LOST", "TRANS_TERM", "RETURN_TO_WORK_DT", "IMMED_NOTIFY_CD",
    "IMMED_NOTIFY", "INVEST_BEGIN_DT", "NARRATIVE", "CLOSED_DOC_NO", "COAL_METAL_IND",
)

# Mine attributes carried by the join (contract step 5).
MINE_JOIN_COLUMNS: tuple[str, ...] = (*MINES_KEEP_COLUMNS, "mine_coal_metal_ind", "_m_mines")

# Columns added by the join steps, in the order they are created.
DERIVED_COLUMNS: tuple[str, ...] = (
    TARGET, "filing_date", "group_id", "accident_group",
    "prev_hours", "prev_employees", "prev_subunits", "prev_year_matched", "_m_prev",
    "same_year_hours", "same_year_employees", "_m_same",
    "type_at_time", "prev_incidents", "prev_valid_cases", "prev_lost_time_cases",
    # sensitivity-run columns beyond the contract (never model inputs)
    "prev_incidents_resolved", "prev_valid_cases_resolved", "prev_lost_time_cases_resolved",
    "prev_year_incomplete_at_filing",
)

EXPECTED_BASE_COLUMNS: tuple[str, ...] = (*RAW_ACCIDENT_COLUMNS, *MINE_JOIN_COLUMNS, *DERIVED_COLUMNS)

# Free text and person or company names are removed before the base is written to parquet:
# they are not features, they hold the only non-ASCII bytes, and they triple the file size.
# EQUIP_MFR_NAME is a categorical feature and is kept although it ends in _NAME.
PERSIST_DROP_COLUMNS: tuple[str, ...] = (
    "NARRATIVE", "DIRECTIONS_TO_MINE", "NEAREST_TOWN", "CONTROLLER_NAME", "OPERATOR_NAME",
)
REQUIRED_BASE_COLUMNS: tuple[str, ...] = tuple(
    c for c in EXPECTED_BASE_COLUMNS if c not in PERSIST_DROP_COLUMNS
)

# Reference figures from decisions.json (target section), snapshot 2026-10-09.
EXPECTED_N = 101_609
EXPECTED_POSITIVES = 49_367
EXPECTED_PREVALENCE = EXPECTED_POSITIVES / EXPECTED_N
TYPE_AT_TIME_LEVELS: tuple[str, ...] = ("Underground", "Surface", "Facility")
VALID_DEGREE_CODES: tuple[str, ...] = (*POSITIVE_CODES, *NEGATIVE_CODES)


class SchemaError(ValueError):
    """Raised when the base table does not satisfy the contract in docs/interfaces.md."""


def validate_base(
    df: pd.DataFrame,
    *,
    check_counts: bool = True,
    n_tolerance: float = 0.01,
    prevalence_tolerance: float = 0.01,
) -> None:
    """Assert the structural contract of the base table; collect every failure before raising.

    `check_counts=False` skips the comparison with the profiled N and prevalence so that the
    same checks run on small synthetic frames in the tests.
    """
    problems: list[str] = []

    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        raise SchemaError("; ".join(problems))

    n = len(df)
    if n == 0:
        raise SchemaError("base table is empty")
    if check_counts:
        if abs(n - EXPECTED_N) / EXPECTED_N > n_tolerance:
            problems.append(f"N = {n:,} differs from the profiled {EXPECTED_N:,} by more than {n_tolerance:.0%}")
        prevalence = df[TARGET].mean()
        if abs(prevalence - EXPECTED_PREVALENCE) > prevalence_tolerance:
            problems.append(
                f"prevalence {prevalence:.4f} differs from the profiled {EXPECTED_PREVALENCE:.4f} "
                f"by more than {prevalence_tolerance:.2f}"
            )

    if not df["MINE_ID"].astype("string").str.len().eq(7).all():
        problems.append("MINE_ID must be a 7-character zero-padded string")
    if df["DOCUMENT_NO"].duplicated().any():
        problems.append("DOCUMENT_NO is not unique")

    for column in (TARGET, "group_id", "accident_group", "prev_year_matched", "type_at_time"):
        if df[column].isna().any():
            problems.append(f"{column} contains NaN")
    if not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET} must be 0/1")
    if not df[TARGET_SOURCE_COL].isin(VALID_DEGREE_CODES).all():
        problems.append(f"{TARGET_SOURCE_COL} outside {VALID_DEGREE_CODES}")
    expected_target = df[TARGET_SOURCE_COL].isin(POSITIVE_CODES).astype("int8")
    if not df[TARGET].eq(expected_target).all():  # no int cast here: a NaN target is reported, not a crash
        problems.append(f"{TARGET} disagrees with {TARGET_SOURCE_COL}")

    cal_yr = df["CAL_YR"]
    if cal_yr.isna().any() or not cal_yr.between(YEAR_MIN, YEAR_MAX).all():
        problems.append(f"CAL_YR outside {YEAR_MIN}-{YEAR_MAX}")
    if df["ACCIDENT_DT"].isna().any():
        problems.append("ACCIDENT_DT contains NaT")

    if not df["type_at_time"].isin(TYPE_AT_TIME_LEVELS).all():
        problems.append(f"type_at_time outside {TYPE_AT_TIME_LEVELS}")
    if df["prev_year_matched"].dtype != bool:
        problems.append("prev_year_matched must be bool")
    else:
        matched = df["prev_year_matched"]
        for column in ("prev_hours", "prev_employees", "prev_incidents", "prev_valid_cases", "prev_lost_time_cases"):
            if not df[column].isna().eq(~matched).all():
                problems.append(f"{column} must be NaN exactly when prev_year_matched is False")
            resolved = f"{column}_resolved"
            if resolved in df.columns:
                if not df[resolved].isna().eq(~matched).all():
                    problems.append(f"{resolved} must be NaN exactly when prev_year_matched is False")
                elif (df[resolved] > df[column]).any():
                    problems.append(f"{resolved} must not exceed {column}")
    if df["prev_year_incomplete_at_filing"].dtype != bool:
        problems.append("prev_year_incomplete_at_filing must be bool")
    if (df["_m_mines"] != "both").any():
        problems.append("every row must match a mine record")

    if problems:
        raise SchemaError("; ".join(problems))
