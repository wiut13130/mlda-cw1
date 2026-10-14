"""Typed readers for the three MSHA pipe-delimited extracts.

The files are `|`-separated with double-quoted strings, Windows line endings and a
latin-1 byte range (a handful of Spanish narratives and some mojibake in free-text
directions). Every column is read as text first so that nothing is silently coerced,
then the columns the codebooks declare as NUMBER or DATE are converted explicitly.
MSHA uses the placeholders "?", "? " and "NO VALUE FOUND" for invalid or missing
codes; those become proper missing values here so that later steps can count and
impute them like any other gap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlda.config import SOURCES

ENCODING = "latin-1"
PLACEHOLDERS = {"", "?", "? ", "NO VALUE FOUND", "No Value Found", "NULL"}
DATE_FORMAT = "%m/%d/%Y"

ACCIDENT_NUMERIC = [
    "CAL_YR", "CAL_QTR", "FISCAL_YR", "FISCAL_QTR", "NO_INJURIES",
    "TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "SCHEDULE_CHARGE", "DAYS_RESTRICT", "DAYS_LOST",
]
ACCIDENT_DATES = ["ACCIDENT_DT", "RETURN_TO_WORK_DT", "INVEST_BEGIN_DT"]
# Times are stored as zero-padded "HHMM" strings; they are decoded in features.py.
ACCIDENT_TIMES = ["ACCIDENT_TIME", "SHIFT_BEGIN_TIME"]

MINES_NUMERIC = [
    "DAYS_PER_WEEK", "HOURS_PER_SHIFT", "PROD_SHIFTS_PER_DAY", "MAINT_SHIFTS_PER_DAY",
    "NO_EMPLOYEES", "AVG_MINE_HEIGHT", "METHANE_LIBERATION", "NO_PRODUCING_PITS",
    "NO_NONPRODUCING_PITS", "NO_TAILING_PONDS", "MILES_FROM_OFFICE", "LONGITUDE", "LATITUDE",
]
MINES_DATES = ["CURRENT_STATUS_DT", "CURRENT_CONTROLLER_BEGIN_DT", "CURRENT_103I_DT"]

PRODUCTION_NUMERIC = ["CALENDAR_YR", "ANNUAL_HRS", "ANNUAL_COAL_PROD", "AVG_ANNUAL_EMPL", "AVG_EMPLOYEE_HOURS"]


def read_msha_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one extract as text, stripping padding and turning MSHA placeholders into NA."""
    frame = pd.read_csv(
        path,
        sep="|",
        quotechar='"',
        dtype=str,
        keep_default_na=False,
        encoding=ENCODING,
        usecols=usecols,
        engine="c",
        low_memory=False,
    )
    for column in frame.columns:
        values = frame[column].str.strip()
        frame[column] = values.mask(values.isin(PLACEHOLDERS))
    return frame


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in (c for c in columns if c in frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _to_dates(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in (c for c in columns if c in frame.columns):
        frame[column] = pd.to_datetime(frame[column], format=DATE_FORMAT, errors="coerce")


def load_accidents(path: Path | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    """Accident/injury records: one row per injured person (or per accident when no injury)."""
    frame = read_msha_table(path or SOURCES["accidents"].extracted_path, usecols)
    _to_numeric(frame, ACCIDENT_NUMERIC)
    _to_dates(frame, ACCIDENT_DATES)
    for column in (c for c in ["CAL_YR", "CAL_QTR", "FISCAL_YR", "FISCAL_QTR"] if c in frame.columns):
        frame[column] = frame[column].astype("Int64")
    return frame


def load_mines(path: Path | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    """Mine master table: one row per MINE_ID with the mine's *current* attributes."""
    frame = read_msha_table(path or SOURCES["mines"].extracted_path, usecols)
    _to_numeric(frame, MINES_NUMERIC)
    _to_dates(frame, MINES_DATES)
    return frame


def load_production(path: Path | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    """Annual employment and production: one row per MINE_ID x calendar year x subunit."""
    frame = read_msha_table(path or SOURCES["production"].extracted_path, usecols)
    _to_numeric(frame, PRODUCTION_NUMERIC)
    if "CALENDAR_YR" in frame.columns:
        frame["CALENDAR_YR"] = frame["CALENDAR_YR"].astype("Int64")
    return frame
