"""Shared fixtures: raw-data skip helper, the real base table, and synthetic MSHA-like frames.

The synthetic builders return frames with exactly the columns and dtypes that `mlda.data.load`
produces, so every downstream test can run without the raw extracts. `synthetic_base()` is the
output of `build_base_table` on those frames: 400 rows, every column of the contract, realistic
codes, missing values and a few multi-victim accident groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlda.cleaning import apply_error_rules
from mlda.config import RANDOM_STATE, SOURCES, YEAR_MAX, YEAR_MIN
from mlda.data.join import build_base_table
from mlda.data.load import (
    ACCIDENT_DATES,
    ACCIDENT_NUMERIC,
    MINES_DATES,
    MINES_NUMERIC,
    PRODUCTION_NUMERIC,
    load_accidents,
    load_mines,
    load_production,
)
from mlda.data.schema import RAW_ACCIDENT_COLUMNS
from mlda.features import FeatureFrame, feature_frame

FIXTURES = Path(__file__).parent / "fixtures"
RAW_SOURCE_NAMES = ("accidents", "mines", "production")


def _fixture_columns(name: str) -> list[str]:
    """Column order of a raw extract, read from the committed 300-row fixture header."""
    return pd.read_csv(FIXTURES / name, sep="|", nrows=0, encoding="latin-1").columns.tolist()


MINE_COLUMNS = _fixture_columns("mines_sample.txt")
PRODUCTION_COLUMNS = _fixture_columns("production_sample.txt")

# One complete, rule-free accident row. Tests override only the fields they care about.
ACCIDENT_DEFAULTS: dict[str, object] = {
    "MINE_ID": "0100001", "CONTROLLER_ID": "C000001", "CONTROLLER_NAME": "Controller One",
    "OPERATOR_ID": "O000001", "OPERATOR_NAME": "Operator One", "CONTRACTOR_ID": np.nan,
    "DOCUMENT_NO": None, "SUBUNIT_CD": "01", "SUBUNIT": "UNDERGROUND", "ACCIDENT_DT": "2015-06-15",
    "CAL_YR": None, "CAL_QTR": None, "FISCAL_YR": None, "FISCAL_QTR": None,
    "ACCIDENT_TIME": "0900", "DEGREE_INJURY_CD": "03", "DEGREE_INJURY": "DAYS AWAY FROM WORK ONLY",
    "FIPS_STATE_CD": "54", "UG_LOCATION_CD": "01", "UG_LOCATION": "FACE",
    "UG_MINING_METHOD_CD": "01", "UG_MINING_METHOD": "CONTINUOUS MINING",
    "MINING_EQUIP_CD": np.nan, "MINING_EQUIP": np.nan, "EQUIP_MFR_CD": np.nan,
    "EQUIP_MFR_NAME": np.nan, "EQUIP_MODEL_NO": np.nan, "SHIFT_BEGIN_TIME": "0700",
    "CLASSIFICATION_CD": "09", "CLASSIFICATION": "HANDLING OF MATERIALS",
    "ACCIDENT_TYPE_CD": "20", "ACCIDENT_TYPE": "OVER-EXERTION IN LIFTING OBJECTS", "NO_INJURIES": 1,
    "TOT_EXPER": 10.0, "MINE_EXPER": 5.0, "JOB_EXPER": 3.0, "OCCUPATION_CD": "004",
    "OCCUPATION": "Roof bolter", "ACTIVITY_CD": "028", "ACTIVITY": "HANDLING SUPPLIES/MATERIALS",
    "INJURY_SOURCE_CD": "109", "INJURY_SOURCE": "METAL, NEC", "NATURE_INJURY_CD": "330",
    "NATURE_INJURY": "SPRAIN,STRAIN RUPT DISC", "INJ_BODY_PART_CD": "420", "INJ_BODY_PART": "BACK",
    "SCHEDULE_CHARGE": 0, "DAYS_RESTRICT": 0, "DAYS_LOST": 5, "TRANS_TERM": "N",
    "RETURN_TO_WORK_DT": np.nan, "IMMED_NOTIFY_CD": np.nan, "IMMED_NOTIFY": np.nan,
    "INVEST_BEGIN_DT": np.nan, "NARRATIVE": "Synthetic narrative.", "CLOSED_DOC_NO": np.nan,
    "COAL_METAL_IND": "C",
}

MINE_DEFAULTS: dict[str, object] = {
    "MINE_ID": "0100001", "CURRENT_MINE_NAME": "Synthetic Mine", "COAL_METAL_IND": "C",
    "CURRENT_MINE_TYPE": "Underground", "CURRENT_MINE_STATUS": "Active",
    "CURRENT_STATUS_DT": "2010-01-01", "CURRENT_CONTROLLER_ID": "C000001",
    "CURRENT_CONTROLLER_NAME": "Controller One", "CURRENT_OPERATOR_ID": "O000001",
    "CURRENT_OPERATOR_NAME": "Operator One", "STATE": "WV", "BOM_STATE_CD": "46",
    "FIPS_CNTY_CD": "039", "FIPS_CNTY_NM": "Kanawha", "CONG_DIST_CD": "02",
    "COMPANY_TYPE": "Corporation", "CURRENT_CONTROLLER_BEGIN_DT": "2005-01-01", "DISTRICT": "C04",
    "OFFICE_CD": "C0406", "OFFICE_NAME": "Field Office", "PRIMARY_SIC_CD": "121100",
    "PRIMARY_SIC": "Coal (Bituminous)", "PRIMARY_SIC_CD_1": "1211", "PRIMARY_SIC_CD_SFX": "00",
    "PRIMARY_CANVASS_CD": "2", "PRIMARY_CANVASS": "Coal", "CURRENT_103I": "Never Had 103I Status",
    "CURRENT_103I_DT": np.nan, "PORTABLE_OPERATION": "N", "DAYS_PER_WEEK": 5, "HOURS_PER_SHIFT": 8,
    "PROD_SHIFTS_PER_DAY": 2, "MAINT_SHIFTS_PER_DAY": 1, "NO_EMPLOYEES": 50, "PART48_TRAINING": "N",
    "LONGITUDE": -81.5, "LATITUDE": 38.3, "PILLAR_RECOVERY_USED": "N", "HIGHWALL_MINER_USED": "N",
    "MULTIPLE_PITS": "N", "MINERS_REP_IND": "N", "SAFETY_COMMITTEE_IND": "N", "MILES_FROM_OFFICE": 30,
    "DIRECTIONS_TO_MINE": "Synthetic directions", "NEAREST_TOWN": "Town",
}

PRODUCTION_DEFAULTS: dict[str, object] = {
    "MINE_ID": "0100001", "CURR_MINE_NM": "Synthetic Mine", "STATE_ABBR": "WV", "SUBUNIT_CD": "01",
    "SUBUNIT_DESC": "UNDERGROUND", "CALENDAR_YR": 2014, "ANNUAL_HRS": 100_000,
    "ANNUAL_COAL_PROD": np.nan, "AVG_ANNUAL_EMPL": 50, "AVG_EMPLOYEE_HOURS": 2_000, "C_M_IND": "C",
}

STATES = [("WV", "54"), ("KY", "21"), ("PA", "42"), ("TX", "48"), ("NV", "32"), ("AZ", "04"), ("MN", "27")]
MINE_KINDS = ["Underground"] * 12 + ["Surface"] * 20 + ["Facility"] * 8
ACCIDENT_SUBUNITS = {
    "Underground": ["01", "01", "01", "03", "30"],
    "Surface": ["03", "03", "03", "30", "99"],
    "Facility": ["30", "30", "99", "17"],
}
PRODUCTION_SUBUNITS = {"Underground": ["01", "03", "30"], "Surface": ["03", "30"], "Facility": ["30", "99"]}
VALID_CODES = ["01", "02", "03", "04", "05", "06"]
VALID_CODE_WEIGHTS = [0.005, 0.01, 0.36, 0.11, 0.2, 0.315]
EXCLUDED_CODES = ["00", "07", "08", "09", "10"]


def document_no(filing: pd.Timestamp, sequence: int) -> str:
    """Injury form number: type 2, Julian filing date YYYYDDD, four-digit sequence."""
    return f"2{filing:%Y%j}{sequence % 10_000:04d}"


def _coerce(frame: pd.DataFrame, numeric: list[str], dates: list[str], int_columns: tuple[str, ...]) -> pd.DataFrame:
    """Apply the same dtype conversions as mlda.data.load to a frame of Python values."""
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in dates:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in int_columns:
        frame[column] = frame[column].astype("Int64")
    typed = set(numeric) | set(dates) | set(int_columns)
    for column in frame.columns:
        if column not in typed:  # text columns are object with NaN, as read_msha_table returns them
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame


def accidents_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Accidents-like frame from per-row overrides of ACCIDENT_DEFAULTS.

    `ACCIDENT_DT` drives CAL_YR, CAL_QTR and the fiscal fields; an optional `filing_date` key
    (not a column) sets the Julian date inside DOCUMENT_NO, which defaults to accident + 7 days.
    """
    records: list[dict[str, object]] = []
    for sequence, overrides in enumerate(rows):
        unknown = set(overrides) - set(RAW_ACCIDENT_COLUMNS) - {"filing_date"}
        if unknown:
            raise KeyError(f"unknown accident columns {sorted(unknown)}")
        row = {**ACCIDENT_DEFAULTS, **overrides}
        accident = pd.Timestamp(row["ACCIDENT_DT"])
        row["ACCIDENT_DT"] = accident
        row["CAL_YR"], row["CAL_QTR"] = accident.year, accident.quarter
        row["FISCAL_YR"] = accident.year + (1 if accident.month >= 10 else 0)
        row["FISCAL_QTR"] = ((accident.month - 10) % 12) // 3 + 1
        filing = row.pop("filing_date", None)
        if row.get("DOCUMENT_NO") is None:
            filing_ts = pd.Timestamp(filing) if filing is not None else accident + pd.Timedelta(days=7)
            row["DOCUMENT_NO"] = document_no(filing_ts, sequence)
        records.append(row)
    frame = pd.DataFrame.from_records(records, columns=list(RAW_ACCIDENT_COLUMNS))
    return _coerce(frame, ACCIDENT_NUMERIC, ACCIDENT_DATES, ("CAL_YR", "CAL_QTR", "FISCAL_YR", "FISCAL_QTR"))


def mines_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Mines-like frame (59 columns) from per-row overrides of MINE_DEFAULTS."""
    records = []
    for overrides in rows:
        unknown = set(overrides) - set(MINE_COLUMNS)
        if unknown:
            raise KeyError(f"unknown mine columns {sorted(unknown)}")
        records.append({**dict.fromkeys(MINE_COLUMNS, np.nan), **MINE_DEFAULTS, **overrides})
    frame = pd.DataFrame.from_records(records, columns=MINE_COLUMNS)
    return _coerce(frame, MINES_NUMERIC, MINES_DATES, ())


def production_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """MinesProdYearly-like frame (11 columns) from per-row overrides of PRODUCTION_DEFAULTS."""
    records = []
    for overrides in rows:
        unknown = set(overrides) - set(PRODUCTION_COLUMNS)
        if unknown:
            raise KeyError(f"unknown production columns {sorted(unknown)}")
        records.append({**PRODUCTION_DEFAULTS, **overrides})
    frame = pd.DataFrame.from_records(records, columns=PRODUCTION_COLUMNS)
    return _coerce(frame, PRODUCTION_NUMERIC, [], ("CALENDAR_YR",))


def _random_date(rng: np.random.Generator, year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(rng.integers(0, 365)))


def _hhmm(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(0, 24)):02d}{int(rng.integers(0, 60)):02d}"


def _accident_row(rng: np.random.Generator, mine: dict[str, object], year: int) -> dict[str, object]:
    """One in-window base row with realistic codes and the error patterns the cleaning rules target."""
    kind = str(mine["kind"])
    subunit = str(rng.choice(ACCIDENT_SUBUNITS[kind]))
    underground = subunit in ("01", "02")
    tot = float(np.round(rng.lognormal(1.8, 0.9), 2))
    mine_exper = float(np.round(tot * rng.uniform(0, 1), 2))
    job = float(np.round(tot * rng.uniform(0, 1), 2))
    u = rng.uniform()
    if u < 0.03:
        job = float(np.round(tot + rng.uniform(1, 10), 2))  # E1
    elif u < 0.035:
        mine_exper = float(np.round(tot + rng.uniform(1, 5), 2))  # E2
    if rng.uniform() < 0.015:
        tot = np.nan
    accident_time = _hhmm(rng)
    t = rng.uniform()
    if t < 0.03:
        accident_time = "9999"  # E5
    elif t < 0.035:
        accident_time = "2400"  # E6
    shift = str(rng.choice(["0700", "0600", "1500", "2300", "0800"]))
    s = rng.uniform()
    if s < 0.01:
        shift = "9999"  # E8
    elif s < 0.03:
        shift = "2400"  # E9
    elif s < 0.04:
        shift = "700"  # E10
    elif s < 0.045:
        shift = np.nan
    code = str(rng.choice(VALID_CODES, p=VALID_CODE_WEIGHTS))
    positive = code in ("01", "02", "03", "04")
    lag_days = int(rng.integers(60, 400)) if rng.uniform() < 0.05 else int(rng.integers(1, 25))
    accident = _random_date(rng, year)
    return {
        "MINE_ID": mine["id"], "COAL_METAL_IND": mine["coal_metal"], "FIPS_STATE_CD": mine["fips"],
        "ACCIDENT_DT": accident, "filing_date": accident + pd.Timedelta(days=lag_days),
        "SUBUNIT_CD": subunit, "SUBUNIT": {"01": "UNDERGROUND", "03": "STRIP, QUARY, OPEN PIT"}.get(subunit, "OTHER"),
        "UG_LOCATION_CD": str(rng.choice(["01", "02", "03"])) if underground else np.nan,
        "UG_MINING_METHOD_CD": str(rng.choice(["01", "02", "03"])) if underground else np.nan,
        "MINING_EQUIP_CD": np.nan if rng.uniform() < 0.5 else str(rng.choice(["052", "009", "010"])),
        "EQUIP_MFR_NAME": np.nan if rng.uniform() < 0.5 else str(rng.choice(["Caterpillar", "Joy", "Komatsu"])),
        "ACCIDENT_TIME": accident_time, "SHIFT_BEGIN_TIME": shift,
        "DEGREE_INJURY_CD": code, "TOT_EXPER": tot, "MINE_EXPER": mine_exper, "JOB_EXPER": job,
        "NO_INJURIES": int(rng.choice([1, 1, 1, 1, 2, 3])) if rng.uniform() < 0.05 else 1,
        "CONTRACTOR_ID": "C123" if rng.uniform() < 0.12 else np.nan,
        "IMMED_NOTIFY_CD": str(rng.choice(["01", "04", "13"])) if rng.uniform() < 0.03 else np.nan,
        "NATURE_INJURY_CD": str(rng.choice(["330", "220", "160", "180", "320", "400"])),
        "INJ_BODY_PART_CD": str(rng.choice(["420", "340", "330", "130", "520"])),
        "CLASSIFICATION_CD": str(rng.choice(["09", "18", "12", "17", "07"])),
        "ACCIDENT_TYPE_CD": str(rng.choice(["20", "02", "31", "10"])),
        "OCCUPATION": str(rng.choice(["Roof bolter", "Laborer", "Mechanic", "Truck driver"])),
        "ACTIVITY_CD": str(rng.choice(["028", "054", "006"])),
        "INJURY_SOURCE_CD": str(rng.choice(["109", "076", "018"])),
        "DAYS_LOST": int(rng.integers(1, 60)) if positive else 0,
    }


def synthetic_sources(n_rows: int = 400, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic (accidents, mines, production) frames whose base table has exactly `n_rows` rows.

    Forty mines of three types; rows concentrated on a few mines; 12 duplicated (MINE_ID,
    ACCIDENT_DT) pairs for multi-victim groups; extra accident rows outside the window or with
    excluded codes; three mines with missing production years so that `prev_year_matched` is
    False for some rows; one invalid and one American Samoa coordinate pair.
    """
    rng = np.random.default_rng(seed)
    mines: list[dict[str, object]] = []
    for i, kind in enumerate(MINE_KINDS):
        state, fips = STATES[i % len(STATES)]
        mines.append(
            {"id": f"01{i + 1:05d}", "kind": kind, "coal_metal": "C" if i < 16 else "M", "state": state,
             "fips": fips, "size": float(np.round(rng.lognormal(3.2, 1.0))) + 1}
        )

    weights = 1 / np.arange(1, len(mines) + 1)
    weights /= weights.sum()
    rows: list[dict[str, object]] = []
    n_multi = 12
    # Rows whose previous year has no production (mines 38-40 below), so prev_year_matched is
    # False for a known handful of rows whatever the random draw does.
    rows += [_accident_row(rng, mines[37], int(rng.integers(YEAR_MIN, YEAR_MAX + 1))) for _ in range(5)]
    rows += [_accident_row(rng, mines[38], year) for year in (2015, 2016, 2017)]
    rows += [_accident_row(rng, mines[39], YEAR_MIN) for _ in range(2)]
    while len(rows) < n_rows - n_multi:
        mine = mines[int(rng.choice(len(mines), p=weights))]
        rows.append(_accident_row(rng, mine, int(rng.integers(YEAR_MIN, YEAR_MAX + 1))))
    for _ in range(n_multi):
        source = rows[int(rng.integers(0, len(rows)))]
        mine = next(m for m in mines if m["id"] == source["MINE_ID"])
        twin = _accident_row(rng, mine, int(source["ACCIDENT_DT"].year))
        twin["ACCIDENT_DT"], twin["ACCIDENT_TIME"] = source["ACCIDENT_DT"], source["ACCIDENT_TIME"]
        twin["filing_date"] = source["filing_date"]
        rows.append(twin)
    rows[5]["DOCUMENT_NO"] = "2XXXXXXX0001"  # unparsable filing date
    rows[6]["DOCUMENT_NO"] = "2XXXXXXX0002"
    rows[7]["TOT_EXPER"], rows[7]["JOB_EXPER"] = 65.0, 2.0  # E3
    rows[8]["NO_INJURIES"] = 12  # E21

    extra: list[dict[str, object]] = []
    for _ in range(50):  # history year for the lag features
        mine = mines[int(rng.choice(len(mines), p=weights))]
        row = _accident_row(rng, mine, YEAR_MIN - 1)
        row["DEGREE_INJURY_CD"] = str(rng.choice(VALID_CODES + ["00", "07"]))
        extra.append(row)
    for _ in range(10):
        extra.append(_accident_row(rng, mines[int(rng.integers(0, len(mines)))], YEAR_MIN - 2))
    for _ in range(8):
        row = _accident_row(rng, mines[int(rng.integers(0, len(mines)))], YEAR_MAX + 1)
        row["ACCIDENT_DT"] = pd.Timestamp(year=YEAR_MAX + 1, month=1, day=1) + pd.Timedelta(days=int(rng.integers(0, 270)))
        extra.append(row)
    for _ in range(30):
        row = _accident_row(rng, mines[int(rng.integers(0, len(mines)))], int(rng.integers(YEAR_MIN, YEAR_MAX + 1)))
        row["DEGREE_INJURY_CD"] = str(rng.choice(EXCLUDED_CODES))
        extra.append(row)
    for _ in range(4):
        row = _accident_row(rng, mines[int(rng.integers(0, len(mines)))], int(rng.integers(YEAR_MIN, YEAR_MAX + 1)))
        row["DEGREE_INJURY_CD"] = np.nan
        extra.append(row)
    accidents = accidents_frame(rows + extra)

    mine_rows: list[dict[str, object]] = []
    for i, mine in enumerate(mines):
        kind = str(mine["kind"])
        status: dict[str, object] = {"CURRENT_103I": "Never Had 103I Status", "CURRENT_103I_DT": np.nan}
        if kind == "Underground":
            status = [
                {"CURRENT_103I": "Removed From 103I Status", "CURRENT_103I_DT": "2012-03-01"},
                {"CURRENT_103I": "Removed From 103I Status", "CURRENT_103I_DT": "2024-06-01"},
                {"CURRENT_103I": "Inspect Once Every 5-days", "CURRENT_103I_DT": "2011-01-15"},
                {"CURRENT_103I": "Inspect Once Every 10-days", "CURRENT_103I_DT": "2025-09-01"},
                {"CURRENT_103I": "Never Had 103I Status", "CURRENT_103I_DT": np.nan},
            ][i % 5]
        elif i % 7 == 0:
            status = {"CURRENT_103I": np.nan, "CURRENT_103I_DT": np.nan}
        coal = mine["coal_metal"] == "C"
        mine_rows.append(
            {
                "MINE_ID": mine["id"], "COAL_METAL_IND": mine["coal_metal"], "CURRENT_MINE_TYPE": kind,
                "STATE": mine["state"], "PRIMARY_CANVASS": "Coal" if coal else str(rng.choice(["Stone", "SandAndGravel", "Metal"])),
                "PRIMARY_SIC": "Coal (Bituminous)" if coal else "Crushed, Broken Stone NEC",
                "DISTRICT": "C04" if coal else "NE", "NO_EMPLOYEES": int(mine["size"]),
                "LATITUDE": float(np.round(rng.uniform(30, 48), 5)), "LONGITUDE": float(np.round(rng.uniform(-120, -75), 5)),
                "SAFETY_COMMITTEE_IND": str(rng.choice(["Y", "N"])), "PART48_TRAINING": str(rng.choice(["Y", "N"])),
                "PORTABLE_OPERATION": "Y" if (not coal and rng.uniform() < 0.2) else "N",
                **status,
            }
        )
    mine_rows[36].update({"LATITUDE": 91.0, "LONGITUDE": -80.0})  # E23 invalid pair
    mine_rows[37].update({"STATE": "AS", "LATITUDE": -14.3, "LONGITUDE": -170.7})  # exempt
    mine_rows[38].update({"LATITUDE": np.nan, "LONGITUDE": np.nan})
    mines_df = mines_frame(mine_rows)

    production_rows: list[dict[str, object]] = []
    for i, mine in enumerate(mines):
        if i == 37:
            continue  # no production at all: every row falls back to the accident SUBUNIT_CD
        for year in range(YEAR_MIN - 2, YEAR_MAX + 1):
            if i == 38 and 2014 <= year <= 2016:
                continue
            if i == 39 and year == YEAR_MIN - 1:
                continue
            subunits = PRODUCTION_SUBUNITS[str(mine["kind"])]
            employees = max(1.0, float(mine["size"]) * rng.uniform(0.8, 1.2))
            for subunit in subunits:
                share = 1 / len(subunits)
                emp = max(1, int(round(employees * share)))
                production_rows.append(
                    {
                        "MINE_ID": mine["id"], "STATE_ABBR": mine["state"], "SUBUNIT_CD": subunit,
                        "CALENDAR_YR": year, "ANNUAL_HRS": int(emp * rng.uniform(1_500, 2_200)),
                        "AVG_ANNUAL_EMPL": emp, "AVG_EMPLOYEE_HOURS": 1_900, "C_M_IND": mine["coal_metal"],
                    }
                )
    production = production_frame(production_rows)
    return accidents, mines_df, production


def synthetic_base(n_rows: int = 400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """The contract base table built from `synthetic_sources` (400 rows by default)."""
    return build_base_table(*synthetic_sources(n_rows=n_rows, seed=seed))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_data_available() -> bool:
    """Skip the requesting test when the extracted MSHA text files are absent."""
    missing = [SOURCES[n].extracted_path.name for n in RAW_SOURCE_NAMES if not SOURCES[n].extracted_path.exists()]
    if missing:
        pytest.skip(f"raw MSHA extracts absent: {', '.join(missing)} (run `uv run mlda-download`)")
    return True


@pytest.fixture(scope="session")
def real_base(raw_data_available: bool) -> pd.DataFrame:
    """Base table built once per session from the real extracts (nothing is written to disk)."""
    return build_base_table(load_accidents(), load_mines(), load_production())


@pytest.fixture(scope="session")
def real_clean(real_base: pd.DataFrame) -> pd.DataFrame:
    """The real base table after the error rules (E1-E25), built once per session."""
    return apply_error_rules(real_base)[0]


@pytest.fixture(scope="session")
def real_features(real_clean: pd.DataFrame) -> FeatureFrame:
    """Model inputs, labels, groups and meta of the real data (X has the 55 FEATURE_SPEC columns)."""
    return feature_frame(real_clean)


@pytest.fixture
def synthetic_sources_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return synthetic_sources()


@pytest.fixture
def synthetic_base_frame() -> pd.DataFrame:
    return synthetic_base()
