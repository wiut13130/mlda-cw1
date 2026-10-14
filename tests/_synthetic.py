"""Synthetic analytic base table for tests that must not depend on data/raw.

``synthetic_base`` builds a frame with every column of the base table described in
docs/interfaces.md section 1: all 57 raw accident columns, the join outputs
(lost_time, filing_date, group_id, accident_group, mine attributes, previous-year
production and case history, type_at_time) and the ``_m_mines`` indicator. Codes are
realistic, placeholders are already NaN as ``mlda.data.load`` leaves them, several
rows share a mine-year (group_id) and a same-day accident_group, and a few rows
trigger each error rule (9999 and 2400 times, a three-character shift time, job
experience above total, more than ten injuries, invalid coordinates, an unmatched
previous year, a mine-year without valid cases). Values are random but seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlda.config import RANDOM_STATE

SUBUNIT_CODES = ["01", "02", "03", "04", "05", "06", "12", "17", "30", "99"]
SUBUNIT_WEIGHTS = [0.34, 0.01, 0.22, 0.02, 0.03, 0.03, 0.02, 0.05, 0.25, 0.03]
STATES = {"01": "AL", "21": "KY", "42": "PA", "54": "WV", "04": "AZ", "32": "NV", "60": "AS"}
OCCUPATIONS = [
    "Laborer, Bullgang, Utility man",
    "Mechanic, Repairman, Helper",
    "Truck driver",
    "Continuous miner operator",
    "Roof bolter, Rock bolter",
    "Electrician",
    "Front-end loader operator",
    "Maintenance man",
    "Shuttle car operator",
    "Welder",
    "Supervisor, Foreman",
    "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator",
]
MANUFACTURERS = ["Caterpillar", "Joy", "Komatsu", "Not Reported", "Not on this list", "Volvo", "Eimco"]
NATURE_CODES = ["330", "220", "160", "180", "320", "310", "400", "170", "100", "210"]
BODY_PART_CODES = ["340", "330", "130", "320", "420", "500", "510", "100", "450", "700"]
SOURCE_CODES = ["076", "040", "065", "110", "095", "120", "011", "052", "070", "130"]
ACCIDENT_TYPE_CODES = ["02", "04", "07", "08", "09", "13", "14", "15", "17", "20"]
CLASSIFICATION_CODES = ["09", "17", "18", "12", "07", "19", "06", "03", "14", "20"]
ACTIVITY_CODES = ["054", "038", "070", "049", "018", "062", "020", "083", "011", "090"]
EQUIP_CODES = ["52", "16", "18", "07", "27", "48", "60", "33"]
UG_LOCATION_CODES = ["01", "02", "03", "04", "05", "06"]
UG_METHOD_CODES = ["01", "02", "03", "05", "06"]
STATUS_103I = ["Never Had 103I Status", "Removed From 103I Status", "5 Day 103I Status", "Hazard 103I Status", None]


def _mine_table(rng: np.random.Generator, n_mines: int) -> pd.DataFrame:
    """One row per synthetic mine with the Mines.txt attributes the base table keeps."""
    ids = rng.choice(np.arange(100_003, 4_899_999), size=n_mines, replace=False)
    fips = rng.choice(list(STATES), size=n_mines)
    coal = rng.random(n_mines) < 0.4
    latitude = rng.uniform(30, 48, n_mines).round(5)
    longitude = rng.uniform(-120, -75, n_mines).round(5)
    latitude[0], longitude[0] = 0.0, 0.0  # E23: invalid pair
    latitude[1], longitude[1] = np.nan, np.nan
    samoa = np.flatnonzero(fips == "60")
    latitude[samoa], longitude[samoa] = -14.3, -170.7  # outside the box but exempt
    status = rng.choice(STATUS_103I, size=n_mines, p=[0.6, 0.15, 0.1, 0.05, 0.1])
    status_dt = pd.to_datetime("2008-01-01") + pd.to_timedelta(rng.integers(0, 6500, n_mines), unit="D")
    return pd.DataFrame(
        {
            "MINE_ID": [f"{i:07d}" for i in ids],
            "FIPS_STATE_CD": fips,
            "STATE": [STATES[f] for f in fips],
            "COAL_METAL_IND": np.where(coal, "C", "M"),
            "PRIMARY_CANVASS": np.where(coal, "Coal", rng.choice(["Stone", "Sand & gravel", "Metal", "Nonmetal"], n_mines)),
            "PRIMARY_SIC": np.where(coal, "Bituminous Coal (Underground)", "Crushed, Broken Stone NEC"),
            "DISTRICT": rng.choice(["C04", "C06", "NE", "SE", "WE", "RM"], n_mines),
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "PORTABLE_OPERATION": rng.choice(["Y", "N"], n_mines, p=[0.1, 0.9]),
            "SAFETY_COMMITTEE_IND": rng.choice(["Y", "N"], n_mines, p=[0.3, 0.7]),
            "PART48_TRAINING": rng.choice(["Y", "N"], n_mines, p=[0.6, 0.4]),
            "CURRENT_103I": status,
            "CURRENT_103I_DT": np.where(pd.isna(status) | (status == "Never Had 103I Status"), pd.NaT, status_dt),
            "CURRENT_MINE_TYPE": np.where(coal, "Underground", "Surface"),
        }
    )


def _hhmm(hours: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    return np.array([f"{h:02d}{m:02d}" for h, m in zip(hours, minutes, strict=True)], dtype=object)


def _type_from_subunits(subunits: frozenset[str]) -> str:
    if subunits & {"01", "02"}:
        return "Underground"
    if subunits <= {"30", "99", "17"}:
        return "Facility"
    return "Surface"


def synthetic_base(n_rows: int = 400, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Seeded base table with ``n_rows`` rows following docs/interfaces.md section 1."""
    rng = np.random.default_rng(random_state)
    n_mines = max(8, n_rows // 40)
    mines = _mine_table(rng, n_mines)

    mine_pos = rng.integers(0, n_mines, n_rows)
    year = rng.integers(2010, 2026, n_rows)
    day = rng.choice(np.array([20, 130, 240, 350]), n_rows)
    # Every tenth row repeats the mine and date of the row before it: multi-victim groups.
    dup = rng.choice(np.arange(1, n_rows), size=n_rows // 10, replace=False)
    mine_pos[dup], year[dup], day[dup] = mine_pos[dup - 1], year[dup - 1], day[dup - 1]
    accident_dt = pd.to_datetime(pd.DataFrame({"year": year, "month": 1, "day": 1})) + pd.to_timedelta(day, unit="D")
    mine = mines.iloc[mine_pos].reset_index(drop=True)

    hours = rng.integers(0, 24, n_rows)
    minutes = rng.integers(0, 60, n_rows)
    accident_time = _hhmm(hours, minutes)
    accident_time[rng.random(n_rows) < 0.03] = "9999"  # E5
    accident_time[rng.random(n_rows) < 0.005] = "2400"  # E6
    shift_time = rng.choice(np.array(["0700", "0600", "0800", "1500", "1900", "2200", "2300", "0000"], dtype=object), n_rows)
    shift_time[rng.random(n_rows) < 0.01] = "9999"  # E8
    shift_time[rng.random(n_rows) < 0.02] = "2400"  # E9
    shift_time[rng.random(n_rows) < 0.01] = "700"  # E10
    shift_time[rng.random(n_rows) < 0.005] = None

    lost_time = (rng.random(n_rows) < 0.486).astype("int8")
    degree = np.where(
        lost_time == 1,
        rng.choice(["01", "02", "03", "04"], n_rows, p=[0.01, 0.02, 0.75, 0.22]),
        rng.choice(["05", "06"], n_rows, p=[0.4, 0.6]),
    )
    subunit = rng.choice(SUBUNIT_CODES, n_rows, p=SUBUNIT_WEIGHTS)
    underground = subunit == "01"
    ug_location = np.where(underground, rng.choice(UG_LOCATION_CODES, n_rows), None)
    ug_method = np.where(underground & (rng.random(n_rows) > 0.03), rng.choice(UG_METHOD_CODES, n_rows), None)
    has_equipment = rng.random(n_rows) > 0.54
    equip = np.where(has_equipment, rng.choice(EQUIP_CODES, n_rows), None)
    manufacturer = np.where(has_equipment, rng.choice(MANUFACTURERS, n_rows), None)
    occupation = rng.choice(OCCUPATIONS[:-1], n_rows)
    occupation[(rng.random(n_rows) < 0.02) & (year <= 2014)] = OCCUPATIONS[-1]  # E28
    activity = rng.choice(ACTIVITY_CODES, n_rows).astype(object)
    activity[rng.random(n_rows) < 0.01] = None  # E16

    tot_exper = np.round(rng.lognormal(1.6, 0.9, n_rows), 2)
    mine_exper = np.round(tot_exper * rng.random(n_rows), 2)
    job_exper = np.round(tot_exper * rng.random(n_rows), 2)
    job_exper[rng.random(n_rows) < 0.03] += 8.0  # E1: job above total
    mine_exper[rng.random(n_rows) < 0.005] += 5.0  # E2
    tot_exper[0] = 65.0  # E3
    for column in (tot_exper, mine_exper, job_exper):
        column[rng.random(n_rows) < 0.015] = np.nan
    no_injuries = np.ones(n_rows)
    no_injuries[rng.random(n_rows) < 0.02] = 2
    no_injuries[1] = 15  # E21

    contractor = np.where(rng.random(n_rows) < 0.12, rng.choice(["1KJ", "Y4F", "C2M"], n_rows), None)
    notify = np.where(rng.random(n_rows) < 0.035, rng.choice(["01", "04", "08", "13"], n_rows), None)

    # Previous-year production and case history for the mine-year.
    matched = rng.random(n_rows) > 0.025
    prev_hours = np.round(rng.lognormal(11, 1.2, n_rows)).clip(200, 3_000_000)
    prev_employees = np.round(prev_hours / rng.uniform(1500, 2500, n_rows)).clip(1, None)
    boundary = rng.choice(np.arange(n_rows), size=8, replace=False)
    prev_employees[boundary] = np.repeat([5, 20, 50, 200], 2)
    prev_incidents = rng.poisson(prev_hours / 200_000 * 4).astype(float)
    prev_valid = np.floor(prev_incidents * rng.uniform(0.6, 1.0, n_rows))
    prev_lost = np.floor(prev_valid * rng.uniform(0.2, 0.9, n_rows))
    subunit_sets = [
        frozenset({"01", "30"}) if t == "Underground" else frozenset({"30", "99"}) if rng.random() < 0.15 else frozenset({"03", "30"})
        for t in mine["CURRENT_MINE_TYPE"]
    ]
    type_at_time = np.array([_type_from_subunits(s) if m else ("Underground" if u else "Surface") for s, m, u in zip(subunit_sets, matched, underground, strict=True)])
    for column in (prev_hours, prev_employees, prev_incidents, prev_valid, prev_lost):
        column[~matched] = np.nan

    document_no = np.array([f"2{dt.year}{dt.dayofyear:03d}{i:04d}" for i, dt in enumerate(accident_dt + pd.Timedelta(days=5))])
    filing_date = accident_dt + pd.Timedelta(days=5)
    quarter = accident_dt.dt.quarter.to_numpy()
    fiscal_year = np.where(accident_dt.dt.month >= 10, year + 1, year)

    frame = pd.DataFrame(
        {
            "MINE_ID": mine["MINE_ID"].to_numpy(),
            "CONTROLLER_ID": "C" + mine["MINE_ID"].str[1:].to_numpy(),
            "CONTROLLER_NAME": "Controller " + mine["MINE_ID"].str[-3:].to_numpy(),
            "OPERATOR_ID": "O" + mine["MINE_ID"].str[1:].to_numpy(),
            "OPERATOR_NAME": "Operator " + mine["MINE_ID"].str[-3:].to_numpy(),
            "CONTRACTOR_ID": contractor,
            "DOCUMENT_NO": document_no,
            "SUBUNIT_CD": subunit,
            "SUBUNIT": "SUBUNIT " + subunit,
            "ACCIDENT_DT": accident_dt,
            "CAL_YR": pd.array(year, dtype="Int64"),
            "CAL_QTR": pd.array(quarter, dtype="Int64"),
            "FISCAL_YR": pd.array(fiscal_year, dtype="Int64"),
            "FISCAL_QTR": pd.array((quarter % 4) + 1, dtype="Int64"),
            "ACCIDENT_TIME": accident_time,
            "DEGREE_INJURY_CD": degree,
            "DEGREE_INJURY": "DEGREE " + degree,
            "FIPS_STATE_CD": mine["FIPS_STATE_CD"].to_numpy(),
            "UG_LOCATION_CD": ug_location,
            "UG_LOCATION": np.where(pd.isna(ug_location), None, "LOCATION"),
            "UG_MINING_METHOD_CD": ug_method,
            "UG_MINING_METHOD": np.where(pd.isna(ug_method), None, "METHOD"),
            "MINING_EQUIP_CD": equip,
            "MINING_EQUIP": np.where(has_equipment, "EQUIPMENT", None),
            "EQUIP_MFR_CD": np.where(has_equipment, "018", None),
            "EQUIP_MFR_NAME": manufacturer,
            "EQUIP_MODEL_NO": None,
            "SHIFT_BEGIN_TIME": shift_time,
            "CLASSIFICATION_CD": rng.choice(CLASSIFICATION_CODES, n_rows),
            "CLASSIFICATION": "CLASSIFICATION",
            "ACCIDENT_TYPE_CD": rng.choice(ACCIDENT_TYPE_CODES, n_rows),
            "ACCIDENT_TYPE": "ACCIDENT TYPE",
            "NO_INJURIES": no_injuries,
            "TOT_EXPER": tot_exper,
            "MINE_EXPER": mine_exper,
            "JOB_EXPER": job_exper,
            "OCCUPATION_CD": rng.choice(["004", "036", "054", "119", "212"], n_rows),
            "OCCUPATION": occupation,
            "ACTIVITY_CD": activity,
            "ACTIVITY": "ACTIVITY",
            "INJURY_SOURCE_CD": rng.choice(SOURCE_CODES, n_rows),
            "INJURY_SOURCE": "SOURCE",
            "NATURE_INJURY_CD": rng.choice(NATURE_CODES, n_rows),
            "NATURE_INJURY": "NATURE",
            "INJ_BODY_PART_CD": rng.choice(BODY_PART_CODES, n_rows),
            "INJ_BODY_PART": "BODY PART",
            "SCHEDULE_CHARGE": np.where(degree == "01", 6000.0, 0.0),
            "DAYS_RESTRICT": np.where(np.isin(degree, ["04", "05"]), rng.integers(1, 60, n_rows), 0).astype(float),
            "DAYS_LOST": np.where(np.isin(degree, ["03", "04"]), rng.integers(1, 120, n_rows), 0).astype(float),
            "TRANS_TERM": rng.choice(["Y", "N", None], n_rows, p=[0.3, 0.3, 0.4]),
            "RETURN_TO_WORK_DT": accident_dt + pd.to_timedelta(rng.integers(0, 90, n_rows), unit="D"),
            "IMMED_NOTIFY_CD": notify,
            "IMMED_NOTIFY": np.where(pd.isna(notify), None, "NOTIFY"),
            "INVEST_BEGIN_DT": pd.NaT,
            "NARRATIVE": "Synthetic narrative for testing.",
            "CLOSED_DOC_NO": np.where(rng.random(n_rows) < 0.5, document_no, None),
            "COAL_METAL_IND": mine["COAL_METAL_IND"].to_numpy(),
            # Join outputs (docs/interfaces.md section 1).
            "lost_time": lost_time,
            "filing_date": filing_date,
            "group_id": mine["MINE_ID"].to_numpy() + "_" + year.astype(str),
            "accident_group": mine["MINE_ID"].to_numpy() + "_" + accident_dt.dt.strftime("%Y-%m-%d").to_numpy(),
            "PRIMARY_CANVASS": mine["PRIMARY_CANVASS"].to_numpy(),
            "PRIMARY_SIC": mine["PRIMARY_SIC"].to_numpy(),
            "DISTRICT": mine["DISTRICT"].to_numpy(),
            "LATITUDE": mine["LATITUDE"].to_numpy(),
            "LONGITUDE": mine["LONGITUDE"].to_numpy(),
            "PORTABLE_OPERATION": mine["PORTABLE_OPERATION"].to_numpy(),
            "SAFETY_COMMITTEE_IND": mine["SAFETY_COMMITTEE_IND"].to_numpy(),
            "PART48_TRAINING": mine["PART48_TRAINING"].to_numpy(),
            "CURRENT_103I": mine["CURRENT_103I"].to_numpy(),
            "CURRENT_103I_DT": pd.to_datetime(mine["CURRENT_103I_DT"]).to_numpy(),
            "STATE": mine["STATE"].to_numpy(),
            "CURRENT_MINE_TYPE": mine["CURRENT_MINE_TYPE"].to_numpy(),
            "mine_coal_metal_ind": mine["COAL_METAL_IND"].to_numpy(),
            "_m_mines": "both",
            "prev_hours": prev_hours,
            "prev_employees": prev_employees,
            "prev_subunits": [s if m else None for s, m in zip(subunit_sets, matched, strict=True)],
            "prev_year_matched": matched,
            "same_year_hours": prev_hours * 1.02,
            "same_year_employees": prev_employees,
            "type_at_time": type_at_time,
            "prev_incidents": prev_incidents,
            "prev_valid_cases": prev_valid,
            "prev_lost_time_cases": prev_lost,
        }
    )
    frame["ACCIDENT_TIME"] = frame["ACCIDENT_TIME"].astype(object)
    return frame
