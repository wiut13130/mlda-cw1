"""Error-correction rules E1-E12 and E21-E25 of docs/profile/decisions.json, plus outlier flags.

`apply_error_rules` is a pure frame -> frame step that runs after the join and before feature
engineering. It corrects values that the profiling showed to be impossible (job experience
longer than total experience, clock times of 9999, coordinates in the ocean), decodes the
HHMM time fields, derives the dated 103(i) status and winsorises the lagged rates. It never
imputes: missing values stay missing so that the imputers inside the model Pipeline are
fitted on training rows only. Every rule reports the number of rows it touched; that log is
Table 4 of the report.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlda.config import RATE_CAP, RATE_PER_HOURS

log = logging.getLogger(__name__)

# Last-modified date of the analysed MSHA files (data/raw/SNAPSHOT.json); no accident can be later.
SNAPSHOT_DATE = pd.Timestamp("2026-10-09")
MAX_EXPERIENCE_YEARS = 60.0
NO_INJURIES_CAP = 10
MAX_HOURS_INTO_SHIFT = 16.0
MINUTES_PER_DAY = 1440
LAT_RANGE = (17.5, 72.0)
LON_RANGE = (-180.0, -64.0)
COORD_EXEMPT_STATE = "AS"  # American Samoa lies outside the box but is a real MSHA location
LOW_HOURS_THRESHOLD = 2_000  # E25: rates from mines with fewer hours than one full-time worker
NEVER_103I = "Never Had 103I Status"
REMOVED_103I = "Removed From 103I Status"

# Rule id, condition and action wording, copied from decisions.json so the report table matches.
RULES: dict[str, tuple[str, str]] = {
    "E1_job_exper_gt_total": ("JOB_EXPER > TOT_EXPER", "set JOB_EXPER NaN; exper_inconsistent = 1"),
    "E2_mine_exper_gt_total": (
        "MINE_EXPER > TOT_EXPER",
        "set MINE_EXPER NaN; exper_inconsistent = 1",
    ),
    "E3_exper_gt_60": (
        "TOT_EXPER > 60 or MINE_EXPER > 60 or JOB_EXPER > 60",
        "set offending value NaN",
    ),
    "E4_negative_exper": ("any experience < 0", "assert none"),
    "E5_accident_time_code": ("9990 <= ACCIDENT_TIME <= 9999", "NaN"),
    "E6_accident_time_2400": ("ACCIDENT_TIME == 2400", "set 0000"),
    "E7_accident_time_other_invalid": (
        "ACCIDENT_TIME blank, minutes >= 60 or > 2400 outside codes",
        "assert none",
    ),
    "E8_shift_time_code_or_blank": (
        "9990 <= SHIFT_BEGIN_TIME <= 9999 or NaN (blank masked by load.py)",
        "NaN",
    ),
    "E9_shift_time_2400": ("SHIFT_BEGIN_TIME == 2400", "set 0000"),
    "E10_shift_time_short": (
        "len(SHIFT_BEGIN_TIME) == 3 (e.g. '700')",
        "parse as integer HHMM (700 -> 07:00); do not string-slice",
    ),
    "E11_shift_wrap": ("acc_min < shift_min", "handled by mod 1440 in hours_into_shift"),
    "E12_hours_into_shift_gt_16": ("hours_into_shift > 16", "NaN"),
    "E21_no_injuries_cap": ("NO_INJURIES > 10", "cap at 10"),
    "E22_dates": (
        "ACCIDENT_DT unparsable or after snapshot; CAL_YR/CAL_QTR != derived",
        "assert none",
    ),
    "E23_coordinates": ("not (17.5<=lat<=72 and -180<=lon<=-64) and STATE != 'AS'", "NaN"),
    "E24_mine_attr_not_in_force": (
        "CURRENT_103I_DT > ACCIDENT_DT and not 'Never Had'; (CURRENT_CONTROLLER_BEGIN_DT > ACCIDENT_DT)",
        "103I: any status dated after the accident ('Removed From 103I Status' included) or blank "
        "-> 'Unknown', because nothing dated later was knowable at filing; COMPANY_TYPE: feature dropped",
    ),
    "E25_unstable_rates": (
        "prev_hours < 2000",
        "winsorise rates at 50 per 200k h then log1p; keep raw counts",
    ),
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "TOT_EXPER",
    "MINE_EXPER",
    "JOB_EXPER",
    "ACCIDENT_TIME",
    "SHIFT_BEGIN_TIME",
    "NO_INJURIES",
    "DEGREE_INJURY_CD",
    "ACCIDENT_DT",
    "CAL_YR",
    "CAL_QTR",
    "LATITUDE",
    "LONGITUDE",
    "STATE",
    "CURRENT_103I",
    "CURRENT_103I_DT",
    "prev_hours",
    "prev_incidents",
    "prev_lost_time_cases",
)


def parse_hhmm(raw: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Decode an HHMM time field to float HHMM and count what was corrected.

    MSHA stores the 24-hour clock as a NUMBER, so a leading zero can be lost ("700" is 07:00);
    parsing the whole value as an integer handles that, whereas slicing characters would not.
    9990-9999 are "not reported" codes and become NaN; 2400 is midnight and becomes 0; anything
    that is not a clock time (minutes >= 60, > 2400) also becomes NaN.
    """
    text = raw.astype("string")
    numeric = pd.to_numeric(text, errors="coerce").astype("float64")
    blank = numeric.isna()
    code = numeric.between(9990, 9999)
    midnight = numeric.eq(2400)
    valid_clock = numeric.between(0, 2359) & (numeric % 100 < 60)
    other_invalid = ~blank & ~code & ~midnight & ~valid_clock
    short = text.str.len().eq(3).fillna(False).astype(bool)

    clean = numeric.where(valid_clock, np.nan)
    clean = clean.mask(midnight, 0.0)
    counts = {
        "blank": int(blank.sum()),
        "code": int(code.sum()),
        "midnight": int(midnight.sum()),
        "short": int(short.sum()),
        "other_invalid": int(other_invalid.sum()),
    }
    return clean, counts


def _minutes(hhmm: pd.Series) -> pd.Series:
    return (hhmm // 100) * 60 + hhmm % 100


def status_103i_at_accident(
    current_103i: pd.Series, status_dt: pd.Series, accident_dt: pd.Series
) -> pd.Series:
    """Four-level 103(i) status in force at the accident date (decisions.json features.engineered).

    Mines.txt holds the status today and the date it was entered. A status entered on or before
    the accident date was still current at the accident, because it is the latest one. A status
    dated after the accident was not knowable at filing, so it becomes "Unknown", and that
    includes "Removed From 103I Status": the removal happened a median 4.3 years later and the
    date the mine was placed in 103(i) is not in the file, so "in force at the accident" cannot
    be checked. A blank status is also "Unknown".
    """
    known = current_103i.notna().to_numpy()
    never = current_103i.eq(NEVER_103I).fillna(False).to_numpy()
    removed = current_103i.eq(REMOVED_103I).fillna(False).to_numpy()
    dated_before = (status_dt.notna() & (status_dt <= accident_dt)).to_numpy() & known
    levels = np.where(
        never,
        "Never had 103I",
        np.where(
            dated_before, np.where(removed, "Formerly 103I", "In 103I at accident"), "Unknown"
        ),
    )
    return pd.Series(levels, index=current_103i.index, name="status_103i_at_accident")


def apply_error_rules(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply rules E1-E12 and E21-E25 and return (corrected frame, rule log).

    Columns changed in place: TOT_EXPER, MINE_EXPER, JOB_EXPER (impossible values -> NaN),
    ACCIDENT_TIME and SHIFT_BEGIN_TIME (strings -> float HHMM, codes -> NaN, 2400 -> 0),
    LATITUDE and LONGITUDE (invalid pairs -> NaN). Columns added: exper_inconsistent (int8),
    hours_into_shift (float hours, NaN when a time is invalid or the value exceeds 16),
    no_injuries_capped, latitude, longitude (validated copies), status_103i_at_accident,
    prev_incident_rate_200k_capped and prev_lost_time_rate_200k_capped (per 200,000 h,
    winsorised at RATE_CAP, not log-transformed; `mlda.features` applies log1p).
    The log has columns rule, condition, action, rows_affected.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in base.columns]
    if missing:
        raise KeyError(f"apply_error_rules needs columns {missing}")
    df = base.copy()
    affected: dict[str, int] = {}

    # E1-E4: experience fields. Flag before blanking so the flag records the original state.
    tot, mine, job = df["TOT_EXPER"], df["MINE_EXPER"], df["JOB_EXPER"]
    job_gt = (job > tot).fillna(False)
    mine_gt = (mine > tot).fillna(False)
    df["exper_inconsistent"] = (job_gt | mine_gt).astype("int8")
    df.loc[job_gt, "JOB_EXPER"] = np.nan
    df.loc[mine_gt, "MINE_EXPER"] = np.nan
    affected["E1_job_exper_gt_total"] = int(job_gt.sum())
    affected["E2_mine_exper_gt_total"] = int(mine_gt.sum())

    too_long = pd.DataFrame(
        {c: df[c] > MAX_EXPERIENCE_YEARS for c in ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER")}
    )
    affected["E3_exper_gt_60"] = int(too_long.any(axis=1).sum())
    negative = pd.DataFrame({c: df[c] < 0 for c in ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER")})
    affected["E4_negative_exper"] = int(negative.any(axis=1).sum())
    if affected["E4_negative_exper"]:
        log.warning(
            "E4: %d rows with negative experience (expected 0); set to NaN",
            affected["E4_negative_exper"],
        )
    for column in ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER"):
        df.loc[too_long[column].fillna(False) | negative[column].fillna(False), column] = np.nan

    # E5-E10: clock times.
    accident_time, acc_counts = parse_hhmm(df["ACCIDENT_TIME"])
    shift_time, shift_counts = parse_hhmm(df["SHIFT_BEGIN_TIME"])
    df["ACCIDENT_TIME"] = accident_time
    df["SHIFT_BEGIN_TIME"] = shift_time
    affected["E5_accident_time_code"] = acc_counts["code"]
    affected["E6_accident_time_2400"] = acc_counts["midnight"]
    affected["E7_accident_time_other_invalid"] = acc_counts["blank"] + acc_counts["other_invalid"]
    if affected["E7_accident_time_other_invalid"]:
        log.warning(
            "E7: %d invalid ACCIDENT_TIME values (expected 0); set to NaN",
            affected["E7_accident_time_other_invalid"],
        )
    affected["E8_shift_time_code_or_blank"] = shift_counts["code"] + shift_counts["blank"]
    affected["E9_shift_time_2400"] = shift_counts["midnight"]
    affected["E10_shift_time_short"] = shift_counts["short"]
    if shift_counts["other_invalid"]:
        log.warning(
            "%d SHIFT_BEGIN_TIME values are not clock times; set to NaN",
            shift_counts["other_invalid"],
        )

    # E11-E12: time into shift; the modulo handles shifts that start before midnight.
    acc_min, shift_min = _minutes(accident_time), _minutes(shift_time)
    hours_into_shift = ((acc_min - shift_min) % MINUTES_PER_DAY) / 60
    affected["E11_shift_wrap"] = int((acc_min < shift_min).fillna(False).sum())
    too_far = (hours_into_shift > MAX_HOURS_INTO_SHIFT).fillna(False)
    affected["E12_hours_into_shift_gt_16"] = int(too_far.sum())
    df["hours_into_shift"] = hours_into_shift.mask(too_far, np.nan)

    # E21: victim count.
    over_cap = (df["NO_INJURIES"] > NO_INJURIES_CAP).fillna(False)
    affected["E21_no_injuries_cap"] = int(over_cap.sum())
    df["no_injuries_capped"] = df["NO_INJURIES"].clip(upper=NO_INJURIES_CAP).astype("float64")
    zero_with_injury = (df["NO_INJURIES"].eq(0) & df["DEGREE_INJURY_CD"].notna()).sum()
    if zero_with_injury:
        log.warning(
            "E21: %d rows have NO_INJURIES = 0 with an injury code (expected 0)",
            int(zero_with_injury),
        )

    # E22: dates are only checked; nothing is changed.
    accident_dt = df["ACCIDENT_DT"]
    bad_dates = (
        accident_dt.isna()
        | (accident_dt > SNAPSHOT_DATE)
        | df["CAL_YR"].astype("float64").ne(accident_dt.dt.year.astype("float64"))
        | df["CAL_QTR"].astype("float64").ne(accident_dt.dt.quarter.astype("float64"))
    )
    affected["E22_dates"] = int(bad_dates.fillna(True).sum())
    if affected["E22_dates"]:
        log.warning("E22: %d rows with inconsistent dates (expected 0)", affected["E22_dates"])

    # E23: coordinates outside the US box (American Samoa exempt).
    lat, lon = df["LATITUDE"], df["LONGITUDE"]
    present = lat.notna() | lon.notna()
    inside = lat.between(*LAT_RANGE) & lon.between(*LON_RANGE)
    exempt = df["STATE"].eq(COORD_EXEMPT_STATE).fillna(False)
    invalid = present & ~inside & ~exempt
    affected["E23_coordinates"] = int(invalid.sum())
    df.loc[invalid, ["LATITUDE", "LONGITUDE"]] = np.nan
    df["latitude"] = df["LATITUDE"].astype("float64")
    df["longitude"] = df["LONGITUDE"].astype("float64")

    # E24: dated mine attribute. Rows affected = status not demonstrably in force at the accident
    # (dated after it, or blank), which is exactly the "Unknown" level: 22,368 on the base.
    df["status_103i_at_accident"] = status_103i_at_accident(
        df["CURRENT_103I"], df["CURRENT_103I_DT"], accident_dt
    )
    not_never = ~df["CURRENT_103I"].eq(NEVER_103I).fillna(False)
    dated_before = (
        df["CURRENT_103I"].notna()
        & df["CURRENT_103I_DT"].notna()
        & (df["CURRENT_103I_DT"] <= accident_dt)
    )
    dated_after = df["CURRENT_103I_DT"].notna() & (df["CURRENT_103I_DT"] > accident_dt)
    affected["E24_mine_attr_not_in_force"] = int((not_never & ~dated_before).sum())
    log.info(
        "E24: %d rows not demonstrably in 103I status at the accident (%d dated after it) -> Unknown",
        affected["E24_mine_attr_not_in_force"],
        int((dated_after & not_never).sum()),
    )

    # E25: lagged rates per 200,000 hours, winsorised at RATE_CAP (log1p is applied in features).
    hours = df["prev_hours"].where(df["prev_hours"] > 0)
    for count_col, rate_col in (
        ("prev_incidents", "prev_incident_rate_200k_capped"),
        ("prev_lost_time_cases", "prev_lost_time_rate_200k_capped"),
    ):
        df[rate_col] = (df[count_col] / hours * RATE_PER_HOURS).clip(upper=RATE_CAP)
    low_hours = (df["prev_hours"] < LOW_HOURS_THRESHOLD).fillna(False)
    affected["E25_unstable_rates"] = int(low_hours.sum())
    log.info(
        "E25: %d rows with prev_hours < %d; incident rate capped on %d rows",
        affected["E25_unstable_rates"],
        LOW_HOURS_THRESHOLD,
        int((df["prev_incidents"] / hours * RATE_PER_HOURS > RATE_CAP).fillna(False).sum()),
    )

    rule_log = pd.DataFrame(
        {
            "rule": list(RULES),
            "condition": [RULES[r][0] for r in RULES],
            "action": [RULES[r][1] for r in RULES],
            "rows_affected": [affected[r] for r in RULES],
        }
    )
    return df, rule_log


def flag_outliers(series: pd.Series, method: str = "iqr", k: float | None = None) -> pd.Series:
    """Boolean mask of outliers by Tukey fences (IQR, default k = 1.5) or Z-score (default k = 3).

    Missing values are never flagged. The Z-score uses the sample standard deviation (ddof = 1),
    the same estimator reported by `describe_numeric`. Used by the EDA tables and the app only;
    the model Pipeline fits its own clipper on training rows.
    """
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    if method == "iqr":
        k = 1.5 if k is None else k
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        mask = (values < q1 - k * iqr) | (values > q3 + k * iqr)
    elif method == "zscore":
        k = 3.0 if k is None else k
        std = values.std(ddof=1)
        if not std or np.isnan(std):
            mask = pd.Series(False, index=values.index)
        else:
            mask = ((values - values.mean()) / std).abs() > k
    else:
        raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}")
    return mask.fillna(False).astype(bool).rename(series.name)


__all__ = [
    "RULES",
    "REQUIRED_COLUMNS",
    "SNAPSHOT_DATE",
    "apply_error_rules",
    "flag_outliers",
    "parse_hhmm",
    "status_103i_at_accident",
]
