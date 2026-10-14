"""Row-wise feature engineering for the lost-time injury model.

``engineer_features`` turns the analytic base table (``mlda.data.join``, optionally
passed through ``mlda.cleaning``) into the exact model-input frame that the
ColumnTransformer in ``mlda.preprocessing`` and the Streamlit inference page expect:
23 categorical columns as strings, 20 numeric columns as floats with NaN allowed and
12 binary int8 flags with no NaN, in the order fixed by ``FEATURE_SPEC``.

Every formula comes from ``docs/profile/decisions.json`` (``features.engineered``).
The error rules that touch these inputs (E1-E12, E21, E23, E28-E30) are applied here
as well, idempotently: on a frame that ``mlda.cleaning`` has already corrected they
change nothing, and on a raw inference row they give the same result the training
data received. Nothing is fitted, so the same code runs on 100k rows or on one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mlda.cleaning import (
    COORD_EXEMPT_STATE,
    LAT_RANGE,
    LON_RANGE,
    MAX_EXPERIENCE_YEARS,
    MAX_HOURS_INTO_SHIFT,
    NO_INJURIES_CAP,
    status_103i_at_accident,
)
from mlda.config import (
    ACCIDENT_CATEGORICAL,
    ACCIDENT_GROUP_KEY,
    GROUP_KEY,
    PLACEHOLDER_AS_LEVEL,
    RATE_CAP,
    RATE_PER_HOURS,
    TARGET,
)
from mlda.data.load import PLACEHOLDERS

# --------------------------------------------------------------------------- spec

CATEGORICAL_FEATURES: tuple[str, ...] = (
    *ACCIDENT_CATEGORICAL,
    "PRIMARY_CANVASS",
    "type_at_time",
    "status_103i_at_accident",
    "mine_size_band",
    "SAFETY_COMMITTEE_IND",
    "PART48_TRAINING",
    "PORTABLE_OPERATION",
    "accident_weekday",
    "accident_month",
)
NUMERIC_FEATURES: tuple[str, ...] = (
    "TOT_EXPER",
    "MINE_EXPER",
    "JOB_EXPER",
    "log1p_tot_exper",
    "exper_ratio_mine",
    "exper_ratio_job",
    "no_injuries_capped",
    "accident_hour",
    "shift_begin_hour",
    "hours_into_shift",
    "latitude",
    "longitude",
    "prev_hours",
    "log1p_prev_hours",
    "prev_employees",
    "prev_hours_per_employee",
    "prev_incidents",
    "prev_incident_rate_200k",
    "prev_lost_time_rate_200k",
    "prev_lost_time_share",
)
BINARY_FEATURES: tuple[str, ...] = (
    "is_night",
    "is_weekend",
    "new_to_mine",
    "exper_inconsistent",
    "is_contractor",
    "immed_reportable",
    "prev_year_matched",
    "tot_exper_missing",
    "accident_hour_missing",
    "hours_into_shift_missing",
    "prev_share_missing",
    "coords_missing",
)
FEATURE_SPEC: dict[str, list[str]] = {
    "categorical": list(CATEGORICAL_FEATURES),
    "numeric": list(NUMERIC_FEATURES),
    "binary": list(BINARY_FEATURES),
}
FEATURE_COLUMNS: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

# Columns of the base table that the formulas read directly.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    *ACCIDENT_CATEGORICAL,
    "PRIMARY_CANVASS",
    "SAFETY_COMMITTEE_IND",
    "PART48_TRAINING",
    "PORTABLE_OPERATION",
    "ACCIDENT_DT",
    "ACCIDENT_TIME",
    "SHIFT_BEGIN_TIME",
    "TOT_EXPER",
    "MINE_EXPER",
    "JOB_EXPER",
    "NO_INJURIES",
    "LATITUDE",
    "LONGITUDE",
    "CONTRACTOR_ID",
    "IMMED_NOTIFY_CD",
    "prev_hours",
    "prev_employees",
    "prev_incidents",
    "prev_valid_cases",
    "prev_lost_time_cases",
    "prev_year_matched",
)
# Outputs of mlda.cleaning / mlda.data.join that are used when present and rebuilt
# from these source columns when absent (for example on a raw inference row).
DERIVED_IF_ABSENT: dict[str, tuple[str, ...]] = {
    "exper_inconsistent": ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER"),
    "no_injuries_capped": ("NO_INJURIES",),
    "status_103i_at_accident": ("CURRENT_103I", "CURRENT_103I_DT", "ACCIDENT_DT"),
    "type_at_time": ("SUBUNIT_CD",),
}
META_COLUMNS: tuple[str, ...] = (
    "CAL_YR",
    "COAL_METAL_IND",
    "MINE_ID",
    "ACCIDENT_DT",
    ACCIDENT_GROUP_KEY,
)

# --------------------------------------------------------------------------- constants

# The thresholds of E3, E12, E21, E23 and the 103(i) rule are defined once, in mlda.cleaning.
MIDNIGHT_CODE = 2400  # E6 / E9: 2400 is midnight, mapped to 00:00
NIGHT_HOURS = frozenset({0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23})
WEEKEND_DAYS = frozenset({5, 6})
MINE_SIZE_BINS: list[float] = [0, 5, 20, 50, 200, np.inf]
MINE_SIZE_LABELS: list[str] = ["1-4", "5-19", "20-49", "50-199", "200+"]
LATITUDE_RANGE = LAT_RANGE  # E23
LONGITUDE_RANGE = LON_RANGE
COORDINATE_EXEMPT_STATE = COORD_EXEMPT_STATE  # American Samoa lies outside the box but is valid
NOT_MARKED_NOTIFY_CODE = "13"  # E29: retired "Not marked" code, 2010-2012 only
DISCONTINUED_OCCUPATION_PREFIX = "Discontinued 1/31/2014"  # E28
UNDERGROUND_SUBUNITS = frozenset({"01", "02"})
FACILITY_SUBUNITS = frozenset({"30", "99", "17"})
MISSING_LEVEL = "missing"
PLACEHOLDER_LEVEL = "none"


# --------------------------------------------------------------------------- helpers


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"base table is missing required columns: {missing}")


def _num(series: pd.Series) -> pd.Series:
    """Coerce to float64 with NaN for anything that is not a number."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _text(series: pd.Series) -> pd.Series:
    """Stripped nullable strings; blanks and the MSHA placeholders count as missing.

    ``mlda.data.load`` already masks the placeholders ("?", "NO VALUE FOUND", ...), so on
    training data this changes nothing. A raw inference row that skipped the loader gets the
    same treatment here, which keeps ``immed_reportable`` and the "none" levels from inverting.
    """
    text = series.astype("string").str.strip()
    return text.mask(text.isin(PLACEHOLDERS))


def _categorical(series: pd.Series, fill: str) -> pd.Series:
    """String levels with missing values replaced by ``fill`` (object dtype, no NaN)."""
    return _text(series).fillna(fill).astype(object)


def _int_level(values: pd.Series) -> pd.Series:
    """Integer-valued floats (weekday, month) as string levels, NaN -> "missing"."""
    return values.astype("Int64").astype("string").fillna(MISSING_LEVEL).astype(object)


def _binary(mask: pd.Series) -> pd.Series:
    """Boolean mask (NaN allowed) -> int8 0/1 with missing treated as 0."""
    return mask.fillna(False).astype(bool).astype("int8")


def parse_time_minutes(values: pd.Series) -> pd.Series:
    """Decode HHMM clock codes into minutes after midnight (float, NaN when invalid).

    Values arrive as zero-padded strings from ``load.py``, as plain integers from the
    inference form, or already decoded by ``mlda.cleaning``; all three parse the same
    way because the text is converted to a number first and never sliced (E10: "700"
    means 07:00). Codes 9990-9999 mean "not reported" (E5, E8), 2400 is midnight
    (E6, E9) and anything with minutes >= 60 or outside 0-2359 is invalid (E7).
    """
    code = _num(values)
    hours = np.floor(code / 100)
    minutes = code - hours * 100
    valid = code.between(0, 2359) & (minutes < 60) & (code == np.floor(code))
    total = (hours * 60 + minutes).where(valid)
    return total.mask(code == MIDNIGHT_CODE, 0.0)


def _experience(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Apply E1-E4 to the three experience columns and return them with the flag.

    The flag is taken from ``mlda.cleaning`` when its column exists and is combined
    with the raw comparison so that an uncleaned inference row is flagged too.
    """
    tot, mine, job = (_num(frame[c]) for c in ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER"))
    job_over = job > tot
    mine_over = mine > tot
    inconsistent = job_over | mine_over
    if "exper_inconsistent" in frame.columns:
        inconsistent |= _num(frame["exper_inconsistent"]).fillna(0) > 0
    job = job.mask(job_over)  # E1
    mine = mine.mask(mine_over)  # E2
    tot, mine, job = (s.mask((s > MAX_EXPERIENCE_YEARS) | (s < 0)) for s in (tot, mine, job))
    return tot, mine, job, inconsistent


def _coordinates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """E23: coordinates outside the US box are NaN unless the mine is in American Samoa."""
    lat, lon = _num(frame["LATITUDE"]), _num(frame["LONGITUDE"])
    valid = lat.between(*LATITUDE_RANGE) & lon.between(*LONGITUDE_RANGE)
    if "STATE" in frame.columns:
        exempt = (_text(frame["STATE"]) == COORDINATE_EXEMPT_STATE).fillna(False).astype(bool)
        valid |= exempt & lat.notna() & lon.notna()
    return lat.where(valid), lon.where(valid)


def _rate_per_200k(count: pd.Series, hours: pd.Series) -> pd.Series:
    """MSHA incidence rate per 200,000 hours, winsorised at RATE_CAP (E25), then log1p."""
    rate = (count / hours * RATE_PER_HOURS).where(hours > 0)
    return np.log1p(rate.clip(upper=RATE_CAP))


def _status_103i(frame: pd.DataFrame) -> pd.Series:
    """Four-level 103(i) status in force at the accident date (E24), from ``mlda.cleaning``."""
    if "status_103i_at_accident" in frame.columns:
        return _categorical(frame["status_103i_at_accident"], MISSING_LEVEL)
    return status_103i_at_accident(
        _text(frame["CURRENT_103I"]),
        pd.to_datetime(frame["CURRENT_103I_DT"], errors="coerce"),
        pd.to_datetime(frame["ACCIDENT_DT"], errors="coerce"),
    ).astype(object)


def _type_at_time(frame: pd.DataFrame) -> pd.Series:
    """Mine type at the accident date; falls back to the accident SUBUNIT_CD."""
    if "type_at_time" in frame.columns:
        return _categorical(frame["type_at_time"], MISSING_LEVEL)
    subunit = _text(frame["SUBUNIT_CD"])
    levels = np.select(
        [
            subunit.isin(UNDERGROUND_SUBUNITS).fillna(False),
            subunit.isin(FACILITY_SUBUNITS).fillna(False),
        ],
        ["Underground", "Facility"],
        default="Surface",
    )
    return pd.Series(levels, index=frame.index, dtype=object)


# --------------------------------------------------------------------------- public API


def engineer_features(base_clean: pd.DataFrame) -> pd.DataFrame:
    """Return the model inputs listed in FEATURE_SPEC, in that order, for every row.

    Accepts the base table before or after ``mlda.cleaning`` and a single inference
    row alike; the input frame is not modified. Categorical columns are strings with
    NaN replaced by "none" for the structural placeholders in PLACEHOLDER_AS_LEVEL
    and by "missing" otherwise, numeric columns are float64 with NaN left for the
    imputer, and binary columns are int8 without NaN.
    """
    _require_columns(base_clean, REQUIRED_INPUT_COLUMNS)
    for derived, sources in DERIVED_IF_ABSENT.items():
        if derived not in base_clean.columns:
            _require_columns(base_clean, sources)
    frame = base_clean
    out: dict[str, pd.Series] = {}

    # Categorical codes and mine attributes.
    for column in (
        *ACCIDENT_CATEGORICAL,
        "PRIMARY_CANVASS",
        "SAFETY_COMMITTEE_IND",
        "PART48_TRAINING",
        "PORTABLE_OPERATION",
    ):
        fill = PLACEHOLDER_LEVEL if column in PLACEHOLDER_AS_LEVEL else MISSING_LEVEL
        out[column] = _categorical(frame[column], fill)
    # E28: the retired scoop-car title has no successor code, so it is mapped to the
    # "missing" level. That level is a training-only one-hot column (all zero after 2014);
    # the temporal evaluation reports it in its all-zero check.
    discontinued = out["OCCUPATION"].str.startswith(DISCONTINUED_OCCUPATION_PREFIX)
    out["OCCUPATION"] = out["OCCUPATION"].mask(discontinued, MISSING_LEVEL)
    out["type_at_time"] = _type_at_time(frame)
    out["status_103i_at_accident"] = _status_103i(frame)

    # Clock times (E5-E12).
    accident_minutes = parse_time_minutes(frame["ACCIDENT_TIME"])
    shift_minutes = parse_time_minutes(frame["SHIFT_BEGIN_TIME"])
    accident_hour = np.floor(accident_minutes / 60)
    shift_begin_hour = np.floor(shift_minutes / 60)
    hours_into_shift = np.mod(accident_minutes - shift_minutes, 1440) / 60  # E11 wrap
    hours_into_shift = hours_into_shift.where(hours_into_shift <= MAX_HOURS_INTO_SHIFT)  # E12

    # Calendar.
    accident_dt = pd.to_datetime(frame["ACCIDENT_DT"], errors="coerce")
    weekday = accident_dt.dt.weekday.astype("float64")
    out["accident_weekday"] = _int_level(weekday)
    out["accident_month"] = _int_level(accident_dt.dt.month.astype("float64"))

    # Worker experience (E1-E4, E30).
    tot, mine, job, inconsistent = _experience(frame)
    denominator_ok = tot > 0
    out["TOT_EXPER"], out["MINE_EXPER"], out["JOB_EXPER"] = tot, mine, job
    out["log1p_tot_exper"] = np.log1p(tot)
    out["exper_ratio_mine"] = (mine / tot).where(denominator_ok).clip(0, 1)
    out["exper_ratio_job"] = (job / tot).where(denominator_ok).clip(0, 1)

    injuries_source = (
        "no_injuries_capped" if "no_injuries_capped" in frame.columns else "NO_INJURIES"
    )
    out["no_injuries_capped"] = _num(frame[injuries_source]).clip(upper=NO_INJURIES_CAP)
    out["accident_hour"] = accident_hour
    out["shift_begin_hour"] = shift_begin_hour
    out["hours_into_shift"] = hours_into_shift
    out["latitude"], out["longitude"] = _coordinates(frame)

    # Previous-year exposure and case history from the mine-year join.
    prev_hours = _num(frame["prev_hours"]).where(lambda s: s >= 0)
    prev_employees = _num(frame["prev_employees"])
    prev_incidents = _num(frame["prev_incidents"])
    prev_valid = _num(frame["prev_valid_cases"])
    prev_lost = _num(frame["prev_lost_time_cases"])
    out["prev_hours"] = prev_hours
    out["log1p_prev_hours"] = np.log1p(prev_hours)
    out["prev_employees"] = prev_employees
    out["prev_hours_per_employee"] = (prev_hours / prev_employees).where(prev_employees > 0)
    out["prev_incidents"] = prev_incidents
    out["prev_incident_rate_200k"] = _rate_per_200k(prev_incidents, prev_hours)
    out["prev_lost_time_rate_200k"] = _rate_per_200k(prev_lost, prev_hours)
    share = (prev_lost / prev_valid).where(prev_valid > 0)
    out["prev_lost_time_share"] = share
    band = pd.cut(prev_employees, MINE_SIZE_BINS, right=False, labels=MINE_SIZE_LABELS)
    out["mine_size_band"] = band.astype(object).where(band.notna(), MISSING_LEVEL).astype(object)

    # Binary flags.
    notify = _text(frame["IMMED_NOTIFY_CD"])
    out["is_night"] = _binary(accident_hour.isin(NIGHT_HOURS))
    out["is_weekend"] = _binary(weekday.isin(WEEKEND_DAYS))
    out["new_to_mine"] = _binary(mine < 1)
    out["exper_inconsistent"] = _binary(inconsistent)
    out["is_contractor"] = _binary(_text(frame["CONTRACTOR_ID"]).notna())
    out["immed_reportable"] = _binary(
        notify.notna() & (notify != NOT_MARKED_NOTIFY_CODE)
    )  # E17, E29
    out["prev_year_matched"] = _binary(frame["prev_year_matched"])
    out["tot_exper_missing"] = _binary(tot.isna())
    out["accident_hour_missing"] = _binary(accident_hour.isna())
    out["hours_into_shift_missing"] = _binary(hours_into_shift.isna())
    out["prev_share_missing"] = _binary(share.isna())
    out["coords_missing"] = _binary(out["latitude"].isna() | out["longitude"].isna())

    features = pd.DataFrame(out, index=frame.index).loc[:, list(FEATURE_COLUMNS)]
    for column in NUMERIC_FEATURES:
        features[column] = features[column].astype("float64")
    return features


class FeatureFrame(NamedTuple):
    """Model inputs plus the columns needed for grouping, stratified reporting and splits."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    meta: pd.DataFrame


def feature_frame(base_clean: pd.DataFrame) -> FeatureFrame:
    """Split the base table into X (FEATURE_SPEC columns), y, groups and meta.

    ``groups`` is the mine-year key used by the grouped split and CV; ``meta`` keeps
    CAL_YR, COAL_METAL_IND, MINE_ID, ACCIDENT_DT and accident_group for the temporal
    split and for reporting metrics by sector. All four share the base table's index.
    """
    _require_columns(base_clean, (TARGET, GROUP_KEY, *META_COLUMNS))
    X = engineer_features(base_clean)
    y = _num(base_clean[TARGET]).astype("int8").rename(TARGET)
    groups = base_clean[GROUP_KEY].astype(str).rename(GROUP_KEY)
    meta = base_clean.loc[:, list(META_COLUMNS)].copy()
    return FeatureFrame(X, y, groups, meta)


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless transformer wrapping ``engineer_features`` for use inside a Pipeline.

    Nothing is learnt, so ``fit`` only returns self; this lets the saved model take a
    raw base-table row (one accident from the inference form) and produce the same
    columns the ColumnTransformer was fitted on.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        """No-op: the feature formulas have no parameters to learn."""
        _require_columns(X, REQUIRED_INPUT_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply ``engineer_features`` to one or many base-table rows."""
        return engineer_features(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Output names are fixed by FEATURE_SPEC regardless of the input columns."""
        return np.asarray(FEATURE_COLUMNS, dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        return True
