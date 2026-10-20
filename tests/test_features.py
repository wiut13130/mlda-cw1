"""Feature formulas from docs/profile/decisions.json, checked on hand-built rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _synthetic import synthetic_base
from sklearn.base import clone

from mlda import cleaning
from mlda import features as features_module
from mlda.config import PLACEHOLDER_AS_LEVEL
from mlda.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_SPEC,
    NUMERIC_FEATURES,
    FeatureEngineer,
    engineer_features,
    feature_frame,
    parse_time_minutes,
)


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return synthetic_base(400)


@pytest.fixture(scope="module")
def features(base: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(base)


def _row(base: pd.DataFrame, **overrides: object) -> pd.DataFrame:
    """One base row with selected columns overwritten (times as the file stores them)."""
    row = base.iloc[[3]].copy()
    for column, value in overrides.items():
        row[column] = [value]
    return row


# --------------------------------------------------------------------------- spec


def test_columns_match_feature_spec_in_order(features: pd.DataFrame) -> None:
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert FEATURE_COLUMNS == tuple(FEATURE_SPEC["categorical"] + FEATURE_SPEC["numeric"] + FEATURE_SPEC["binary"])
    assert (len(CATEGORICAL_FEATURES), len(NUMERIC_FEATURES), len(BINARY_FEATURES)) == (20, 20, 12)
    assert len(FEATURE_COLUMNS) == 52
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_snapshot_only_mines_flags_are_not_model_inputs(base: pd.DataFrame, features: pd.DataFrame) -> None:
    """The Mines table holds these three flags as at the snapshot, so their value at filing is unknown.

    They stay in the base table (the data explorer shows them) but must reach neither the model
    inputs nor the columns the formulas read; the output must not change when they are removed.
    """
    flags = ("SAFETY_COMMITTEE_IND", "PART48_TRAINING", "PORTABLE_OPERATION")
    assert all(flag in base.columns for flag in flags)
    assert set(flags).isdisjoint(FEATURE_COLUMNS) and set(flags).isdisjoint(features.columns)
    assert set(flags).isdisjoint(features_module.REQUIRED_INPUT_COLUMNS)
    pd.testing.assert_frame_equal(engineer_features(base.drop(columns=list(flags))), features)


def test_dtypes_and_missingness_by_block(features: pd.DataFrame) -> None:
    categorical = features[FEATURE_SPEC["categorical"]]
    assert categorical.isna().sum().sum() == 0
    assert all(categorical[c].map(type).eq(str).all() for c in categorical.columns)
    numeric = features[FEATURE_SPEC["numeric"]]
    assert all(numeric[c].dtype == "float64" for c in numeric.columns)
    binary = features[FEATURE_SPEC["binary"]]
    assert all(binary[c].dtype == "int8" for c in binary.columns)
    assert binary.isna().sum().sum() == 0
    assert binary.isin([0, 1]).all().all()


# --------------------------------------------------------------------------- clock times


def test_parse_time_minutes_handles_every_code() -> None:
    codes = pd.Series(["0740", "700", "2400", "9999", "9990", "1260", None, 130, 7.0])
    expected = [460.0, 420.0, 0.0, np.nan, np.nan, np.nan, np.nan, 90.0, 7.0]
    assert np.allclose(parse_time_minutes(codes).to_numpy(), expected, equal_nan=True)


def test_hours_into_shift_wraps_past_midnight(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, SHIFT_BEGIN_TIME="2200", ACCIDENT_TIME="0130"))
    assert out["hours_into_shift"].iloc[0] == pytest.approx(3.5)
    assert out["hours_into_shift_missing"].iloc[0] == 0
    assert out["shift_begin_hour"].iloc[0] == 22
    assert out["accident_hour"].iloc[0] == 1


def test_midnight_code_2400_is_hour_zero(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, ACCIDENT_TIME="2400", SHIFT_BEGIN_TIME="2200"))
    assert out["accident_hour"].iloc[0] == 0
    assert out["is_night"].iloc[0] == 1
    assert out["hours_into_shift"].iloc[0] == pytest.approx(2.0)


def test_not_reported_time_9999_is_missing_with_flag(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, ACCIDENT_TIME="9999", SHIFT_BEGIN_TIME="0700"))
    assert np.isnan(out["accident_hour"].iloc[0])
    assert out["accident_hour_missing"].iloc[0] == 1
    assert np.isnan(out["hours_into_shift"].iloc[0])
    assert out["hours_into_shift_missing"].iloc[0] == 1
    assert out["is_night"].iloc[0] == 0
    assert out["shift_begin_hour"].iloc[0] == 7


def test_three_character_shift_time_parses_as_integer(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, SHIFT_BEGIN_TIME="700", ACCIDENT_TIME="0930"))
    assert out["shift_begin_hour"].iloc[0] == 7
    assert out["hours_into_shift"].iloc[0] == pytest.approx(2.5)


def test_hours_into_shift_above_16_is_missing_but_hour_kept(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, SHIFT_BEGIN_TIME="0700", ACCIDENT_TIME="0030"))
    assert np.isnan(out["hours_into_shift"].iloc[0])
    assert out["hours_into_shift_missing"].iloc[0] == 1
    assert out["accident_hour"].iloc[0] == 0
    exactly_16 = engineer_features(_row(base, SHIFT_BEGIN_TIME="0700", ACCIDENT_TIME="2300"))
    assert exactly_16["hours_into_shift"].iloc[0] == pytest.approx(16.0)


def test_is_night_follows_the_hour_set(base: pd.DataFrame) -> None:
    for time, expected in (("0300", 1), ("2015", 1), ("0559", 1), ("0600", 0), ("1200", 0), ("1759", 0), ("1800", 1)):
        assert engineer_features(_row(base, ACCIDENT_TIME=time))["is_night"].iloc[0] == expected, time


# --------------------------------------------------------------------------- mine-year features


def test_mine_size_band_boundaries_are_left_closed(base: pd.DataFrame) -> None:
    cases = {4.9: "1-4", 5: "5-19", 19.9: "5-19", 20: "20-49", 50: "50-199", 199: "50-199", 200: "200+", np.nan: "missing"}
    for employees, band in cases.items():
        out = engineer_features(_row(base, prev_employees=employees))
        assert out["mine_size_band"].iloc[0] == band, employees


def test_prev_lost_time_share_missing_without_valid_cases(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, prev_valid_cases=0.0, prev_lost_time_cases=0.0, prev_incidents=2.0))
    assert np.isnan(out["prev_lost_time_share"].iloc[0])
    assert out["prev_share_missing"].iloc[0] == 1
    out = engineer_features(_row(base, prev_valid_cases=4.0, prev_lost_time_cases=3.0))
    assert out["prev_lost_time_share"].iloc[0] == pytest.approx(0.75)
    assert out["prev_share_missing"].iloc[0] == 0


def test_rates_are_winsorised_and_guard_zero_hours(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, prev_hours=100_000.0, prev_incidents=3.0, prev_lost_time_cases=1.0))
    assert out["prev_incident_rate_200k"].iloc[0] == pytest.approx(np.log1p(6.0))
    assert out["prev_lost_time_rate_200k"].iloc[0] == pytest.approx(np.log1p(2.0))
    capped = engineer_features(_row(base, prev_hours=100.0, prev_incidents=3.0))
    assert capped["prev_incident_rate_200k"].iloc[0] == pytest.approx(np.log1p(50.0))
    zero_hours = engineer_features(_row(base, prev_hours=0.0, prev_incidents=3.0, prev_employees=0.0))
    for column in ("prev_incident_rate_200k", "prev_lost_time_rate_200k", "prev_hours_per_employee"):
        assert np.isnan(zero_hours[column].iloc[0]), column
    assert zero_hours["log1p_prev_hours"].iloc[0] == 0.0


def test_unmatched_previous_year_sets_flag_and_nan(base: pd.DataFrame) -> None:
    out = engineer_features(
        _row(base, prev_year_matched=False, prev_hours=np.nan, prev_employees=np.nan, prev_incidents=np.nan, prev_valid_cases=np.nan, prev_lost_time_cases=np.nan)
    )
    assert out["prev_year_matched"].iloc[0] == 0
    assert out["mine_size_band"].iloc[0] == "missing"
    assert out[["prev_hours", "log1p_prev_hours", "prev_employees", "prev_hours_per_employee", "prev_incidents"]].isna().all().all()


# --------------------------------------------------------------------------- categorical fills


def test_placeholder_columns_fill_none_and_others_missing(base: pd.DataFrame) -> None:
    row = _row(base, UG_LOCATION_CD=None, EQUIP_MFR_NAME=None, ACTIVITY_CD=None, SUBUNIT_CD=None, PRIMARY_CANVASS=None)
    out = engineer_features(row)
    for column in ("UG_LOCATION_CD", "EQUIP_MFR_NAME", "ACTIVITY_CD"):
        assert column in PLACEHOLDER_AS_LEVEL and out[column].iloc[0] == "none", column
    for column in ("SUBUNIT_CD", "PRIMARY_CANVASS"):
        assert out[column].iloc[0] == "missing", column


def test_calendar_features_are_string_levels(base: pd.DataFrame) -> None:
    saturday = engineer_features(_row(base, ACCIDENT_DT=pd.Timestamp("2021-06-05")))
    assert saturday["accident_weekday"].iloc[0] == "5"
    assert saturday["accident_month"].iloc[0] == "6"
    assert saturday["is_weekend"].iloc[0] == 1
    monday = engineer_features(_row(base, ACCIDENT_DT=pd.Timestamp("2021-06-07")))
    assert monday["accident_weekday"].iloc[0] == "0"
    assert monday["is_weekend"].iloc[0] == 0


def test_discontinued_occupation_title_is_treated_as_missing(base: pd.DataFrame) -> None:
    out = engineer_features(_row(base, OCCUPATION="Discontinued 1/31/2014. Several other codes cover Scoop Car Operator"))
    assert out["OCCUPATION"].iloc[0] == "missing"
    kept = engineer_features(_row(base, OCCUPATION="Electrician"))
    assert kept["OCCUPATION"].iloc[0] == "Electrician"


# --------------------------------------------------------------------------- experience and flags


def test_experience_rules_and_zero_denominator_guard(base: pd.DataFrame) -> None:
    inconsistent = engineer_features(_row(base, TOT_EXPER=5.0, MINE_EXPER=2.0, JOB_EXPER=9.0))
    assert np.isnan(inconsistent["JOB_EXPER"].iloc[0])
    assert inconsistent["exper_inconsistent"].iloc[0] == 1
    assert np.isnan(inconsistent["exper_ratio_job"].iloc[0])
    assert inconsistent["exper_ratio_mine"].iloc[0] == pytest.approx(0.4)
    assert inconsistent["log1p_tot_exper"].iloc[0] == pytest.approx(np.log1p(5.0))
    assert inconsistent["tot_exper_missing"].iloc[0] == 0

    zero_total = engineer_features(_row(base, TOT_EXPER=0.0, MINE_EXPER=0.0, JOB_EXPER=0.0))
    assert np.isnan(zero_total["exper_ratio_mine"].iloc[0]) and np.isnan(zero_total["exper_ratio_job"].iloc[0])
    assert zero_total["new_to_mine"].iloc[0] == 1

    implausible = engineer_features(_row(base, TOT_EXPER=70.0, MINE_EXPER=np.nan, JOB_EXPER=1.0))
    assert np.isnan(implausible["TOT_EXPER"].iloc[0])
    assert implausible["tot_exper_missing"].iloc[0] == 1
    assert implausible["new_to_mine"].iloc[0] == 0


def test_notification_and_contractor_flags(base: pd.DataFrame) -> None:
    assert engineer_features(_row(base, IMMED_NOTIFY_CD=None))["immed_reportable"].iloc[0] == 0
    assert engineer_features(_row(base, IMMED_NOTIFY_CD="13"))["immed_reportable"].iloc[0] == 0
    assert engineer_features(_row(base, IMMED_NOTIFY_CD="01"))["immed_reportable"].iloc[0] == 1
    assert engineer_features(_row(base, CONTRACTOR_ID=None))["is_contractor"].iloc[0] == 0
    assert engineer_features(_row(base, CONTRACTOR_ID=""))["is_contractor"].iloc[0] == 0
    assert engineer_features(_row(base, CONTRACTOR_ID="1KJ"))["is_contractor"].iloc[0] == 1


def test_no_injuries_is_capped_at_ten(base: pd.DataFrame) -> None:
    assert engineer_features(_row(base, NO_INJURIES=36.0))["no_injuries_capped"].iloc[0] == 10
    assert engineer_features(_row(base, NO_INJURIES=2.0))["no_injuries_capped"].iloc[0] == 2


# --------------------------------------------------------------------------- mine attributes


def test_coordinates_outside_the_box_are_missing_unless_american_samoa(base: pd.DataFrame) -> None:
    invalid = engineer_features(_row(base, LATITUDE=0.0, LONGITUDE=0.0, STATE="WV"))
    assert np.isnan(invalid["latitude"].iloc[0]) and invalid["coords_missing"].iloc[0] == 1
    valid = engineer_features(_row(base, LATITUDE=37.5, LONGITUDE=-81.2, STATE="WV"))
    assert valid["latitude"].iloc[0] == pytest.approx(37.5) and valid["coords_missing"].iloc[0] == 0
    samoa = engineer_features(_row(base, LATITUDE=-14.3, LONGITUDE=-170.7, STATE="AS"))
    assert samoa["longitude"].iloc[0] == pytest.approx(-170.7) and samoa["coords_missing"].iloc[0] == 0


def test_status_103i_levels_follow_the_date_rule(base: pd.DataFrame) -> None:
    accident = pd.Timestamp("2018-06-01")
    cases = [
        ("Never Had 103I Status", pd.NaT, "Never had 103I"),
        ("5 Day 103I Status", pd.Timestamp("2016-01-01"), "In 103I at accident"),
        ("Removed From 103I Status", pd.Timestamp("2016-01-01"), "Formerly 103I"),
        ("Removed From 103I Status", pd.Timestamp("2020-01-01"), "Unknown"),  # removal not knowable at filing
        ("5 Day 103I Status", pd.Timestamp("2020-01-01"), "Unknown"),
        (None, pd.NaT, "Unknown"),
        (None, pd.Timestamp("2016-01-01"), "Unknown"),  # a date without a status is not "In 103I"
    ]
    for status, status_dt, expected in cases:
        row = _row(base, CURRENT_103I=status, CURRENT_103I_DT=status_dt, ACCIDENT_DT=accident).drop(columns="status_103i_at_accident", errors="ignore")
        assert engineer_features(row)["status_103i_at_accident"].iloc[0] == expected, (status, status_dt)


def test_type_at_time_falls_back_to_subunit_when_absent(base: pd.DataFrame) -> None:
    for subunit, expected in (("01", "Underground"), ("02", "Underground"), ("30", "Facility"), ("17", "Facility"), ("03", "Surface")):
        row = _row(base, SUBUNIT_CD=subunit).drop(columns="type_at_time")
        assert engineer_features(row)["type_at_time"].iloc[0] == expected, subunit
    kept = engineer_features(_row(base, SUBUNIT_CD="03", type_at_time="Underground"))
    assert kept["type_at_time"].iloc[0] == "Underground"


# --------------------------------------------------------------------------- API behaviour


def test_input_frame_is_not_mutated_and_missing_columns_raise(base: pd.DataFrame) -> None:
    before = base.copy(deep=True)
    engineer_features(base)
    pd.testing.assert_frame_equal(base, before)
    with pytest.raises(KeyError, match="prev_hours"):
        engineer_features(base.drop(columns=["prev_hours"]))
    with pytest.raises(KeyError, match="CURRENT_103I"):
        engineer_features(base.drop(columns=["status_103i_at_accident", "CURRENT_103I"], errors="ignore"))


def test_rules_are_idempotent_on_a_cleaned_frame(base: pd.DataFrame, features: pd.DataFrame) -> None:
    """A frame that mlda.cleaning has already corrected must give the same features."""
    cleaned = base.copy()
    cleaned["ACCIDENT_TIME"] = parse_time_minutes(base["ACCIDENT_TIME"]).pipe(lambda m: (m // 60) * 100 + m % 60)
    cleaned["SHIFT_BEGIN_TIME"] = parse_time_minutes(base["SHIFT_BEGIN_TIME"]).pipe(lambda m: (m // 60) * 100 + m % 60)
    for column in ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER"):
        cleaned[column] = features[column]
    cleaned["exper_inconsistent"] = features["exper_inconsistent"]
    cleaned["no_injuries_capped"] = features["no_injuries_capped"]
    cleaned["status_103i_at_accident"] = features["status_103i_at_accident"]
    cleaned["LATITUDE"], cleaned["LONGITUDE"] = features["latitude"], features["longitude"]
    pd.testing.assert_frame_equal(engineer_features(cleaned), features)


def test_feature_engineer_is_stateless_and_works_on_one_row(base: pd.DataFrame, features: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    assert engineer.fit(base.iloc[:50]) is engineer
    one_row = engineer.transform(base.iloc[[11]])
    pd.testing.assert_frame_equal(one_row, features.iloc[[11]])
    assert list(engineer.get_feature_names_out()) == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(clone(engineer).fit_transform(base), features)


def test_feature_frame_returns_aligned_parts(base: pd.DataFrame) -> None:
    X, y, groups, meta = feature_frame(base)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert y.dtype == "int8" and set(y.unique()) <= {0, 1} and y.name == "lost_time"
    assert groups.name == "group_id" and groups.equals(base["group_id"].astype(str))
    assert list(meta.columns) == ["CAL_YR", "COAL_METAL_IND", "MINE_ID", "ACCIDENT_DT", "accident_group"]
    assert X.index.equals(base.index) and y.index.equals(base.index) and meta.index.equals(base.index)


def test_placeholders_that_skipped_the_loader_are_treated_as_missing(base: pd.DataFrame) -> None:
    """mlda.data.load masks "?" and friends; a raw inference row gets the same treatment here."""
    row = _row(base, IMMED_NOTIFY_CD="? ", UG_LOCATION_CD="?", EQUIP_MFR_NAME="NO VALUE FOUND", MINING_EQUIP_CD="")
    out = engineer_features(row)
    assert out["immed_reportable"].iloc[0] == 0  # not 1: the literal "? " is not a notification code
    for column in ("UG_LOCATION_CD", "EQUIP_MFR_NAME", "MINING_EQUIP_CD"):
        assert out[column].iloc[0] == "none", column
    assert engineer_features(_row(base, IMMED_NOTIFY_CD="01"))["immed_reportable"].iloc[0] == 1
    assert engineer_features(_row(base, IMMED_NOTIFY_CD="13"))["immed_reportable"].iloc[0] == 0  # E29


def test_thresholds_and_the_103i_rule_are_defined_once_in_cleaning() -> None:
    assert features_module.MAX_EXPERIENCE_YEARS == cleaning.MAX_EXPERIENCE_YEARS
    assert features_module.MAX_HOURS_INTO_SHIFT == cleaning.MAX_HOURS_INTO_SHIFT
    assert features_module.NO_INJURIES_CAP == cleaning.NO_INJURIES_CAP
    assert features_module.LATITUDE_RANGE == cleaning.LAT_RANGE
    assert features_module.LONGITUDE_RANGE == cleaning.LON_RANGE
    assert features_module.COORDINATE_EXEMPT_STATE == cleaning.COORD_EXEMPT_STATE
    assert features_module.status_103i_at_accident is cleaning.status_103i_at_accident


def test_retired_scoop_car_title_maps_to_the_missing_level(base: pd.DataFrame) -> None:
    """E28 is realised as the "missing" level (not the infrequent bucket); see data_profile.md section 5."""
    title = "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator"
    assert engineer_features(_row(base, OCCUPATION=title))["OCCUPATION"].iloc[0] == "missing"
    assert engineer_features(_row(base, OCCUPATION="Electrician"))["OCCUPATION"].iloc[0] == "Electrician"


def test_real_feature_counts_match_decisions(real_features: features_module.FeatureFrame) -> None:
    X = real_features.X
    assert X.shape == (101_609, 52) and list(X.columns) == list(FEATURE_COLUMNS)
    assert int(X["immed_reportable"].sum()) == 3_665
    assert X["mine_size_band"].value_counts().to_dict() == {
        "200+": 36_801, "50-199": 30_483, "20-49": 16_654, "5-19": 14_180, "1-4": 2_285, "missing": 1_206,
    }
    assert X["status_103i_at_accident"].value_counts().to_dict() == {
        "Never had 103I": 61_397, "Unknown": 22_368, "In 103I at accident": 9_344, "Formerly 103I": 8_500,
    }
    assert round(100 * X["hours_into_shift_missing"].mean(), 2) == 3.80
    assert int((X["OCCUPATION"] == "missing").sum()) == 533  # the retired scoop-car title, 2010-2014 only
    assert int(X["prev_incidents"].notna().sum()) == 100_403 and int((X["prev_incidents"] == 0).sum()) == 17_822
    assert int(X["prev_lost_time_share"].notna().sum()) == 81_469
    assert int(X["prev_share_missing"].sum()) == 20_140
    assert X["type_at_time"].value_counts().to_dict() == {"Surface": 51_053, "Underground": 40_379, "Facility": 10_177}
