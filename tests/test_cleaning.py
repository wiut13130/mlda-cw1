"""Error-correction rules E1-E12 and E21-E25 on hand-built rows; outlier flags on a known vector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlda.cleaning import (
    REQUIRED_COLUMNS,
    RULES,
    apply_error_rules,
    flag_outliers,
    parse_hhmm,
    status_103i_at_accident,
)
from mlda.config import RATE_CAP

# A row that triggers no rule: every test overrides one or two fields.
CLEAN_ROW: dict[str, object] = {
    "TOT_EXPER": 10.0, "MINE_EXPER": 5.0, "JOB_EXPER": 3.0, "ACCIDENT_TIME": "0900",
    "SHIFT_BEGIN_TIME": "0700", "NO_INJURIES": 1, "DEGREE_INJURY_CD": "03",
    "ACCIDENT_DT": "2015-06-15", "CAL_YR": 2015, "CAL_QTR": 2, "LATITUDE": 38.3, "LONGITUDE": -81.5,
    "STATE": "WV", "CURRENT_103I": "Never Had 103I Status", "CURRENT_103I_DT": np.nan,
    "prev_hours": 100_000.0, "prev_incidents": 5.0, "prev_lost_time_cases": 2.0,
}


def rows(*overrides: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame([{**CLEAN_ROW, **o} for o in overrides], columns=list(REQUIRED_COLUMNS))
    frame["ACCIDENT_DT"] = pd.to_datetime(frame["ACCIDENT_DT"])
    frame["CURRENT_103I_DT"] = pd.to_datetime(frame["CURRENT_103I_DT"])
    frame["CAL_YR"] = frame["CAL_YR"].astype("Int64")
    frame["CAL_QTR"] = frame["CAL_QTR"].astype("Int64")
    for column in ("ACCIDENT_TIME", "SHIFT_BEGIN_TIME", "CURRENT_103I", "STATE"):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame


def affected(log: pd.DataFrame) -> dict[str, int]:
    return dict(zip(log["rule"], log["rows_affected"], strict=True))


def test_log_lists_every_rule_in_order_with_contract_columns() -> None:
    _, log = apply_error_rules(rows({}))
    assert list(log.columns) == ["rule", "condition", "action", "rows_affected"]
    assert log["rule"].tolist() == list(RULES)
    assert log["rule"].str.fullmatch(r"E(\d|1[0-2]|2[1-5])_[a-z0-9_]+").all()
    assert (log["rows_affected"] == 0).all()


def test_clean_row_is_untouched_and_input_not_mutated() -> None:
    frame = rows({})
    snapshot = frame.copy()
    clean, _ = apply_error_rules(frame)
    pd.testing.assert_frame_equal(frame, snapshot)
    assert clean.loc[0, "exper_inconsistent"] == 0
    assert clean.loc[0, "ACCIDENT_TIME"] == 900.0 and clean.loc[0, "SHIFT_BEGIN_TIME"] == 700.0
    assert clean.loc[0, "hours_into_shift"] == 2.0
    assert clean.loc[0, "no_injuries_capped"] == 1
    assert clean.loc[0, "status_103i_at_accident"] == "Never had 103I"
    assert clean.loc[0, "prev_incident_rate_200k_capped"] == 10.0
    assert clean.loc[0, "prev_lost_time_rate_200k_capped"] == 4.0


def test_missing_required_column_raises() -> None:
    with pytest.raises(KeyError, match="prev_hours"):
        apply_error_rules(rows({}).drop(columns=["prev_hours"]))


def test_experience_rules_e1_to_e4() -> None:
    frame = rows(
        {"JOB_EXPER": 12.0},  # E1
        {"MINE_EXPER": 12.0},  # E2
        {"TOT_EXPER": 65.0},  # E3
        {"MINE_EXPER": -1.0},  # E4
        {"TOT_EXPER": np.nan, "JOB_EXPER": 20.0},  # NaN total: no comparison possible, no rule
    )
    clean, log = apply_error_rules(frame)
    counts = affected(log)
    assert counts["E1_job_exper_gt_total"] == 1 and counts["E2_mine_exper_gt_total"] == 1
    assert counts["E3_exper_gt_60"] == 1 and counts["E4_negative_exper"] == 1
    assert np.isnan(clean.loc[0, "JOB_EXPER"]) and clean.loc[0, "TOT_EXPER"] == 10.0
    assert np.isnan(clean.loc[1, "MINE_EXPER"])
    assert clean["exper_inconsistent"].tolist() == [1, 1, 0, 0, 0]
    assert clean["exper_inconsistent"].dtype == np.int8
    assert np.isnan(clean.loc[2, "TOT_EXPER"]) and clean.loc[2, "JOB_EXPER"] == 3.0
    assert np.isnan(clean.loc[3, "MINE_EXPER"])
    assert clean.loc[4, "JOB_EXPER"] == 20.0


def test_time_rules_e5_to_e12() -> None:
    frame = rows(
        {"ACCIDENT_TIME": "9999"},  # E5
        {"ACCIDENT_TIME": "2400", "SHIFT_BEGIN_TIME": "2300"},  # E6; 00:00 after a 23:00 start wraps (E11)
        {"ACCIDENT_TIME": "0975"},  # E7 minutes >= 60
        {"ACCIDENT_TIME": np.nan},  # E7 blank
        {"SHIFT_BEGIN_TIME": "9999"},  # E8
        {"SHIFT_BEGIN_TIME": np.nan},  # E8
        {"SHIFT_BEGIN_TIME": "2400"},  # E9
        {"SHIFT_BEGIN_TIME": "700"},  # E10 lost leading zero
        {"ACCIDENT_TIME": "0100", "SHIFT_BEGIN_TIME": "2300"},  # E11 wrap past midnight
        {"ACCIDENT_TIME": "0100", "SHIFT_BEGIN_TIME": "0800"},  # E12 17 h into shift (also a wrap)
    )
    clean, log = apply_error_rules(frame)
    counts = affected(log)
    assert counts["E5_accident_time_code"] == 1
    assert counts["E6_accident_time_2400"] == 1
    assert counts["E7_accident_time_other_invalid"] == 2
    assert counts["E8_shift_time_code_or_blank"] == 2
    assert counts["E9_shift_time_2400"] == 1
    assert counts["E10_shift_time_short"] == 1
    assert counts["E11_shift_wrap"] == 3
    assert counts["E12_hours_into_shift_gt_16"] == 1

    acc, shift, into = clean["ACCIDENT_TIME"], clean["SHIFT_BEGIN_TIME"], clean["hours_into_shift"]
    assert np.isnan(acc[0]) and np.isnan(into[0])
    assert acc[1] == 0.0 and into[1] == 1.0
    assert np.isnan(acc[2]) and np.isnan(acc[3])
    assert np.isnan(shift[4]) and np.isnan(shift[5]) and np.isnan(into[4])
    assert shift[6] == 0.0 and into[6] == 9.0
    assert shift[7] == 700.0 and into[7] == 2.0
    assert into[8] == 2.0
    assert np.isnan(into[9]) and acc[9] == 100.0  # accident_hour stays available


def test_parse_hhmm_counts() -> None:
    clean, counts = parse_hhmm(pd.Series(["0740", "700", "2400", "9999", "1299", np.nan, "0000"], dtype=object))
    assert clean.iloc[0] == 740.0 and clean.iloc[1] == 700.0 and clean.iloc[2] == 0.0
    assert np.isnan(clean.iloc[3]) and np.isnan(clean.iloc[4]) and np.isnan(clean.iloc[5])
    assert clean.iloc[6] == 0.0
    assert counts == {"blank": 1, "code": 1, "midnight": 1, "short": 1, "other_invalid": 1}


def test_victim_count_dates_and_coordinates_e21_to_e23() -> None:
    frame = rows(
        {"NO_INJURIES": 12},  # E21
        {"CAL_YR": 2014},  # E22 year disagrees with ACCIDENT_DT
        {"ACCIDENT_DT": "2027-01-01", "CAL_YR": 2027, "CAL_QTR": 1},  # E22 after the snapshot
        {"LATITUDE": 91.0, "LONGITUDE": -80.0},  # E23 invalid pair
        {"LATITUDE": -14.3, "LONGITUDE": -170.7, "STATE": "AS"},  # exempt
        {"LATITUDE": np.nan, "LONGITUDE": np.nan},  # missing is not invalid
    )
    clean, log = apply_error_rules(frame)
    counts = affected(log)
    assert counts["E21_no_injuries_cap"] == 1 and clean.loc[0, "no_injuries_capped"] == 10
    assert clean.loc[0, "NO_INJURIES"] == 12  # raw column untouched
    assert counts["E22_dates"] == 2
    assert counts["E23_coordinates"] == 1
    assert np.isnan(clean.loc[3, "LATITUDE"]) and np.isnan(clean.loc[3, "longitude"])
    assert clean.loc[4, "latitude"] == -14.3 and clean.loc[4, "longitude"] == -170.7
    assert np.isnan(clean.loc[5, "latitude"])


def test_status_103i_rule_e24() -> None:
    frame = rows(
        {"CURRENT_103I": "Never Had 103I Status"},
        {"CURRENT_103I": "Inspect Once Every 5-days", "CURRENT_103I_DT": "2012-01-01"},
        {"CURRENT_103I": "Removed From 103I Status", "CURRENT_103I_DT": "2013-05-05"},
        {"CURRENT_103I": "Removed From 103I Status", "CURRENT_103I_DT": "2016-01-01"},  # removed after
        {"CURRENT_103I": "Inspect Once Every 10-days", "CURRENT_103I_DT": "2017-01-01"},  # set after
        {"CURRENT_103I": np.nan, "CURRENT_103I_DT": np.nan},  # blank
        {"CURRENT_103I": np.nan, "CURRENT_103I_DT": "2012-01-01"},  # date without a status
    )
    clean, log = apply_error_rules(frame)
    # A status dated after the accident was not knowable at filing, so a later removal does not
    # turn into "In 103I at accident" (the placement date is not in the file).
    assert clean["status_103i_at_accident"].tolist() == [
        "Never had 103I", "In 103I at accident", "Formerly 103I", "Unknown", "Unknown", "Unknown", "Unknown",
    ]
    assert affected(log)["E24_mine_attr_not_in_force"] == 4
    direct = status_103i_at_accident(frame["CURRENT_103I"], frame["CURRENT_103I_DT"], frame["ACCIDENT_DT"])
    assert direct.tolist() == clean["status_103i_at_accident"].tolist()


def test_rate_winsorising_e25() -> None:
    frame = rows(
        {"prev_hours": 1_000.0, "prev_incidents": 1.0, "prev_lost_time_cases": 0.0},  # 200 per 200k h -> capped
        {"prev_hours": np.nan, "prev_incidents": np.nan, "prev_lost_time_cases": np.nan},  # unmatched year
        {"prev_hours": 0.0, "prev_incidents": 0.0, "prev_lost_time_cases": 0.0},  # zero denominator
    )
    clean, log = apply_error_rules(frame)
    assert affected(log)["E25_unstable_rates"] == 2  # hours 1,000 and hours 0 are both below 2,000
    assert clean.loc[0, "prev_incident_rate_200k_capped"] == RATE_CAP
    assert clean.loc[0, "prev_lost_time_rate_200k_capped"] == 0.0
    assert clean.loc[1:2, ["prev_incident_rate_200k_capped", "prev_lost_time_rate_200k_capped"]].isna().all().all()


def test_flag_outliers_on_known_vector() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, np.nan], name="x")
    # Q1 = 3.25, Q3 = 7.75, IQR = 4.5: upper fence 14.5, so only 100 is flagged
    iqr = flag_outliers(values)
    assert iqr.dtype == bool and iqr.name == "x"
    assert iqr.tolist() == [False] * 9 + [True, False]
    assert flag_outliers(values, method="iqr", k=30).sum() == 0
    # mean 14.5, sample sd 30.15: z(100) = 2.84, so k = 3 flags nothing and k = 2.5 flags 100
    assert flag_outliers(values, method="zscore").sum() == 0
    assert flag_outliers(values, method="zscore", k=2.5).tolist() == [False] * 9 + [True, False]
    assert flag_outliers(pd.Series([5.0, 5.0, 5.0]), method="zscore").sum() == 0
    with pytest.raises(ValueError, match="method"):
        flag_outliers(values, method="mad")


# ---------------------------------------------------------------------------
# Real data (skipped when the extracts are absent)
# ---------------------------------------------------------------------------

# Rows affected on the final base, from docs/profile/decisions.json error_rules (E24 is the
# 22,368 rows whose 103(i) status is not demonstrably in force at the accident).
EXPECTED_ROWS_AFFECTED = {
    "E1_job_exper_gt_total": 3_731, "E2_mine_exper_gt_total": 83, "E3_exper_gt_60": 3,
    "E4_negative_exper": 0, "E5_accident_time_code": 3_091, "E6_accident_time_2400": 326,
    "E7_accident_time_other_invalid": 0, "E8_shift_time_code_or_blank": 519, "E9_shift_time_2400": 1_833,
    "E10_shift_time_short": 48, "E11_shift_wrap": 11_538, "E12_hours_into_shift_gt_16": 662,
    "E21_no_injuries_cap": 33, "E22_dates": 0, "E23_coordinates": 3,
    "E24_mine_attr_not_in_force": 22_368, "E25_unstable_rates": 764,
}


def test_real_rule_counts_match_decisions(real_base: pd.DataFrame) -> None:
    clean, log = apply_error_rules(real_base)
    assert affected(log) == EXPECTED_ROWS_AFFECTED
    # "Unknown" is exactly the E24 count: 14,277 blank or later-dated statuses plus the 8,091
    # "Removed" statuses dated after the accident, which no longer count as "In 103I at accident".
    assert clean["status_103i_at_accident"].value_counts().to_dict() == {
        "Never had 103I": 61_397, "Unknown": 22_368, "In 103I at accident": 9_344, "Formerly 103I": 8_500,
    }
    assert int(clean["exper_inconsistent"].sum()) == 3_771
    assert int(clean["ACCIDENT_TIME"].notna().sum()) == 98_518
    assert int(clean["SHIFT_BEGIN_TIME"].notna().sum()) == 101_090
    assert int(clean["hours_into_shift"].notna().sum()) == 97_752
    assert int(clean["latitude"].isna().sum()) == 199
