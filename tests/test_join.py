"""Base-table contract (docs/interfaces.md section 1) on hand-built frames, plus the real-data counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import accidents_frame, mines_frame, production_frame

from mlda.config import POSITIVE_CODES, YEAR_MAX, YEAR_MIN
from mlda.data.join import (
    build_base_table,
    case_history_table,
    filing_date,
    persistable,
    previous_year_incomplete,
    type_from_subunit_code,
    type_from_subunits,
)
from mlda.data.schema import (
    EXPECTED_BASE_COLUMNS,
    EXPECTED_N,
    EXPECTED_POSITIVES,
    SchemaError,
    validate_base,
)

VALID = ("01", "02", "03", "04", "05", "06")


def _mines(*mine_ids: str) -> pd.DataFrame:
    return mines_frame([{"MINE_ID": m} for m in mine_ids])


def _production(mine_id: str, year: int, subunits: dict[str, tuple[int, int]]) -> list[dict[str, object]]:
    """Production rows for one mine-year: subunit -> (hours, employees)."""
    return [
        {"MINE_ID": mine_id, "CALENDAR_YR": year, "SUBUNIT_CD": s, "ANNUAL_HRS": h, "AVG_ANNUAL_EMPL": e}
        for s, (h, e) in subunits.items()
    ]


# ---------------------------------------------------------------------------
# Synthetic base
# ---------------------------------------------------------------------------


def test_synthetic_base_has_contract_columns_and_400_rows(synthetic_base_frame: pd.DataFrame) -> None:
    assert len(synthetic_base_frame) == 400
    assert set(EXPECTED_BASE_COLUMNS) <= set(synthetic_base_frame.columns)
    validate_base(synthetic_base_frame, check_counts=False)
    # realistic structure: missing values, multi-victim groups, some unmatched previous years
    assert synthetic_base_frame["TOT_EXPER"].isna().any()
    assert synthetic_base_frame["accident_group"].duplicated().any()
    assert 0 < synthetic_base_frame["prev_year_matched"].mean() < 1


def test_row_filters_keep_only_window_and_valid_codes(synthetic_sources_frames) -> None:
    accidents, mines, production = synthetic_sources_frames
    expected = accidents[accidents["CAL_YR"].between(YEAR_MIN, YEAR_MAX) & accidents["DEGREE_INJURY_CD"].isin(VALID)]
    assert len(accidents) > len(expected) == 400
    base = build_base_table(accidents, mines, production)
    assert len(base) == len(expected)
    assert set(base["DOCUMENT_NO"]) == set(expected["DOCUMENT_NO"])
    steps = pd.DataFrame(base.attrs["row_filter_log"])
    assert steps["step"].iloc[0] == "all rows" and steps["kept"].iloc[0] == len(accidents)
    assert steps["removed"].sum() == len(accidents) - len(base)
    assert steps["kept"].iloc[-1] == len(base)


def test_joins_preserve_row_count_and_keep_every_raw_column(synthetic_sources_frames) -> None:
    accidents, mines, production = synthetic_sources_frames
    base = build_base_table(accidents, mines, production)
    assert (base["_m_mines"] == "both").all()
    assert set(accidents.columns) <= set(base.columns)
    assert base.attrs["coal_metal_ind_agreement"] == 1.0
    assert (base["mine_coal_metal_ind"] == base["COAL_METAL_IND"]).all()


def test_duplicate_mine_id_raises(synthetic_sources_frames) -> None:
    accidents, mines, production = synthetic_sources_frames
    duplicated = pd.concat([mines, mines.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        build_base_table(accidents, duplicated, production)


def test_missing_mine_record_raises(synthetic_sources_frames) -> None:
    accidents, mines, production = synthetic_sources_frames
    with pytest.raises(ValueError, match="no record in Mines"):
        build_base_table(accidents, mines.iloc[1:], production)


def test_target_and_group_keys(synthetic_base_frame: pd.DataFrame) -> None:
    base = synthetic_base_frame
    assert base["lost_time"].dtype == np.int8
    assert base["lost_time"].eq(base["DEGREE_INJURY_CD"].isin(POSITIVE_CODES).astype("int8")).all()
    assert base["group_id"].eq(base["MINE_ID"] + "_" + base["CAL_YR"].astype(str)).all()
    assert base["accident_group"].str.fullmatch(r"\d{7}_\d{4}-\d{2}-\d{2}").all()
    # every same-day group lies inside one mine-year
    assert base.groupby("accident_group")["group_id"].nunique().eq(1).all()


def test_filing_date_parses_julian_and_rejects_garbage() -> None:
    parsed = filing_date(pd.Series(["220152230015", "2XXXXXXX0001", None, "220100010001"]))
    assert parsed.iloc[0] == pd.Timestamp("2015-08-11")
    assert parsed.iloc[1:3].isna().all()
    assert parsed.iloc[3] == pd.Timestamp("2010-01-01")


# ---------------------------------------------------------------------------
# Lagged case history
# ---------------------------------------------------------------------------


def test_lagged_counts_use_only_cases_filed_before_the_accident() -> None:
    mine, other = "0100001", "0100002"
    accidents = accidents_frame(
        [
            {"MINE_ID": mine, "ACCIDENT_DT": "2015-06-15", "DEGREE_INJURY_CD": "03"},  # the scored row
            # t-1 cases that must count: filed before 2015-06-15
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-03-01", "DEGREE_INJURY_CD": "03", "filing_date": "2014-03-10"},
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-11-20", "DEGREE_INJURY_CD": "06", "filing_date": "2015-01-05"},
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-12-28", "DEGREE_INJURY_CD": "00", "filing_date": "2015-01-08"},
            # t-1 cases that must not count
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-12-30", "DEGREE_INJURY_CD": "03", "filing_date": "2015-06-15"},  # same day
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-12-31", "DEGREE_INJURY_CD": "04", "filing_date": "2015-09-01"},  # after
            {"MINE_ID": mine, "ACCIDENT_DT": "2014-07-01", "DEGREE_INJURY_CD": "03", "DOCUMENT_NO": "2XXXXXXX0001"},
            {"MINE_ID": mine, "ACCIDENT_DT": "2013-05-05", "DEGREE_INJURY_CD": "03", "filing_date": "2013-05-10"},  # t-2
            {"MINE_ID": other, "ACCIDENT_DT": "2014-05-05", "DEGREE_INJURY_CD": "03", "filing_date": "2014-05-10"},
        ]
    )
    production = production_frame(
        _production(mine, 2014, {"01": (100_000, 50)}) + _production(mine, 2013, {"01": (90_000, 45)})
        + _production(mine, 2012, {"01": (90_000, 45)}) + _production(other, 2013, {"03": (10_000, 5)})
    )
    base = build_base_table(accidents, _mines(mine, other), production).set_index("DOCUMENT_NO")
    scored = base.loc[accidents.loc[0, "DOCUMENT_NO"]]
    assert scored["prev_year_matched"]
    assert scored["prev_incidents"] == 3
    assert scored["prev_valid_cases"] == 2
    assert scored["prev_lost_time_cases"] == 1


def _closing_doc(date: str) -> str:
    """CLOSED_DOC_NO as MSHA writes it: type 3, Julian YYYYDDD, sequence."""
    return f"3{pd.Timestamp(date):%Y%j}0001"


def test_resolved_before_variant_drops_cases_still_open_at_the_accident() -> None:
    mine = "0100001"
    base_case = {"MINE_ID": mine, "DEGREE_INJURY_CD": "03", "ACCIDENT_DT": "2014-03-01", "filing_date": "2014-03-10"}
    accidents = accidents_frame(
        [
            {"MINE_ID": mine, "ACCIDENT_DT": "2015-06-15", "DEGREE_INJURY_CD": "03"},  # the scored row
            {**base_case, "RETURN_TO_WORK_DT": "2014-04-01", "CLOSED_DOC_NO": _closing_doc("2014-04-10")},  # resolved
            {**base_case, "RETURN_TO_WORK_DT": "2015-08-01"},  # worker back only after the accident
            {**base_case, "RETURN_TO_WORK_DT": "2014-04-01", "CLOSED_DOC_NO": _closing_doc("2015-07-01")},  # closed later
            dict(base_case),  # no dates at all: counted, as the contract counts the case by its filing date
            {**base_case, "filing_date": "2015-09-01"},  # filed after the accident: never counted
        ]
    )
    production = production_frame(_production(mine, 2014, {"01": (100_000, 50)}))
    base = build_base_table(accidents, _mines(mine), production).set_index("DOCUMENT_NO")
    scored = base.loc[accidents.loc[0, "DOCUMENT_NO"]]
    assert (scored["prev_incidents"], scored["prev_lost_time_cases"]) == (4, 4)
    assert (scored["prev_incidents_resolved"], scored["prev_valid_cases_resolved"]) == (2, 2)
    assert scored["prev_lost_time_cases_resolved"] == 2
    history = case_history_table(accidents)
    assert {"return_to_work_dt", "closed_date"} <= set(history.columns)
    assert history["closed_date"].notna().sum() == 2
    validate_base(base.reset_index(), check_counts=False)


def test_previous_year_incomplete_flag_marks_forms_filed_by_15_january() -> None:
    filing = pd.Series(pd.to_datetime(["2015-01-02", "2015-01-15", "2015-01-16", "2015-06-01", None]))
    year = pd.Series([2015] * 5)
    assert previous_year_incomplete(filing, year).tolist() == [True, True, False, False, False]
    mine = "0100001"
    accidents = accidents_frame(
        [
            {"MINE_ID": mine, "ACCIDENT_DT": "2015-01-05", "filing_date": "2015-01-12"},
            {"MINE_ID": mine, "ACCIDENT_DT": "2015-01-05", "filing_date": "2015-02-20"},
        ]
    )
    production = production_frame(_production(mine, 2014, {"01": (100_000, 50)}))
    base = build_base_table(accidents, _mines(mine), production)
    assert base["prev_year_incomplete_at_filing"].tolist() == [True, False]


def test_lagged_counts_zero_when_matched_without_cases_and_nan_when_unmatched() -> None:
    quiet, unmatched = "0100001", "0100002"
    accidents = accidents_frame(
        [
            {"MINE_ID": quiet, "ACCIDENT_DT": "2015-06-15"},
            {"MINE_ID": unmatched, "ACCIDENT_DT": "2015-06-15"},
            # a 2014 case at the unmatched mine: ignored because there is no 2014 production
            {"MINE_ID": unmatched, "ACCIDENT_DT": "2014-02-02", "filing_date": "2014-02-09"},
        ]
    )
    production = production_frame(_production(quiet, 2014, {"03": (10_000, 5)}) + _production(unmatched, 2013, {"03": (10_000, 5)}))
    base = build_base_table(accidents, _mines(quiet, unmatched), production)
    counts = ["prev_incidents", "prev_valid_cases", "prev_lost_time_cases"]
    quiet_row = base[base["MINE_ID"] == quiet].iloc[0]
    assert quiet_row["prev_year_matched"] and (quiet_row[counts] == 0).all()
    row_2015 = base[(base["MINE_ID"] == unmatched) & (base["CAL_YR"] == 2015)].iloc[0]
    assert not row_2015["prev_year_matched"]
    assert row_2015[[*counts, "prev_hours", "prev_employees"]].isna().all()
    row_2014 = base[(base["MINE_ID"] == unmatched) & (base["CAL_YR"] == 2014)].iloc[0]
    assert row_2014["prev_year_matched"] and (row_2014[counts] == 0).all()


# ---------------------------------------------------------------------------
# Production aggregates and type_at_time
# ---------------------------------------------------------------------------


def test_previous_and_same_year_aggregates() -> None:
    mine = "0100001"
    accidents = accidents_frame([{"MINE_ID": mine, "ACCIDENT_DT": "2015-06-15"}])
    production = production_frame(
        _production(mine, 2014, {"01": (100_000, 50), "03": (20_000, 10)})
        + _production(mine, 2015, {"01": (80_000, 40), "30": (5_000, 3)})
    )
    row = build_base_table(accidents, _mines(mine), production).iloc[0]
    assert row["prev_hours"] == 120_000 and row["prev_employees"] == 60
    assert row["prev_subunits"] == "01|03"
    assert row["same_year_hours"] == 85_000 and row["same_year_employees"] == 43
    assert row["_m_prev"] == "both" and row["_m_same"] == "both"


def test_type_at_time_rules() -> None:
    ids = [f"010000{i}" for i in range(1, 8)]
    accidents = accidents_frame(
        [
            {"MINE_ID": ids[0], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "03"},  # {01,03} -> Underground
            {"MINE_ID": ids[1], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "03"},  # {30,99} -> Facility
            {"MINE_ID": ids[2], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "01"},  # {17} -> Facility
            {"MINE_ID": ids[3], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "01"},  # {03,30} -> Surface
            {"MINE_ID": ids[4], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "02"},  # no t-1 production -> fallback
            {"MINE_ID": ids[5], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "99"},
            {"MINE_ID": ids[6], "ACCIDENT_DT": "2015-06-15", "SUBUNIT_CD": "03"},
        ]
    )
    production = production_frame(
        _production(ids[0], 2014, {"01": (1, 1), "03": (1, 1)})
        + _production(ids[1], 2014, {"30": (1, 1), "99": (1, 1)})
        + _production(ids[2], 2014, {"17": (1, 1)})
        + _production(ids[3], 2014, {"03": (1, 1), "30": (1, 1)})
        + _production(ids[4], 2015, {"01": (1, 1)})  # same year only: not usable at filing
    )
    base = build_base_table(accidents, _mines(*ids), production).set_index("MINE_ID")
    assert base.loc[ids, "type_at_time"].tolist() == [
        "Underground", "Facility", "Facility", "Surface", "Underground", "Facility", "Surface",
    ]
    assert base.loc[ids, "prev_year_matched"].tolist() == [True, True, True, True, False, False, False]


def test_type_rules_as_pure_functions() -> None:
    assert type_from_subunits({"01", "03", "30"}) == "Underground"
    assert type_from_subunits({"02"}) == "Underground"
    assert type_from_subunits({"30", "99", "17"}) == "Facility"
    assert type_from_subunits({"03", "30"}) == "Surface"
    assert type_from_subunits({"06"}) == "Surface"
    assert [type_from_subunit_code(c) for c in ("01", "02", "30", "99", "17", "03", np.nan)] == [
        "Underground", "Underground", "Facility", "Facility", "Facility", "Surface", "Surface",
    ]


# ---------------------------------------------------------------------------
# Schema and persistence
# ---------------------------------------------------------------------------


def test_validate_base_detects_broken_frames(synthetic_base_frame: pd.DataFrame) -> None:
    base = synthetic_base_frame
    with pytest.raises(SchemaError, match="N ="):
        validate_base(base)  # count check against the profiled 101,609 rows
    broken = base.copy()
    broken.loc[0, "lost_time"] = np.nan
    with pytest.raises(SchemaError, match="lost_time contains NaN"):
        validate_base(broken, check_counts=False)
    broken = base.copy()
    broken.loc[0, "MINE_ID"] = "123"
    with pytest.raises(SchemaError, match="7-character"):
        validate_base(broken, check_counts=False)
    broken = base.copy()
    broken.loc[0, "CAL_YR"] = YEAR_MIN - 1
    with pytest.raises(SchemaError, match="CAL_YR outside"):
        validate_base(broken, check_counts=False)
    with pytest.raises(SchemaError, match="missing columns"):
        validate_base(base.drop(columns=["type_at_time"]), check_counts=False)
    broken = base.copy()
    broken.loc[broken["prev_year_matched"], "prev_incidents"] = np.nan
    with pytest.raises(SchemaError, match="prev_incidents must be NaN exactly when"):
        validate_base(broken, check_counts=False)
    broken = base.copy()
    row = broken.index[broken["prev_year_matched"]][0]
    broken.loc[row, "prev_incidents_resolved"] = broken.loc[row, "prev_incidents"] + 1
    with pytest.raises(SchemaError, match="prev_incidents_resolved must not exceed prev_incidents"):
        validate_base(broken, check_counts=False)


def test_persistable_drops_text_and_names_but_keeps_equipment_name(synthetic_base_frame: pd.DataFrame) -> None:
    slim = persistable(synthetic_base_frame)
    for column in ("NARRATIVE", "CONTROLLER_NAME", "OPERATOR_NAME"):
        assert column not in slim.columns
    assert "EQUIP_MFR_NAME" in slim.columns
    validate_base(slim, check_counts=False)


def test_parquet_round_trip_keeps_dtypes(tmp_path, synthetic_base_frame: pd.DataFrame) -> None:
    path = tmp_path / "base.parquet"
    slim = persistable(synthetic_base_frame)
    slim.to_parquet(path, compression="zstd", index=False)
    back = pd.read_parquet(path)
    validate_base(back, check_counts=False)
    assert back["CAL_YR"].dtype == "Int64"
    assert back["lost_time"].dtype == np.int8
    assert back["prev_year_matched"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(back["filing_date"])
    pd.testing.assert_frame_equal(back, slim.reset_index(drop=True), check_dtype=True)


# ---------------------------------------------------------------------------
# Real data (skipped when the extracts are absent)
# ---------------------------------------------------------------------------


def test_real_base_matches_data_profile(real_base: pd.DataFrame) -> None:
    validate_base(real_base)
    assert len(real_base) == EXPECTED_N == 101_609
    assert int(real_base["lost_time"].sum()) == EXPECTED_POSITIVES == 49_367
    assert int(real_base["prev_year_matched"].sum()) == 100_403
    assert real_base["type_at_time"].value_counts().to_dict() == {
        "Surface": 51_053, "Underground": 40_379, "Facility": 10_177,
    }
    assert int(real_base["prev_incidents"].notna().sum()) == 100_403
    assert real_base.attrs["coal_metal_ind_agreement"] == 1.0
    assert int(real_base["filing_date"].isna().sum()) == 8
    assert real_base["group_id"].nunique() == 32_732
    assert real_base["accident_group"].nunique() == 98_693


def test_real_base_sensitivity_columns_and_lag_counts(real_base: pd.DataFrame) -> None:
    """Counts the filing-date rule produces (strict <): 17,822 true zeros, not the 17,820 of a <= cut."""
    assert int((real_base["prev_incidents"] == 0).sum()) == 17_822
    assert int(real_base["prev_year_incomplete_at_filing"].sum()) == 1_643
    matched = real_base["prev_year_matched"]
    for column in ("prev_incidents", "prev_valid_cases", "prev_lost_time_cases"):
        assert (real_base.loc[matched, f"{column}_resolved"] <= real_base.loc[matched, column]).all()
        assert real_base.loc[matched, f"{column}_resolved"].sum() < real_base.loc[matched, column].sum()
    share = real_base["prev_lost_time_cases"] / real_base["prev_valid_cases"].where(real_base["prev_valid_cases"] > 0)
    assert int(share.notna().sum()) == 81_469  # 20,140 rows (19.82 %) have no valid case filed before the accident
