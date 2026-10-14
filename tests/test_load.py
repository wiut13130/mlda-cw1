"""Typed loading of the MSHA extracts, exercised on 300-row fixtures committed with the repo.

The fixtures are samples of the public files. Their name columns (mine, controller and operator
names, which can be the names of private individuals) hold placeholders built from the ID columns,
so no personal or company name from the register is stored in the repository.
"""

from pathlib import Path

import pandas as pd
import pytest

from mlda.data.load import (
    ACCIDENT_DATES,
    ACCIDENT_NUMERIC,
    PLACEHOLDERS,
    load_accidents,
    load_mines,
    load_production,
    read_msha_table,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def accidents() -> pd.DataFrame:
    return load_accidents(FIXTURES / "accidents_sample.txt")


def test_accidents_shape_and_columns(accidents: pd.DataFrame) -> None:
    assert accidents.shape == (300, 57)
    assert accidents.columns[0] == "MINE_ID" and accidents.columns[-1] == "COAL_METAL_IND"


def test_placeholders_become_missing(accidents: pd.DataFrame) -> None:
    for column in ["UG_LOCATION_CD", "UG_LOCATION", "IMMED_NOTIFY_CD"]:
        assert not accidents[column].dropna().isin(PLACEHOLDERS).any()
    # the first fixture row has an unknown underground location, so NA must appear
    assert accidents["UG_LOCATION_CD"].isna().any()


def test_numeric_and_date_conversion(accidents: pd.DataFrame) -> None:
    for column in ACCIDENT_NUMERIC:
        assert pd.api.types.is_numeric_dtype(accidents[column]), column
    for column in ACCIDENT_DATES:
        assert pd.api.types.is_datetime64_any_dtype(accidents[column]), column
    assert accidents["CAL_YR"].between(2000, 2026).all()
    assert (accidents["ACCIDENT_DT"].dt.year == accidents["CAL_YR"]).all()


def test_mine_id_keeps_leading_zeros(accidents: pd.DataFrame) -> None:
    assert accidents["MINE_ID"].str.len().eq(7).all()
    assert accidents["MINE_ID"].iloc[0] == "0100003"


def test_times_stay_as_hhmm_strings(accidents: pd.DataFrame) -> None:
    assert accidents["ACCIDENT_TIME"].dropna().str.fullmatch(r"\d{4}").all()


def test_mines_fixture() -> None:
    mines = load_mines(FIXTURES / "mines_sample.txt")
    assert mines.shape == (300, 59)
    assert mines["MINE_ID"].is_unique
    assert pd.api.types.is_numeric_dtype(mines["NO_EMPLOYEES"])
    assert pd.api.types.is_float_dtype(mines["LATITUDE"])
    assert mines["LATITUDE"].dropna().between(-90, 90).all()


def test_production_fixture() -> None:
    production = load_production(FIXTURES / "production_sample.txt")
    assert production.shape == (300, 11)
    assert production["CALENDAR_YR"].dtype == "Int64"
    assert production["ANNUAL_HRS"].ge(0).all()
    assert not production.duplicated(["MINE_ID", "CALENDAR_YR", "SUBUNIT_CD"]).any()


def test_usecols_restricts_columns() -> None:
    frame = read_msha_table(FIXTURES / "accidents_sample.txt", usecols=["MINE_ID", "CAL_YR"])
    assert list(frame.columns) == ["MINE_ID", "CAL_YR"]
