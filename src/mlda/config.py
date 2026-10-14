"""Project-wide constants: paths, random seed, data sources and their checksums.

Everything that the notebook, the CLI and the Streamlit app need to agree on lives
here so that there is exactly one definition of each value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STUDENT_ID = "13130"
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
CV_DIR = REPORTS_DIR / "cv"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
FIGURES_DIR = REPORTS_DIR / "figures"

MSHA_BASE_URL = "https://arlweb.msha.gov/OpenGovernmentData/DataSets/"


@dataclass(frozen=True)
class Source:
    """One downloadable MSHA file, pinned to the snapshot taken on 14 Oct 2026."""

    filename: str
    sha256: str
    size_bytes: int
    member: str | None = None  # text file inside the zip, if the source is a zip

    @property
    def url(self) -> str:
        return MSHA_BASE_URL + self.filename

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename

    @property
    def extracted_path(self) -> Path:
        return RAW_DIR / (self.member or self.filename)


# Snapshot taken 2026-10-14T12:06:01Z; MSHA refreshes these files weekly, so the
# hashes pin the exact version analysed in the report (see data/raw/SNAPSHOT.json).
SOURCES: dict[str, Source] = {
    "accidents": Source(
        "Accidents.zip",
        "62d0c861a5c3a8e76bd06de79ac60e4dcc643bca148501cb3fc752fa76c6687d",
        52_269_752,
        member="Accidents.txt",
    ),
    "mines": Source(
        "Mines.zip",
        "1a0957ff00496855289725e351797d1737faad53c48ef7bf5ac51d725e997e02",
        7_308_890,
        member="Mines.txt",
    ),
    "production": Source(
        "MinesProdYearly.zip",
        "c0fdca86d1756fc03b3f22dbc9430a02cefae9a8afac5d8e9e8503f590a3c342",
        7_072_831,
        member="MinesProdYearly.txt",
    ),
    "accidents_codebook": Source(
        "Accidents_Definition_File.txt",
        "c7681808dd372b6c5bd95fdf70b01e856e0d2501b7e5d7cc9185f4c4e1f876e5",
        10_214,
    ),
    "mines_codebook": Source(
        "Mines_Definition_File.txt",
        "efd2bc2c493207a464be7c4e4795ffd08a221588b76b716ee334a75dfc14f1f9",
        10_285,
    ),
    "production_codebook": Source(
        "MinesProdYearly_Definition_File.txt",
        "2931d666cf837770bdb12889e1f6fd13b024eed87f52c264a95b4e37c290ab65",
        2_023,
    ),
}


# ---------------------------------------------------------------------------
# Modelling decisions fixed after data profiling (docs/data_profile.md,
# docs/profile/decisions.json). Changing anything here changes the whole study.
# ---------------------------------------------------------------------------

TARGET = "lost_time"
TARGET_SOURCE_COL = "DEGREE_INJURY_CD"
# Lost-time injury: fatality, permanent disability, days away from work (with or
# without restricted activity). Negatives: restricted activity only, or no lost time.
POSITIVE_CODES = ("01", "02", "03", "04")
NEGATIVE_CODES = ("05", "06")
# Excluded: 00 accident without injury, 07 occupational illness, 08 natural causes,
# 09 non-employee, 10 all other cases including first aid.
EXCLUDED_CODES = ("00", "07", "08", "09", "10")
YEAR_MIN = 2010  # earlier years use older code lists and reporting practice
YEAR_MAX = 2025  # 2026 is partial and its degree codes are still provisional

GROUP_KEY = "group_id"  # MINE_ID + "_" + CAL_YR: rows from one mine-year never straddle train/test
ACCIDENT_GROUP_KEY = "accident_group"  # MINE_ID + "_" + ACCIDENT_DT: multi-victim accidents
SECTOR_COL = "COAL_METAL_IND"
TEST_FRACTION = 0.2
N_SPLITS = 5
TEMPORAL_SPLIT_YEAR = 2020  # secondary evaluation: train 2010-2019, test 2020-2025

# Columns that are recorded after the outcome is known, or that define the label.
LEAKAGE_COLUMNS = (
    "DEGREE_INJURY_CD", "DEGREE_INJURY", "DAYS_LOST", "DAYS_RESTRICT", "SCHEDULE_CHARGE",
    "TRANS_TERM", "RETURN_TO_WORK_DT", "CLOSED_DOC_NO", "INVEST_BEGIN_DT", "NARRATIVE",
)
# Mine attributes that describe the mine *today* rather than at the accident date, or that
# are operational details not reliably in force at the time of the accident.
MINES_CURRENT_STATE_COLUMNS = (
    "CURRENT_MINE_STATUS", "CURRENT_STATUS_DT", "CURRENT_MINE_TYPE", "COMPANY_TYPE",
    "CURRENT_CONTROLLER_BEGIN_DT", "NO_EMPLOYEES", "DAYS_PER_WEEK", "HOURS_PER_SHIFT",
    "PROD_SHIFTS_PER_DAY", "MAINT_SHIFTS_PER_DAY", "AVG_MINE_HEIGHT", "METHANE_LIBERATION",
    "MINE_GAS_CATEGORY_CD", "NO_PRODUCING_PITS", "NO_NONPRODUCING_PITS", "NO_TAILING_PONDS",
    "MILES_FROM_OFFICE",
)
# Mine columns carried into the base table (stable attributes or dated so they can be checked).
MINES_KEEP_COLUMNS = (
    "PRIMARY_CANVASS", "PRIMARY_SIC", "DISTRICT", "LATITUDE", "LONGITUDE", "PORTABLE_OPERATION",
    "SAFETY_COMMITTEE_IND", "PART48_TRAINING", "CURRENT_103I", "CURRENT_103I_DT", "STATE",
    "CURRENT_MINE_TYPE",  # EDA comparison only, never a feature
)

# Raw accident columns used directly as categorical features (descriptions replace codes
# where one description spans several code series).
ACCIDENT_CATEGORICAL = (
    "SUBUNIT_CD", "FIPS_STATE_CD", "UG_LOCATION_CD", "UG_MINING_METHOD_CD", "MINING_EQUIP_CD",
    "EQUIP_MFR_NAME", "CLASSIFICATION_CD", "ACCIDENT_TYPE_CD", "OCCUPATION", "ACTIVITY_CD",
    "INJURY_SOURCE_CD", "NATURE_INJURY_CD", "INJ_BODY_PART_CD", "COAL_METAL_IND",
)
# "?" in these columns means "not applicable" (e.g. underground location at a surface mine),
# so the placeholder is kept as its own level "none" instead of being imputed.
PLACEHOLDER_AS_LEVEL = ("UG_LOCATION_CD", "UG_MINING_METHOD_CD", "MINING_EQUIP_CD", "EQUIP_MFR_NAME", "ACTIVITY_CD")
ACCIDENT_NUMERIC_RAW = ("TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "NO_INJURIES")

ONE_HOT_MIN_FREQUENCY = 0.001  # levels rarer than 0.1 % of training rows share an "infrequent" bucket
RATE_PER_HOURS = 200_000  # MSHA convention: incidence rate per 200,000 employee-hours
RATE_CAP = 50.0  # winsorise lagged rates from tiny mines before log1p

# Business framing: cost of a missed lost-time case (FN) versus an unnecessary review (FP).
COST_FP_USD = 200
COST_FN_USD = 4_000
COST_RATIOS = (2, 3, 5, 10, 20, 30)
OPERATING_SPECIFICITY = 0.40  # capacity operating point: reviewers can handle flagging ~75 % of cases
SUCCESS_CRITERIA = {
    "roc_auc_min": 0.80,
    "pr_auc_min": 0.75,
    "recall_min_at_operating_point": 0.92,
    "recall_at_50pct_flag_rate_min": 0.78,
    "brier_max": 0.20,
    "cost_pass_ratios": (2, 3, 5),
}
