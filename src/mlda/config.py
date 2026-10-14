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
