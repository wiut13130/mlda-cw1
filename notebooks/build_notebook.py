"""Generate notebooks/01_end_to_end.ipynb from the cell definitions below.

The notebook is a graded artefact, so it is written as code: every markdown justification and
every code cell lives in this file, which makes the notebook reproducible and reviewable in a
diff. The generator only assembles cells; nothing here fits a model. Execute the result with

    uv run python notebooks/build_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb

The default run repeats the hyper-parameter search live for the three inexpensive models
(Gaussian naive Bayes, logistic regression, decision tree), loads the cross-validation tables
that ``uv run mlda-train`` wrote for the other five, and refits every selected pipeline (several
minutes). ``FULL_RUN=1`` repeats all eight searches inside the notebook (about an hour).
``NOTEBOOK_MODELS=logreg,hist_gb`` restricts the zoo to a subset for quick development checks;
leave it unset for the graded run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from mlda.config import PROJECT_ROOT, RAW_DIR, STUDENT_ID

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_end_to_end.ipynb"
SECTIONS = ("A", "B", "C", "D", "E", "F")

# Cells are collected here in reading order as ("markdown" | "code", source) pairs.
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Append a markdown cell (leading and trailing blank lines are stripped)."""
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell (leading and trailing blank lines are stripped)."""
    CELLS.append(("code", text.strip("\n")))


def snapshot_text() -> tuple[str, str]:
    """Snapshot timestamp from data/raw/SNAPSHOT.json as (ISO string, readable date and time)."""
    with (RAW_DIR / "SNAPSHOT.json").open(encoding="utf-8") as handle:
        stamp = json.load(handle)["snapshot_utc"]
    year, month, rest = stamp.split("-", 2)
    day, clock = rest.split("T")
    months = [
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    ]
    readable = f"{int(day)} {months[int(month) - 1]} {year}, {clock[:5]} UTC"
    return stamp, readable


# ---------------------------------------------------------------------------
# Title and environment
# ---------------------------------------------------------------------------


def title_cells() -> None:
    """Title cell (identity, snapshot date, how to run) and the environment cell that prints the versions.

    They come first so that a reader sees what was run, on which data, and how to reproduce it
    before any result appears.
    """
    stamp, readable = snapshot_text()
    md(
        f"""
# Predicting lost-time mine injuries from MSHA open data

| | |
|---|---|
| **Module** | 6COSC017C Machine Learning and Data Analytics, Coursework 1 (CW1), WIUT |
| **Student ID** | {STUDENT_ID} |
| **Data snapshot** | MSHA Open Government Data, pulled {readable} (`{stamp}` in `data/raw/SNAPSHOT.json`); the three data archives were published by MSHA on 9 October 2026 |
| **Task** | Binary classification: for each injured person on an MSHA Form 7000-1 (accident years 2010-2025, N = 101,609) predict whether the case is a *lost-time injury* (`lost_time` = 1, 48.6 % of cases) or not |
| **Models** | Logistic regression, k-nearest neighbours, Gaussian naive Bayes, decision tree, random forest, histogram gradient boosting, RBF support vector machine, multi-layer perceptron, compared against four reference rules |
| **Features** | 52 model inputs: 20 categorical, 20 numeric, 12 binary |

**Contents.** [A. Load dataset](#A.-Load-dataset) | [B. Exploratory data analysis](#B.-Exploratory-data-analysis) |
[C. Data preparation](#C.-Data-preparation) | [D. Model training](#D.-Model-training) |
[E. Model evaluation](#E.-Model-evaluation) | [F. Deployment and reproducibility](#F.-Deployment-and-reproducibility)

**Reading guide.** The notebook follows the structure of the brief: **A** loads and joins the data,
**B** explores the training rows, **C** prepares the data, **D** trains and tunes the models,
**E** evaluates them on the held-out test fold and **F** describes deployment and reproducibility.
Every group of code cells is preceded by a markdown cell that says why the step is needed and
what the output shows. Numbers that depend on the model runs are printed by the code and are not
typed into the text.

**How to run.**

```
uv sync --locked --all-groups
uv run mlda-download      # fetch the raw MSHA files and verify their SHA-256 hashes
uv run mlda-train         # the full study, with the temporal and sensitivity studies (about an hour and a half on 8 cores)
uv run python notebooks/build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb
```

**The `FULL_RUN` switch.** By default (variable unset) the notebook runs the hyper-parameter
search itself, with `mlda.tuning.tune`, for the three models whose search is cheap (Gaussian
naive Bayes, logistic regression and the decision tree) and checks that it reproduces the tables
stored by `uv run mlda-train` within 0.001 of each mean score. For the other five models (k-nearest neighbours, random forest,
histogram gradient boosting, support vector machine and perceptron) it loads the stored
cross-validation tables and selected hyper-parameters. It then refits every selected pipeline on
the training split, so training happens here, and the whole notebook runs in under ten minutes on eight workers (section F.6 lists the stage times of the executed run). With
`FULL_RUN=1` (PowerShell: `$env:FULL_RUN = "1"`) the notebook repeats all eight grid and
randomised searches itself, which takes about an hour.
"""
    )
    code(
        r"""
%matplotlib inline
import gc
import io
import json
import logging
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

# Quiet the library chatter that is not about the data: sklearn and pandas FutureWarnings.
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning"  # worker processes too

# The filters have to be active before the libraries are imported, so these imports follow them (E402).
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
import statsmodels  # noqa: E402
from IPython.display import Image, Markdown, display  # noqa: E402

import mlda  # noqa: E402
from mlda.config import PROJECT_ROOT, RANDOM_STATE, STUDENT_ID  # noqa: E402
from mlda.pipeline import StageTimer  # noqa: E402

matplotlib.rcParams["figure.dpi"] = 120      # sharper inline figures than the 72 dpi default
pd.set_option("display.float_format", "{:,.3f}".format)   # every table rounded to 3 decimals
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)

FULL_RUN = os.environ.get("FULL_RUN") == "1"
timer = StageTimer()          # wall-clock seconds per stage, summarised in the last cell
NOTEBOOK_STARTED = time.perf_counter()


def show(fig, alt: str = "Figure") -> None:
    # Figures from mlda.plots are matplotlib Figure objects. They are rendered to PNG with a text
    # description, so that the HTML export carries alternative text for every image.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=fig.dpi)
    display(Image(data=buffer.getvalue(), format="png", alt=alt))


def note(text: str) -> None:
    display(Markdown(text))


def show_table(frame: pd.DataFrame, caption: str | None = None) -> None:
    if caption:
        note(f"**{caption}**")
    display(frame)


with open(PROJECT_ROOT / "data" / "raw" / "SNAPSHOT.json", encoding="utf-8") as handle:
    SNAPSHOT_UTC = json.load(handle)["snapshot_utc"]

environment = pd.DataFrame(
    {
        "value": {
            "Python": platform.python_version(),
            "platform": platform.platform(),
            "CPU cores": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "mlda (this project)": getattr(mlda, "__version__", "0.1.0"),
            "RANDOM_STATE": RANDOM_STATE,
            "STUDENT_ID": STUDENT_ID,
            "FULL_RUN": FULL_RUN,
            "data snapshot (UTC)": SNAPSHOT_UTC,
            "project root": str(PROJECT_ROOT),
        }
    }
)
environment["value"] = environment["value"].astype(str)
display(environment)
"""
    )


# ---------------------------------------------------------------------------
# A. Load dataset
# ---------------------------------------------------------------------------


def section_a() -> None:
    """Section A: verify the raw files, load the three tables, filter, define the target, build and validate the base table."""
    md(
        """
# A. Load dataset

The study uses three open-data tables from the US Mine Safety and Health Administration (MSHA).
They form a hierarchy: **Mines** holds one row per mine, **MinesProdYearly** holds the employee
hours of each mine, year and subunit (Form 7000-2), and **Accidents** holds one row per injured
person on a Form 7000-1. The unit of analysis is the injured person, so *Accidents* is the left
table and each of the other two is attached with a **left join**, which cannot drop an accident.
Every join has to be many-to-one (many accidents per mine, many accidents per mine-year), and the
cells below check that, count the matches and record what happens to the rows that do not match.
"""
    )
    md(
        """
## A.1 Raw files and checksums

MSHA replaces these files every week, so a rerun next month would silently analyse different
rows. `ensure_raw_files()` recomputes the SHA-256 hash of every file and compares it with the
hash pinned in `mlda.config.SOURCES`; it downloads a file only if it is missing, and refuses one
whose hash differs. The table shows the verification result for the snapshot used in this work.
"""
    )
    code(
        r"""
from mlda.config import RAW_DIR, SOURCES
from mlda.data.download import ensure_raw_files, sha256_of

raw_paths = ensure_raw_files()     # verifies every SHA-256 against the pinned value; downloads only if needed
rows = []
for key, source in SOURCES.items():
    actual = sha256_of(source.path)
    rows.append(
        {
            "source": key,
            "file": source.filename,
            "size (MB)": round(source.path.stat().st_size / 1e6, 2),
            "sha256 (first 12)": actual[:12],
            "matches pinned hash": actual == source.sha256,
        }
    )
checksums = pd.DataFrame(rows)
show_table(checksums, "Raw files verified against the pinned SHA-256 hashes")
assert checksums["matches pinned hash"].all(), "a raw file differs from the analysed snapshot"

with open(RAW_DIR / "SNAPSHOT.json", encoding="utf-8") as handle:
    snapshot = json.load(handle)
print("snapshot taken (UTC):", snapshot["snapshot_utc"])
for name in ("Accidents.zip", "Mines.zip", "MinesProdYearly.zip"):
    print(f"  {name:22s} published by MSHA: {snapshot[name]['last_modified']}")
"""
    )
    md(
        """
## A.2 Load the three tables

`mlda.data.load` reads each pipe-delimited file as text first (so that nothing is coerced
silently), strips padding, turns the MSHA placeholders (`?`, `NO VALUE FOUND`, blanks) into real
missing values and then converts the columns that the codebook declares as numbers or dates. The
cells print the shape and the data types of each table and the first rows, which is the first
check that the parsing did what the codebook says.
"""
    )
    code(
        r"""
from mlda.data import load

with timer.stage("A.2 load the three tables"):
    accidents = load.load_accidents()
    mines = load.load_mines()
    production = load.load_production()

tables = {
    "Accidents": (accidents, "DOCUMENT_NO", "one injured person on one Form 7000-1"),
    "Mines": (mines, "MINE_ID", "one mine (attributes as of the snapshot)"),
    "MinesProdYearly": (production, "MINE_ID + CALENDAR_YR + SUBUNIT_CD", "one mine, year and subunit"),
}
overview = pd.DataFrame(
    [
        {
            "table": name,
            "rows": len(frame),
            "columns": frame.shape[1],
            "primary key": key,
            "grain": grain,
            "memory (MB)": round(frame.memory_usage(deep=True).sum() / 1e6),
        }
        for name, (frame, key, grain) in tables.items()
    ]
).set_index("table")
show_table(overview, "Shape of the three tables")

dtype_counts = pd.concat(
    {name: frame.dtypes.astype(str).value_counts() for name, (frame, _, _) in tables.items()}, axis=1
).fillna(0).astype(int)
show_table(dtype_counts, "Columns per data type (object = text)")
"""
    )
    code(
        r"""
from mlda.data.schema import PERSIST_DROP_COLUMNS

# Company names and free text (narratives, directions) are left out of the displays; they are
# not features and are not stored in the processed data either. Transposed so that the columns read down the page.
accident_view = accidents.drop(columns=[c for c in PERSIST_DROP_COLUMNS if c in accidents.columns])
show_table(accident_view.head(3).T, "Accidents.head(3), transposed (names and narrative omitted)")
mine_columns = [
    "MINE_ID", "COAL_METAL_IND", "CURRENT_MINE_TYPE", "CURRENT_MINE_STATUS", "STATE", "DISTRICT", "PRIMARY_SIC",
    "PRIMARY_CANVASS", "CURRENT_103I", "CURRENT_103I_DT", "PORTABLE_OPERATION", "SAFETY_COMMITTEE_IND",
    "PART48_TRAINING", "LATITUDE", "LONGITUDE",
]
show_table(mines[mine_columns].head(3), "Mines.head(3), selected columns (59 in the file)")
show_table(production.head(), "MinesProdYearly.head()")
"""
    )
    md(
        """
## A.3 Keys and the many-to-one rule

A join is many-to-one only if the key is unique on the "one" side: `MINE_ID` in *Mines* and
(`MINE_ID`, `CALENDAR_YR`) after the production rows are summed over subunits. If the "one" side
held duplicates, a left join would silently multiply accident rows and inflate the sample. The
table checks the keys, and the second cell shows that `validate="many_to_one"` makes pandas
raise instead of duplicating rows.
"""
    )
    code(
        r"""
key_checks = pd.DataFrame(
    [
        {"table": "Accidents", "key": "DOCUMENT_NO", "unique": bool(accidents["DOCUMENT_NO"].is_unique)},
        {"table": "Mines", "key": "MINE_ID", "unique": bool(mines["MINE_ID"].is_unique)},
        {
            "table": "MinesProdYearly",
            "key": "MINE_ID + CALENDAR_YR + SUBUNIT_CD",
            "unique": not production.duplicated(["MINE_ID", "CALENDAR_YR", "SUBUNIT_CD"]).any(),
        },
        {
            "table": "Accidents and Mines",
            "key": "MINE_ID is a 7-character string in both",
            "unique": bool(accidents["MINE_ID"].str.len().eq(7).all() and mines["MINE_ID"].str.len().eq(7).all()),
        },
    ]
)
show_table(key_checks, "Key checks before joining")
assert key_checks["unique"].all()

# What validate="many_to_one" protects against: a right table with a duplicated key.
duplicated_right = pd.concat([mines[["MINE_ID", "STATE"]].head(1)] * 2, ignore_index=True)
try:
    accidents.head(3).merge(duplicated_right, on="MINE_ID", how="left", validate="many_to_one")
except pd.errors.MergeError as error:
    print("pandas refused the join:", error)
"""
    )
    md(
        """
## A.4 Row filters and the target

Each row filter is applied on its own so that the number of rows it removes can be reported
(the report reproduces this table). The window is accident years **2010-2025**: earlier years use
older code lists, and 2026 is incomplete and its degree codes are still provisional. Degree codes
00 (accident without injury), 07 (occupational illness), 08 (natural causes), 09 (non-employee)
and 10 (first aid and other) are not injuries to an employee that the model should rank, so they
are removed. What remains are the six codes 01-06.
"""
    )
    code(
        r"""
from mlda.config import (
    NEGATIVE_CODES,
    POSITIVE_CODES,
    TARGET,
    TARGET_SOURCE_COL,
    YEAR_MAX,
    YEAR_MIN,
)
from mlda.data import join
from mlda.data.schema import EXPECTED_N

kept, filter_log = join.filter_rows(accidents)
show_table(filter_log, "Row-filter sequence (rows removed and rows kept after each rule)")
positives = int(kept[TARGET_SOURCE_COL].isin(POSITIVE_CODES).sum())
print(f"N = {len(kept):,} rows, {positives:,} lost-time cases, prevalence {positives / len(kept):.1%}")
assert len(kept) == EXPECTED_N
"""
    )
    md(
        """
**Target definition.** `lost_time` = 1 when MSHA classifies the injury as a fatality, a
permanent disability, or a case with days away from work (codes 01-04); it is 0 for restricted
activity only and for no lost time (codes 05-06). Days away from work is the trigger for wage
replacement in workers' compensation, which is why the line is drawn between 04 and 05. The two
classes are almost balanced (48.6 % positive), so accuracy, ROC-AUC and PR-AUC are all
informative and no resampling is needed; the cost of a missed case enters later through the
decision threshold, not through the label.
"""
    )
    code(
        r"""
window = accidents[accidents["CAL_YR"].between(YEAR_MIN, YEAR_MAX)]
codes = (
    window.groupby(TARGET_SOURCE_COL, dropna=False)
    .agg(codebook_label=("DEGREE_INJURY", "first"), cases=("DOCUMENT_NO", "size"))
    .reset_index()
)
role = np.select(
    [codes[TARGET_SOURCE_COL].isin(POSITIVE_CODES), codes[TARGET_SOURCE_COL].isin(NEGATIVE_CODES)],
    [f"{TARGET} = 1", f"{TARGET} = 0"],
    default="excluded (not an injury to rank)",
)
codes["role in this study"] = role
codes["share of window (%)"] = 100 * codes["cases"] / codes["cases"].sum()
show_table(codes, "DEGREE_INJURY_CD (accident years 2010-2025) and the lost_time target")
"""
    )
    md(
        """
## A.5 Build the analytic base table

`join.build_base_table` performs the three joins and the lagged case counts in the order of the
interface contract:

1. **Mines** (left join on `MINE_ID`, `many_to_one`, indicator `_m_mines`): mine attributes
   that do not change with the accident, such as the commodity, the state and the coordinates.
   Attributes that describe the mine *today* (current status, employee counts, controller dates)
   are not carried over, because they would describe the future of the accident. Three undated
   flags (safety committee, Part 48 training programme, portable operation) are the exception:
   they stay in the base table and in the data explorer, but they are not model inputs, because
   the snapshot gives no historical value and so they were not known when the form was filed.
2. **Production of the previous year** (left join on `MINE_ID` and `CAL_YR - 1`, indicator
   `_m_prev`): hours and employees summed over subunits. The previous year is used on purpose.
   At the time a form is filed the current year's quarterly hours are still incomplete, and a
   current-year case count would include the row itself. Rows without production in the
   previous year are **kept** with missing values and flagged by `prev_year_matched`.
3. **Production of the same year** (indicator `_m_same`): kept for exploratory rates only, never
   used as a model input.
4. **Lagged case counts** from the full accidents file: the cases at the same mine in the
   previous year that were *filed* before the accident (the filing date is encoded in
   `DOCUMENT_NO`), as `prev_incidents`, `prev_valid_cases` and `prev_lost_time_cases`.
   They are zero when the mine had production but no such case and missing when there is no
   previous-year production.

The first cell prints the two merge functions from `src/mlda/data/join.py`, so the join
statements themselves (left joins with `validate="many_to_one"`, an indicator column and an
asserted row count) can be read here. The log lines printed by the second cell are written by
the join code while it runs.
"""
    )
    code(
        r"""
import inspect

# The merge statements of joins J1 (Mines) and J2/J3 (production of the previous and the same year).
for merge_function in (join._join_mines, join._join_production):
    print(inspect.getsource(merge_function))
"""
    )
    code(
        r"""
# Show the INFO messages of the join code in the cell output.
join_logger = logging.getLogger("mlda.data.join")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("  %(message)s"))
join_logger.addHandler(handler)
join_logger.setLevel(logging.INFO)

with timer.stage("A.5 build the base table"):
    base = join.build_base_table(accidents, mines, production)

join_logger.removeHandler(handler)
print(f"\nbase table: {base.shape[0]:,} rows x {base.shape[1]} columns")
"""
    )
    code(
        r"""
n_rows = len(base)
matched_mines = int((base["_m_mines"] == "both").sum())
matched_prev = int(base["prev_year_matched"].sum())
matched_same = int((base["_m_same"] == "both").sum())
join_summary = pd.DataFrame(
    [
        {
            "join": "J1 accidents -> Mines",
            "keys": "MINE_ID",
            "role": "mine attributes (features)",
            "rows before": len(kept),
            "rows after": n_rows,
            "matched": matched_mines,
        },
        {
            "join": "J2 accidents -> production, previous year",
            "keys": "MINE_ID, CAL_YR - 1",
            "role": "exposure and lagged cases (features)",
            "rows before": len(kept),
            "rows after": n_rows,
            "matched": matched_prev,
        },
        {
            "join": "J3 accidents -> production, same year",
            "keys": "MINE_ID, CAL_YR",
            "role": "exploration only",
            "rows before": len(kept),
            "rows after": n_rows,
            "matched": matched_same,
        },
    ]
).set_index("join")
join_summary["unmatched (kept)"] = join_summary["rows after"] - join_summary["matched"]
join_summary["match rate (%)"] = 100 * join_summary["matched"] / join_summary["rows after"]
show_table(join_summary, "Join diagnostics (a left join keeps every accident row)")
assert (join_summary["rows before"] == join_summary["rows after"]).all()   # no accident lost or duplicated
print(f"COAL_METAL_IND agrees between Accidents and Mines on {base.attrs['coal_metal_ind_agreement']:.2%} of rows")

by_match = (
    base["prev_year_matched"]
    .map({True: "matched (t-1 production found)", False: "unmatched (kept with NaN)"})
    .value_counts()
    .rename("rows")
    .to_frame()
)
show_table(by_match, "Rows with and without previous-year production (their outcomes are compared on the training rows in B.12)")
"""
    )
    code(
        r"""
lag_columns = ["prev_hours", "prev_employees", "prev_incidents", "prev_valid_cases", "prev_lost_time_cases"]
show_table(base[lag_columns].describe().T, "Previous-year exposure and lagged case counts")

missing_matches_flag = base["prev_incidents"].isna().eq(~base["prev_year_matched"]).all()
print("lagged counts are NaN exactly where there is no previous-year production:", bool(missing_matches_flag))
print(f"rows with previous-year production but no earlier case (true zeros): {int(base['prev_incidents'].eq(0).sum()):,}")

show_table(
    base[
        ["MINE_ID", "CAL_YR", "COAL_METAL_IND", "mine_coal_metal_ind", "_m_mines", "prev_hours", "prev_employees",
         "prev_year_matched", "type_at_time", "prev_incidents", "prev_valid_cases", "prev_lost_time_cases", TARGET]
    ].head(8),
    "Joined columns for the first rows of the base table",
)
"""
    )
    md(
        """
## A.6 Validate the base table and compare it with the stored copy

`validate_base` collects every violation of the data contract (expected columns, N and
prevalence within tolerance, `MINE_ID` of 7 characters, no missing target or group, years inside
the window, lagged counts missing exactly when the previous year is unmatched) and raises if there
is any. Passing it is the gate between loading and exploring. The last cell compares the table
built live with `data/processed/base.parquet`, which `mlda-train` wrote: they should be identical,
because the whole load is deterministic.
"""
    )
    code(
        r"""
from mlda.config import PROCESSED_DIR
from mlda.data.schema import validate_base

validate_base(base)      # raises SchemaError listing every violation if the contract is broken
print("validate_base passed: columns, N, prevalence, keys, years and lag-count rules are all satisfied")

stored_base = pd.read_parquet(PROCESSED_DIR / "base.parquet")
live_base = join.persistable(base)
differing = [c for c in stored_base.columns if c not in live_base.columns or not live_base[c].equals(stored_base[c])]
print(f"live base table vs data/processed/base.parquet: same shape {live_base.shape == stored_base.shape}, "
      f"columns that differ: {differing if differing else 'none'}")

# The raw tables are no longer needed; free their memory before the modelling sections.
del accidents, mines, production, kept, window, stored_base, live_base
gc.collect()
"""
    )


# ---------------------------------------------------------------------------
# B. Exploratory data analysis
# ---------------------------------------------------------------------------


def section_b() -> None:
    """Section B: exploratory analysis of the training rows only, so that no test row informs a modelling choice."""
    md(
        """
# B. Exploratory data analysis

**Why the training rows only.** The exploration decides which transformations, encodings and
features the models receive. If the held-out rows took part in those decisions (a log transform
chosen because a test-set histogram looked skewed, a feature kept because it correlated with the
test labels), the test score would no longer be an honest estimate of performance on new data.
The split is therefore made first, by `mlda.split.make_split`, and persisted in
`data/processed/split_indices.json`; every table and figure in this section uses the training
rows of that split.

**Which columns.** The model inputs are built by `engineer_features` (52 columns; section C
explains each formula and the error rules behind it). The exploration uses these columns because
they are what the models see. Section B.1 first looks at four raw columns to show why the error
rules exist.
"""
    )
    code(
        r"""
from mlda.config import GROUP_KEY, SECTOR_COL
from mlda.features import FEATURE_SPEC, engineer_features
from mlda.split import load_split_indices

split = load_split_indices()                      # the persisted grouped, stratified split
train_idx, test_idx = split.train_idx, split.test_idx

X = engineer_features(base)                       # 52 model inputs, one row per accident
y = base[TARGET].to_numpy(dtype=int)
groups = base[GROUP_KEY].to_numpy(dtype=str)      # mine-year: rows of one mine-year never straddle the split
sector = base[SECTOR_COL].to_numpy(dtype=str)     # C = coal, M = metal/non-metal, used for stratified reporting

X_train, y_train, g_train = X.iloc[train_idx], y[train_idx], groups[train_idx]
frame_train = X_train.assign(**{TARGET: y_train})  # features and target together, for the plots
base_train = base.iloc[train_idx]                  # the raw joined columns of the same rows

numeric_cols, categorical_cols, binary_cols = (FEATURE_SPEC[k] for k in ("numeric", "categorical", "binary"))
print(f"training rows: {len(train_idx):,} of {len(base):,} ({len(train_idx) / len(base):.1%}); "
      f"held-out test rows (not touched in this section): {len(test_idx):,}")
print(f"inputs: {len(categorical_cols)} categorical + {len(numeric_cols)} numeric + {len(binary_cols)} binary "
      f"= {X.shape[1]} features")
print(f"lost_time prevalence: training {y_train.mean():.3%}, all rows {y.mean():.3%}")
"""
    )
    md(
        """
## B.1 What the raw columns look like

Four raw columns are enough to see why a cleaning step is unavoidable. Work experience is in
years and can exceed a working life or contradict the total (job experience longer than total
experience); clock times are stored as four-digit codes in which `9999` means
"not reported"; the victim count has a very long tail. The model never sees these raw values; the
error rules of section C turn them into missing values or caps and log how many rows each rule
touched.
"""
    )
    code(
        r"""
from mlda.eda import describe_categorical, describe_numeric

raw_numeric = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "NO_INJURIES", "prev_hours"]
show_table(describe_numeric(base_train, raw_numeric), "Raw numeric columns on the training rows (before any rule)")

accident_time = pd.to_numeric(base_train["ACCIDENT_TIME"], errors="coerce")
code_counts = accident_time.value_counts().head(5).rename_axis("ACCIDENT_TIME code").to_frame("rows")
show_table(code_counts, "Most frequent raw ACCIDENT_TIME codes: 9999 means 'not reported'")
print("rows where JOB_EXPER > TOT_EXPER:", int((base_train["JOB_EXPER"] > base_train["TOT_EXPER"]).sum()))
print("rows where MINE_EXPER > TOT_EXPER:", int((base_train["MINE_EXPER"] > base_train["TOT_EXPER"]).sum()))
"""
    )
    md(
        """
## B.2 Descriptive statistics of the numeric inputs

The first table is pandas' `describe()`. The second, from `mlda.eda.describe_numeric`, adds what
`describe()` lacks. For every numeric input it gives the location (mean, median, 10 % trimmed mean), the
spread (sample variance and sample standard deviation with `ddof = 1`, range, quartiles and IQR)
and the shape (skewness and excess kurtosis). Comparing mean, median and trimmed mean shows how
far a few extreme values pull the mean; a large positive skewness with a mean well above the
median marks a right-skewed variable that may need a log transform for the linear and distance
models. Note the number of rows: missing values are excluded column by column, so `n` differs.
"""
    )
    code(
        r"""
from mlda.config import METRICS_DIR


def matches_stored(live: pd.DataFrame, filename: str, index_col: int | None = None) -> str:
    # Reproducibility check: does the table computed here equal the one written by mlda-train?
    path = METRICS_DIR / filename
    if not path.exists():
        return f"{filename} not found"
    stored = pd.read_csv(path, index_col=index_col)
    numeric_live = live.reset_index(drop=index_col is None).select_dtypes("number")
    numeric_stored = stored.select_dtypes("number")
    if numeric_live.shape != numeric_stored.shape:
        return f"differs from {filename} (shape {numeric_live.shape} vs {numeric_stored.shape})"
    same = np.allclose(numeric_live.to_numpy(float), numeric_stored.to_numpy(float), rtol=1e-5, atol=1e-8, equal_nan=True)
    return f"identical to {filename}" if same else f"differs from {filename}"


show_table(X_train[numeric_cols].describe().T, "pandas describe() of the numeric inputs, for orientation")
numeric_description = describe_numeric(frame_train, numeric_cols)
show_table(numeric_description, "Descriptive statistics of the 20 numeric inputs (training rows): the full table")
print("reproducibility:", matches_stored(numeric_description, "eda_describe_numeric.csv", index_col=0))
"""
    )
    md(
        """
## B.3 Descriptive statistics of the categorical inputs

For a categorical input the useful summary is its **cardinality** and how dominant its most
frequent level is. The number of levels predicts the width of the one-hot encoding (a column with
193 occupations adds 193 indicator columns), and a high share for the mode warns of a nearly
constant column. The structural placeholder `none` (for example, no underground location at a
surface mine) is a genuine level and often the mode.
"""
    )
    code(
        r"""
categorical_description = describe_categorical(X_train, categorical_cols)
show_table(categorical_description, "Categorical inputs: levels and the most frequent level (training rows)")
print("one-hot width before rare levels are pooled:", int(categorical_description["n_levels"].sum()), "indicator columns")
"""
    )
    md(
        """
## B.4 Target distribution

The bar chart shows the class balance of the training rows. A balanced target has two practical
consequences: plain accuracy is not misleading (a majority-class rule scores only about 51 %), and
resampling methods such as SMOTE are not needed. The class weight of logistic regression is still
offered as a hyper-parameter, so the search can show whether re-weighting helps.
"""
    )
    code(
        r"""
from mlda import plots

show(plots.target_distribution(y_train), "Bar chart of the two outcome classes in the training rows, with counts and percentages")
balance = pd.DataFrame(
    {
        "rows": [len(train_idx), len(test_idx), len(y)],
        "lost_time = 1": [int(y_train.sum()), int(y[test_idx].sum()), int(y.sum())],
        "prevalence": [y_train.mean(), y[test_idx].mean(), y.mean()],
    },
    index=["training", "test", "all"],
)
show_table(balance, "Class balance by split")
"""
    )
    md(
        """
## B.5 Distributions and class differences

**Histograms** with a kernel density estimate show the shape of each numeric input, split by
outcome. Right-skewed inputs (hours, employees, experience) justify the log-transformed
companions already included in the feature set. **Box plots** by class show whether the median
and spread differ between lost-time and other cases; with fliers hidden the boxes stay readable
for variables with a long tail. Overlapping boxes and curves mean a variable separates the
classes only weakly on its own, which is expected for most accident circumstances.
"""
    )
    code(
        r"""
with timer.stage("B.5 histograms and box plots"):
    # two figures of two columns each: the long axis labels of a three-column grid overlap
    for group, columns in (("worker and accident", numeric_cols[:10]), ("mine and exposure", numeric_cols[10:])):
        show(plots.histograms_grid(frame_train, columns, hue=TARGET, ncols=2),
             f"Grid of histograms with density curves of the {group} inputs, split by outcome class")
    show(plots.class_boxplots(frame_train, numeric_cols, TARGET, showfliers=False), "Grid of box plots of the numeric inputs by outcome class")
"""
    )
    md(
        """
## B.6 Association with the target and between inputs

**Point-biserial correlation** is Pearson's r between a numeric input and the 0/1 target, so it
is the right coefficient for numeric-versus-binary. **Pearson** measures linear association and
**Spearman** rank (monotone) association; where they disagree by more than 0.1 the relation is
monotone but not linear, or outliers pull Pearson, which is a hint to transform or to prefer a
rank-based or tree model. The pair plot shows the five numeric inputs with the largest
|point-biserial r| against each other, coloured by outcome.
"""
    )
    code(
        r"""
from mlda.eda import correlations, point_biserial

biserial = point_biserial(frame_train, numeric_cols, TARGET)
show_table(biserial, "Point-biserial correlation with lost_time (strongest first)")
print("reproducibility:", matches_stored(biserial, "eda_point_biserial.csv"))

corr = correlations(frame_train, numeric_cols)
show(plots.correlation_heatmaps(corr.pearson, corr.spearman), "Pearson and Spearman correlation heatmaps of the numeric inputs")
show_table(corr.disagreements.head(12), "Pairs where |Pearson - Spearman| exceeds 0.1 (largest gap first)")
print(f"{len(corr.disagreements)} pairs disagree by more than 0.1")

top_numeric = biserial["feature"].head(5).tolist()     # the same five as the report figure (mlda.pipeline.run_eda)
with timer.stage("B.6 pair plot"):
    show(plots.pairplot_top(frame_train, top_numeric, TARGET), "Pair plot of the numeric inputs most correlated with the target, coloured by outcome")
"""
    )
    md(
        """
## B.7 Categorical inputs against the target

**Cramer's V** (bias-corrected) is the association between a categorical input and the target,
between 0 (independent) and 1 (the level determines the class). It is the categorical analogue of
the correlation table above and it is comparable across inputs with different numbers of levels
because of the bias correction. The bar charts show the lost-time rate per level of the eight
strongest inputs, with the overall rate marked, and levels with fewer than 100 rows pooled.
"""
    )
    code(
        r"""
from mlda.eda import class_rate_by_level, cramers_v_table

cramers = cramers_v_table(frame_train, categorical_cols, TARGET)
show_table(cramers, "Cramer's V with lost_time (strongest first)")
print("reproducibility:", matches_stored(cramers, "eda_cramers_v.csv"))

top_categorical = cramers["feature"].head(8).tolist()
show(plots.categorical_vs_target(frame_train, top_categorical, TARGET), "Bar charts of the lost-time rate per level for the categorical inputs most associated with the target")
show_table(class_rate_by_level(frame_train, SECTOR_COL, TARGET), "Lost-time rate by commodity (C = coal, M = metal/non-metal)")
"""
    )
    md(
        """
## B.8 Collinearity: variance inflation factors

A variance inflation factor (VIF) above about 5-10 means an input is almost a linear combination
of the others. That does not hurt prediction by trees, but it makes the coefficients of logistic
regression unstable and inflates distances in k-nearest neighbours. The factors are computed on
the training rows, with missing values filled by the median only for this diagnostic.
"""
    )
    code(
        r"""
from mlda.eda import vif_table

vif = vif_table(X_train[numeric_cols])
show_table(vif.head(10), "Largest variance inflation factors")
print("reproducibility:", matches_stored(vif, "eda_vif.csv"))
"""
    )
    md(
        """
## B.9 Missing values

The table counts the missing values in the raw joined columns of the training rows, and the
figure ranks the most incomplete ones. Two patterns matter. Some columns are missing for a
structural reason (an underground location is not applicable at a surface mine), so the gap
carries information and becomes its own level later. Others are reporting gaps (worker
experience, clock times) and need imputation plus an indicator. Codes such as
`9999` are not yet counted as missing here; the rules of section C convert them.
"""
    )
    code(
        r"""
from mlda.eda import missingness_table

missing = missingness_table(base_train)
show_table(missing[missing["n_missing"] > 0].head(20), "Raw columns with the most missing values (training rows)")
print(f"{int((missing['n_missing'] > 0).sum())} of {len(missing)} raw columns have at least one missing value")
show(plots.missingness(base_train), "Bar chart of the percentage of missing values in each raw column")
"""
    )
    md(
        """
## B.10 Normality and outliers

A normal **Q-Q plot** compares the quantiles of a variable with those of a normal distribution; a
straight line means normality. The log transform pulls the right tail of experience and hours
towards the line. For outliers two standard rules are compared: **Tukey's IQR fences**
(Q1 - 1.5 IQR, Q3 + 1.5 IQR) and the **Z-score rule** (|z| > 3, with the sample standard
deviation). On a skewed variable the IQR rule flags more points on the long side, because the mean
and standard deviation are themselves inflated by the tail. Outliers are not deleted: they are
real mines and real workers, and the error rules already remove impossible values. The
preprocessor offers optional clipping, which section C evaluates.
"""
    )
    code(
        r"""
from mlda.cleaning import flag_outliers
from mlda.pipeline import OUTLIER_COLUMNS, QQ_COLUMNS

show(plots.qq_plots(frame_train, list(QQ_COLUMNS)), "Normal Q-Q plots of numeric inputs and their log transforms")

outlier_rows = []
for column in numeric_cols:
    series = frame_train[column]
    iqr_flags, z_flags = flag_outliers(series, "iqr"), flag_outliers(series, "zscore")
    outlier_rows.append(
        {
            "feature": column,
            "IQR rule (1.5)": int(iqr_flags.sum()),
            "Z-score rule (3)": int(z_flags.sum()),
            "IQR share (%)": 100 * iqr_flags.mean(),
            "Z-score share (%)": 100 * z_flags.mean(),
        }
    )
outliers = pd.DataFrame(outlier_rows).set_index("feature")
show_table(outliers.sort_values("IQR rule (1.5)", ascending=False), "Rows flagged as outliers by each rule (training rows)")

after = pd.DataFrame({raw: frame_train[logged].to_numpy() for raw, logged in OUTLIER_COLUMNS.items()})
show(plots.outlier_fences(frame_train, list(OUTLIER_COLUMNS), after=after, after_labels=OUTLIER_COLUMNS),
     "Histograms of experience and previous-year hours with IQR and Z-score outlier fences, before and after the log transform")
"""
    )
    md(
        """
## B.11 Principal components

A principal component analysis of the standardised numeric inputs shows whether the classes
separate in the directions of largest variance. The first two components are plotted for a
sample of rows, coloured by outcome. Heavy overlap means no simple linear projection separates
the classes, so a model has to combine many weak signals, which agrees with the modest
correlations above. The explained-variance table is fitted on the training rows only.
"""
    )
    code(
        r"""
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

show(plots.pca_projection(X_train[numeric_cols], y_train), "Scatter plot of the first two principal components of the numeric inputs, coloured by outcome")

scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_train[numeric_cols]))
pca = PCA(n_components=6, random_state=RANDOM_STATE).fit(scaled)
explained = pd.DataFrame(
    {
        "explained variance ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    },
    index=[f"PC{i + 1}" for i in range(6)],
)
show_table(explained, "Variance explained by the first six principal components")
"""
    )
    md(
        """
## B.12 Evidence for three claims that later sections use

Three statements further down depend on the training rows and are checked here instead of being
asserted. (1) A blank value can carry signal, so missing-value indicators are kept (section C.3).
(2) Operators code restricted duty (`DEGREE_INJURY_CD` 05, "restricted activity only") at
different rates in coal and in metal and non-metal mines, which is why the label is partly a
property of practice (sections E.8 and F.4). (3) The coal lost-time rate rises over the period
while restricted-duty coding falls, which is the label drift that section E.13 tests. All three
tables use the training rows only; the note under them states the numbers.
"""
    )
    code(
        r"""
evidence = base_train[["CAL_YR", SECTOR_COL]].assign(
    lost_time=y_train,
    restricted_duty=(base_train["DEGREE_INJURY_CD"] == "05").to_numpy(),
    experience_blank=X_train["tot_exper_missing"].to_numpy() == 1,
    unmatched_year=~X_train["prev_year_matched"].to_numpy().astype(bool),
)
indicator_rows = {
    "worker experience blank": evidence["experience_blank"],
    "no previous-year production": evidence["unmatched_year"],
}
indicator_table = pd.DataFrame(
    {
        label: {
            "rows": int(flag.sum()),
            "lost-time rate where the indicator is 1": evidence.loc[flag, "lost_time"].mean(),
            "lost-time rate where it is 0": evidence.loc[~flag, "lost_time"].mean(),
        }
        for label, flag in indicator_rows.items()
    }
).T.astype({"rows": int})
show_table(indicator_table, "Lost-time rate by missing-value indicator (training rows)")

sector_summary = evidence.groupby(SECTOR_COL).agg(
    rows=("lost_time", "size"), lost_time_rate=("lost_time", "mean"), restricted_duty_share=("restricted_duty", "mean")
)
sector_summary.index = sector_summary.index.map({"C": "coal", "M": "metal / non-metal"})
show_table(sector_summary, "Lost-time rate and share of restricted-duty cases (code 05) by commodity (training rows)")

by_year = (
    evidence.groupby(["CAL_YR", SECTOR_COL])
    .agg(lost_time_rate=("lost_time", "mean"), restricted_duty_share=("restricted_duty", "mean"))
    .unstack(SECTOR_COL)
)
by_year.columns = [f"{measure.replace('_', ' ')}, {'coal' if sector == 'C' else 'metal'}" for measure, sector in by_year.columns]
show_table(by_year, "Lost-time rate and restricted-duty share by accident year and commodity (training rows)")

early = evidence[evidence["CAL_YR"] <= YEAR_MIN + 2].groupby(SECTOR_COL)[["lost_time", "restricted_duty"]].mean()
late = evidence[evidence["CAL_YR"] >= YEAR_MAX - 2].groupby(SECTOR_COL)[["lost_time", "restricted_duty"]].mean()
blank = indicator_table.loc["worker experience blank"]
note(
    f"**What the tables show.** Where total experience is blank, {blank.iloc[1]:.1%} of the {int(blank.iloc[0]):,} training cases are lost-time cases "
    f"against {blank.iloc[2]:.1%} elsewhere, so the indicator carries signal. Restricted duty is "
    f"{sector_summary.loc['coal', 'restricted_duty_share']:.1%} of the coal cases and {sector_summary.loc['metal / non-metal', 'restricted_duty_share']:.1%} of "
    f"the metal and non-metal cases, so the sectors draw the line between lost time and restricted duty in different places. Between "
    f"{YEAR_MIN}-{YEAR_MIN + 2} and {YEAR_MAX - 2}-{YEAR_MAX} the coal lost-time rate moves from {early.loc['C', 'lost_time']:.1%} to {late.loc['C', 'lost_time']:.1%} "
    f"while the coal share of restricted-duty cases moves from {early.loc['C', 'restricted_duty']:.1%} to {late.loc['C', 'restricted_duty']:.1%}; "
    f"in metal and non-metal mines the lost-time rate moves from {early.loc['M', 'lost_time']:.1%} to {late.loc['M', 'lost_time']:.1%}."
)
"""
    )
    md(
        """
## B.13 What the exploration shows, and what it changes

The cell below reads the tables above and states the findings in numbers, so the sentences cannot
drift from the outputs. The markdown after it lists the decisions that follow for the preparation
and the modelling.
"""
    )
    code(
        r"""
skewed = numeric_description[numeric_description["skewness"].abs() > 1].sort_values("skewness", ascending=False)
heavy_tail = outliers.sort_values("IQR rule (1.5)", ascending=False).head(3)
weak = biserial[biserial["r"].abs() < 0.1]
coal_rate = class_rate_by_level(frame_train, SECTOR_COL, TARGET).set_index("level")["positive_rate_pct"]
collinear = vif[vif["vif"] > 10]
note(
    f"**Target.** {len(y_train):,} training rows, {y_train.mean():.1%} lost time: the classes are balanced. "
    f"**Shape.** {len(skewed)} of {len(numeric_cols)} numeric inputs have a skewness above 1 in absolute value "
    f"(largest: `{skewed.index[0]}`, {skewed['skewness'].iloc[0]:.1f}); the log-transformed columns are close to symmetric "
    f"(`log1p_tot_exper` {numeric_description.loc['log1p_tot_exper', 'skewness']:.2f}, "
    f"`log1p_prev_hours` {numeric_description.loc['log1p_prev_hours', 'skewness']:.2f}). "
    f"**Outliers.** The IQR rule flags the most rows for `{heavy_tail.index[0]}` ({heavy_tail['IQR share (%)'].iloc[0]:.1f} % against "
    f"{heavy_tail['Z-score share (%)'].iloc[0]:.1f} % by the Z-score rule), which is the effect of a long right tail on the mean and the standard deviation. "
    f"**Signal.** Only {int((biserial['r'].abs() >= 0.1).sum())} of {len(biserial)} numeric inputs have |r| of at least 0.1 with the target "
    f"(strongest: `{biserial['feature'].iloc[0]}`, r = {biserial['r'].iloc[0]:.2f}); among the categorical inputs the injury "
    f"descriptors lead (`{cramers['feature'].iloc[0]}` V = {cramers['cramers_v'].iloc[0]:.2f}, `{cramers['feature'].iloc[1]}` "
    f"V = {cramers['cramers_v'].iloc[1]:.2f}). The commodity matters: {coal_rate.get('C', float('nan')):.1f} % of coal cases and "
    f"{coal_rate.get('M', float('nan')):.1f} % of metal and non-metal cases are lost-time cases. "
    f"**Collinearity.** {len(collinear)} numeric inputs have a VIF above 10 ({', '.join(f'`{f}`' for f in collinear['feature'])}); "
    f"they carry the same size information. **Missing values.** {int((numeric_description['missing_pct'] > 0).sum())} of the "
    f"{len(numeric_cols)} numeric inputs have gaps, at most {numeric_description['missing_pct'].max():.1f} % of the rows. "
    f"**Projection.** The first two principal components explain {explained['explained variance ratio'].iloc[:2].sum():.0%} of the variance of the numeric inputs "
    f"and the classes overlap heavily in that plane."
)
"""
    )
    md(
        """
**Consequences for the next sections.** (1) Several numeric inputs are strongly right-skewed, so the
feature set carries log-transformed copies, and the models that use distances or a linear
boundary receive standardised inputs. (2) The outliers are real mines and real workers, so they
are kept; only impossible values are corrected (section C.2), and clipping is tested rather than
assumed (C.7). (3) The exposure columns are almost collinear, which argues for regularisation in
logistic regression and does not matter for trees. (4) No numeric input separates the classes on its
own, and the injury descriptors are the strongest categorical signals, so categorical encoding
carries most of the load: one-hot encoding with a pooled bucket for rare levels. (5) The commodity
changes the base rate, so every result is also reported by commodity. (6) Missing values are common
only for a few groups, and their pattern is informative, so imputation is paired with indicator
columns.
"""
    )


# ---------------------------------------------------------------------------
# C. Data preparation
# ---------------------------------------------------------------------------


def section_c() -> None:
    """Section C: duplicates, error rules, missing-value plan, features, leakage check, preprocessing and the grouped split."""
    md(
        """
# C. Data preparation

Preparation turns the joined table into model inputs without letting information from the test
fold leak in. The order is: (1) check for duplicates, (2) correct impossible values with
documented, numbered rules, (3) decide how every group of missing values is handled, (4) engineer
features that use only what is known when the form is filed, (5) test for leakage, (6) build the
preprocessing (imputation, scaling, encoding, optional outlier clipping) as one
`ColumnTransformer` that sits **inside** the model pipeline, so that it is re-fitted on the
training part of every cross-validation fold, and (7) verify the grouped, stratified split.
Every fitted statistic (a median, a scaler's mean, the one-hot levels) is learnt from training
rows only.
"""
    )
    md(
        """
## C.1 Duplicates

Each row is one injured person on one form, so a repeated row would mean a repeated person. The
check counts exact duplicates and the duplicates that remain when the identifiers that make every
form unique (the document numbers and the narrative) are ignored. Rows that agree on everything
else are distinct victims of the same event (for example two workers hurt in one accident), so
they are kept; the nested key `accident_group` (mine and accident date) keeps them on the same
side of the split.
"""
    )
    code(
        r"""
identifier_columns = ["DOCUMENT_NO", "CLOSED_DOC_NO", "NARRATIVE"]
descriptive_columns = [c for c in base.columns if c not in identifier_columns]
duplicates = pd.DataFrame(
    [
        {"check": "exact duplicate rows (all columns)", "rows": int(base.duplicated().sum())},
        {"check": "DOCUMENT_NO values that repeat", "rows": int(base["DOCUMENT_NO"].duplicated().sum())},
        {"check": "rows identical apart from the document numbers and narrative", "rows": int(base.duplicated(subset=descriptive_columns).sum())},
    ]
).set_index("check")
show_table(duplicates, "Duplicate checks on the base table")
assert duplicates.loc["exact duplicate rows (all columns)", "rows"] == 0
assert duplicates.loc["DOCUMENT_NO values that repeat", "rows"] == 0
"""
    )
    md(
        """
## C.2 Error-correction rules (the decision log)

The profiling of the raw data found values that cannot be true: job experience longer than total
experience, experience above 60 years, clock-time codes that mean "not reported", a victim count
above ten, coordinates outside the United States, mine attributes that are dated after the
accident, and rates computed from almost no hours. Each problem became a numbered rule, E1 to E30.
The function `apply_error_rules` applies the rules that change values in the joined table (E1-E12
and E21-E25) and returns a log with the condition, the action and the number of rows affected;
the first table below is that log, computed here. The remaining rules (E13-E20 and E26-E30) are
applied when the files are loaded and the features are built: placeholders kept as levels,
repairs of the code lists, type coercion. The second table lists them from the profiling record
`docs/profile/decisions.json`, with the row counts found when the data were profiled.
**Together the two tables are the decision log of the project**: they record what was changed,
why, and how many rows it touched, and none of the rules uses the label.
"""
    )
    code(
        r"""
from mlda.cleaning import apply_error_rules

clean, rule_log = apply_error_rules(base)
with pd.option_context("display.max_colwidth", None):
    show_table(rule_log, f"Error rules E1-E12 and E21-E25: condition, action and rows affected (all {len(base):,} rows)")
print("reproducibility:", matches_stored(rule_log.drop(columns=["rule", "condition", "action"]), "error_rules.csv"))

experience = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"]
effect = pd.DataFrame(
    {
        "raw max": base[experience].max(),
        "corrected max": clean[experience].max(),
        "raw missing": base[experience].isna().sum(),
        "corrected missing": clean[experience].isna().sum(),
    }
)
show_table(effect, "Effect of rules E1-E3 on the experience columns")

decisions = json.loads((PROJECT_ROOT / "docs" / "profile" / "decisions.json").read_text(encoding="utf-8"))
documented = pd.DataFrame(decisions["error_rules"])
documented["id"] = documented["name"].str.extract(r"^(E\d+)_", expand=False)
applied_here = set(rule_log["rule"].str.extract(r"^(E\d+)_", expand=False))
elsewhere = documented[~documented["id"].isin(applied_here)].set_index("id")[["name", "condition", "action", "rows_affected"]]
with pd.option_context("display.max_colwidth", None):
    show_table(elsewhere, "Rules applied in load and features (record of the data profiling; row counts as found then)")
assert len(applied_here) + len(elsewhere) == len(documented) == 30, "every documented rule is in exactly one of the two tables"
print(f"{len(applied_here)} rules applied by apply_error_rules, {len(elsewhere)} applied in load and features: {len(documented)} in total")
"""
    )
    md(
        """
## C.3 Missing-value plan

Missing values are treated by group, because the reason for the gap differs. A **structural**
gap (an underground location at a surface mine) is not a missing measurement, so the placeholder
becomes its own level `none`. A **reporting** gap (worker experience, accident hour) is imputed
with the **median**, which is the safer centre for right-skewed variables, and an explicit
**indicator column** records that the value was imputed, because missingness itself can carry
signal (section B.12 compares the lost-time rate where experience is blank with the rest). **Unmatched joins** keep their
rows and carry the indicator `prev_year_matched`. All imputation statistics are learnt inside the
pipeline on training rows. Columns that are mostly empty and carry no information are dropped.
The percentages in the table are measured on the training rows.
"""
    )
    code(
        r"""
def span(values):
    low, high = f"{values.min():.1f}", f"{values.max():.1f}"
    return low if low == high else f"{low} to {high}"


def pct_missing(columns):
    return span(X_train[columns].isna().mean() * 100)


def pct_level(columns, level):
    return span((X_train[columns] == level).mean() * 100)


numeric_plan = [
    ("Worker experience", ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "exper_ratio_mine", "exper_ratio_job", "log1p_tot_exper"],
     "median imputation", "tot_exper_missing, exper_inconsistent"),
    ("Time of day", ["accident_hour", "shift_begin_hour", "hours_into_shift"], "median imputation",
     "accident_hour_missing, hours_into_shift_missing"),
    ("Coordinates", ["latitude", "longitude"], "median imputation", "coords_missing"),
    ("Previous-year exposure", ["prev_hours", "log1p_prev_hours", "prev_employees", "prev_hours_per_employee", "prev_incidents",
                               "prev_incident_rate_200k", "prev_lost_time_rate_200k"],
     "median imputation; unmatched rows kept", "prev_year_matched"),
    ("Previous-year lost-time share", ["prev_lost_time_share"], "median imputation; missing when no valid case was filed before the accident",
     "prev_share_missing"),
]
plan_rows = [
    {"group": group, "columns": ", ".join(cols), "missing in training rows (%)": pct_missing(cols), "treatment": treatment, "indicator column": flag}
    for group, cols, treatment, flag in numeric_plan
]
structural = ["UG_LOCATION_CD", "UG_MINING_METHOD_CD", "MINING_EQUIP_CD", "EQUIP_MFR_NAME", "ACTIVITY_CD"]
plan_rows.append(
    {"group": "Structural categorical", "columns": ", ".join(structural),
     "missing in training rows (%)": f"level 'none' {pct_level(structural, 'none')}",
     "treatment": "placeholder kept as the level 'none' (not applicable)", "indicator column": "implicit"}
)
other_cat = [c for c in categorical_cols if c not in structural]
plan_rows.append(
    {"group": "Other categorical", "columns": f"{len(other_cat)} columns", "missing in training rows (%)": f"level 'missing' {pct_level(other_cat, 'missing')}",
     "treatment": "constant level 'missing'; unseen levels fall into an infrequent bucket", "indicator column": "implicit"}
)
plan_rows.append(
    {"group": "Dated mine attribute", "columns": "status_103i_at_accident",
     "missing in training rows (%)": f"level 'Unknown' {(X_train['status_103i_at_accident'] == 'Unknown').mean() * 100:.1f}",
     "treatment": "status dated after the accident, or blank, becomes the level 'Unknown'", "indicator column": "implicit"}
)
plan_rows.append(
    {"group": "Dropped (sparse)", "columns": "EQUIP_MODEL_NO",
     "missing in training rows (%)": f"{base_train['EQUIP_MODEL_NO'].isna().mean() * 100:.1f}",
     "treatment": "dropped: free-text model number, mostly empty", "indicator column": "none"}
)
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame(plan_rows).set_index("group"), "Missing-value plan by column group")

indicators = X_train[binary_cols].mean().mul(100).rename("share of rows = 1 (%)").to_frame()
show_table(indicators, "Binary columns on the training rows (indicators and flags)")
"""
    )
    md(
        """
## C.4 Feature engineering

`engineer_features` builds the 52 inputs row by row, with no fitted state: 20 categorical, 20
numeric and 12 binary columns, listed in `FEATURE_SPEC`. The design rule is **only what is known
when the form is filed**. The accident circumstances, the worker's experience and the mine's
stable attributes qualify; the mine's history qualifies only as the previous year, and only cases
filed before the accident. The safety-committee, Part 48 training and portable-operation flags
of the Mines table were dropped because the snapshot gives no historical value for them. The
error rules are part of these formulas, which is why computing the
features from the raw base table and from the corrected table gives the identical frame (the
cell asserts this).
"""
    )
    code(
        r"""
from mlda.features import FeatureFrame, feature_frame

parts: FeatureFrame = feature_frame(clean)     # X, y, groups and the reporting columns, from the corrected table
assert parts.X.equals(X), "features differ between the raw and the corrected base table"
assert np.array_equal(parts.y.to_numpy(dtype=int), y) and np.array_equal(parts.groups.to_numpy(dtype=str), groups)
meta = parts.meta

spec_table = pd.DataFrame(
    {
        "kind": [kind for kind in ("categorical", "numeric", "binary") for _ in FEATURE_SPEC[kind]],
        "feature": [name for kind in ("categorical", "numeric", "binary") for name in FEATURE_SPEC[kind]],
    }
)
spec_table["dtype"] = [str(X[name].dtype) for name in spec_table["feature"]]
spec_table["levels (categorical)"] = pd.array(
    [X[name].nunique() if kind == "categorical" else None for kind, name in zip(spec_table["kind"], spec_table["feature"], strict=True)],
    dtype="Int64",
)
counts = spec_table["kind"].value_counts().rename("features").to_frame().loc[["categorical", "numeric", "binary"]]
show_table(counts, "FEATURE_SPEC: features per kind")
print(f"total {len(spec_table)} features; X has shape {X.shape}")
with pd.option_context("display.max_rows", 80):
    show_table(spec_table.set_index("feature"), "FEATURE_SPEC listing")
"""
    )
    code(
        r"""
formulas = pd.DataFrame(
    [
        ("log1p_tot_exper", "log(1 + TOT_EXPER)", "experience is right-skewed; the log spreads the short careers"),
        ("exper_ratio_mine, exper_ratio_job", "MINE_EXPER / TOT_EXPER and JOB_EXPER / TOT_EXPER, clipped to [0, 1]", "newness to the mine or the job, whatever the career length"),
        ("hours_into_shift", "(accident time - shift start) mod 24 h, in hours; missing if > 16", "fatigue and shift position; the modulo handles shifts that start before midnight"),
        ("accident_hour, shift_begin_hour, is_night, is_weekend", "hour of the clock time; night = 18:00-05:59; Saturday or Sunday", "time-of-day and calendar effects"),
        ("new_to_mine", "1 if MINE_EXPER < 1 year", "workers new to a site"),
        ("prev_hours, log1p_prev_hours, prev_employees, prev_hours_per_employee", "previous-year employee-hours and average employees of the mine", "exposure and mine size"),
        ("prev_incident_rate_200k, prev_lost_time_rate_200k", "log(1 + min(cases / hours x 200,000, 50))", "MSHA's incidence rate per 200,000 hours; capped for tiny mines (rule E25)"),
        ("prev_lost_time_share", "previous-year lost-time cases / valid cases, filed before the accident", "how often this mine's cases end in lost time"),
        ("mine_size_band, type_at_time, status_103i_at_accident", "employees band; Underground / Surface / Facility; 103(i) status in force", "mine context at the time, not today"),
        ("is_contractor, immed_reportable", "contractor present; immediate-notification code present and valid", "who was hurt and how the event was reported"),
    ],
    columns=["feature(s)", "formula", "reason"],
).set_index("feature(s)")
with pd.option_context("display.max_colwidth", None):
    show_table(formulas, "Engineered features: formula and reason")

show_table(
    pd.concat(
        [base[["TOT_EXPER", "ACCIDENT_TIME", "SHIFT_BEGIN_TIME", "prev_hours"]].head(5).add_prefix("raw "),
         X[["log1p_tot_exper", "accident_hour", "hours_into_shift", "log1p_prev_hours", "is_night"]].head(5)],
        axis=1,
    ),
    "Five rows: raw columns next to the engineered features derived from them",
)
"""
    )
    md(
        """
## C.5 Leakage check

**Target leakage** means that a model input contains the answer, or something recorded after the
outcome is known. In this data the outcome columns (days lost, days restricted, the schedule
charge, the return-to-work date, the closing document, the investigation date, the narrative
written after the case) would make the problem trivial and useless, because none of them exists
when the form is filed. The mine's *current* state (status today, employee count today) is a
second, quieter leak: it describes the mine after the accident. The cell asserts that none of
these columns is in `X`, and that no single input is almost a copy of the target.
"""
    )
    code(
        r"""
from mlda.config import LEAKAGE_COLUMNS, MINES_CURRENT_STATE_COLUMNS

forbidden = set(LEAKAGE_COLUMNS) | set(MINES_CURRENT_STATE_COLUMNS) | {TARGET, TARGET_SOURCE_COL}
overlap = sorted(forbidden & set(X.columns))
assert not overlap, f"leakage columns in the feature frame: {overlap}"
print(f"none of the {len(forbidden)} leakage or current-state columns is among the {X.shape[1]} features")

strongest = float(biserial["r"].abs().max())
assert strongest < 0.95, "one numeric input is almost a copy of the target"
print(f"strongest numeric association with the target: |r| = {strongest:.3f} ({biserial['feature'].iloc[0]})")
print(f"strongest categorical association: Cramer's V = {cramers['cramers_v'].max():.3f} ({cramers['feature'].iloc[0]})")

reasons = {
    "DEGREE_INJURY_CD": "defines the label",
    "DEGREE_INJURY": "text of the label",
    "DAYS_LOST": "days away from work, known only after recovery",
    "DAYS_RESTRICT": "days of restricted duty, known only after recovery",
    "SCHEDULE_CHARGE": "days charged for permanent loss, assigned after the case",
    "TRANS_TERM": "transfer or termination, recorded after the injury",
    "RETURN_TO_WORK_DT": "date the worker came back, after the injury",
    "CLOSED_DOC_NO": "closing document, filed when the case is resolved",
    "INVEST_BEGIN_DT": "investigation date, after the accident",
    "NARRATIVE": "free text written after the case",
}
leak_table = pd.DataFrame({"excluded because": [reasons[c] for c in LEAKAGE_COLUMNS]}, index=pd.Index(LEAKAGE_COLUMNS, name="outcome column"))
show_table(leak_table, "Outcome columns kept out of the features")
print("current-state mine columns kept out:", ", ".join(MINES_CURRENT_STATE_COLUMNS))
"""
    )
    md(
        """
## C.6 The preprocessing pipeline

`build_preprocessor` returns an unfitted `ColumnTransformer` with three branches: **numeric**
columns are cast to float32, optionally clipped, imputed and scaled; **categorical** columns are
imputed with the constant level "missing" and one-hot encoded, with levels rarer than 0.1 % of
the training rows pooled into an *infrequent* bucket (which is also where a level that never
appeared in training goes at prediction time); **binary** columns pass through. The default
options are the median imputer, the standard scaler, one-hot encoding and no clipping. Tree
models override the scaler with "none", because their splits do not depend on scale. The first
table gives the rationale of each option; the cells after it fit the transformer on the
training rows and check what comes out.
"""
    )
    code(
        r"""
from mlda.preprocessing import DEFAULTS, OPTIONS, build_preprocessor, get_feature_names

option_rows = [
    {"option": name, "value": str(value), "default": DEFAULTS[name] == value, "rationale": text}
    for name, choices in OPTIONS.items()
    for value, text in choices.items()
]
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame(option_rows).set_index(["option", "value"]), "Preprocessing options and their rationale (default marked)")

preprocessor = build_preprocessor()              # median imputer, standard scaler, one-hot encoder, no clipping
display(preprocessor)                            # the branch structure as scikit-learn draws it
"""
    )
    code(
        r"""
with timer.stage("C.6 fit the preprocessor"):
    preprocessor.fit(X_train)                    # every statistic is learnt from the TRAINING rows
X_test, y_test, g_test, sector_test = X.iloc[test_idx], y[test_idx], groups[test_idx], sector[test_idx]
Xt_train = preprocessor.transform(X_train)
Xt_test = preprocessor.transform(X_test)
names = get_feature_names(preprocessor)

n_num, n_bin = len(numeric_cols), len(binary_cols)
n_cat = Xt_train.shape[1] - n_num - n_bin
widths = pd.DataFrame(
    {
        "input columns": [n_num, len(categorical_cols), n_bin, X.shape[1]],
        "output columns": [n_num, n_cat, n_bin, Xt_train.shape[1]],
    },
    index=["numeric (impute + scale)", "categorical (one-hot)", "binary (pass-through)", "total"],
)
show_table(widths, "Width of the design matrix")
print(f"training matrix {Xt_train.shape}, test matrix {Xt_test.shape}, dtype {Xt_train.dtype}")
print("missing values after transformation (train / test):", int(np.isnan(Xt_train).sum()), "/", int(np.isnan(Xt_test).sum()))
print("one-hot columns that are 'infrequent' buckets:", sum(n.endswith("_infrequent_sklearn") for n in names))

# Proof that the fitted statistics come from the training rows: medians of the imputer, mean and scale of the scaler.
numeric_branch = preprocessor.named_transformers_["num"]
learnt = pd.DataFrame(
    {
        "imputer median": numeric_branch["imputer"].statistics_,
        "training median": X_train[numeric_cols].median().to_numpy(),
        "scaler mean": numeric_branch["scaler"].mean_,
        "scaler std": np.sqrt(numeric_branch["scaler"].var_),
    },
    index=numeric_cols,
)
assert np.allclose(learnt["imputer median"], learnt["training median"], rtol=1e-4)
show_table(learnt.head(8), "Statistics learnt on the training rows (first 8 numeric inputs)")
check = pd.DataFrame(
    {"mean": Xt_train[:, :n_num].mean(axis=0), "std": Xt_train[:, :n_num].std(axis=0)}, index=numeric_cols
).describe().loc[["min", "max"]]
show_table(check, "Scaled numeric training columns: range of the column means and standard deviations (about 0 and 1)")
"""
    )
    code(
        r"""
from mlda.models import MODEL_ZOO

overrides = pd.DataFrame(
    {"preprocessing": {name: str(spec.preprocess_overrides or "default") for name, spec in MODEL_ZOO.items()}}
)
show_table(overrides, "Model-specific preprocessing (trees and boosting skip the scaler)")
"""
    )
    md(
        """
## C.7 Do the preprocessing choices matter?

The defaults come from the data profile, but a choice that is only argued should also be tested.
The cell compares seven preprocessing variants with logistic regression, which is the model most
sensitive to scaling and outliers. The comparison uses a 12,000-row sample of **training mine-years**
and three grouped, stratified folds, scored by average precision, so no test row is involved.
The k-nearest-neighbour imputer was left out because it costs far more than the others on
100,000 rows. Differences smaller than the fold-to-fold standard deviation should not be read as
a ranking.
"""
    )
    code(
        r"""
from sklearn.model_selection import cross_val_score

from mlda.models import build_pipeline
from mlda.split import cv_splitter
from mlda.tuning import stratified_group_subsample

sample = stratified_group_subsample(y_train, g_train, 12_000)
variants = {
    "median + standard scaler + one-hot (chosen)": {},
    "mean imputer": {"numeric_imputer": "mean"},
    "robust scaler": {"scaler": "robust"},
    "min-max scaler": {"scaler": "minmax"},
    "IQR clipping": {"outlier_rule": "iqr_clip"},
    "Z-score clipping": {"outlier_rule": "zscore_clip"},
    "ordinal instead of one-hot": {"encoder": "ordinal"},
}
ablation_rows = []
with timer.stage("C.7 preprocessing ablation"):
    for label, options in variants.items():
        pipeline = build_pipeline(MODEL_ZOO["logreg"], build_preprocessor(**options))
        scores = cross_val_score(
            pipeline, X_train.iloc[sample], y_train[sample], groups=g_train[sample],
            cv=cv_splitter(3), scoring="average_precision", n_jobs=3,
        )
        ablation_rows.append({"variant": label, "mean AP": scores.mean(), "std over folds": scores.std(ddof=1)})
ablation = pd.DataFrame(ablation_rows).set_index("variant")
show_table(ablation, f"Logistic regression, {len(sample):,} training rows, 3 grouped folds: average precision")
"""
    )
    md(
        """
**Reading the table.** The imputer, the scaler and the clipping rule change the score by less than
one fold standard deviation for logistic regression, so none of them earns extra complexity and the
defaults stay: the error rules have already removed the impossible values, which is why clipping
adds nothing. The encoder is the exception. Replacing one-hot columns by integer codes costs
several points of average precision, because the codes impose an order that a linear model reads
as magnitude. One-hot encoding is therefore used for every model in the zoo. The ordinal option was
compared with logistic regression only; the tree models also receive the one-hot matrix, and
whether integer codes would suit them better was not tested.
"""
    )
    md(
        """
## C.8 The grouped, stratified split

A random split would put the rows of one mine-year on both sides, and most rows have a sibling
from the same mine-year (the cell prints the share) whose outcome the model could simply remember. The split therefore keeps
every **mine-year** (`MINE_ID` + year) together and stratifies on the label:
`StratifiedGroupKFold(5, shuffle=True, random_state=42)` takes one fold of mine-years as the 20 %
test set. The indices are persisted in `data/processed/split_indices.json`, so the notebook, the
command-line run and the app all use the same rows. The cell checks the invariants (disjoint
mine-years, no multi-victim accident divided, prevalence within 1 percentage point) and rebuilds
the split from scratch to confirm that it is reproducible.
"""
    )
    code(
        r"""
from mlda.split import check_split, make_split

accident_groups = meta["accident_group"].to_numpy(dtype=str)
print(f"rows that share their mine-year with at least one other row: {pd.Series(groups).duplicated(keep=False).mean():.1%}")
check_split(split, y, groups, accident_groups)      # raises AssertionError if any invariant fails

straddling_years = len(np.intersect1d(groups[train_idx], groups[test_idx]))
straddling_accidents = len(np.intersect1d(accident_groups[train_idx], accident_groups[test_idx]))
gap_pp = {name: 100 * abs(y[idx].mean() - y.mean()) for name, idx in (("train", train_idx), ("test", test_idx))}
checks = pd.DataFrame(
    {
        "check": ["row positions shared by train and test", "mine-years in both sets", "accident groups (mine + date) in both sets",
                  "train prevalence gap (pp)", "test prevalence gap (pp)", "tolerance (pp)"],
        "value": [len(np.intersect1d(train_idx, test_idx)), straddling_years, straddling_accidents,
                  round(gap_pp["train"], 3), round(gap_pp["test"], 3), 1.0],
    }
).set_index("check")
show_table(checks, "check_split: invariants of the persisted split (all passed)")

summary = pd.DataFrame(
    {
        "rows": [len(train_idx), len(test_idx)],
        "share of rows (%)": [100 * len(train_idx) / len(y), 100 * len(test_idx) / len(y)],
        "mine-years": [len(np.unique(groups[train_idx])), len(np.unique(groups[test_idx]))],
        "mines": [meta["MINE_ID"].iloc[train_idx].nunique(), meta["MINE_ID"].iloc[test_idx].nunique()],
        "lost-time rate": [y_train.mean(), y_test.mean()],
        "coal share": [(sector[train_idx] == "C").mean(), (sector_test == "C").mean()],
    },
    index=["training", "test"],
)
show_table(summary, "Training and test fold")

rebuilt = make_split(X, y, groups, accident_groups=accident_groups)
print("split rebuilt from the seed equals the persisted split:",
      bool(np.array_equal(rebuilt.train_idx, split.train_idx) and np.array_equal(rebuilt.test_idx, split.test_idx)))
print(f"X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")
"""
    )
    md(
        """
**Why mine-years and not rows or mines.** Section 3.9 of the report rests on three measurements,
computed here on all in-scope rows. (1) The mean label of the rows of the same mine-year filed on
a *later* date ranks a row by ROC-AUC: a random row split would hand the model these outcomes.
(2) Four planned numeric inputs (coordinates and the previous year's hours and employees) are
almost a mine-year fingerprint, so KNN and tree ensembles could memorise the mine-year. (3) A
split grouped by mine alone would test only unseen mines, yet most reports of 2020-2025 come from
mines that had already filed before 2020. The labels serve only to measure the leak; none of these
quantities is a model input.
"""
    )
    code(
        r"""
from sklearn.metrics import roc_auc_score

# (1) mean label of the later-dated rows of the same mine-year
by_day = pd.DataFrame({"g": groups, "day": base["ACCIDENT_DT"].to_numpy(), "y": y})
per_day = (
    by_day.groupby(["g", "day"])["y"].agg(["sum", "count"]).reset_index()
    .sort_values(["g", "day"], ascending=[True, False])
)
per_day["later_sum"] = per_day.groupby("g")["sum"].cumsum() - per_day["sum"]
per_day["later_n"] = per_day.groupby("g")["count"].cumsum() - per_day["count"]
siblings = by_day.merge(per_day[["g", "day", "later_sum", "later_n"]], on=["g", "day"], how="left", validate="many_to_one")
has_later = siblings["later_n"].to_numpy() > 0
sibling_score = (siblings["later_sum"] / siblings["later_n"]).to_numpy()[has_later]
sibling_auc = roc_auc_score(siblings["y"].to_numpy()[has_later], sibling_score)

# (2) share of rows whose (latitude, longitude, prev_hours, prev_employees) tuple belongs to one mine-year only
fingerprint = X[["latitude", "longitude", "prev_hours", "prev_employees"]].astype(str).agg("|".join, axis=1).to_numpy()
mine_years_per_tuple = pd.Series(groups).groupby(fingerprint).nunique()
identified = float(pd.Series(fingerprint).map(mine_years_per_tuple).eq(1).mean())

# (3) share of 2020-2025 rows at mines that filed at least one in-scope report in 2010-2019
years = pd.to_numeric(base["CAL_YR"]).to_numpy()
mine_ids = base["MINE_ID"].astype(str).to_numpy()
seen_share = float(np.isin(mine_ids[years >= 2020], np.unique(mine_ids[years < 2020])).mean())

leak_checks = pd.DataFrame(
    {
        "value": [sibling_auc, int(has_later.sum()), identified, len(np.unique(fingerprint)), len(np.unique(groups)), seen_share],
    },
    index=[
        "ROC-AUC of the later-dated siblings' mean label",
        "rows with at least one later-dated sibling",
        "share of rows whose four-feature tuple identifies one mine-year",
        "distinct four-feature tuples",
        "mine-years",
        "share of 2020-2025 rows at mines seen in 2010-2019",
    ],
)
show_table(leak_checks, "Three measurements behind the mine-year grouping (all in-scope rows)")
print(f"later-dated siblings rank a row with ROC-AUC {sibling_auc:.3f}; the four-feature tuple identifies the mine-year "
      f"for {identified:.1%} of rows; {seen_share:.1%} of 2020-2025 rows come from mines seen before 2020")
"""
    )


# ---------------------------------------------------------------------------
# D. Model training
# ---------------------------------------------------------------------------


def section_d() -> None:
    """Section D: model zoo, hyper-parameter search (loaded, or repeated with FULL_RUN=1) and the live refit of each selected pipeline."""
    md(
        """
# D. Model training

**Protocol.** The test fold stays untouched. Hyper-parameters are chosen by **5-fold
cross-validation on the training rows**, with the same `StratifiedGroupKFold` as the split
(mine-years never straddle a fold, class share preserved). The search is a `GridSearchCV` where
the grid is small, and a `RandomizedSearchCV` for the random forest, whose grid is large and each
fit expensive. Seven metrics are recorded for every candidate and the search **refits on average
precision** (the area under the precision-recall curve), which is the primary metric here: it
rewards a good ranking of the positive class and does not depend on a threshold. Training and
validation scores are both kept, because the gap between them is the diagnostic for
**overfitting** (training much better than validation) and **underfitting** (both poor).

**Models.** Eight classifiers from the module, from simple to flexible, so that the extra
flexibility has to earn its keep: logistic regression, k-nearest neighbours, Gaussian naive
Bayes, decision tree, random forest, histogram gradient boosting, an RBF support vector machine
and a multi-layer perceptron. Class imbalance is not a problem here (48.6 % positive), so no
resampling is used; `class_weight` is offered to logistic regression as a hyper-parameter to
test whether re-weighting helps.
"""
    )
    code(
        r"""
from mlda.config import CV_DIR
from mlda.models import MODEL_ZOO, build_pipeline, default_preprocessor
from mlda.persistence import read_manifest
from mlda.split import cv_splitter
from mlda.tuning import default_n_jobs, estimator_params, fit_cap, stratified_group_subsample, tune

# NOTEBOOK_MODELS (comma-separated names) restricts the zoo for quick development checks; unset for the graded run.
requested = [m for m in os.environ.get("NOTEBOOK_MODELS", "").split(",") if m]
MODEL_NAMES = [m for m in MODEL_ZOO if not requested or m in requested]
N_JOBS = default_n_jobs()
cv = cv_splitter()
print("cross-validation scheme:", cv)

zoo_rows = []
for name in MODEL_NAMES:
    spec = MODEL_ZOO[name]
    cap = fit_cap(spec, quick=False)
    zoo_rows.append(
        {
            "model": name,
            "taught (week)": spec.taught_in_week,
            "pipeline": " -> ".join(f"{step} ({type(part).__name__})" for step, part in build_pipeline(spec, default_preprocessor(spec)).steps),
            "search": f"{spec.search} search, {spec.n_candidates} candidates",
            "parameter grid": "; ".join(f"{key.split('__')[-1]}: {values}" for key, values in spec.param_grid.items()),
            "preprocessing": str(spec.preprocess_overrides or "default"),
            "rows fitted": f"{cap:,} (whole mine-years, stratified)" if cap else "all training rows",
        }
    )
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame(zoo_rows).set_index("model"), "Model zoo: pipeline, search type, grid and fitting rows")
    show_table(pd.DataFrame({"description": {n: MODEL_ZOO[n].description for n in MODEL_NAMES}}), "What each model is")
note("**The pipeline of the first model** (every model has the same two steps: the preprocessing that is fitted inside each fold, then the classifier):")
display(build_pipeline(MODEL_ZOO[MODEL_NAMES[0]], default_preprocessor(MODEL_ZOO[MODEL_NAMES[0]])))
"""
    )
    md(
        """
## D.1 Hyper-parameter search

The search is `mlda.tuning.tune`: a `GridSearchCV` (a `RandomizedSearchCV` for the random forest)
over the grid of the model, five grouped folds, seven metrics with training scores, a refit on
average precision, and then out-of-fold probabilities on the same folds, from which the decision
thresholds of section E are chosen. The first cell builds the search object for one model in the
way `tune` does and displays it, so that the set-up can be read: the pipeline that is cloned for
every candidate and fold, the grid, the scorers and the refit metric.

With `FULL_RUN=1` the second cell runs `tune` for every model (about an hour). In the default mode
it runs `tune` live for the three models whose search is cheap (Gaussian naive Bayes, logistic
regression and the decision tree; a few minutes on eight workers) and compares each result with
the table that `uv run mlda-train` stored, which was produced with exactly the same code, folds
and seed. The tree and naive Bayes reproduce the stored scores exactly on any machine; the lbfgs
solver of logistic regression differs in the last decimals between BLAS builds (the Windows and
Linux wheels of NumPy and SciPy), so mean validation scores are compared within 0.001, a third of
the smallest fold standard deviation, and the stored choice must lie within that tolerance of the
live best. For the other five models it loads the stored table of every candidate with its
per-fold scores (`reports/cv/<model>_cv_results.csv`) and the selected parameters
(`models/manifest.json`). Section D.3 refits every selected pipeline live.
"""
    )
    code(
        r"""
from sklearn.model_selection import GridSearchCV

from mlda.tuning import REFIT_METRIC, SCORING

grid_models = [name for name in MODEL_NAMES if MODEL_ZOO[name].search == "grid"]
if grid_models:
    demo = MODEL_ZOO["logreg" if "logreg" in grid_models else grid_models[0]]
    search_setup = GridSearchCV(
        build_pipeline(demo, default_preprocessor(demo)), demo.param_grid,
        scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS, return_train_score=True, error_score="raise",
    )
    fits = demo.n_candidates * cv.get_n_splits()
    note(f"**The search that `tune` builds for `{demo.name}`:** {demo.n_candidates} candidates x {cv.get_n_splits()} folds = {fits} fits, "
         f"scored on {len(SCORING)} metrics, refitted on `{REFIT_METRIC}`, with the training score kept for the overfitting diagnosis.")
    display(search_setup)
    print("scorers:", ", ".join(SCORING), "| refit on:", REFIT_METRIC)
"""
    )
    code(
        r"""
def read_cv_table(name: str) -> pd.DataFrame:
    # Restore the parameter columns with their real types (None, int, tuple) from the JSON in 'params'.
    frame = pd.read_csv(CV_DIR / f"{name}_cv_results.csv")
    parsed = frame["params"].map(json.loads)
    for key in parsed.iloc[0]:
        values = [tuple(p[key]) if isinstance(p[key], list) else p[key] for p in parsed]
        frame[f"param_{key}"] = pd.Series(values, index=frame.index, dtype="object")   # object keeps None as None, not NaN
    return frame


def as_tuple_params(params: dict) -> dict:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in params.items()}


LIVE_SEARCH = [name for name in ("gaussian_nb", "logreg", "decision_tree") if name in MODEL_NAMES]   # the cheap searches
manifest = read_manifest()
tuned = {}
for name in MODEL_NAMES if FULL_RUN else LIVE_SEARCH:
    with timer.stage(f"D.1 tune {name}"):
        tuned[name] = tune(MODEL_ZOO[name], X_train, y_train, g_train, cv, n_jobs=N_JOBS)
    print(f"{name:14s} best average precision {tuned[name].cv_results_df.iloc[0]['mean_test_average_precision']:.4f} "
          f"in {tuned[name].elapsed_seconds:.0f} s: {estimator_params(tuned[name])}")

cv_tables, best_params = {}, {}
for name in MODEL_NAMES:
    if name in tuned:                       # searched in this notebook
        cv_tables[name], best_params[name] = tuned[name].cv_results_df, tuned[name].best_params
    else:                                   # written by mlda-train with the same code, folds and seed
        cv_tables[name] = read_cv_table(name)
        best_params[name] = as_tuple_params(manifest[name]["best_params"])

quick_flags = [name for name in MODEL_NAMES if manifest[name]["extra"].get("quick")]
if quick_flags:
    print("NOTE: the stored results of", quick_flags, "come from a quick run (two candidates, 3 folds); "
          "run `uv run mlda-train` for the full search")
print(f"searched in this notebook: {', '.join(tuned) or 'none'}; loaded from reports/cv and models/manifest.json: "
      f"{', '.join(n for n in MODEL_NAMES if n not in tuned) or 'none'}")
SCORE_TOLERANCE = 1e-3                      # a third of the smallest fold standard deviation in the stored tables
stored_environment = json.loads((METRICS_DIR / "run_manifest.json").read_text(encoding="utf-8"))["environment"]
agreement = {}
for name in tuned:                          # the live search should reproduce the stored table, candidate by candidate
    live, stored = tuned[name].cv_results_df, read_cv_table(name)
    both = live.merge(stored, on="params", suffixes=("", "_stored"), validate="one_to_one")
    score_columns = [c for c in live.columns if c.startswith("mean_test_")]
    stored_choice = as_tuple_params(manifest[name]["best_params"])
    stored_choice_plain = {key.rsplit("__", 1)[-1]: value for key, value in stored_choice.items()}   # the params column has no step prefix
    is_stored_choice = [as_tuple_params(json.loads(p)) == stored_choice_plain for p in live["params"]]
    live_best = float(live["mean_test_average_precision"].max())
    live_score_of_stored_choice = (float(live.loc[is_stored_choice, "mean_test_average_precision"].max())
                                   if any(is_stored_choice) else float("nan"))
    agreement[name] = {
        "candidates (live)": len(live),
        "candidates (stored)": len(stored),
        "same candidates": len(both) == len(live) == len(stored),
        "largest difference in a mean validation score": max(float((both[c] - both[f"{c}_stored"]).abs().max()) for c in score_columns),
        "same selected parameters": stored_choice == tuned[name].best_params,
        "stored choice within tolerance of the live best": live_best - live_score_of_stored_choice < SCORE_TOLERANCE,
    }
agreement = pd.DataFrame(agreement).T
show_table(agreement, "The live search against the table that mlda-train stored")
print(agreement.to_string())
print(f"stored tables: {stored_environment['platform']}, numpy {stored_environment['numpy_version']}; "
      f"this run: {platform.platform()}, numpy {np.__version__}")
reproduced = bool(
    agreement["same candidates"].astype(bool).all()
    and (agreement["largest difference in a mean validation score"].astype(float) < SCORE_TOLERANCE).all()
    and agreement["stored choice within tolerance of the live best"].astype(bool).all()
)
print(f"every search run in this notebook reproduces the stored table and the selected parameters within {SCORE_TOLERANCE}: {reproduced}")
if not FULL_RUN:                            # the same code, folds and seed: a difference beyond floating-point drift is an error
    assert reproduced, "the live search differs from the stored table"
"""
    )
    md(
        """
**Reading the result.** The first table lists, for each model, the selected parameters and the
mean and standard deviation over the five validation folds of every metric. A standard deviation
that is small next to the gap between two models means the ranking is stable; one that is as
large as the gap means the two cannot be separated on these folds. The second table shows the
three best candidates of each model with the **train minus validation** average precision: a
large positive gap marks a candidate that has memorised the training folds (typical of an
unpruned tree or a very flexible forest), and the selected candidate is usually not the one with
the smallest gap but the one with the best validation score.
"""
    )
    code(
        r"""
from mlda.evaluation import summarise_cv

cv_summary = summarise_cv(pd.concat(cv_tables.values(), ignore_index=True))
cv_summary = cv_summary.loc[[m for m in MODEL_NAMES if m in cv_summary.index]]
cv_summary["neg_brier_score_mean"] = -cv_summary["neg_brier_score_mean"]   # report the Brier score itself (lower is better)

labels = {
    "average_precision": "PR-AUC (AP)", "roc_auc": "ROC-AUC", "f1": "F1", "recall": "recall", "precision": "precision",
    "accuracy": "accuracy", "neg_brier_score": "Brier",
}
pretty = pd.DataFrame(index=cv_summary.index)
for key, label in labels.items():
    pretty[label] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(cv_summary[f"{key}_mean"], cv_summary[f"{key}_std"], strict=True)]
pretty.insert(0, "selected parameters", [str({k.split("__")[-1]: v for k, v in best_params[n].items()}) for n in pretty.index])
pretty = pretty.loc[cv_summary["average_precision_mean"].sort_values(ascending=False).index]
with pd.option_context("display.max_colwidth", None):
    show_table(pretty, "5-fold grouped cross-validation on the training rows: mean +/- standard deviation (best candidate per model)")

top_rows = []
for name in MODEL_NAMES:
    for rank, (_, row) in enumerate(cv_tables[name].head(3).iterrows(), start=1):
        top_rows.append(
            {
                "model": name,
                "rank": rank,
                "parameters": str({k.split("__")[-1]: v for k, v in json.loads(row["params"]).items()}),
                "validation AP": row["mean_test_average_precision"],
                "std": row["std_test_average_precision"],
                "training AP": row["mean_train_average_precision"],
                "train - validation": row["mean_train_average_precision"] - row["mean_test_average_precision"],
            }
        )
with pd.option_context("display.max_colwidth", None, "display.max_rows", 40):
    show_table(pd.DataFrame(top_rows).set_index(["model", "rank"]), "Three best candidates per model and the overfitting gap")

gaps = pd.Series(
    {n: cv_tables[n].iloc[0]["mean_train_average_precision"] - cv_tables[n].iloc[0]["mean_test_average_precision"] for n in MODEL_NAMES}
).sort_values(ascending=False)
weakest = cv_summary["average_precision_mean"].idxmin()
note(
    f"**Overfitting diagnosis.** The train minus validation gap of the selected candidates runs from {gaps.iloc[-1]:.3f} (`{gaps.index[-1]}`) "
    f"to {gaps.iloc[0]:.3f} (`{gaps.index[0]}`). The weakest model on validation is `{weakest}` (average precision "
    f"{cv_summary.loc[weakest, 'average_precision_mean']:.3f}, gap {gaps[weakest]:.3f}): "
    + ("a small gap with a low score is underfitting, so the model is too restrictive for this data."
       if gaps[weakest] < 0.05 else "a large gap is overfitting, so the model memorises the training folds.")
)
if best_params.get("knn", {}).get("clf__weights") == "distance":
    note(f"**The gap of `knn` is not a fair measure.** The selected model weights neighbours by distance, so every training row is its own "
         f"nearest neighbour at distance zero and its training average precision is {cv_tables['knn'].iloc[0]['mean_train_average_precision']:.3f} by "
         f"construction. Compare its validation score with the others, not its gap.")

run_rows = {}
for name in MODEL_NAMES:
    stored = json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    run_rows[name] = {
        "candidates": stored["n_candidates"],
        "rows in the search": stored["n_fit_rows"],
        "folds": stored["cv_folds"],
        "search seconds (mlda-train)": stored["tuning_seconds"],
        "search seconds (this notebook)": tuned[name].elapsed_seconds if name in tuned else np.nan,
    }
show_table(pd.DataFrame(run_rows).T.astype({"candidates": int, "rows in the search": int, "folds": int}), "Cost of the search: stored by mlda-train (reports/metrics/<model>.json) and repeated here")
"""
    )
    md(
        """
## D.2 Validation curves and the learning curve

A **validation curve** plots the training and the validation score against one hyper-parameter
while the others stay at their selected values. Where the two curves diverge the model starts to
overfit; where both are low it underfits. The panels use the hyper-parameter with the most grid
values for each model (a model whose grid has a single value per parameter has no curve). The
**learning curve** of the best model plots both scores against the number of training rows: if
the validation curve is still rising, more data would help; if the curves have met, more data
would not. The curve is computed by `mlda-train` on a sample of the training rows, because a curve
at full size needs about twenty-five fits of the final pipeline; the cell states the sample size,
so the curve is read for its trend and not for its level. The cell after the figures quotes, for
every panel, the training and validation score at the selected value.
"""
    )
    code(
        r"""
from types import SimpleNamespace

from mlda import plots
from mlda.pipeline import validation_panel


def in_value_order(panel: dict) -> dict:
    # validation_panel keeps the order of the candidate table when a parameter mixes numbers with None
    # (max_depth: 12, None, 8, 4), which draws a zig-zag. Numbers ascending with None last reads as a curve.
    labels = [str(value) for value in panel["x"]]
    try:
        keys = [(label == "None", 0.0 if label == "None" else float(label)) for label in labels]
    except ValueError:                       # a categorical parameter keeps the order of the table
        return panel
    order = sorted(range(len(labels)), key=keys.__getitem__)
    return {**panel, **{key: np.asarray(panel[key])[order] for key in ("x", "train_mean", "train_std", "val_mean", "val_std")}}


panels = {}
for name in MODEL_NAMES:
    # validation_panel only needs the table of candidates and the selected parameters
    holder = SimpleNamespace(cv_results_df=cv_tables[name], best_params=best_params[name])
    panel = validation_panel(holder)
    if panel is not None:
        panels[name] = in_value_order(panel)
print("models with a validation curve:", ", ".join(panels))
if panels:
    show(plots.curves_grid(panels, kind="validation"), "Training and validation average precision against one hyper-parameter, one panel per model")
    leader = cv_summary["average_precision_mean"].idxmax()
    if leader in panels:                      # the single-curve builder, for the model ranked first
        p = panels[leader]
        note(f"**Validation curve of `{leader}`** against `{p['param_name']}`")
        show(plots.validation_curve(p["x"], p["train_mean"], p["train_std"], p["val_mean"], p["val_std"], p["param_name"], best_value=p["best_value"]),
             f"Training and validation average precision of {leader} against {p['param_name']}")

panel_rows = []
for model_name, p in panels.items():
    labels = [str(value) for value in p["x"]]
    chosen = labels.index(str(p["best_value"])) if str(p["best_value"]) in labels else int(np.argmax(p["val_mean"]))
    panel_rows.append(
        {
            "model": model_name,
            "parameter": p["param_name"],
            "selected value": labels[chosen],
            "training AP": p["train_mean"][chosen],
            "validation AP": p["val_mean"][chosen],
            "train - validation": p["train_mean"][chosen] - p["val_mean"][chosen],
            "best validation value on the curve": labels[int(np.argmax(p["val_mean"]))],
        }
    )
if panel_rows:
    panel_table = pd.DataFrame(panel_rows).set_index("model")
    show_table(panel_table, "The validation curves at the selected value")
    artefact = {"knn"} if best_params.get("knn", {}).get("clf__weights") == "distance" else set()   # training score 1.000 by construction
    fair = panel_table[~panel_table.index.isin(artefact)]
    widest = fair["train - validation"].idxmax()
    agree = int((panel_table["selected value"] == panel_table["best validation value on the curve"]).sum())
    points = pd.DataFrame(
        [
            {"model": m, "parameter": p["param_name"], "value": str(v), "training AP": t, "validation AP": a, "gap": t - a}
            for m, p in panels.items() if m not in artefact
            for v, t, a in zip(p["x"], p["train_mean"], p["val_mean"], strict=True)
        ]
    )
    top = points.loc[points["gap"].idxmax()]
    selected_there = panel_table.loc[top["model"], "selected value"] == top["value"]
    note(
        f"**Reading the curves.** In {agree} of {len(panel_table)} panels the selected value has the highest validation score on its curve; the other "
        f"parameters are held at the selected candidate's values in each panel. Setting aside `knn`, whose training score is an artefact of distance weighting "
        f"(see the note above), `{widest}` has the widest gap at its selected value of `{fair.loc[widest, 'parameter']}` ({fair.loc[widest, 'selected value']}): "
        f"training {fair.loc[widest, 'training AP']:.3f} against validation {fair.loc[widest, 'validation AP']:.3f}, so the validation score is the figure to quote. "
        f"The widest gap anywhere on these curves is {top['gap']:.3f} for `{top['model']}` at {top['parameter']} = {top['value']} (training {top['training AP']:.3f}, "
        f"validation {top['validation AP']:.3f})"
        + (", which is the selected value." if selected_there else
           f"; it was not selected, and its validation score is {top['validation AP']:.3f} against {panel_table.loc[top['model'], 'validation AP']:.3f} at the selected value.")
    )

run_info = json.loads((METRICS_DIR / "run_manifest.json").read_text(encoding="utf-8"))
for path in sorted(CV_DIR.glob("*_learning_curve.csv")):
    curve = pd.read_csv(path)
    curve_model = path.name.replace("_learning_curve.csv", "")
    sample_size = run_info.get("learning_curve", {}).get("n_rows")
    note(f"**Learning curve of `{curve_model}`** (stored by `mlda-train`): grouped folds on a sample of "
         f"{sample_size:,} of the {len(y_train):,} training rows, so the largest training size is {int(curve['train_size'].max()):,} rows "
         f"per fit and not the size of the real fits." if sample_size else f"**Learning curve of `{curve_model}`** (stored by `mlda-train`)")
    show(plots.learning_curve(curve["train_size"], curve["train_mean"], curve["train_std"], curve["val_mean"], curve["val_std"]),
         "Training and validation average precision against the number of training rows")
    last, before = curve.iloc[-1], curve.iloc[-2]
    full_score = cv_summary.loc[curve_model, "average_precision_mean"] if curve_model in cv_summary.index else float("nan")
    note(
        f"From {int(before['train_size']):,} to {int(last['train_size']):,} rows the validation average precision goes from {before['val_mean']:.3f} to "
        f"{last['val_mean']:.3f} (standard deviation over folds {last['val_std']:.3f}), so the curve is "
        + ("still rising. " if last["val_mean"] - before["val_mean"] > last["val_std"] else "no longer rising by more than the spread between folds. ")
        + f"The training score at the last point is {last['train_mean']:.3f}, a gap of {last['train_mean'] - last['val_mean']:.3f}. "
        f"The five-fold score of the same model on all {len(y_train):,} training rows is {full_score:.3f}, "
        f"{abs(full_score - last['val_mean']):.3f} {'above' if full_score > last['val_mean'] else 'below'} the last point of the curve: "
        f"the sample shows the trend, not the level at full size."
    )
"""
    )
    md(
        """
## D.3 Refit the selected pipelines on the training split

Whatever the mode, every selected pipeline is **trained here**, on the training rows only. In the
default mode a fresh pipeline is built from the model zoo (`build_pipeline`, with the preprocessing
the model prefers), the selected parameters are set, and it is fitted. Models with a row cap
(the SVM and k-nearest neighbours) are fitted on the same stratified sample of whole mine-years
that the search used, because the SVM's training time grows faster than linearly with the number
of rows and k-nearest neighbours needs the distance from every scored row to every stored row.
The random forest, k-nearest neighbours and the SVM use several processes; the number of workers
does not change their results. The table times every fit.
"""
    )
    code(
        r"""
PARALLEL_FIT = {"random_forest", "knn", "svc_rbf"}      # n_jobs speeds these up without changing the fitted model
fitted, fit_seconds, fit_rows = {}, {}, {}
if FULL_RUN:
    for name in MODEL_NAMES:
        fitted[name] = tuned[name].best_pipeline            # already refitted by the search
        fit_seconds[name] = tuned[name].elapsed_seconds     # search plus refit
        fit_rows[name] = len(tuned[name].fit_idx)
else:
    with timer.stage("D.3 refit the selected pipelines"):
        for name in MODEL_NAMES:
            spec = MODEL_ZOO[name]
            cap = fit_cap(spec, quick=False)
            fit_idx = np.arange(len(y_train)) if cap is None else stratified_group_subsample(y_train, g_train, cap)
            pipeline = build_pipeline(spec, default_preprocessor(spec)).set_params(**best_params[name])
            if name in PARALLEL_FIT:
                pipeline.set_params(clf__n_jobs=N_JOBS)
            started = time.perf_counter()
            pipeline.fit(X_train.iloc[fit_idx], y_train[fit_idx])
            fit_seconds[name] = time.perf_counter() - started
            fit_rows[name] = len(fit_idx)
            fitted[name] = pipeline
            print(f"{name:14s} fitted on {len(fit_idx):6,d} rows in {fit_seconds[name]:6.1f} s")

fit_table = pd.DataFrame(
    {
        "estimator": [type(fitted[n][-1]).__name__ for n in MODEL_NAMES],
        "rows fitted": [fit_rows[n] for n in MODEL_NAMES],
        "seconds": [fit_seconds[n] for n in MODEL_NAMES],
        "design-matrix columns": [fitted[n][0].transform(X_train.iloc[:5]).shape[1] for n in MODEL_NAMES],
    },
    index=MODEL_NAMES,
)
show_table(fit_table, "Training time of each selected pipeline" + (" (search plus refit)" if FULL_RUN else " (refit only)"))
"""
    )


# ---------------------------------------------------------------------------
# E. Model evaluation
# ---------------------------------------------------------------------------


def section_e() -> None:
    """Section E: held-out evaluation, thresholds and costs, tests, success criteria, the model decision and robustness checks."""
    md(
        """
# E. Model evaluation

**Protocol.** Every fitted pipeline scores the held-out test fold once. Nothing in this section
is fitted on test rows, and no threshold is tuned on them: the decision thresholds were chosen on
the **out-of-fold probabilities of the training rows** (`thresholds.json`) and are applied to the
test fold unchanged. Besides the models, four reference rules are scored the same way, so the
questions are which model is best and whether any model beats what a safety officer could do by
hand. The lookup baseline predicts the training lost-time rate of the cell
nature of injury x body part x commodity.

**Metrics.** The brief's columns are **TPR** (true positive rate = **recall** = TP / (TP + FN),
the share of lost-time cases that are caught), **TNR** (true negative rate = specificity = TN /
(TN + FP)), **AUC** (area under the ROC curve, threshold-free), **precision** (TP / (TP + FP)),
**F1** (harmonic mean of precision and recall), **PR-AUC** (average precision, the primary
metric) and **accuracy**. Balanced accuracy, the Matthews correlation coefficient (MCC) and the
Brier score (mean squared error of the probabilities, a calibration measure) are added. TPR and
Recall are the same number; both columns are shown because the brief lists both.

**Two operating points.** At the default threshold of 0.5 the classifier is treated as a plain
yes/no rule. The business setting is different: a missed lost-time case costs far more than an
unnecessary review (assumed USD 4,000 against USD 200, a 20:1 ratio), and reviewers can handle
about three of every four cases. The **capacity operating point** is therefore the out-of-fold
threshold at which **specificity is 0.40** (about 75 % of cases are flagged); the success
criteria S1-S6 were fixed before modelling and are tested at that point.
"""
    )
    md(
        """
## E.1 Score the test fold

Each fitted pipeline predicts the probability of lost time for the 20,320 test rows, and the four
reference rules do the same. The first table is a reproducibility check: it sets the scores
computed here against the test predictions that `mlda-train` stored. They agree when the stored run
used the same fitting rows and parameters; differences of this kind are the first thing to look
at if a number in the report and a number in the notebook disagree.
"""
    )
    code(
        r"""
from mlda.baselines import fit_baselines
from mlda.config import COST_RATIOS, METRICS_DIR, PREDICTIONS_DIR
from mlda.evaluation import (
    bootstrap_ci,
    by_sector,
    classification_metrics,
    compare_models,
    cost_from_counts,
    curves,
    mcnemar_test,
    paired_fold_test,
    threshold_for_flag_rate,
    threshold_sweep,
)
from mlda.operating import (
    FLAG_KEY,
    LOOKUP_BASELINE,
    OPERATING_KEY,
    all_positive_floor,
    evaluate_on_test,
    model_criteria,
    oof_thresholds,
)


def predict_proba_chunks(pipeline, frame, workers=N_JOBS):
    # Score row chunks in threads: libsvm and numpy release the GIL, and the result equals one predict_proba call.
    pieces = np.array_split(np.arange(len(frame)), workers * 2)
    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(lambda ix: pipeline.predict_proba(frame.iloc[ix])[:, 1], pieces))
    return np.concatenate(parts)


THREADED_PREDICT = {"svc_rbf"}                  # the SVM needs about 100 s for 20,000 rows on one thread
proba, predict_seconds = {}, {}
with timer.stage("E.1 score the test fold"):
    for name in MODEL_NAMES:
        started = time.perf_counter()
        if name in THREADED_PREDICT:
            proba[name] = predict_proba_chunks(fitted[name], X_test)
        else:
            proba[name] = fitted[name].predict_proba(X_test)[:, 1]
        predict_seconds[name] = time.perf_counter() - started
    if "svc_rbf" in MODEL_NAMES:                # chunked scoring must not change the numbers
        sample_rows = X_test.iloc[:300]
        assert np.allclose(predict_proba_chunks(fitted["svc_rbf"], sample_rows), fitted["svc_rbf"].predict_proba(sample_rows)[:, 1])

baselines = fit_baselines(X_train, y_train)     # the four reference rules, fitted on the training rows only
for name, model in baselines.items():
    proba[name] = model.predict_proba(X_test)[:, 1]
ALL_NAMES = [*MODEL_NAMES, *baselines]

stored_thresholds = json.loads((METRICS_DIR / "thresholds.json").read_text(encoding="utf-8"))
thresholds = {name: stored_thresholds[name] for name in ALL_NAMES}
if FULL_RUN:                                    # thresholds from the out-of-fold probabilities of this notebook's own search
    for name in MODEL_NAMES:
        thresholds[name] = oof_thresholds(y_train, tuned[name].oof_proba)

evaluations = {name: evaluate_on_test(y_test, proba[name], sector_test, thresholds[name]) for name in ALL_NAMES}

# Reproducibility: the live scores against the test predictions that mlda-train stored.
stored_predictions = pd.read_parquet(PREDICTIONS_DIR / "test_predictions.parquet")
assert np.array_equal(stored_predictions["row"].to_numpy(), test_idx)
stored_baselines = json.loads((METRICS_DIR / "baselines.json").read_text(encoding="utf-8"))


def stored_auc(name):
    if name in baselines:
        return stored_baselines[name]["test"]["at_0.5"]["roc_auc"]
    return json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))["test"]["at_0.5"]["roc_auc"]


reproduce = pd.DataFrame(
    {
        name: {
            "ROC-AUC (this notebook)": evaluations[name]["at_0.5"]["roc_auc"],
            "ROC-AUC (stored by mlda-train)": stored_auc(name),
            "largest |p difference|": float(np.abs(proba[name] - stored_predictions[f"proba_{name}"].to_numpy()).max()),
        }
        for name in ALL_NAMES
    }
).T
reproduce["largest |p difference|"] = reproduce["largest |p difference|"].map("{:.1e}".format)
show_table(reproduce, "Live test scores against the stored predictions (identical when the stored run used the same fit rows)")
show_table(
    pd.DataFrame({f"seconds to score {len(X_test):,} test rows": predict_seconds}),
    "Scoring time",
)
"""
    )
    md(
        """
## E.2 Comparison at threshold 0.5

The table is in the order the brief asks for: TPR, TNR, AUC, Recall, Precision, F1, PR-AUC and
Accuracy, sorted by PR-AUC, with the models and the four reference rules together. A model is
only worth deploying if it beats the lookup rule, which needs no training beyond a table of
rates, and the two trivial rules show what accuracy and F1 mean when nothing is learnt (the
all-positive rule has an F1 of about 0.65 without looking at a single feature). The second table
adds balanced accuracy, MCC and Brier. The third compares the cross-validated estimate with the
test score: a test score far below the cross-validation mean would indicate overfitting to the
training folds, and a close match shows that the selection procedure was honest.
"""
    )
    code(
        r"""
lookup_name = LOOKUP_BASELINE
at_half = compare_models({name: evaluations[name]["at_0.5"] for name in ALL_NAMES})
show_table(at_half, "Held-out test fold, threshold 0.5 (columns in the order of the brief; sorted by PR-AUC)")
note(f"*Footnote: {at_half.attrs['footnote']}*")

extras = compare_models({name: evaluations[name]["at_0.5"] for name in ALL_NAMES}, extra_columns=True)[
    ["Balanced accuracy", "MCC", "Brier", "Log loss", "Flag rate"]
]
show_table(extras, "Further metrics at threshold 0.5 (Brier: lower is better)")

ranking = list(cv_summary["average_precision_mean"].sort_values(ascending=False).index)   # selection uses training folds only
best = ranking[0]
generalisation = pd.DataFrame(
    {
        "CV average precision (mean)": cv_summary["average_precision_mean"],
        "CV std over folds": cv_summary["average_precision_std"],
        "test PR-AUC": [evaluations[n]["at_0.5"]["pr_auc"] for n in cv_summary.index],
    },
    index=cv_summary.index,
)
generalisation["test minus CV"] = generalisation["test PR-AUC"] - generalisation["CV average precision (mean)"]
show_table(generalisation.loc[ranking], "Cross-validation estimate against the held-out test score")
print(f"model ranked first by cross-validated average precision: {best}")

by_pr = at_half["PR-AUC"].drop(["majority", "all_positive"])
beating = [n for n in MODEL_NAMES if at_half.loc[n, "PR-AUC"] > at_half.loc[lookup_name, "PR-AUC"]]
note(
    f"**Reading the tables.** `{by_pr.index[0]}` has the highest test PR-AUC ({by_pr.iloc[0]:.3f}) and ROC-AUC "
    f"{at_half.loc[by_pr.index[0], 'AUC']:.3f}. {len(beating)} of the {len(MODEL_NAMES)} models beat the lookup rule on PR-AUC "
    f"({at_half.loc[lookup_name, 'PR-AUC']:.3f}); the others are {', '.join(f'`{n}`' for n in MODEL_NAMES if n not in beating) or 'none'}. "
    f"Plain accuracy at 0.5 ranges from {at_half.loc[MODEL_NAMES, 'Accuracy'].min():.3f} to {at_half.loc[MODEL_NAMES, 'Accuracy'].max():.3f} across the models, "
    f"against {at_half.loc['majority', 'Accuracy']:.3f} for the majority rule. The largest difference between the cross-validation estimate and the test score is "
    f"{generalisation['test minus CV'].abs().max():.3f} (`{generalisation['test minus CV'].abs().idxmax()}`)."
)
"""
    )
    md(
        """
## E.3 Operating point at specificity 0.40

The same metrics at the capacity operating point: the threshold is the one that gave a
specificity of 0.40 on the out-of-fold training probabilities, and the test fold shows what that
threshold does on unseen rows. The specificity actually reached on the test fold should be close
to 0.40; recall is the share of lost-time cases caught when about three of four cases are
reviewed. The two constant rules are left out, because a rule that gives every case the same
score has no threshold that reaches a given specificity.
"""
    )
    code(
        r"""
informative = [name for name in ALL_NAMES if name not in ("majority", "all_positive")]
at_operating = compare_models(
    {name: evaluations[name][f"at_{OPERATING_KEY}"] for name in informative}, extra_columns=True
)
show_table(at_operating, "Test fold at the out-of-fold threshold for specificity 0.40 (sorted by PR-AUC)")
model_rows = at_operating.loc[MODEL_NAMES]

# The thresholds above flag different shares of the cases, so recall is also compared at one common flag rate:
# the cut-off is taken from the scores alone (the top share of test scores), and the labels only count the hits.
common_rate = float(model_rows["Flag rate"].median())
at_common = pd.DataFrame(
    {
        name: classification_metrics(y_test, proba[name], threshold_for_flag_rate(y_test, proba[name], common_rate))
        for name in MODEL_NAMES
    }
).T[["flag_rate", "recall", "specificity", "precision"]]
at_common.columns = ["Flag rate", "TPR", "TNR", "Precision"]
show_table(at_common.sort_values("TPR", ascending=False), f"Models at the same review workload: the {common_rate:.1%} of test cases with the highest scores are flagged")
note(
    f"**Reading the table.** On the test fold the specificity of the models at this threshold runs from {model_rows['TNR'].min():.3f} to "
    f"{model_rows['TNR'].max():.3f} (target 0.40), and the share of cases flagged from {model_rows['Flag rate'].min():.1%} to {model_rows['Flag rate'].max():.1%}. "
    f"Recall runs from {model_rows['TPR'].min():.3f} (`{model_rows['TPR'].idxmin()}`) to {model_rows['TPR'].max():.3f} (`{model_rows['TPR'].idxmax()}`), "
    f"and the lookup rule reaches {at_operating.loc[lookup_name, 'TPR']:.3f} at a flag rate of {at_operating.loc[lookup_name, 'Flag rate']:.1%}. "
    f"The flag rates differ (`{model_rows['Flag rate'].idxmax()}` flags {model_rows['Flag rate'].max():.1%} and `{model_rows['Flag rate'].idxmin()}` "
    f"{model_rows['Flag rate'].min():.1%}), so these recalls do not compare models at equal workload. At the same workload ({common_rate:.1%} flagged) recall runs from "
    f"{at_common['TPR'].min():.3f} (`{at_common['TPR'].idxmin()}`) to {at_common['TPR'].max():.3f} (`{at_common['TPR'].idxmax()}`); that is the fairer measure of "
    f"how many more lost-time cases one model catches than another for the same review effort."
)
"""
    )
    md(
        """
## E.4 ROC and precision-recall curves

The **ROC curve** plots the true positive rate against the false positive rate (1 - TNR) as the
threshold moves; the area under it is the probability that a random lost-time case is scored
above a random other case, and the diagonal is a coin flip. The **precision-recall curve** shows
precision against recall; because the classes are balanced its baseline is the prevalence (0.49),
and it shows more clearly than the ROC curve what a high recall costs in precision. Models that
lie on top of each other cannot be separated by these plots; the numbers in E.2 and the tests in
E.9 decide.
"""
    )
    code(
        r"""
shown = [*ranking, LOOKUP_BASELINE]
model_curves = {name: curves(y_test, proba[name]) for name in ALL_NAMES}
show(plots.roc_curves({n: model_curves[n]["roc"] for n in shown}, {n: evaluations[n]["at_0.5"]["roc_auc"] for n in shown}),
     "ROC curves of the models and the lookup baseline on the test fold")
show(plots.pr_curves({n: model_curves[n]["pr"] for n in shown}, float(y_test.mean()), {n: evaluations[n]["at_0.5"]["pr_auc"] for n in shown}),
     "Precision-recall curves of the models and the lookup baseline on the test fold")
"""
    )
    md(
        """
## E.5 Confusion matrices

A confusion matrix counts the four outcomes of a decision rule: true negatives and true positives
on the diagonal, false positives (unnecessary reviews) and false negatives (missed lost-time
cases) off it. The percentages are row percentages, so the bottom-right cell is the TPR and the
top-left cell the TNR. The four panels compare the best model and the lookup baseline at the
default threshold and at the capacity operating point. Moving to the operating point trades
specificity for recall in both; the question is how much recall each model buys for the same
review workload.
"""
    )
    code(
        r"""
def confusion(metrics):
    return np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])


lookup = LOOKUP_BASELINE
matrices = {
    f"{plots.display_name(best)}\nthreshold 0.50": confusion(evaluations[best]["at_0.5"]),
    f"{plots.display_name(best)}\nspecificity 0.40 threshold {thresholds[best][OPERATING_KEY]:.3f}": confusion(evaluations[best][f"at_{OPERATING_KEY}"]),
    "Lookup baseline\nthreshold 0.50": confusion(evaluations[lookup]["at_0.5"]),
    f"Lookup baseline\nspecificity 0.40 threshold {thresholds[lookup][OPERATING_KEY]:.3f}": confusion(evaluations[lookup][f"at_{OPERATING_KEY}"]),
}
show(plots.confusion_matrices(matrices), "Four confusion matrices: the best model and the lookup baseline at threshold 0.5 and at the specificity 0.40 threshold")
constant_rules = ("majority", "all_positive")          # constant scores have no threshold for a given specificity
count_rows = {}
for name in ALL_NAMES:
    row = {("threshold 0.5", k.upper()): evaluations[name]["at_0.5"][k] for k in ("tn", "fp", "fn", "tp")}
    row.update(
        {("specificity 0.40", k.upper()): (np.nan if name in constant_rules else evaluations[name][f"at_{OPERATING_KEY}"][k]) for k in ("tn", "fp", "fn", "tp")}
    )
    count_rows[name] = row
count_table = pd.DataFrame(count_rows).T
count_table.columns = pd.MultiIndex.from_tuples(count_table.columns)
show_table(count_table.astype("Int64"), "Confusion counts of every model and rule on the test fold (TN, FP, FN, TP)")

at_op_best, at_op_lookup = evaluations[best][f"at_{OPERATING_KEY}"], evaluations[lookup][f"at_{OPERATING_KEY}"]
note(
    f"**Reading the matrices.** At the capacity operating point `{best}` flags {at_op_best['flag_rate']:.1%} of the cases and misses {at_op_best['fn']:,} of "
    f"{at_op_best['tp'] + at_op_best['fn']:,} lost-time cases ({1 - at_op_best['recall']:.1%}); the lookup rule flags {at_op_lookup['flag_rate']:.1%} and misses "
    f"{at_op_lookup['fn']:,} ({1 - at_op_lookup['recall']:.1%}). The model therefore catches {at_op_lookup['fn'] - at_op_best['fn']:,} more lost-time cases and causes "
    f"{abs(at_op_best['fp'] - at_op_lookup['fp']):,} {'fewer' if at_op_best['fp'] < at_op_lookup['fp'] else 'more'} unnecessary reviews."
)
"""
    )
    md(
        """
## E.6 Threshold, cost and the review-everyone floor

The expected cost per 1,000 cases of a decision rule is 1000 x (FN x C_FN + FP x C_FP) / N with
C_FP = USD 200 and C_FN = ratio x C_FP. The cost ratio is an assumption, so the analysis covers
2:1, 3:1, 5:1, 10:1, 20:1 and 30:1. A rule that **reviews every case** costs the same at every
ratio (it has no missed cases), and a classifier is worth using only where it costs less than that
floor. As the ratio grows, the cheapest policy moves towards reviewing every case, so the saving
over the floor shrinks and the cost curve of any model flattens onto it. The curve below shows the
cost against the threshold for each ratio, with the three operating points marked. The table gives, for each ratio, the cost at the out-of-fold optimal
threshold on the test fold, with the floor and the lookup rule for comparison.
"""
    )
    code(
        r"""
floor = all_positive_floor(y_test)                       # cost of reviewing every case, by ratio
lookup_cost = {key: value["cost_per_1000"] for key, value in evaluations[lookup]["cost_optimal"].items()}
cost_rows = []
for ratio in COST_RATIOS:
    key = str(int(ratio))
    entry = evaluations[best]["cost_optimal"][key]
    cost_rows.append(
        {
            "FN : FP cost ratio": f"{int(ratio)} : 1",
            "threshold (out-of-fold optimum)": entry["threshold"],
            f"cost per 1,000 ({best})": entry["cost_per_1000"],
            "cost per 1,000 (lookup)": lookup_cost[key],
            "cost per 1,000 (review everyone)": floor[key],
            "saving against the floor": floor[key] - entry["cost_per_1000"],
            "recall": entry["recall"],
            "specificity": entry["specificity"],
            "flag rate": entry["flag_rate"],
        }
    )
cost_table_frame = pd.DataFrame(cost_rows).set_index("FN : FP cost ratio")
dollar_columns = [c for c in cost_table_frame.columns if c.startswith("cost per 1,000") or c == "saving against the floor"]
show_table(cost_table_frame.astype({c: float for c in dollar_columns}).round({c: 0 for c in dollar_columns}).astype({c: int for c in dollar_columns}),
           "Expected cost per 1,000 cases on the test fold (USD)")
low, high = cost_table_frame.iloc[0], cost_table_frame.iloc[-1]
note(
    f"**Reading the table.** At 2:1 the out-of-fold optimal threshold gives a cost of {low[f'cost per 1,000 ({best})']:,.0f} against "
    f"{low['cost per 1,000 (review everyone)']:,.0f} for reviewing everyone, a saving of {low['saving against the floor'] / low['cost per 1,000 (review everyone)']:.0%}; "
    f"at 30:1 the saving is {high['saving against the floor'] / high['cost per 1,000 (review everyone)']:.1%} and the optimal threshold ({high['threshold (out-of-fold optimum)']:.3f}) flags "
    f"{high['flag rate']:.1%} of the cases. The lookup rule costs {low['cost per 1,000 (lookup)']:,.0f} at 2:1, so the model's saving over the lookup rule there is "
    f"{(low['cost per 1,000 (lookup)'] - low[f'cost per 1,000 ({best})']) / low['cost per 1,000 (lookup)']:.0%}."
)

at_twenty = cost_table_frame.loc["20 : 1"]
capacity_flag = evaluations[best][f"at_{OPERATING_KEY}"]["flag_rate"]
note(
    f"**At the assumed ratio of 20:1** the out-of-fold optimum saves USD {at_twenty['saving against the floor']:,.0f} per 1,000 cases against reviewing everyone "
    f"({at_twenty['saving against the floor'] / at_twenty['cost per 1,000 (review everyone)']:.1%} of the floor) and flags {at_twenty['flag rate']:.1%} of the cases, "
    + (f"above the {capacity_flag:.1%} flagged at the capacity operating point, so the cost optimum asks reviewers to take more cases than the capacity constraint allows"
       if at_twenty["flag rate"] > capacity_flag else f"within the {capacity_flag:.1%} flagged at the capacity operating point")
    + ". The saving over the floor is small at this ratio, which is why the headline operating point is the capacity point (specificity 0.40) and not the cost optimum."
)

sweep = threshold_sweep(y_test, proba[best])
chosen = {
    "Specificity 0.40": thresholds[best][OPERATING_KEY],
    "Flag rate 50 %": thresholds[best][FLAG_KEY],
    "Cost-optimal 3:1": thresholds[best]["cost_optimal"]["3"],
}
floor_at_two = cost_from_counts(0, int((y_test == 0).sum()), int(y_test.size), COST_RATIOS[0])
show(plots.threshold_cost_curve(sweep, chosen=chosen, cost_floor=floor_at_two),
     "Classification rates and expected cost per 1,000 cases against the decision threshold for six cost ratios")
"""
    )
    md(
        """
## E.7 Calibration

A model is **calibrated** if, among the cases it scores at 0.7, about 70 % really are lost-time
cases. The reliability diagram bins the predicted probabilities and plots the observed lost-time
rate in each bin; points on the diagonal are calibrated. A point below the diagonal means that the
model over-estimates the risk in that bin and a point above it that the model under-estimates it.
A curve flatter than the diagonal (above it at low scores and below it at high scores) is
over-confident; a steeper curve is under-confident. The Brier score summarises the same property
in one number. Calibration matters here because the cost
thresholds assume that a probability means what it says: a threshold of 0.25 is only the right
one for a 3:1 cost ratio if the scores are calibrated.
"""
    )
    code(
        r"""
show(plots.calibration({n: model_curves[n]["calibration"] for n in shown}, {n: evaluations[n]["at_0.5"]["brier"] for n in shown}),
     "Reliability diagram: predicted probability against observed lost-time rate for each model")
calibration_rows = {}
for name in shown:
    bins = model_curves[name]["calibration"].dropna()
    if len(bins) < 3:                        # a constant score has one bin and no slope
        continue
    slope = float(np.polyfit(bins["mean_predicted"], bins["fraction_positive"], 1, w=np.sqrt(bins["count"]))[0])
    low = bins[bins["mean_predicted"] < 0.25]
    calibration_rows[name] = {
        "slope of observed on predicted": slope,
        "observed minus predicted below 0.25": float((low["fraction_positive"] - low["mean_predicted"]).mean()) if len(low) else np.nan,
        "reading": "over-confident (flatter than the diagonal)" if slope < 0.9 else "under-confident (steeper than the diagonal)" if slope > 1.1 else "close to the diagonal",
    }
slopes = pd.DataFrame(calibration_rows).T.astype({"slope of observed on predicted": float, "observed minus predicted below 0.25": float})
show_table(slopes, "Calibration slope of each model: the weighted slope of the observed rate on the predicted probability (1 = on the diagonal)")
over_confident = slopes.index[slopes["slope of observed on predicted"] < 0.9].tolist()
under_confident = slopes.index[slopes["slope of observed on predicted"] > 1.1].tolist()
calibration_sentence = (
    f"Over-confident (flatter than the diagonal): {', '.join(f'`{n}`' for n in over_confident) or 'none'}. "
    f"Under-confident (steeper than the diagonal): {', '.join(f'`{n}`' for n in under_confident) or 'none'}. "
    f"Points far from the diagonal in a thin bin carry little weight, which is why the slope is weighted by the bin size."
)
brier = pd.DataFrame({"Brier score": {n: evaluations[n]["at_0.5"]["brier"] for n in shown}})
brier["constant predictor at the prevalence"] = float(y_test.mean() * (1 - y_test.mean()))
show_table(brier, "Brier score on the test fold (the constant predictor scores 0.25)")
brier_models = brier["Brier score"].loc[MODEL_NAMES]
worse_than_constant = [n for n in MODEL_NAMES if brier_models[n] > 0.25]
note(
    f"**Reading the diagram.** The lowest Brier score is `{brier_models.idxmin()}` ({brier_models.min():.3f}) and the highest `{brier_models.idxmax()}` ({brier_models.max():.3f}). "
    + (f"{', '.join(f'`{n}`' for n in worse_than_constant)} {'scores' if len(worse_than_constant) == 1 else 'score'} worse than the constant predictor (0.25), so "
       f"{'its probabilities are' if len(worse_than_constant) == 1 else 'their probabilities are'} unusable as probabilities even if "
       f"{'it ranks' if len(worse_than_constant) == 1 else 'they rank'} cases well. "
       if worse_than_constant else "Every model beats the constant predictor (0.25). ")
    + calibration_sentence
)
"""
    )
    md(
        """
## E.8 Results by commodity

Coal and metal/non-metal mines differ in the mix of injuries and in how often a case ends in lost
time (about 63 % against 39 % in the training rows). A model that is good on average could still
be poor in one sector, so the metrics are stratified by `COAL_METAL_IND`. The label also reflects
how often operators offer restricted duty, which differs between the two sectors (section B.12
gives the share of restricted-duty cases in each), so a gap between the sectors is partly a
property of the label and not only of the model.
"""
    )
    code(
        r"""
columns = ["n", "prevalence", "roc_auc", "pr_auc", "recall", "specificity", "precision", "f1", "brier", "flag_rate"]
for threshold_label, threshold in (("threshold 0.5", 0.5), ("the specificity 0.40 operating threshold", thresholds[best][OPERATING_KEY])):
    table = by_sector(y_test, proba[best], sector_test, threshold)[columns]
    table.index = table.index.map({"C": "coal", "M": "metal / non-metal", "all": "all"})
    show_table(table, f"{best} by commodity at {threshold_label}")

lookup_table = by_sector(y_test, proba[lookup], sector_test, thresholds[lookup][OPERATING_KEY])[columns]
lookup_table.index = lookup_table.index.map({"C": "coal", "M": "metal / non-metal", "all": "all"})
show_table(lookup_table, "Lookup baseline by commodity at its specificity 0.40 operating threshold")
sector_best = by_sector(y_test, proba[best], sector_test, thresholds[best][OPERATING_KEY])
note(
    f"**Reading the tables.** For `{best}` the ROC-AUC is {sector_best.loc['C', 'roc_auc']:.3f} in coal and {sector_best.loc['M', 'roc_auc']:.3f} in metal and non-metal mines "
    f"(the lookup rule: {by_sector(y_test, proba[lookup], sector_test, 0.5).loc['C', 'roc_auc']:.3f} and {by_sector(y_test, proba[lookup], sector_test, 0.5).loc['M', 'roc_auc']:.3f}). "
    f"At the operating threshold the recall is {sector_best.loc['C', 'recall']:.3f} in coal and {sector_best.loc['M', 'recall']:.3f} in metal, but the precision is "
    f"{sector_best.loc['C', 'precision']:.3f} and {sector_best.loc['M', 'precision']:.3f}: in coal, where most cases are lost-time cases, flagging most of them wastes little review effort; in metal and "
    f"non-metal mines the same threshold sends many more cases to review that turn out to be harmless. The Brier score is {sector_best.loc['C', 'brier']:.3f} in coal and "
    f"{sector_best.loc['M', 'brier']:.3f} in metal. In the training rows restricted duty is {sector_summary.loc['coal', 'restricted_duty_share']:.1%} of the coal "
    f"cases and {sector_summary.loc['metal / non-metal', 'restricted_duty_share']:.1%} of the metal and non-metal cases (B.12), so part of the gap between the sectors belongs to the label."
)
"""
    )
    md(
        """
## E.9 Is the best model really better? Statistical tests

Two tests answer the question the metrics cannot. **McNemar's test** compares two classifiers on
the same test rows through the cases on which exactly one of them is right: if the two make the
same number of exclusive errors the difference is chance. The plain test treats rows as
independent, but rows of one mine-year share a mine, a year and often an event, so the p-value
is too small; the **clustered** version (Durkalski et al.) treats each mine-year as one unit and is
the one to quote. The **paired fold test** compares the per-fold validation scores of the two best
models; with five overlapping folds the variance-corrected t-test (Nadeau and Bengio) is the
appropriate one, and the Wilcoxon test cannot go below p = 0.0625 with five folds. The bootstrap
intervals resample whole mine-years, so they are wider and more honest than the row bootstrap.
"""
    )
    code(
        r"""
from types import SimpleNamespace

from mlda.tuning import fold_scores

stored_tests = json.loads((METRICS_DIR / "statistical_tests.json").read_text(encoding="utf-8"))
mcnemar_rows = []
for label, key, t_best, t_lookup in (
    ("threshold 0.5", "at_0.5", 0.5, 0.5),
    ("specificity 0.40 operating threshold", f"at_{OPERATING_KEY}", thresholds[best][OPERATING_KEY], thresholds[lookup][OPERATING_KEY]),
):
    outcome = mcnemar_test(y_test, proba[best] >= t_best, proba[lookup] >= t_lookup, groups=g_test)
    stored = stored_tests["mcnemar_best_vs_lookup"][key]
    mcnemar_rows.append(
        {
            "comparison": f"{best} against the lookup rule, {label}",
            "only the model right": outcome["n_a_only"],
            "only the lookup right": outcome["n_b_only"],
            "accuracy (model)": outcome["accuracy_a"],
            "accuracy (lookup)": outcome["accuracy_b"],
            "p, rows independent": f"{outcome['p_value']:.2e}",
            "p, clustered by mine-year": f"{outcome['clustered_p_value']:.2e}",
            "stored clustered p": f"{stored['clustered_p_value']:.2e}" if stored["model_a"] == best else "n/a (other model)",
        }
    )
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame(mcnemar_rows).set_index("comparison"), "McNemar's test, model against the lookup baseline")

first, second = ranking[0], ranking[1]
comparable = fit_rows[first] == fit_rows[second] == len(y_train)       # both searched on all training rows
scores_first = fold_scores(SimpleNamespace(cv_results_df=cv_tables[first]))
scores_second = fold_scores(SimpleNamespace(cv_results_df=cv_tables[second]))
fold_table = pd.DataFrame({first: scores_first, second: scores_second}, index=[f"fold {i + 1}" for i in range(len(scores_first))])
show_table(fold_table, "Validation average precision of the two best models, fold by fold")
if comparable:
    paired = paired_fold_test(scores_first, scores_second)
    keys = ["n_folds", "mean_difference", "std_difference", "t_p_value", "corrected_t_p_value", "wilcoxon_p_value"]
    paired_table = pd.Series({k: paired[k] for k in keys}, name="value").to_frame()
    show_table(paired_table, f"Paired fold test: {first} against {second}")
else:
    print(f"{first} and {second} were not fitted on the same rows (sample cap), so their folds are not comparable; no p-value is given")

# 1,000 resamples with the seed of mlda-train, so the intervals equal those of statistical_tests.json and of the report.
N_BOOT = 1000
stored_boot = stored_tests["bootstrap_best_model"]
bootstrap_rows = []
for metric, threshold in (("roc_auc", 0.5), ("pr_auc", 0.5), ("recall", thresholds[best][OPERATING_KEY])):
    for resampling, cluster, stored_key in (("rows", None, "rows"), ("mine-years", g_test, "mine_years")):
        interval = bootstrap_ci(y_test, proba[best], metric, threshold, n_boot=N_BOOT, groups=cluster)
        stored_metric = metric if metric != "recall" else f"recall_at_{OPERATING_KEY}"
        stored_interval = stored_boot["metrics"].get(stored_metric, {}).get(stored_key, {}) if stored_boot["model"] == best else {}
        bootstrap_rows.append(
            {"metric": metric if metric != "recall" else "recall at specificity 0.40", "resampled units": resampling,
             "estimate": interval["estimate"], "95 % lower": interval["lower"], "95 % upper": interval["upper"],
             "stored lower": stored_interval.get("lower", np.nan), "stored upper": stored_interval.get("upper", np.nan)}
        )
show_table(pd.DataFrame(bootstrap_rows).set_index(["metric", "resampled units"]),
           f"Bootstrap 95 % intervals of {best} ({N_BOOT:,} resamples), beside the intervals stored by mlda-train")
"""
    )
    md(
        """
## E.10 Success criteria

The criteria were fixed before any model was fitted (S1-S6; S7 is the statistical comparison of
E.9). They are tested on the test fold with the thresholds chosen on training folds. S4 is the
cost criterion: at the ratios 2:1, 3:1 and 5:1 the model must cost less than both the
review-everyone floor and the lookup rule; at 10:1 and above the possible saving over the floor
is small, so those ratios are reported in E.6 but not scored. S6 also requires the Brier score to stay below the limit in
each sector on the temporal split, where the model was evaluated there. The table shows the value
and the verdict; the last cell recomputes every verdict and compares it with
`reports/metrics/success_criteria.json`.
"""
    )
    code(
        r"""
stored_criteria = json.loads((METRICS_DIR / "success_criteria.json").read_text(encoding="utf-8"))
temporal = json.loads((METRICS_DIR / "temporal_split.json").read_text(encoding="utf-8"))
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame({"criterion": stored_criteria["criteria"]}), "Success criteria fixed before modelling")

criteria = {}
for name in [*MODEL_NAMES, lookup]:
    temporal_brier = temporal["models"].get(name, {}).get("with_status_103i", {}).get("brier_by_sector", {})
    criteria[name] = model_criteria(
        evaluations[name], thresholds[name], floor, lookup_cost, is_lookup=name == lookup, temporal_brier=temporal_brier
    )


def verdict(value, passed):
    return f"{value:.3f} ({'pass' if passed else 'fail'})"


criteria_rows = {}
for name, checks in criteria.items():
    s4 = checks["S4"]["by_ratio"]
    counted = [r for r in s4 if s4[r]["counts_for_pass"]]
    s4_passed = sum(s4[r]["below_floor"] and s4[r]["below_lookup"] for r in counted)
    criteria_rows[name] = {
        "S1 ROC-AUC": verdict(checks["S1"]["value"], checks["S1"]["pass"]),
        "S2 PR-AUC": verdict(checks["S2"]["value"], checks["S2"]["pass"]),
        "S3 recall at spec. 0.40": verdict(checks["S3"]["value"], checks["S3"]["pass"]),
        "S4 ratios beaten (of 3)": "n/a (the reference)" if checks["S4"]["pass"] is None else f"{s4_passed} of {len(counted)} ({'pass' if checks['S4']['pass'] else 'fail'})",
        "S5 recall at 50 % flag rate": verdict(checks["S5"]["value"], checks["S5"]["pass"]),
        "S6 Brier": verdict(checks["S6"]["value"], checks["S6"]["pass"]),
        "S1-S6 all passed": "yes" if checks["all_pass"] else "no",
    }
show_table(pd.DataFrame(criteria_rows).T, "Success criteria on the test fold")

differences = [
    (name, key)
    for name in criteria
    for key in ("S1", "S2", "S3", "S4", "S5", "S6")
    if criteria[name][key]["pass"] != stored_criteria["models"][name][key]["pass"]
]
print("verdicts recomputed here against success_criteria.json:", "identical" if not differences else f"differ for {differences}")

# S5 was planned as recall among the half of the cases with the highest scores and is scored above at the
# out-of-fold threshold for a 50 % flag rate. The cut-off below comes from the test scores alone (labels only
# count the hits), so it shows how much of each S5 result depends on flagging more or less than half.
s5_rows = {}
for name in [*MODEL_NAMES, lookup]:
    top_half = classification_metrics(y_test, proba[name], threshold_for_flag_rate(y_test, proba[name], 0.5))
    s5_rows[name] = {
        "S5 as scored: recall": criteria[name]["S5"]["value"],
        "flag rate": criteria[name]["S5"]["flag_rate_achieved"],
        "S5 as planned (top half of test scores): recall": top_half["recall"],
        "flag rate ": top_half["flag_rate"],
    }
show_table(pd.DataFrame(s5_rows).T.sort_values("S5 as planned (top half of test scores): recall", ascending=False),
           "S5 two ways: at the out-of-fold threshold for a 50 % flag rate, and among the top half of the test scores")
"""
    )
    md(
        """
## E.11 The final-model decision

The model is chosen by a rule fixed in advance, not by looking at which one scores highest on the
test fold. **1. Primary rule:** the highest cross-validated average precision on the training
folds (this is also how `mlda-train` names its best model, and the model the app loads by
default). **2. Evidence checks on the test fold:** the success criteria S1-S6 and the clustered
McNemar test against the lookup rule. They confirm or challenge the choice, and a criterion that
the selected model misses is reported as a shortfall, not hidden. **3. Near ties:** when the
runner-up lies within one fold standard deviation of the leader, the table is read for
calibration (Brier score), stability across folds and commodities, interpretability, and the
cost of running the model (file size and prediction time), because the model will be deployed in
a small web app. The table puts the evidence for every model side by side; the
interpretability column is a judgement, not a measurement.
"""
    )
    code(
        r"""
INTERPRETABILITY = {
    "logreg": "high (coefficients, odds ratios)", "decision_tree": "high (readable rules)", "gaussian_nb": "high (per-feature likelihoods)",
    "random_forest": "medium (importances, SHAP)", "hist_gb": "medium (importances, SHAP)",
    "knn": "low (neighbours only)", "svc_rbf": "low (kernel)", "mlp": "low (weights)",
}
decision = pd.DataFrame(index=ranking)
decision["CV AP (mean)"] = cv_summary["average_precision_mean"]
decision["CV AP (std)"] = cv_summary["average_precision_std"]
decision["test PR-AUC"] = [evaluations[n]["at_0.5"]["pr_auc"] for n in decision.index]
decision["test ROC-AUC"] = [evaluations[n]["at_0.5"]["roc_auc"] for n in decision.index]
decision["test Brier"] = [evaluations[n]["at_0.5"]["brier"] for n in decision.index]
decision["recall at spec. 0.40"] = [evaluations[n][f"at_{OPERATING_KEY}"]["recall"] for n in decision.index]
decision["S1-S6 passed"] = [sum(bool(criteria[n][k]["pass"]) for k in ("S1", "S2", "S3", "S4", "S5", "S6")) for n in decision.index]
decision["fit seconds"] = [fit_seconds[n] for n in decision.index]
decision["ms to score 1,000 rows"] = [1000 * predict_seconds[n] / len(X_test) * 1000 for n in decision.index]
decision["model file (MB)"] = [manifest[n]["size_bytes"] / 1e6 for n in decision.index]
decision["interpretability"] = [INTERPRETABILITY[n] for n in decision.index]
show_table(decision, "Evidence for the model decision (rows ordered by cross-validated average precision)")

winner, runner_up = ranking[0], ranking[1]
gap = decision.loc[winner, "CV AP (mean)"] - decision.loc[runner_up, "CV AP (mean)"]
tie_width = max(decision.loc[winner, "CV AP (std)"], decision.loc[runner_up, "CV AP (std)"])
near_tie = gap < tie_width
criteria_names = ("S1", "S2", "S3", "S4", "S5", "S6")
failed = [k for k in criteria_names if not criteria[winner][k]["pass"]]
cluster_p = mcnemar_rows[0]["p, clustered by mine-year"]
winner_ci = next(r for r in bootstrap_rows if r["metric"] == "pr_auc" and r["resampled units"] == "mine-years")
runner_pr = decision.loc[runner_up, "test PR-AUC"]
runner_beats = [k for k in failed if criteria[runner_up][k]["pass"]]

parts = [
    f"**Decision.** By the primary rule the selected model is **`{winner}`**: its cross-validated average precision is "
    f"{decision.loc[winner, 'CV AP (mean)']:.4f} (std {decision.loc[winner, 'CV AP (std)']:.4f}), {gap:.4f} ahead of `{runner_up}` "
    f"({decision.loc[runner_up, 'CV AP (mean)']:.4f}, std {decision.loc[runner_up, 'CV AP (std)']:.4f}), which is "
    f"{'inside' if near_tie else 'outside'} one fold standard deviation, so the ranking is "
    f"{'a near tie that the tie-breakers must settle' if near_tie else 'clear enough to decide'}.",
    f"On the held-out fold it reaches ROC-AUC {decision.loc[winner, 'test ROC-AUC']:.3f}, PR-AUC {decision.loc[winner, 'test PR-AUC']:.3f} and Brier "
    f"{decision.loc[winner, 'test Brier']:.3f}, and at the capacity operating point it catches "
    f"{decision.loc[winner, 'recall at spec. 0.40']:.1%} of the lost-time cases. It passes "
    f"{int(decision.loc[winner, 'S1-S6 passed'])} of the 6 success criteria"
    + ("." if not failed else f" and misses {', '.join(failed)}.")
    + f" Against the lookup rule the clustered McNemar p-value at threshold 0.5 is {cluster_p}"
    + f" ({'significant' if float(cluster_p) < 0.05 else 'not significant'} at the 5 % level).",
    f"The test PR-AUC of `{runner_up}` ({runner_pr:.3f}) "
    + (f"lies inside the clustered 95 % interval of `{winner}` ({winner_ci['95 % lower']:.3f} to {winner_ci['95 % upper']:.3f}), so the two cannot be separated on the test fold."
       if winner_ci["95 % lower"] <= runner_pr <= winner_ci["95 % upper"] else
       f"lies outside the clustered 95 % interval of `{winner}` ({winner_ci['95 % lower']:.3f} to {winner_ci['95 % upper']:.3f})."),
    f"{'Tie-breakers' if near_tie else 'Further criteria'}: Brier {decision.loc[winner, 'test Brier']:.3f} against {decision.loc[runner_up, 'test Brier']:.3f} for `{runner_up}`; "
    f"interpretability {decision.loc[winner, 'interpretability']} against {decision.loc[runner_up, 'interpretability']}; "
    f"model file {decision.loc[winner, 'model file (MB)']:.2f} MB against {decision.loc[runner_up, 'model file (MB)']:.2f} MB and "
    f"{decision.loc[winner, 'ms to score 1,000 rows']:.0f} ms against {decision.loc[runner_up, 'ms to score 1,000 rows']:.0f} ms per 1,000 rows "
    + ("(both inside the 20 MB budget for one model file in the web app)." if max(decision.loc[[winner, runner_up], "model file (MB)"]) <= 20 else "(at least one is above the 20 MB budget for one model file in the web app)."),
]
oversize = decision.index[decision["model file (MB)"] > 20].tolist()
parts.append(
    "Model files above the 20 MB budget for one file: "
    + (", ".join(f"`{n}` ({decision.loc[n, 'model file (MB)']:.1f} MB)" for n in oversize) if oversize else "none")
    + f". The {len(decision)} files together take {decision['model file (MB)'].sum():.1f} MB (budget 60 MB)."
)
if runner_beats:
    parts.append(
        f"`{runner_up}` passes {', '.join(runner_beats)}, which `{winner}` misses. The shortfall is reported here; it does not change the choice, "
        f"because the primary rule was fixed before the test fold was looked at."
    )
note(" ".join(parts))
"""
    )
    md(
        """
## E.12 What drives the predictions

**Permutation importance** shuffles one input column of the test fold at a time and measures how
much the average precision of the fitted model drops; a large drop means the model relies on that
column. It is computed on held-out rows and on the *raw* inputs, so one-hot columns of the same
variable are not split apart. It is read from the table that `mlda-train` stored for the best model
(5,000 test rows, five repeats) and is recomputed here when that file is missing. The
**coefficients** of the logistic regression are read on standardised inputs, so their size is
comparable across numeric inputs; the odds ratio is the multiplicative change in the odds of lost
time for a one-standard-deviation increase (or for the presence of a one-hot level). The largest
coefficients often belong to rare one-hot levels (a nature-of-injury code that is always lost time,
a state with few rows), which are estimated from few cases; the table of numeric and binary
inputs is easier to read. Importance is association, not cause: a high importance for
`NATURE_INJURY_CD` means the type of injury is informative, not that it can be changed.
"""
    )
    code(
        r"""
from mlda.explain import coefficient_table, permutation_importance_table

permutation_path = METRICS_DIR / f"permutation_importance_{best}.csv"
if permutation_path.exists():
    permutation = pd.read_csv(permutation_path)
    note(f"Permutation importance of `{best}` read from `{permutation_path.name}` (stored by mlda-train).")
else:
    with timer.stage("E.12 permutation importance"):
        permutation = permutation_importance_table(fitted[best], X_test, y_test, max_rows=3000, n_repeats=3, n_jobs=N_JOBS)
    note(f"Permutation importance of `{best}` computed here on 3,000 test rows (three repeats).")
show_table(permutation.head(15), "Permutation importance: drop in average precision when the input is shuffled")
show(plots.feature_importance(permutation.rename(columns={"importance_mean": "importance", "importance_std": "std"})),
     f"Bar chart of the permutation importance of the inputs of {best}")

for model_name in dict.fromkeys(["logreg", best]):
    if model_name not in fitted:
        continue
    table = coefficient_table(fitted[model_name])
    if table is None:
        continue
    kind = table.attrs.get("kind", "coefficient")
    show_table(table.head(15), f"{model_name}: largest {kind.replace('_', ' ')}s")
    if kind == "coefficient":
        numeric_only = table[table["feature"].isin(numeric_cols + binary_cols)]
        show_table(numeric_only.head(10), f"{model_name}: largest coefficients of the numeric and binary inputs (read these first)")
        show(plots.feature_importance(table.rename(columns={"coefficient": "importance"}), value_label="Logistic regression coefficient (standardised inputs)"),
             "Bar chart of the largest logistic regression coefficients")

shap_path = METRICS_DIR / f"shap_summary_{best}.csv"
if shap_path.exists():
    show_table(pd.read_csv(shap_path).head(10), f"Mean absolute SHAP value per transformed column ({best}; 500 test rows, stored by mlda-train)")
"""
    )
    md(
        """
## E.13 Temporal split: train on 2010-2019, test on 2020-2025

The primary split is random over mine-years, so the test fold is drawn from the same years as
the training rows. A forecast, however, is always about the future. The temporal study repeats the
whole protocol on the accident years **before 2020** only (re-tuning, thresholds from out-of-fold
probabilities, model ranking) and scores 2020-2025 once, so nothing from the later period
influences any choice. It reports the two best models with the dated 103(i) status and the
immediate-notification flag included and removed, because those two inputs depend on how the form
was filled. A shift of the Brier score or of the specificity at the fixed operating threshold
points to **label drift** (the coal lost-time rate rises over the period while restricted-duty
coding falls; the yearly rates of the training rows are in section B.12), not only to a worse model.
"""
    )
    code(
        r"""
variants_label = {
    "with_status_103i": "all inputs",
    "without_status_103i": "without the 103(i) status",
    "without_immed_reportable": "without immed_reportable",
}
temporal_rows = []
for model_name, entry in temporal["models"].items():
    for variant, label in variants_label.items():
        test = entry[variant]["test"]
        temporal_rows.append(
            {
                "model": model_name,
                "split and inputs": f"temporal 2020-2025, {label}",
                "ROC-AUC": test["at_0.5"]["roc_auc"],
                "PR-AUC": test["at_0.5"]["pr_auc"],
                "Brier": test["at_0.5"]["brier"],
                "recall at the operating threshold": test[f"at_{OPERATING_KEY}"]["recall"],
                "specificity at the operating threshold": test[f"at_{OPERATING_KEY}"]["specificity"],
                "Brier coal": entry[variant]["brier_by_sector"].get("C"),
                "Brier metal": entry[variant]["brier_by_sector"].get("M"),
            }
        )
    if model_name in evaluations:
        half = evaluations[model_name]["at_0.5"]
        sectors = evaluations[model_name]["by_sector"]["at_0.5"]
        temporal_rows.append(
            {
                "model": model_name,
                "split and inputs": "primary random split (section E.2)",
                "ROC-AUC": half["roc_auc"],
                "PR-AUC": half["pr_auc"],
                "Brier": half["brier"],
                "recall at the operating threshold": evaluations[model_name][f"at_{OPERATING_KEY}"]["recall"],
                "specificity at the operating threshold": evaluations[model_name][f"at_{OPERATING_KEY}"]["specificity"],
                "Brier coal": sectors["C"]["brier"],
                "Brier metal": sectors["M"]["brier"],
            }
        )
temporal_table = pd.DataFrame(temporal_rows).set_index(["model", "split and inputs"]).sort_index(level=0, sort_remaining=False)
show_table(temporal_table, "Temporal split against the primary split")
print(f"temporal split: {temporal['n_train']:,} training rows (prevalence {temporal['prevalence_train']:.3f}) and "
      f"{temporal['n_test']:,} test rows (prevalence {temporal['prevalence_test']:.3f})")
drift = temporal["level_drift"]
print(f"level drift: {drift['n_all_zero_in_test_period']} of {drift['n_onehot_columns']} one-hot columns are all zero in 2020-2025: {drift['columns']}")

lead = next(iter(temporal["models"]))
lead_entry = temporal["models"][lead]
temporal_roc = lead_entry["with_status_103i"]["test"]["at_0.5"]["roc_auc"]
spec_op = lead_entry["with_status_103i"]["test"][f"at_{OPERATING_KEY}"]["specificity"]
without_roc = lead_entry["without_status_103i"]["test"]["at_0.5"]["roc_auc"]
note(
    f"**Reading the table.** For `{lead}`, the temporal ROC-AUC is {temporal_roc:.3f}"
    + (f" against {evaluations[lead]['at_0.5']['roc_auc']:.3f} on the primary split" if lead in evaluations else "")
    + f"; with the operating threshold fixed on 2010-2019 data the specificity on 2020-2025 is {spec_op:.3f} (target 0.40). "
    f"Removing the 103(i) status changes the ROC-AUC to {without_roc:.3f} "
    + ("(a change under 0.01, so the result does not hinge on that dated input). " if abs(without_roc - temporal_roc) < 0.01 else "(a visible change, so that dated input matters). ")
    + f"The Brier score by sector ({lead_entry['with_status_103i']['brier_by_sector'].get('C', float('nan')):.3f} coal, "
    f"{lead_entry['with_status_103i']['brier_by_sector'].get('M', float('nan')):.3f} metal) shows where calibration is weaker. "
    f"The drift in the label is visible before any model is fitted: in the training rows the coal lost-time rate is {early.loc['C', 'lost_time']:.1%} in "
    f"{YEAR_MIN}-{YEAR_MIN + 2} and {late.loc['C', 'lost_time']:.1%} in {YEAR_MAX - 2}-{YEAR_MAX}, while the share of restricted-duty cases in coal falls from "
    f"{early.loc['C', 'restricted_duty']:.1%} to {late.loc['C', 'restricted_duty']:.1%}."
)
"""
    )
    md(
        """
## E.14 Sensitivity analyses

Four questions about the **result's dependence on choices** are answered with the selected model.
(a) **Deterministic levels:** a level that is 100 % one class (here a nature-of-injury code) makes
rows trivially easy; removing those rows from the test fold shows how much of the score they
provide. `mlda-train` searched all in-scope rows for such levels; the cell repeats the search on
the training rows only and checks that it finds the same level, so no test label decides which
test rows the check removes. (b) **Altered inputs:** the model is refitted without the immediate-notification flag
(a near copy of the fatality code) and without the 103(i) status, with previous-year case counts
restricted to cases that were already resolved at the accident date, and with previous-year hours
blanked for forms filed before the fourth-quarter report was due. (c) **Grouping by mine:** the
primary split keeps mine-years together, but the same mine can appear on both sides. Splitting by
`MINE_ID` puts every mine on one side only and is the strictest test of generalisation to unseen
mines. A small change means the model has not learnt mine identities.
"""
    )
    code(
        r"""
from mlda.sensitivity import pure_levels

sensitivity = json.loads((METRICS_DIR / "sensitivity.json").read_text(encoding="utf-8"))
show_table(pd.DataFrame(sensitivity["pure_levels"]["found"]), "Levels that are 100 % one class (at least 100 rows), searched on all rows")
training_levels = pure_levels(X_train, y_train)
show_table(training_levels, "The same search on the training rows only")
same_levels = sorted(zip(training_levels["column"], training_levels["level"], strict=True)) == sorted(
    (r["column"], r["level"]) for r in sensitivity["pure_levels"]["found"]
)
print(f"the training rows alone give the same deterministic levels as all rows: {same_levels}")

excluded = sensitivity["excluding_pure_levels"]
pure_rows = []
for model_name in [best, lookup]:
    if model_name not in excluded["models"]:
        continue
    entry = excluded["models"][model_name]
    for label, key in (("all test rows", "with_levels"), (f"without {', '.join(excluded['levels'])}", "without_levels"), ("difference", "delta")):
        pure_rows.append({"model": model_name, "test rows": label, **{k: entry[key][k] for k in ("roc_auc", "pr_auc", "brier", f"recall_at_{OPERATING_KEY}")}})
show_table(pd.DataFrame(pure_rows).set_index(["model", "test rows"]),
           f"Removing the deterministic level ({excluded['n_test_excluded']} of {excluded['n_test']:,} test rows)")

variants = sensitivity["primary_split_variants"]
variant_rows = [{"variant": "reference (selected model, primary split)", **variants["reference"]}]
for name, entry in variants["variants"].items():
    variant_rows.append({"variant": name.replace("_", " "), **entry["test"]})
variant_table = pd.DataFrame(variant_rows).set_index("variant")
show_table(variant_table[["roc_auc", "pr_auc", "brier", f"recall_at_{OPERATING_KEY}", f"specificity_at_{OPERATING_KEY}"]],
           "Altered inputs (the selected model refitted; hyper-parameters not re-tuned)")
with pd.option_context("display.max_colwidth", None):
    show_table(pd.DataFrame({"what changed": {k.replace("_", " "): v["description"] for k, v in variants["variants"].items()}}), "Description of each variant")

grouped = sensitivity["mine_grouped_split"]
grouped_table = pd.DataFrame(
    {
        "primary split (mine-years)": variants["reference"],
        "split grouped by mine": grouped["model"]["test"],
        "difference": grouped["model"]["delta_vs_primary"],
        "lookup rule, grouped by mine": grouped["lookup"]["test"],
    }
)
show_table(grouped_table, f"Split grouped by MINE_ID: {grouped['n_train']:,} training and {grouped['n_test']:,} test rows, {grouped['shared_mines']} mines on both sides")
passes = {}
for name, checks in {sensitivity["model"]: grouped["model"]["criteria"], lookup: grouped["lookup"]["criteria"]}.items():
    passes[name] = {key: checks[key]["pass"] for key in ("S1", "S2", "S3", "S4", "S5", "S6")}
    passes[name]["all passed"] = checks["all_pass"]
show_table(pd.DataFrame(passes).T, "Success criteria under the mine-grouped split (True = passed, None = not applicable)")

worst = max(variants["variants"].items(), key=lambda kv: abs(kv[1]["delta_vs_reference"]["roc_auc"]))
note(
    f"**Reading the tables.** Removing the deterministic level lowers the ROC-AUC of `{sensitivity['model']}` by "
    f"{abs(excluded['models'][sensitivity['model']]['delta']['roc_auc']):.3f}. The largest change among the altered inputs is "
    f"{worst[1]['delta_vs_reference']['roc_auc']:+.4f} ROC-AUC (`{worst[0].replace('_', ' ')}`). Grouping the split by mine moves the "
    f"ROC-AUC by {grouped['model']['delta_vs_primary']['roc_auc']:+.3f} and the recall at the operating point by "
    f"{grouped['model']['delta_vs_primary'][f'recall_at_{OPERATING_KEY}']:+.3f}"
    + (", a small change: most of the skill carries over to mines the model has not seen." if abs(grouped["model"]["delta_vs_primary"]["roc_auc"]) < 0.03
       else ", a large change: part of the skill comes from mines that appear in training.")
)
"""
    )


# ---------------------------------------------------------------------------
# F. Deployment and reproducibility
# ---------------------------------------------------------------------------


def section_f() -> None:
    """Section F: how the artefacts feed the app, the manifest, the reproducibility checklist, limitations, licence and run time."""
    md(
        """
# F. Deployment and reproducibility

The study is deployed as a multi-page **Streamlit** web app (`app/streamlit_app.py`, with the
pages in `app/pages/`). The app does no heavy work: `uv run mlda-train` precomputes the models,
the tables and the figures, and the committed artefacts feed the pages. Only the training page
fits models live, on a sample of at most 20,000 rows with three folds and a small grid. The
deployed address is given in the README ("see README"); the app also runs locally with
`uv run streamlit run app/streamlit_app.py` or in a container (`docker run -p 8501:8501 mlda-cw1`).
"""
    )
    md(
        """
## F.1 How the artefacts feed the app

| Page | What the user does | Artefacts it reads |
|---|---|---|
| Data explorer | read how the sources are joined and the target is defined, filter a sample | `data/processed/base.parquet` |
| Exploratory analysis | descriptive statistics, correlations, missing values, distributions | `reports/metrics/eda_*.csv`, `data/processed/features.parquet` |
| Preprocessing | try imputer, scaler, encoder and outlier options on a sample | `mlda.preprocessing`, `data/processed/features.parquet`, `data/processed/split_indices.json` |
| Model training | tune one model live on a stratified sample (at most 20,000 rows, 3 folds, a reduced grid) | `mlda.tuning.tune`, `mlda.models.MODEL_ZOO`, `reports/metrics/cv_summary.csv` |
| Inference | predict one case, or a CSV of cases, and compare with the mine's history | `models/<name>.joblib`, `reports/metrics/thresholds.json`, `data/processed/mine_year_features.parquet`, `data/processed/mine_case_history.parquet` |
| Evaluation | comparison tables, curves, thresholds, success criteria, tests, importance | `reports/metrics/*.json`, `reports/metrics/*.csv`, `reports/predictions/test_predictions.parquet` |
| About | data card, licence, run manifest, limitations | `models/manifest.json`, `reports/metrics/run_manifest.json` |

The last column is the contract between training and deployment: the pages read these files and
do no heavy work of their own (the training page is the one exception), so the notebook, the
report and the app show the same numbers. The first cell lists the pages registered in
`app/streamlit_app.py`; the second lists the artefacts, with their size.
"""
    )
    code(
        r"""
import re

entrypoint = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
registered = re.findall(r'st\.Page\(\s*"([^"]+)",\s*title="([^"]+)"', entrypoint)
page_table = pd.DataFrame(registered, columns=["page file", "title in the navigation"])
page_table["file exists"] = [(PROJECT_ROOT / "app" / path).exists() for path in page_table["page file"]]
show_table(page_table, "Pages registered in app/streamlit_app.py")
assert page_table["file exists"].all() and len(page_table) >= 5
"""
    )
    code(
        r"""
from mlda.config import FIGURES_DIR, MODELS_DIR

artefacts = [
    PROCESSED_DIR / "base.parquet", PROCESSED_DIR / "features.parquet", PROCESSED_DIR / "split_indices.json",
    PROCESSED_DIR / "mine_year_features.parquet", PROCESSED_DIR / "mine_case_history.parquet",
    METRICS_DIR / "comparison.csv", METRICS_DIR / "comparison_operating_point.csv", METRICS_DIR / "cv_summary.csv",
    METRICS_DIR / "thresholds.json", METRICS_DIR / "success_criteria.json", METRICS_DIR / "statistical_tests.json",
    METRICS_DIR / "temporal_split.json", METRICS_DIR / "sensitivity.json", METRICS_DIR / "baselines.json",
    METRICS_DIR / "explanations.json", METRICS_DIR / "error_rules.csv", METRICS_DIR / "run_manifest.json",
    PREDICTIONS_DIR / "test_predictions.parquet", PREDICTIONS_DIR / "oof_predictions.parquet", MODELS_DIR / "manifest.json",
    *[MODELS_DIR / f"{name}.joblib" for name in MODEL_ZOO],
    *[CV_DIR / f"{name}_cv_results.csv" for name in MODEL_ZOO],
    *sorted(FIGURES_DIR.glob("F*.png")),
]
inventory = pd.DataFrame(
    {
        "artefact": [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in artefacts],
        "exists": [p.exists() for p in artefacts],
        "size (kB)": [round(p.stat().st_size / 1e3, 1) if p.exists() else None for p in artefacts],
    }
).set_index("artefact")
print(f"{int(inventory['exists'].sum())} of {len(inventory)} artefacts present; "
      f"models total {sum(p.stat().st_size for p in MODELS_DIR.glob('*.joblib')) / 1e6:.1f} MB")
missing_artefacts = inventory[~inventory["exists"]]
if len(missing_artefacts):
    show_table(missing_artefacts, "Artefacts that are missing (run `uv run mlda-train`)")
with pd.option_context("display.max_rows", 80):
    show_table(inventory.head(25), "Artefacts read by the app (first 25)")

"""
    )
    md(
        """
## F.2 The manifest: versions, commit and data checksum

Pickled scikit-learn models are only guaranteed to load under the library version that wrote them,
and a model is only reproducible if the code and the data are known. `models/manifest.json`
therefore records, for each saved model, the Python and library versions, the git commit, a digest
of the pinned raw-data snapshot, the time it was saved and the selected parameters. The app reads
the same file on its "About" page. The cell checks the recorded environment against the one running
this notebook and the recorded data digest against the files on disk.
"""
    )
    code(
        r"""
from mlda.persistence import data_checksum

manifest = read_manifest()
manifest_table = pd.DataFrame(
    {
        name: {
            "estimator": entry["estimator"],
            "python": entry["python_version"],
            "scikit-learn": entry["sklearn_version"],
            "pandas": entry["pandas_version"],
            "git commit": (entry["git_sha"] or "n/a")[:10],
            "data digest": (entry["data_checksum"] or "n/a")[:12],
            "saved (UTC)": entry["saved_at_utc"],
            "file (MB)": entry["size_bytes"] / 1e6,
            "quick run": entry["extra"].get("quick"),
        }
        for name, entry in manifest.items()
    }
).T
show_table(manifest_table, "models/manifest.json")

live_digest = data_checksum()["sha256"]
print("digest of the raw files on disk:", live_digest[:12], "| manifest entries with the same digest:",
      sum(entry["data_checksum"] == live_digest for entry in manifest.values()), "of", len(manifest))
print("scikit-learn installed:", sklearn.__version__, "| manifest entries from the same version:",
      sum(entry["sklearn_version"] == sklearn.__version__ for entry in manifest.values()), "of", len(manifest))

commits = {}
for name, entry in manifest.items():
    commits.setdefault((entry["git_sha"] or "n/a")[:10], []).append(name)
if len(commits) > 1:
    detail = "; ".join(f"`{sha}`: {', '.join(names)}" for sha, names in commits.items())
    same_inputs = len({e["data_checksum"] for e in manifest.values()}) == 1 and len({e["sklearn_version"] for e in manifest.values()}) == 1
    note(
        f"**Why the commit column is not constant.** `models/manifest.json` stores the commit that was checked out when each model was saved ({detail}). "
        "A commit made between two saves of one `mlda-train` run therefore gives that run more than one commit. "
        + ("The data digest and the scikit-learn version are the same for every model, so all models were fitted on the same data with the same library."
           if same_inputs else "The data digest or the library version also differs between models, so check which models were fitted together.")
    )
else:
    note(f"Every model was saved at the same commit (`{next(iter(commits))}`).")

run_manifest = json.loads((METRICS_DIR / "run_manifest.json").read_text(encoding="utf-8"))
print(f"mlda-train: quick={run_manifest['quick']}, {run_manifest['cv_folds']} folds, {run_manifest['n_jobs']} workers, "
      f"{run_manifest['total_seconds'] / 60:.1f} minutes in total; best model by cross-validation: {run_manifest['best_model']}")
stage_seconds = pd.DataFrame({"seconds": run_manifest["stage_seconds"]})
show_table(stage_seconds, "Wall-clock time of each stage of mlda-train")
"""
    )
    md(
        """
## F.3 Reproducibility checklist

| Requirement | How it is met |
|---|---|
| Same software | `uv.lock` pins every package; `uv sync --locked --all-groups` recreates the environment; `requirements.txt` is exported from the lock for `pip`; the `Dockerfile` builds the app with uv |
| Same data | `mlda.config.SOURCES` pins the SHA-256 of every raw file; `uv run mlda-download` refuses a file whose hash differs; the snapshot is recorded in `data/raw/SNAPSHOT.json` |
| Same random numbers | `RANDOM_STATE = 42` seeds the split, the cross-validation, every stochastic estimator, the bootstrap and the sampling |
| Same split | the row indices of the split are stored in `data/processed/split_indices.json` and rebuilt from the seed in section C.8 to confirm that they agree |
| Same results | `uv run mlda-train` rewrites the models, tables and figures; this notebook recomputes the tables and refits the models and prints where they differ from the stored ones |
| Traceable models | `models/manifest.json` records versions, git commit and data digest for every model (F.2) |
| Tested | `uv run pytest` runs the unit tests, the quick end-to-end run and this notebook (marked slow); CI repeats them on Ubuntu and Windows |

```
uv sync --locked --all-groups
uv run mlda-download
uv run mlda-train
uv run python notebooks/build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb
uv run jupyter nbconvert --to html notebooks/01_end_to_end.ipynb
uv run pytest
uv run streamlit run app/streamlit_app.py
```
"""
    )
    md(
        """
## F.4 Limitations

- **The label depends on practice, not only on the injury.** The line between lost time and
  restricted duty depends on whether the operator offers light duty, which differs between coal and
  metal/non-metal mines and has changed over the period. The model is suited to reserving and
  return-to-work planning, where that line has a financial meaning, and not to ranking hazards.
- **History needs the mine's previous year.** For 1.2 % of cases there is no previous-year
  production, and the history features are imputed and flagged. A mine that is new has no history.
- **Several mine attributes are undated.** The safety-committee flag, the Part 48 training flag and
  the portable-operation flag describe the mine at the snapshot, not at the accident, and the file
  gives no historical value for them. They were dropped from the model inputs and remain only in
  the base table and the data explorer.
- **The cost ratio is an assumption** (USD 4,000 for a missed case against USD 200 for a review);
  that is why the cost analysis covers 2:1 to 30:1 and why the capacity operating point, not the cost
  optimum, is the headline.
- **Dependence between rows.** Rows from one mine-year are not independent; the split, the
  cross-validation, McNemar's test and the bootstrap all group by mine-year for that reason.
"""
    )
    code(
        r"""
snapshot_flags = ["SAFETY_COMMITTEE_IND", "PART48_TRAINING", "PORTABLE_OPERATION"]
# These Mines-table flags are recorded as at the snapshot date, so their value at the time of an
# accident years earlier is unknown. They are kept in the base table for the data explorer but are
# not model inputs; this cell proves the exclusion on the fitted artefacts.
in_inputs = [c for c in snapshot_flags if c in X.columns]
in_importance = [c for c in snapshot_flags if c in set(permutation["feature"])]
assert not in_inputs and not in_importance, (in_inputs, in_importance)
print(
    "snapshot-only Mines flags excluded from the inputs:", ", ".join(snapshot_flags),
    f"| present in the base table: {all(c in base.columns for c in snapshot_flags)}",
    f"| present among the {len(X.columns)} model inputs: False",
)
"""
    )
    md(
        """
## F.5 Licence, data citation and deployed address

**Code.** MIT licence (`LICENSE` in the repository root).

**Data.** Mine Safety and Health Administration (MSHA), US Department of Labor, *Open Government
Data Sets: Accidents, Mines and MinesProdYearly*, https://arlweb.msha.gov/OpenGovernmentData/DataSets/,
files published 9 October 2026 and retrieved 14 October 2026 (hashes in `data/raw/SNAPSHOT.json`).
The files are works of the United States federal government and carry no copyright in the United
States (17 U.S.C. 105); the data card in `data/README.md` lists the sources, the join keys and the
target definition.

**Deployed app.** see README.
"""
    )
    code(
        r"""
licence_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()
print("LICENSE:", licence_text[0])
print("data card present:", (PROJECT_ROOT / "data" / "README.md").exists())
"""
    )
    md(
        """
## F.6 Run time of this notebook

The stages that took at least five seconds, and the wall-clock time of the whole run. In the
default mode the live searches of section D.1, the refits of section D.3 and the scoring of
section E.1 dominate; the SVM is the slowest model to refit and to score. `FULL_RUN=1` adds the hyper-parameter searches (about an hour).
"""
    )
    code(
        r"""
stage_table = pd.DataFrame({"seconds": timer.seconds})
stage_table["minutes"] = stage_table["seconds"] / 60
slow = stage_table[stage_table["seconds"] >= 5].sort_values("seconds", ascending=False)
show_table(slow, "Stages of this notebook that took at least 5 seconds")
elapsed = time.perf_counter() - NOTEBOOK_STARTED
print(f"wall-clock time of the notebook up to this cell: {elapsed:.0f} s ({elapsed / 60:.1f} minutes); FULL_RUN = {FULL_RUN}")
"""
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build(through: str = "F") -> nbformat.NotebookNode:
    """Assemble the notebook from the title cells and the sections up to ``through``."""
    CELLS.clear()
    title_cells()
    builders = {"A": section_a, "B": section_b, "C": section_c, "D": section_d, "E": section_e, "F": section_f}
    for key in SECTIONS:
        builders[key]()
        if key == through:
            break
    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    for index, (kind, source) in enumerate(CELLS):
        cell = new_code_cell(source) if kind == "code" else new_markdown_cell(source)
        cell.id = f"cell-{index:03d}"           # deterministic ids keep regenerated notebooks diffable
        notebook.cells.append(cell)
    return notebook


def main() -> None:
    """Write the notebook (default notebooks/01_end_to_end.ipynb); ``--through`` builds a prefix for checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=NOTEBOOK_PATH, help="where to write the .ipynb file")
    parser.add_argument("--through", choices=SECTIONS, default="F", help="last section to include")
    args = parser.parse_args()
    notebook = build(args.through)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
