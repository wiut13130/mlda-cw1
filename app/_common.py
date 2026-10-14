"""Shared loaders, descriptions, chart helpers and inference helpers for the Streamlit pages.

Why one module: every page reads the same precomputed artefacts (data/processed, models,
reports) and draws the same few chart shapes, so the loaders and the Altair helpers live here
and the pages stay short enough to read from top to bottom. Sections, in order: artefact readers,
run and model facts, data descriptions, training samples and live jobs, evaluation data, chart
helpers, inference (options, lookups, scoring, explanations), evaluation tables, sidebar.

Every cached loader takes the file's modification time as part of its key. An app that is
running while ``mlda-train`` rewrites the artefacts therefore reads the new files on the next
interaction instead of serving stale numbers. Nothing here fits a model on test rows: the live
pages sample TRAINING rows only, and the evaluation page re-applies stored thresholds to the
stored held-out scores.
"""

# ruff: noqa: E402  (the path shim below has to run before the mlda imports)
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from datetime import date, time
from pathlib import Path
from time import perf_counter
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterGrid

# A page that Streamlit starts on its own (refresh or bookmark) has not been through the entrypoint, so the
# shim is repeated here: the package can then be imported without being installed (Streamlit Community Cloud).
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlda import persistence
from mlda.baselines import BASELINE_ZOO
from mlda.cleaning import status_103i_at_accident
from mlda.config import (
    ACCIDENT_CATEGORICAL,
    COST_FP_USD,
    GROUP_KEY,
    METRICS_DIR,
    MODELS_DIR,
    PLACEHOLDER_AS_LEVEL,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    SOURCES,
    STUDENT_ID,
    TARGET,
    YEAR_MAX,
)
from mlda.data.join import lagged_case_counts, type_from_subunit_code
from mlda.evaluation import curves
from mlda.explain import coefficient_table, shap_summary, transformed_feature_names
from mlda.features import (
    FEATURE_COLUMNS,
    FEATURE_SPEC,
    REQUIRED_INPUT_COLUMNS,
    FeatureEngineer,
)
from mlda.models import MODEL_ZOO, ModelSpec, get_estimator, get_preprocessor
from mlda.operating import OPERATING_KEY
from mlda.preprocessing import build_preprocessor, get_feature_names
from mlda.split import cv_splitter, load_split_indices
from mlda.tuning import (
    estimator_params,
    fold_scores,
    stratified_group_subsample,
    stratified_subsample,
    tune,
)

alt.data_transformers.disable_max_rows()  # chart samples are capped in code, not by Altair

BASE_PATH = PROCESSED_DIR / "base.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
SPLIT_PATH = PROCESSED_DIR / "split_indices.json"
MINE_YEAR_PATH = PROCESSED_DIR / "mine_year_features.parquet"
CASE_HISTORY_PATH = PROCESSED_DIR / "mine_case_history.parquet"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_predictions.parquet"
MSHA_LANDING_PAGE = "https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp"

MAX_TABLE_ROWS = 10_000  # st.dataframe never receives more rows than this
MAX_CHART_ROWS = 20_000  # Altair never receives more rows than this
LIVE_CV_FOLDS = 3
MAX_LIVE_CANDIDATES = 4  # hard cap on the grid points of the live tuning page
# One worker for the live search. Each joblib worker holds its own copy of the sample and of the design
# matrix; measured on a 20,000-row run, four workers peaked at 1.3 to 2.0 GB and one worker at 0.4 to 0.7 GB,
# and Streamlit Community Cloud gives the app less memory than the larger figure.
LIVE_N_JOBS = 1
# The RBF SVM costs about 28 s per candidate even on 4,000 rows (quadratic in rows, plus a five-fold Platt
# calibration inside every fit), so its live grid is cut to two points. The other models need at most 50 s.
LIVE_CANDIDATE_LIMIT = {"svc_rbf": 2}
NONE_LEVEL = "none"  # form option for the structural placeholders (surface accident, no equipment)
CLASS_NAMES = {0: "No lost time", 1: "Lost time"}
MODEL_ORDER = [*MODEL_ZOO, *BASELINE_ZOO]
# seaborn colorblind, so that a model keeps its colour in every chart
MODEL_PALETTE = (
    "#0173B2 #DE8F05 #029E73 #D55E00 #CC78BC #CA9161 "
    "#FBAFE4 #56B4E9 #949494 #000000 #ECE133 #7F7F7F"
).split()
CLASS_SCALE = alt.Scale(domain=list(CLASS_NAMES.values()), range=["#0173B2", "#D55E00"])

# --------------------------------------------------------------------------- artefact readers


def stamp(*paths: Path) -> float:
    """Latest modification time of the given artefacts, 0.0 for a file that does not exist."""
    times = [0.0]
    for path in paths:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times)


@st.cache_data(show_spinner=False)
def _read_json(path: str, version: float) -> Any:
    """Parsed JSON, or None when the file is missing or still being written."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _read_csv(path: str, version: float, index_col: int | None) -> pd.DataFrame | None:
    """Cached CSV read keyed on the file's modification time, so a retrained report refreshes the page."""
    try:
        return pd.read_csv(path, index_col=index_col, encoding="utf-8")
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _read_parquet(
    path: str, version: float, columns: tuple[str, ...] | None, filters: list | None = None
) -> pd.DataFrame | None:
    """Cached parquet read (optionally columns and row filters) keyed on the file's modification time."""
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None, filters=filters)
    except (OSError, ValueError):
        return None


def metrics_json(name: str) -> Any:
    """A file of reports/metrics as parsed JSON (None when absent)."""
    path = METRICS_DIR / name
    return _read_json(str(path), stamp(path))


def metrics_csv(name: str, index_col: int | None = None) -> pd.DataFrame | None:
    """A CSV of reports/metrics as a DataFrame (None when absent)."""
    path = METRICS_DIR / name
    return _read_csv(str(path), stamp(path), index_col)


def parquet(path: Path, columns: tuple[str, ...] | None = None, filters: list | None = None) -> Any:
    """A parquet artefact, optionally restricted to some columns and row filters."""
    return _read_parquet(str(path), stamp(path), columns, filters)


def require(artefact: Any, what: str) -> Any:
    """Stop the page with a pointer to ``mlda-train`` when an artefact is missing or empty."""
    empty = artefact is None or (hasattr(artefact, "__len__") and len(artefact) == 0)
    if empty:
        st.error(f"{what} is not available. Create it with `uv run mlda-train`.")
        st.stop()
    return artefact


def require_files(*paths: Path) -> None:
    """Stop the page when any of the given artefact files does not exist."""
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        st.error(f"Missing artefacts: {', '.join(missing)}. Create them with `uv run mlda-train`.")
        st.stop()


# --------------------------------------------------------------------------- run and model facts


def run_manifest() -> dict[str, Any]:
    """reports/metrics/run_manifest.json, or an empty dict before the first run."""
    return metrics_json("run_manifest.json") or {}


def ranked_models() -> list[str]:
    """Saved models, best first by cross-validated average precision."""
    ranking = run_manifest().get("ranking_by_cv_average_precision") or list(MODEL_ZOO)
    return [name for name in ranking if (MODELS_DIR / f"{name}.joblib").exists()]


def index_of(options: list[str], wanted: str) -> int:
    """Position of ``wanted`` in ``options`` for a widget default, 0 when it is not offered."""
    return options.index(wanted) if wanted in options else 0


def best_model() -> str:
    """Name of the best model according to the run manifest (first saved model otherwise)."""
    names = ranked_models()
    chosen = run_manifest().get("best_model")
    return chosen if chosen in names else (names[0] if names else "")


def model_thresholds(name: str) -> dict[str, Any]:
    """Out-of-fold thresholds of one model from thresholds.json."""
    return (metrics_json("thresholds.json") or {}).get(name, {})


def snapshot_label() -> str:
    """Date and time of the MSHA snapshot that the artefacts were built from."""
    recorded = persistence.data_checksum()["snapshot_utc"]
    recorded = recorded or run_manifest().get("environment", {}).get("data_snapshot_utc")
    return str(recorded).replace("T", " ").replace("Z", " UTC") if recorded else "not recorded"


@st.cache_resource(show_spinner=False)
def _labels() -> tuple[dict[str, str], dict[str, str]]:
    """Model and column label dictionaries of the report figures, loaded once per server process."""
    from mlda import plots  # imported on first use: it pulls in matplotlib and seaborn

    return plots.MODEL_LABELS, plots.LABELS


def model_label(name: str) -> str:
    """Readable model name, shared with the report figures."""
    return _labels()[0].get(name, name.replace("_", " "))


def axis_label(column: str) -> str:
    """Readable column name with units, shared with the report figures."""
    return _labels()[1].get(column, column.replace("_", " "))


@st.cache_resource(show_spinner="Loading model", max_entries=4)
def _pipeline(name: str, version: float) -> Any:
    """Cached load of one saved pipeline; ``version`` is the file's modification time, so a retrained model reloads."""
    return persistence.load_model(name)


def load_pipeline(name: str) -> Any:
    """Fitted Pipeline of one saved model, loaded lazily and cached across sessions."""
    return _pipeline(name, stamp(MODELS_DIR / f"{name}.joblib"))


# --------------------------------------------------------------------------- data descriptions

# origin, meaning of every model input (names and order come from FEATURE_SPEC)
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "SUBUNIT_CD": (
        "Accidents",
        "Mine subunit of the accident: underground, strip or open pit, mill, shop and so on.",
    ),
    "FIPS_STATE_CD": ("Accidents", "State or territory of the mine as a two-digit FIPS code."),
    "UG_LOCATION_CD": (
        "Accidents",
        "Underground location (face, shaft, intersection); 'none' for surface accidents.",
    ),
    "UG_MINING_METHOD_CD": (
        "Accidents",
        "Underground mining method such as longwall; 'none' for surface accidents.",
    ),
    "MINING_EQUIP_CD": (
        "Accidents",
        "Type of mining equipment involved; 'none' when no equipment was involved.",
    ),
    "EQUIP_MFR_NAME": (
        "Accidents",
        "Manufacturer of the equipment involved; 'none' when no equipment was involved.",
    ),
    "CLASSIFICATION_CD": (
        "Accidents",
        "MSHA accident classification, such as handling of materials.",
    ),
    "ACCIDENT_TYPE_CD": (
        "Accidents",
        "Accident type, for example fall of the victim or struck by a falling object.",
    ),
    "OCCUPATION": ("Accidents", "Job title of the injured worker."),
    "ACTIVITY_CD": ("Accidents", "Activity the worker was carrying out when injured."),
    "INJURY_SOURCE_CD": ("Accidents", "Object or substance that directly produced the injury."),
    "NATURE_INJURY_CD": (
        "Accidents",
        "Nature of the injury, for example sprain, fracture or laceration.",
    ),
    "INJ_BODY_PART_CD": ("Accidents", "Part of the body injured."),
    "COAL_METAL_IND": ("Accidents", "Sector: C for coal, M for metal and non-metal."),
    "PRIMARY_CANVASS": (
        "Mines join",
        "Commodity group of the mine: coal, stone, metal, nonmetal or sand and gravel.",
    ),
    "type_at_time": (
        "Production join",
        "Mine type last year (Underground, Surface or Facility) from the subunits that reported.",
    ),
    "status_103i_at_accident": (
        "Mines join",
        "Section 103(i) inspection status in force at the accident date (four levels).",
    ),
    "mine_size_band": (
        "Production join",
        "Band of last year's average employees: 1-4, 5-19, 20-49, 50-199, 200+.",
    ),
    "SAFETY_COMMITTEE_IND": ("Mines join", "Y when the mine has a safety committee."),
    "PART48_TRAINING": ("Mines join", "Y when the mine runs a Part 48 training programme."),
    "PORTABLE_OPERATION": ("Mines join", "Y for a portable operation that moves between sites."),
    "accident_weekday": ("Derived", "Weekday of the accident, 0 for Monday to 6 for Sunday."),
    "accident_month": ("Derived", "Month of the accident, 1 to 12."),
    "TOT_EXPER": ("Accidents", "Total mining experience of the injured worker in years."),
    "MINE_EXPER": ("Accidents", "Years of experience at this mine."),
    "JOB_EXPER": ("Accidents", "Years in the current job."),
    "log1p_tot_exper": ("Derived", "log(1 + total experience), which reduces the right skew."),
    "exper_ratio_mine": (
        "Derived",
        "Mine experience as a share of total experience, between 0 and 1.",
    ),
    "exper_ratio_job": (
        "Derived",
        "Job experience as a share of total experience, between 0 and 1.",
    ),
    "no_injuries_capped": ("Accidents", "People injured in the accident, capped at 10."),
    "accident_hour": ("Derived", "Hour of the accident (0 to 23) decoded from the HHMM time code."),
    "shift_begin_hour": ("Derived", "Hour the shift began (0 to 23)."),
    "hours_into_shift": (
        "Derived",
        "Hours from shift start to the accident; missing above 16 hours.",
    ),
    "latitude": ("Mines join", "Mine latitude in degrees; missing when outside the United States."),
    "longitude": (
        "Mines join",
        "Mine longitude in degrees; missing when outside the United States.",
    ),
    "prev_hours": (
        "Production join",
        "Employee-hours reported by the mine in the previous calendar year.",
    ),
    "log1p_prev_hours": ("Derived", "log(1 + previous-year employee-hours)."),
    "prev_employees": (
        "Production join",
        "Average number of employees in the previous calendar year.",
    ),
    "prev_hours_per_employee": (
        "Derived",
        "Previous-year hours divided by previous-year employees.",
    ),
    "prev_incidents": (
        "Accident history",
        "Cases of any degree filed at the mine last year, before the accident date.",
    ),
    "prev_incident_rate_200k": (
        "Derived",
        "log(1 + previous-year cases per 200,000 hours), rate capped at 50.",
    ),
    "prev_lost_time_rate_200k": (
        "Derived",
        "log(1 + last year's lost-time cases per 200,000 hours), rate capped at 50.",
    ),
    "prev_lost_time_share": (
        "Derived",
        "Lost-time cases as a share of valid cases (01 to 06) last year.",
    ),
    "is_night": ("Derived", "1 when the accident hour is between 18:00 and 05:59."),
    "is_weekend": ("Derived", "1 when the accident happened on a Saturday or Sunday."),
    "new_to_mine": ("Derived", "1 when the worker has under one year at this mine."),
    "exper_inconsistent": (
        "Derived",
        "1 when job or mine experience exceeded total experience (set to missing).",
    ),
    "is_contractor": ("Accidents", "1 when a contractor employed the injured worker."),
    "immed_reportable": (
        "Accidents",
        "1 when the accident needed immediate (15 minute) notification to MSHA.",
    ),
    "prev_year_matched": (
        "Production join",
        "1 when the mine reported hours in the previous year.",
    ),
    "tot_exper_missing": ("Derived", "1 when total experience is missing."),
    "accident_hour_missing": ("Derived", "1 when the accident time was not reported."),
    "hours_into_shift_missing": (
        "Derived",
        "1 when the time into the shift could not be computed.",
    ),
    "prev_share_missing": ("Derived", "1 when the mine had no valid case in the previous year."),
    "coords_missing": ("Derived", "1 when the mine coordinates are missing or invalid."),
}

# file, rows, columns, primary key, grain of the three pinned MSHA files (docs/data_profile.md section 0)
SOURCE_FACTS: tuple[tuple[str, str, int, int, str, str], ...] = (
    (
        "Accidents",
        "accidents",
        275_220,
        57,
        "DOCUMENT_NO",
        "one injured person on one accident form (Form 7000-1)",
    ),
    ("Mines", "mines", 92_060, 59, "MINE_ID", "one mine, attributes as at the snapshot"),
    (
        "Production",
        "production",
        658_922,
        11,
        "MINE_ID + CALENDAR_YR + SUBUNIT_CD",
        "hours and employees per mine, year and subunit",
    ),
)


def source_table() -> pd.DataFrame:
    """Rows, columns, size, key and URL of the three source files (checksums are on the About page)."""
    rows = []
    for title, key, n_rows, n_columns, primary_key, grain in SOURCE_FACTS:
        source = SOURCES[key]
        rows.append(
            {
                "source": title,
                "file": source.filename,
                "rows": n_rows,
                "columns": n_columns,
                "size_mb": round(source.size_bytes / 1e6, 1),
                "primary key": primary_key,
                "one row is": grain,
                "url": source.url,
            }
        )
    return pd.DataFrame(rows)


def data_dictionary() -> pd.DataFrame:
    """Feature name, type, origin and meaning for every model input in FEATURE_SPEC."""
    rows = [
        {
            "feature": name,
            "type": kind,
            "origin": FEATURE_INFO[name][0],
            "meaning": FEATURE_INFO[name][1],
        }
        for kind, names in FEATURE_SPEC.items()
        for name in names
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _base_summary(version: float) -> dict[str, Any] | None:
    """Join match rates, target counts and table size read from base.parquet."""
    columns = ("_m_mines", "_m_prev", "_m_same", "DEGREE_INJURY_CD", "DEGREE_INJURY", "lost_time")
    try:
        base = pd.read_parquet(BASE_PATH, columns=list(columns))
        n_columns = len(pq.ParquetFile(BASE_PATH).schema_arrow.names)
    except (OSError, ValueError):
        return None
    labels = base[["DEGREE_INJURY_CD", "DEGREE_INJURY"]].value_counts().reset_index()
    labels = labels.drop_duplicates("DEGREE_INJURY_CD").set_index("DEGREE_INJURY_CD")[
        "DEGREE_INJURY"
    ]
    counts = base["DEGREE_INJURY_CD"].value_counts().sort_index()
    return {
        "n_rows": len(base),
        "n_columns": n_columns,
        "prevalence": float(base["lost_time"].mean()),
        "matches": {
            key: int((base[key] == "both").sum()) for key in ("_m_mines", "_m_prev", "_m_same")
        },
        "target": pd.DataFrame(
            {
                "code": counts.index,
                "meaning": labels.reindex(counts.index).to_numpy(),
                "rows": counts.to_numpy(),
            }
        ),
    }


def base_summary() -> dict[str, Any] | None:
    """Row and column counts, join match counts and degree-code counts of the base table."""
    return _base_summary(stamp(BASE_PATH))


@st.cache_data(show_spinner=False)
def base_filter_options(version: float) -> dict[str, Any]:
    """States and the year range available in the base table, for the Data Explorer filters."""
    frame = pd.read_parquet(BASE_PATH, columns=["STATE", "CAL_YR"])
    return {
        "states": sorted(frame["STATE"].dropna().unique()),
        "years": (int(frame["CAL_YR"].min()), int(frame["CAL_YR"].max())),
    }


EXPLORER_COLUMNS = (
    "MINE_ID",
    "ACCIDENT_DT",
    "CAL_YR",
    "STATE",
    "COAL_METAL_IND",
    "SUBUNIT",
    "NATURE_INJURY",
    "INJ_BODY_PART",
    "OCCUPATION",
    "TOT_EXPER",
    "NO_INJURIES",
    "DEGREE_INJURY_CD",
    "lost_time",
)


@st.cache_data(show_spinner=False, max_entries=20)
def base_sample(
    sector: str, years: tuple[int, int], states: tuple[str, ...], version: float
) -> tuple[pd.DataFrame, int]:
    """Filtered rows of the base table (at most MAX_TABLE_ROWS, random) and the number of matches."""
    filters: list[tuple[str, str, Any]] = [("CAL_YR", ">=", years[0]), ("CAL_YR", "<=", years[1])]
    if sector != "All":
        filters.append(("COAL_METAL_IND", "==", sector))
    if states:
        filters.append(("STATE", "in", list(states)))
    table = pq.read_table(BASE_PATH, columns=list(EXPLORER_COLUMNS), filters=filters)
    keep = np.sort(np.random.default_rng(RANDOM_STATE).permutation(table.num_rows)[:MAX_TABLE_ROWS])
    shown = table.take(keep).to_pandas()  # only the displayed rows become Python objects
    return shown.sort_values("ACCIDENT_DT").reset_index(drop=True), table.num_rows


# --------------------------------------------------------------------------- training samples


@st.cache_data(show_spinner=False, max_entries=6)
def training_sample(
    n_rows: int,
    seed: int,
    grouped: bool,
    version: float,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Selected input columns, labels and mine-year groups of at most ``n_rows`` TRAINING rows (never test rows).

    ``grouped=True`` draws whole mine-years with ``stratified_group_subsample`` (what the model
    zoo uses when it caps rows); ``False`` is a plain class-stratified draw. Only the sampled rows
    are converted to pandas, so the full 100,000-row table is never held as Python objects.
    """
    split = load_split_indices(SPLIT_PATH)
    labels = pq.read_table(FEATURES_PATH, columns=[TARGET, GROUP_KEY])
    train_rows = split.train_idx
    y = labels[TARGET].to_numpy().astype(int)[train_rows]
    group_codes = (
        labels[GROUP_KEY].dictionary_encode().combine_chunks().indices.to_numpy()
    )  # integer ids, no strings
    groups = group_codes[train_rows]
    if grouped:
        chosen = stratified_group_subsample(y, groups, n_rows, random_state=seed)
    else:
        chosen = stratified_subsample(y, n_rows, random_state=seed)
    table = pq.read_table(FEATURES_PATH, columns=list(columns)).take(train_rows[chosen])
    return table.to_pandas().reset_index(drop=True), y[chosen], groups[chosen]


def sample_for(
    n_rows: int, seed: int, grouped: bool, columns: tuple[str, ...] = FEATURE_COLUMNS
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """``training_sample`` keyed on the modification time of the feature table and the split."""
    return training_sample(n_rows, seed, grouped, stamp(FEATURES_PATH, SPLIT_PATH), columns)


@dataclass(frozen=True)
class PreprocessResult:
    """A fitted-and-applied preprocessor on one training sample, kept in memory (not pickled)."""

    numeric_raw: pd.DataFrame
    n_inputs: int
    matrix: np.ndarray
    names: list[str]
    seconds: float


@st.cache_resource(show_spinner="Fitting the preprocessor on the training sample", max_entries=2)
def preprocess_sample(
    options: tuple[str | None, str, str, str | None, float], n_rows: int, seed: int, version: float
) -> PreprocessResult:
    """Fit ``build_preprocessor(*options)`` on a stratified training sample and transform it."""
    imputer, scaler, encoder, outlier_rule, min_frequency = options
    X, y, _ = training_sample(n_rows, seed, False, version)
    preprocessor = build_preprocessor(imputer, scaler, encoder, outlier_rule, min_frequency)
    started = perf_counter()
    matrix = preprocessor.fit_transform(X, y)
    seconds = perf_counter() - started
    return PreprocessResult(
        X[FEATURE_SPEC["numeric"]],
        X.shape[1],
        np.asarray(matrix),
        get_feature_names(preprocessor),
        seconds,
    )


def live_limit(spec: ModelSpec) -> int:
    """Largest number of grid points the live page searches for one model."""
    return LIVE_CANDIDATE_LIMIT.get(spec.name, MAX_LIVE_CANDIDATES)


def live_grid(spec: ModelSpec) -> list[dict[str, list[Any]]]:
    """At most ``live_limit`` points of the full grid, evenly spaced through ``ParameterGrid`` order.

    The first and the last point of the full grid are always included (as in ``models.quick_grid``) and the
    rest are spread between them, so a live search compares real combinations from the committed search
    space instead of the two extremes only. Each point is a one-value grid that ``tune`` accepts.
    """
    points = list(ParameterGrid(spec.param_grid))
    picks = np.unique(
        np.linspace(0, len(points) - 1, min(live_limit(spec), len(points))).round().astype(int)
    )
    return [{key: [value] for key, value in points[i].items()} for i in picks]


PREPROCESSING_CHOICE_KEY = "preprocessing_choice"  # set by page 3, read by page 4 (not a widget key)
PREPROCESSING_OPTION_NAMES = ("numeric_imputer", "scaler", "encoder", "outlier_rule", "min_frequency")


def preprocessing_choice(session_state: Any) -> dict[str, Any] | None:
    """The options last chosen on the Preprocessing page, or None if that page has not been opened.

    Streamlit drops the state of widgets that a page no longer renders, so page 3 copies its choice
    into a plain session key that survives the switch to the training page.
    """
    choice = session_state.get(PREPROCESSING_CHOICE_KEY)
    if not choice:
        return None
    return {name: choice[name] for name in PREPROCESSING_OPTION_NAMES if name in choice}


@st.cache_data(show_spinner=False, max_entries=24)
def live_tuning(
    model_name: str,
    n_rows: int,
    seed: int,
    version: float,
    preprocessing: tuple[tuple[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Tune one model on a training sample with 3 grouped folds and the reduced grid.

    The reduced grid is ``live_grid`` (at most ``live_limit`` points of the full grid). The search
    runs with LIVE_N_JOBS workers. ``preprocessing`` holds ``build_preprocessor`` options as (name,
    value) pairs, for example the choice made on the Preprocessing page; None keeps the model's own
    preprocessing. The result holds plain tables and numbers so that it can be cached by (model,
    sample size, seed, preprocessing) without pickling a fitted pipeline.
    """
    spec = MODEL_ZOO[model_name]
    X, y, groups = training_sample(n_rows, seed, True, version)
    caps = [cap for cap in (spec.sample_cap, spec.quick_sample_cap) if cap]
    live_spec = replace(
        spec, param_grid=live_grid(spec), search="grid", sample_cap=min(caps) if caps else None
    )
    cv = cv_splitter(n_splits=LIVE_CV_FOLDS, random_state=seed)
    preprocessor = None if preprocessing is None else build_preprocessor(**dict(preprocessing))
    result = tune(live_spec, X, y, groups, cv, quick=False, n_jobs=LIVE_N_JOBS, preprocessor=preprocessor)
    return {
        "cv_results": result.cv_results_df,
        "best_params": estimator_params(result),
        "fold_scores": fold_scores(result).tolist(),
        "seconds": result.elapsed_seconds,
        "n_sample": len(X),
        "n_fit": len(result.fit_idx),
        "n_candidates": result.n_candidates,
    }


# --------------------------------------------------------------------------- evaluation data


def held_out_scores() -> pd.DataFrame | None:
    """Held-out labels, sector and one ``proba_<model>`` column per model and baseline."""
    return parquet(TEST_PREDICTIONS_PATH)


def score_models(scores: pd.DataFrame) -> list[str]:
    """Model and baseline names present in the stored test scores."""
    return [column[len("proba_") :] for column in scores.columns if column.startswith("proba_")]


CURVE_COLUMNS = ("fpr", "tpr", "precision", "recall", "mean_predicted", "fraction_positive")


@st.cache_data(show_spinner=False)
def curve_frame(kind: str, version: float, max_points: int = 300) -> pd.DataFrame:
    """ROC, precision-recall or calibration points of every stored model, thinned for Altair."""
    scores = parquet(TEST_PREDICTIONS_PATH)
    y = scores["y_true"].to_numpy()
    frames = []
    for name in score_models(scores):
        points = curves(y, scores[f"proba_{name}"].to_numpy())[kind]
        points = points[[col for col in CURVE_COLUMNS if col in points.columns]]
        if kind == "calibration":
            points = points.dropna()
        if len(points) > max_points:
            points = points.iloc[np.unique(np.linspace(0, len(points) - 1, max_points).astype(int))]
        frames.append(points.assign(model=name))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- chart helpers


def bar_chart(
    data: pd.DataFrame,
    category: str,
    value: str,
    *,
    title: str | None = None,
    signed: bool = False,
    height: int | None = None,
) -> alt.Chart:
    """Horizontal bars in the row order of ``data``; ``signed=True`` colours positive and negative bars apart."""
    encoding: dict[str, Any] = {
        "y": alt.Y(
            f"{category}:N", sort=data[category].tolist(), title=None, axis=alt.Axis(labelLimit=260)
        ),
        "x": alt.X(f"{value}:Q", title=title or value),
        "tooltip": [category, value],
    }
    if signed:
        encoding["color"] = alt.condition(
            alt.datum[value] > 0, alt.value("#D55E00"), alt.value("#0173B2")
        )
    chart = alt.Chart(data).mark_bar().encode(**encoding)
    return chart.properties(height=height or max(120, 22 * len(data)))


def histogram(
    data: pd.DataFrame, column: str, by: str | None = None, bins: int = 40, title: str | None = None
) -> alt.Chart:
    """Histogram of one numeric column, overlaid by class when ``by`` is given."""
    chart = (
        alt.Chart(data)
        .mark_bar(opacity=0.65 if by else 1.0)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=bins), title=title or axis_label(column)),
            y=alt.Y("count():Q", stack=None, title="Rows"),
        )
    )
    return chart.encode(color=alt.Color(f"{by}:N", scale=CLASS_SCALE, title=None)) if by else chart


def boxplot(data: pd.DataFrame, column: str, by: str) -> alt.Chart:
    """Box plot of one numeric column for each class."""
    return (
        alt.Chart(data)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X(f"{by}:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{column}:Q", title=axis_label(column)),
            color=alt.Color(f"{by}:N", scale=CLASS_SCALE, legend=None),
        )
    )


def heatmap(matrix: pd.DataFrame, title: str) -> alt.Chart:
    """Square correlation heatmap with a diverging scale fixed to [-1, 1]."""
    long = (
        matrix.rename_axis(index="feature_a", columns="feature_b")
        .stack()
        .rename("value")
        .reset_index()
    )
    scale = alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)
    return (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X(
                "feature_b:N",
                title=None,
                sort=list(matrix.columns),
                axis=alt.Axis(labelOverlap=False),
            ),
            y=alt.Y(
                "feature_a:N",
                title=None,
                sort=list(matrix.index),
                axis=alt.Axis(labelOverlap=False),
            ),
            color=alt.Color("value:Q", scale=scale, title=title),
            tooltip=["feature_a", "feature_b", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=560)
    )


def model_scale() -> alt.Scale:
    """Fixed colour for every model label, so that choosing fewer models does not recolour the rest."""
    return alt.Scale(domain=[model_label(name) for name in MODEL_ORDER], range=MODEL_PALETTE)


def line_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    colour: str,
    *,
    x_title: str,
    y_title: str,
    diagonal: bool = False,
) -> alt.Chart:
    """Lines per model on a unit square, with an optional y = x reference."""
    unit = alt.Scale(domain=[0, 1])
    lines = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:Q", scale=unit, title=x_title),
            y=alt.Y(f"{y}:Q", scale=unit, title=y_title),
            color=alt.Color(
                f"{colour}:N",
                scale=model_scale(),
                legend=alt.Legend(orient="bottom", columns=2),
                title=None,
            ),
            tooltip=[
                colour,
                alt.Tooltip(f"{x}:Q", format=".3f"),
                alt.Tooltip(f"{y}:Q", format=".3f"),
            ],
        )
    )
    if not diagonal:
        return lines.properties(height=360)
    reference = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color="grey", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    return (reference + lines).properties(height=360)


# --------------------------------------------------------------------------- inference: options and lookups

# raw form column -> description column of base.parquet (None: the value is its own label)
CODE_FIELDS: dict[str, str | None] = {
    "SUBUNIT_CD": "SUBUNIT",
    "FIPS_STATE_CD": "STATE",
    "UG_LOCATION_CD": "UG_LOCATION",
    "UG_MINING_METHOD_CD": "UG_MINING_METHOD",
    "MINING_EQUIP_CD": "MINING_EQUIP",
    "EQUIP_MFR_NAME": None,
    "CLASSIFICATION_CD": "CLASSIFICATION",
    "ACCIDENT_TYPE_CD": "ACCIDENT_TYPE",
    "OCCUPATION": None,
    "ACTIVITY_CD": "ACTIVITY",
    "INJURY_SOURCE_CD": "INJURY_SOURCE",
    "NATURE_INJURY_CD": "NATURE_INJURY",
    "INJ_BODY_PART_CD": "INJ_BODY_PART",
    "COAL_METAL_IND": None,
    "PRIMARY_CANVASS": None,
    "IMMED_NOTIFY_CD": "IMMED_NOTIFY",
}
NONE_COLUMNS = (
    *PLACEHOLDER_AS_LEVEL,
    "IMMED_NOTIFY_CD",
)  # columns whose "none" option is stored as NaN
SECTOR_LABELS = {"C": "C - Coal", "M": "M - Metal and non-metal"}
STATUS_103I_LEVELS = ("Never had 103I", "In 103I at accident", "Formerly 103I", "Unknown")
MINE_PROFILE_COLUMNS = (
    "PRIMARY_CANVASS",
    "SAFETY_COMMITTEE_IND",
    "PART48_TRAINING",
    "PORTABLE_OPERATION",
    "LATITUDE",
    "LONGITUDE",
    "COAL_METAL_IND",
    "FIPS_STATE_CD",
    "STATE",
    "CURRENT_103I",
    "CURRENT_103I_DT",
)
HISTORY_COLUMNS = (
    "prev_hours",
    "prev_employees",
    "prev_year_matched",
    "prev_incidents",
    "prev_valid_cases",
    "prev_lost_time_cases",
)
RAW_COLUMNS = (
    *REQUIRED_INPUT_COLUMNS,
    "MINE_ID",
    "STATE",
    "type_at_time",
    "status_103i_at_accident",
)
BATCH_REQUIRED = (
    "MINE_ID",
    "ACCIDENT_DT",
    "ACCIDENT_TIME",
    "SHIFT_BEGIN_TIME",
    *ACCIDENT_CATEGORICAL,
    "TOT_EXPER",
    "MINE_EXPER",
    "JOB_EXPER",
    "NO_INJURIES",
)
BATCH_OPTIONAL = ("CONTRACTOR_ID", "IMMED_NOTIFY_CD")
MAX_BATCH_ROWS = 20_000


def _tidy(label: str) -> str:
    """Sentence-case an all-capitals description from the codebook; leave short or mixed-case labels alone."""
    return label.capitalize() if label.isupper() and len(label) > 3 else label


@st.cache_data(show_spinner=False)
def _form_options(version: float) -> dict[str, dict[str, str]]:
    """For each raw categorical field, code -> "code - description" ordered by frequency."""
    options: dict[str, dict[str, str]] = {}
    for column, description in CODE_FIELDS.items():
        columns = [column] if description is None else [column, description]
        table = (
            pq.read_table(BASE_PATH, columns=columns)
            .group_by(columns)
            .aggregate([([], "count_all")])
        )
        counts = table.to_pandas().dropna().sort_values("count_all", ascending=False)
        if description is None:
            options[column] = {str(code): str(code) for code in counts[column].drop_duplicates()}
        else:
            names = counts.drop_duplicates(column).set_index(column)[description]
            options[column] = {
                str(code): f"{code} - {_tidy(str(name))}" for code, name in names.items()
            }
    options["COAL_METAL_IND"] = {
        code: SECTOR_LABELS[code] for code in options["COAL_METAL_IND"] if code in SECTOR_LABELS
    }
    for column in NONE_COLUMNS:
        options[column] = {NONE_LEVEL: "none (not applicable)", **options[column]}
    return options


def form_options() -> dict[str, dict[str, str]]:
    """Selectable levels of every raw categorical field, most frequent first."""
    return _form_options(stamp(BASE_PATH))


def level_select(label: str, column: str, key: str, wanted: str | None = None) -> str:
    """Selectbox over the levels of one raw field, defaulting to ``wanted`` or the most frequent real level."""
    levels = form_options()[column]
    keys = list(levels)
    index = keys.index(wanted) if wanted in keys else int(column in NONE_COLUMNS)
    return st.selectbox(label, keys, index=index, format_func=levels.get, key=key)


@st.cache_data(show_spinner=False)
def _mine_profiles(version: float) -> pd.DataFrame:
    """One row per mine: its latest known attributes and the number of accident rows."""
    columns = ["MINE_ID", "ACCIDENT_DT", *MINE_PROFILE_COLUMNS]
    base = pd.read_parquet(BASE_PATH, columns=columns).sort_values("ACCIDENT_DT")
    counts = base["MINE_ID"].value_counts()
    latest = (
        base.drop_duplicates("MINE_ID", keep="last")
        .drop(columns="ACCIDENT_DT")
        .set_index("MINE_ID")
    )
    return latest.assign(n_cases=counts.reindex(latest.index))


def mine_profiles() -> pd.DataFrame:
    """Latest attributes of every mine in the base table, indexed by MINE_ID."""
    return _mine_profiles(stamp(BASE_PATH))


def with_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Previous-year exposure and case counts for accident rows, built like the training joins.

    ``frame`` needs MINE_ID and ACCIDENT_DT. Hours and employees come from the mine-year table at
    CAL_YR - 1; case counts come from ``mlda.data.join.lagged_case_counts`` over the persisted case
    history (cases filed strictly before the accident date). Rows without previous-year production
    get ``prev_year_matched`` False and NaN counts, exactly as in the training table.
    """
    accident = pd.to_datetime(frame["ACCIDENT_DT"], errors="coerce")
    keys = pd.DataFrame(
        {
            "MINE_ID": frame["MINE_ID"].astype(str).str.zfill(7).to_numpy(),
            "ACCIDENT_DT": accident.to_numpy(),
        }
    )
    keys["CAL_YR"] = keys["ACCIDENT_DT"].dt.year.fillna(0).astype(int)
    mines = sorted(set(keys["MINE_ID"]))
    years = pd.read_parquet(MINE_YEAR_PATH, filters=[("MINE_ID", "in", mines)])
    cases = pd.read_parquet(CASE_HISTORY_PATH, filters=[("MINE_ID", "in", mines)])
    previous = keys.assign(CALENDAR_YR=keys["CAL_YR"] - 1).merge(
        years[["MINE_ID", "CALENDAR_YR", "hours", "employees", "type"]],
        on=["MINE_ID", "CALENDAR_YR"],
        how="left",
        validate="many_to_one",
    )
    matched = previous["hours"].notna().to_numpy()
    counts = lagged_case_counts(keys.assign(prev_year_matched=matched), cases)
    out = pd.DataFrame(
        {
            "prev_hours": previous["hours"].astype(float).to_numpy(),
            "prev_employees": previous["employees"].astype(float).to_numpy(),
            "prev_year_matched": matched,
            "type_at_time": previous["type"].to_numpy(),
        },
        index=frame.index,
    )
    for column in counts.columns:
        out[column] = counts[column].to_numpy()
    return out


def mine_history(mine_id: str, accident_date: date) -> dict[str, Any]:
    """``with_history`` for a single mine and date, as a dictionary."""
    row = pd.DataFrame({"MINE_ID": [mine_id], "ACCIDENT_DT": [pd.Timestamp(accident_date)]})
    return with_history(row).iloc[0].to_dict()


def subunit_type(subunit: pd.Series) -> pd.Series:
    """Fallback mine type from the accident subunit: ``mlda.data.join.type_from_subunit_code`` applied per row.

    It is the rule the training join uses for accidents without a previous-year production row; only
    surrounding whitespace is stripped first, because an uploaded CSV may carry it.
    """
    return subunit.astype("string").str.strip().map(type_from_subunit_code)


def status_103i_for(mine_ids: pd.Series, accident_dates: pd.Series) -> pd.Series:
    """Section 103(i) status in force at each accident, looked up from the mine like the training feature.

    The mine's ``CURRENT_103I`` status and its date come from the latest known profile of the mine
    (``mine_profiles``); ``mlda.cleaning.status_103i_at_accident`` then applies the training rule (a status
    dated after the accident, a blank status or an unknown mine give "Unknown"). Rebuilding it this way
    gives the stored feature for every training row, so the form default and the batch path match training.
    """
    profiles = mine_profiles()
    keys = mine_ids.astype(str).str.strip().str.zfill(7)
    levels = status_103i_at_accident(
        keys.map(profiles["CURRENT_103I"]),
        pd.to_datetime(keys.map(profiles["CURRENT_103I_DT"]), errors="coerce"),
        pd.to_datetime(accident_dates, errors="coerce"),
    )
    return pd.Series(levels.to_numpy(), index=mine_ids.index, name="status_103i_at_accident")


def mine_status_103i(mine_id: str, accident_date: date | pd.Timestamp) -> str:
    """``status_103i_for`` for a single mine and date, as one of STATUS_103I_LEVELS."""
    keys = pd.Series([mine_id])
    return str(status_103i_for(keys, pd.Series([pd.Timestamp(accident_date)])).iloc[0])


def hhmm(clock: time) -> str:
    """Clock time as the zero-padded HHMM code that Accidents.txt uses."""
    return f"{clock.hour:02d}{clock.minute:02d}"


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill the derived mine type where no previous-year production exists and order the columns."""
    frame["type_at_time"] = frame["type_at_time"].where(
        frame["type_at_time"].notna(), subunit_type(frame["SUBUNIT_CD"])
    )
    for column in NONE_COLUMNS:
        if column not in frame.columns:
            continue
        is_none = frame[column].astype("string").eq(NONE_LEVEL).fillna(False).astype(bool)
        frame[column] = frame[column].mask(is_none)
    return frame.loc[:, [*RAW_COLUMNS, *[c for c in frame.columns if c not in RAW_COLUMNS]]]


def raw_case_frame(fields: dict[str, Any]) -> pd.DataFrame:
    """Turn form values into a one-row raw DataFrame with every column FeatureEngineer needs.

    ``fields`` maps raw column names to values. Columns the form leaves out get neutral defaults:
    missing values and no previous-year match. The 103(i) status is looked up from ``MINE_ID`` and the
    accident date unless ``fields`` gives it. The "none" option of a placeholder field becomes NaN,
    which ``engineer_features`` turns back into the "none" level.
    """
    unknown = sorted(set(fields) - set(RAW_COLUMNS))
    if unknown:
        raise KeyError(f"not raw input columns: {unknown}")
    row: dict[str, Any] = dict.fromkeys(RAW_COLUMNS, np.nan)
    row.update({"prev_year_matched": False, "NO_INJURIES": 1})
    row.update(fields)
    frame = pd.DataFrame([row], columns=list(RAW_COLUMNS))
    frame["ACCIDENT_DT"] = pd.to_datetime(frame["ACCIDENT_DT"])
    if "status_103i_at_accident" not in fields:
        frame["status_103i_at_accident"] = status_103i_for(
            frame["MINE_ID"], frame["ACCIDENT_DT"]
        ).to_numpy()
    frame["prev_year_matched"] = frame["prev_year_matched"].astype(bool)
    return _finish(frame)


def coordinate(value: Any, bound: float) -> float:
    """A stored coordinate clipped to the widget range; 0.0 when the mine has none."""
    return 0.0 if value is None or pd.isna(value) else float(np.clip(value, -bound, bound))


def cost_note(probability: float, ratio: int, optimal: float | None) -> str:
    """Expected cost of reviewing against not reviewing one case at a cost ratio of ``ratio``:1."""
    review, skip = (1 - probability) * COST_FP_USD, probability * ratio * COST_FP_USD
    cheaper = "review" if review < skip else "routine handling"
    threshold = "not available" if optimal is None else f"{optimal:.3f}"
    return (
        f"At {ratio}:1 an unnecessary review costs USD {COST_FP_USD} and a missed case USD "
        f"{ratio * COST_FP_USD}. For this "
        f"case reviewing costs USD {review:.0f} in expectation and not reviewing USD {skip:.0f}, so "
        f"{cheaper} is cheaper. "
        f"The break-even probability is {1 / (1 + ratio):.3f} if the score is calibrated; the model's out-of-fold "
        f"cost-optimal threshold is {threshold}."
    )


def score_frame(pipeline: Any, raw: pd.DataFrame) -> np.ndarray:
    """Probability of a lost-time injury for each raw row: FeatureEngineer, then the saved pipeline."""
    return pipeline.predict_proba(FeatureEngineer().transform(raw))[:, 1]


def template_csv() -> str:
    """Five real accident rows from the last year with the raw columns a batch upload needs, as CSV text."""
    columns = (*BATCH_REQUIRED, *BATCH_OPTIONAL)
    base = pd.read_parquet(BASE_PATH, columns=list(columns), filters=[("CAL_YR", "==", YEAR_MAX)])
    return base.head(5).to_csv(index=False)


def prepare_batch(upload: pd.DataFrame) -> pd.DataFrame:
    """Complete an uploaded table of raw accidents so that FeatureEngineer can read it.

    Required columns are listed in BATCH_REQUIRED. Mine attributes, the 103(i) status at the accident
    date and the previous-year history are looked up from MINE_ID and the accident date unless the upload
    already has those columns.
    """
    missing = [column for column in BATCH_REQUIRED if column not in upload.columns]
    if missing:
        raise ValueError(f"the file lacks required columns: {', '.join(missing)}")
    if len(upload) > MAX_BATCH_ROWS:
        raise ValueError(f"the file has {len(upload):,} rows; the limit is {MAX_BATCH_ROWS:,}")
    frame = upload.copy()
    frame["MINE_ID"] = frame["MINE_ID"].astype(str).str.strip().str.zfill(7)
    frame["ACCIDENT_DT"] = pd.to_datetime(frame["ACCIDENT_DT"], errors="coerce")
    if frame["ACCIDENT_DT"].isna().any():
        raise ValueError("ACCIDENT_DT must be a valid date (YYYY-MM-DD) in every row")
    for column, options in form_options().items():
        if column in frame.columns and all(
            code.isdigit() for code in options if code != NONE_LEVEL
        ):
            width = max(len(code) for code in options if code != NONE_LEVEL)
            frame[column] = frame[column].where(
                frame[column].isna(), frame[column].astype(str).str.strip().str.zfill(width)
            )
    profiles = mine_profiles()
    for column in (
        "PRIMARY_CANVASS",
        "SAFETY_COMMITTEE_IND",
        "PART48_TRAINING",
        "PORTABLE_OPERATION",
        "LATITUDE",
        "LONGITUDE",
        "STATE",
    ):
        if column not in frame.columns:
            frame[column] = frame["MINE_ID"].map(profiles[column])
    if not set(HISTORY_COLUMNS) <= set(frame.columns):
        history = with_history(frame)
        for column in [*HISTORY_COLUMNS, "type_at_time"]:
            frame[column] = history[column].to_numpy()
    if "status_103i_at_accident" not in frame.columns:
        frame["status_103i_at_accident"] = status_103i_for(
            frame["MINE_ID"], frame["ACCIDENT_DT"]
        ).to_numpy()
    for column in BATCH_OPTIONAL:
        if column not in frame.columns:
            frame[column] = np.nan
    if "type_at_time" not in frame.columns:
        frame["type_at_time"] = np.nan
    frame["prev_year_matched"] = (
        frame["prev_year_matched"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    )
    return _finish(frame)


# --------------------------------------------------------------------------- inference: explanations


def linear_contributions(pipeline: Any, features: pd.DataFrame) -> pd.DataFrame | None:
    """Coefficient times transformed value for one row of a linear model (None for other models)."""
    estimator = get_estimator(pipeline)
    if not hasattr(estimator, "coef_"):
        return None
    values = np.asarray(get_preprocessor(pipeline).transform(features), dtype=float)[0]
    table = pd.DataFrame(
        {
            "feature": transformed_feature_names(pipeline),
            "value": values,
            "contribution": values * np.asarray(estimator.coef_, dtype=float).ravel(),
        }
    )
    table = table[table["contribution"] != 0]
    return table.reindex(table["contribution"].abs().sort_values(ascending=False).index)


def row_shap(pipeline: Any, features: pd.DataFrame) -> pd.DataFrame | None:
    """SHAP value of every transformed column for one row, or None when no fast explainer exists."""
    summary = shap_summary(pipeline, features, max_rows=1)
    if summary is None:
        return None
    table = pd.DataFrame({"feature": summary.feature_names, "shap": summary.values[0]})
    return table.reindex(table["shap"].abs().sort_values(ascending=False).index)


def global_importance(pipeline: Any) -> pd.DataFrame | None:
    """Impurity importances of a tree model (None for models without ``feature_importances_``)."""
    table = coefficient_table(pipeline)
    return table if table is not None and "importance" in table.columns else None


def shap_supported(pipeline: Any) -> bool:
    """True when ``mlda.explain.shap_summary`` has a fast explainer for the final estimator.

    A tree explainer exists for the tree models (including histogram gradient boosting, which has no
    ``feature_importances_``) and a linear explainer for models with coefficients; the other models would
    fall back to a permutation explainer that is too slow for a web request.
    """
    estimator = get_estimator(pipeline)
    return (
        hasattr(estimator, "coef_")
        or hasattr(estimator, "feature_importances_")
        or isinstance(estimator, HistGradientBoostingClassifier)
    )


def explain_row(
    model_name: str, pipeline: Any, features: pd.DataFrame
) -> tuple[str, pd.DataFrame, str, bool] | None:
    """Title, table, value column and "is specific to this row" for the best explanation of a model.

    A linear model explains the row itself (coefficient x value). Other models show the permutation
    importance computed for the best model, or the impurity importance of a fitted tree model.
    """
    contributions = linear_contributions(pipeline, features)
    if contributions is not None:
        return (
            "Contribution to the log-odds (coefficient x value)",
            contributions,
            "contribution",
            True,
        )
    explanations = metrics_json("explanations.json") or {}
    if explanations.get("best_model") == model_name and "permutation_importance" in explanations:
        top = pd.DataFrame(explanations["permutation_importance"]["top_10"]).rename(
            columns={"importance_mean": "importance"}
        )
        return "Permutation importance on held-out rows (drop in PR-AUC)", top, "importance", False
    impurity = global_importance(pipeline)
    if impurity is not None:
        return "Impurity importance (global, from the fitted trees)", impurity, "importance", False
    return None


# --------------------------------------------------------------------------- evaluation tables


def fmt_p(p: float | None) -> str:
    """A p-value as text; the stored values underflow to 0.0 for large samples."""
    return "n/a" if p is None or pd.isna(p) else ("< 0.0001" if p < 1e-4 else f"{p:.4f}")


def criteria_rows(success: dict[str, Any], model: str) -> list[dict[str, str]]:
    """Code, criterion, observed value and verdict of S1 to S7 for one model."""
    rows = []
    for code, text in success["criteria"].items():
        check = success["models"][model][code]
        if code == "S3":
            observed = (
                f"recall {check['value']:.3f} at specificity {check['specificity_achieved']:.3f}"
            )
        elif code == "S4":
            costs = (
                f"{r}:1 USD {v['cost_per_1000']:,.0f} (floor {v['floor']:,.0f})"
                for r, v in check["by_ratio"].items()
                if v["counts_for_pass"]
            )
            observed = "; ".join(costs)
        elif code == "S5":
            observed = f"recall {check['value']:.3f} at flag rate {check['flag_rate_achieved']:.2f}"
        else:
            observed = "see the tests below" if code == "S7" else f"{check['value']:.3f}"
        verdict = (
            "Reported"
            if code == "S7"
            else "n/a"
            if check["pass"] is None
            else ("Pass" if check["pass"] else "Fail")
        )
        rows.append({"code": code, "criterion": text, "observed": observed, "verdict": verdict})
    return rows


def significance_tables(tests: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar results (best model against the lookup) and bootstrap intervals of the best model."""
    mcnemar = pd.DataFrame(
        [
            {
                "threshold": key,
                "only the model correct": v["n_a_only"],
                "only the lookup correct": v["n_b_only"],
                "p (rows)": fmt_p(v["p_value"]),
                "p (mine-year clusters)": fmt_p(v.get("clustered_p_value")),
            }
            for key, v in tests["mcnemar_best_vs_lookup"].items()
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": round(v["rows"]["estimate"], 4),
                "95 % interval (rows)": f"{v['rows']['lower']:.3f} to {v['rows']['upper']:.3f}",
                "95 % interval (mine-years)": f"{v['mine_years']['lower']:.3f} to {v['mine_years']['upper']:.3f}",
            }
            for name, v in tests["bootstrap_best_model"]["metrics"].items()
        ]
    )
    return mcnemar, bootstrap


def temporal_table(temporal: dict[str, Any]) -> pd.DataFrame:
    """Held-out metrics of the temporal split for each model and input variant."""
    rows = []
    for name, entry in temporal["models"].items():
        for variant in ("with_status_103i", "without_status_103i", "without_immed_reportable"):
            test, brier = entry[variant]["test"], entry[variant]["brier_by_sector"]
            rows.append(
                {
                    "model": model_label(name),
                    "variant": variant,
                    "ROC-AUC": test["at_0.5"]["roc_auc"],
                    "PR-AUC": test["at_0.5"]["pr_auc"],
                    "Brier": test["at_0.5"]["brier"],
                    "Brier coal": brier.get("C"),
                    "Brier metal": brier.get("M"),
                    "recall at operating point": test[f"at_{OPERATING_KEY}"]["recall"],
                }
            )
    return pd.DataFrame(rows)


def sensitivity_table(sensitivity: dict[str, Any]) -> pd.DataFrame:
    """Change in the headline metrics of the selected model under each altered input or split."""
    deltas = {
        name: v["delta_vs_reference"]
        for name, v in sensitivity["primary_split_variants"]["variants"].items()
    }
    deltas["split grouped by MINE_ID"] = sensitivity["mine_grouped_split"]["model"][
        "delta_vs_primary"
    ]
    pure = sensitivity["excluding_pure_levels"]
    if sensitivity["model"] in pure["models"]:
        deltas[f"without {', '.join(pure['levels'])}"] = pure["models"][sensitivity["model"]][
            "delta"
        ]
    keys = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier": "Brier",
        f"recall_at_{OPERATING_KEY}": "Recall at operating point",
    }
    return pd.DataFrame(
        [
            {"variant": name, **{label: d[key] for key, label in keys.items()}}
            for name, d in deltas.items()
        ]
    )


# --------------------------------------------------------------------------- sidebar and entrypoint


def ensure_entrypoint(page: str) -> None:
    """Send a page that Streamlit started on its own back through the entrypoint, then return to it.

    Because the app has a ``pages/`` folder, a page URL that is opened first (a refresh, a bookmark) is run
    directly by Streamlit's folder-based navigation. The page configuration, the shared sidebar and the page
    list would then be missing, so the page asks the entrypoint to run once and come back.
    """
    if not st.session_state.get("entrypoint_ran"):
        st.session_state["return_to"] = (
            Path(page).resolve().relative_to(Path(__file__).resolve().parent).as_posix()
        )
        st.switch_page("streamlit_app.py")


def render_sidebar() -> None:
    """Project title, student ID, snapshot date and links, shown beside every page."""
    with st.sidebar:
        st.markdown("### Lost-time injury triage")
        st.caption(f"6COSC017C coursework 1 | student ID {STUDENT_ID}")
        st.caption(f"MSHA snapshot: {snapshot_label()}")
        st.link_button("MSHA open data", MSHA_LANDING_PAGE, icon=":material/open_in_new:")
        st.link_button(
            "Accidents codebook", SOURCES["accidents_codebook"].url, icon=":material/menu_book:"
        )
        st.caption("Decision support for case review. It does not replace it.")
