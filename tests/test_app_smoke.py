"""Smoke tests for the Streamlit app, run with ``streamlit.testing.v1.AppTest``.

The tests start the real entrypoint, visit every page, fill the inference form, change a
preprocessing option and run one live tuning job. They read the committed artefacts
(data/processed, models, reports) and are skipped when those are missing, as in a fresh clone
before ``uv run mlda-train``. Two further tests check that the inference path rebuilds the
training features: the lookup of mine history from MINE_ID and the accident date, and the
FeatureEngineer applied to raw rows, must agree with ``data/processed/features.parquet``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid
from streamlit.testing.v1 import AppTest

from mlda.config import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, RANDOM_STATE
from mlda.features import BINARY_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, FeatureEngineer
from mlda.models import MODEL_ZOO

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP = APP_DIR / "streamlit_app.py"
REQUIRED = (
    PROCESSED_DIR / "base.parquet",
    PROCESSED_DIR / "features.parquet",
    PROCESSED_DIR / "split_indices.json",
    PROCESSED_DIR / "mine_year_features.parquet",
    PROCESSED_DIR / "mine_case_history.parquet",
    METRICS_DIR / "run_manifest.json",
    MODELS_DIR / "manifest.json",
    PREDICTIONS_DIR / "test_predictions.parquet",
)
pytestmark = pytest.mark.skipif(
    any(not path.exists() for path in REQUIRED),
    reason="artefacts missing: run `uv run mlda-train` first",
)

PAGES = {
    "pages/1_Data_Explorer.py": "Data explorer",
    "pages/2_EDA.py": "Exploratory analysis",
    "pages/3_Preprocessing.py": "Preprocessing choices",
    "pages/4_Model_Training.py": "Model training",
    "pages/5_Inference.py": "Inference",
    "pages/6_Evaluation.py": "Evaluation",
    "pages/7_About.py": "About this project",
}
SAMPLE_ROWS = 300


def new_app() -> AppTest:
    """A fresh session of the app; the generous timeout covers the first model and data loads."""
    return AppTest.from_file(str(APP), default_timeout=240)


def metrics_of(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def score_default_case(model: str | None = None, **selections: str) -> float:
    """Open the inference page, apply form selections, submit and return the probability shown."""
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    if model:
        app.selectbox(key="inf_model").select(model).run()
    for key, value in selections.items():
        app.selectbox(key=key).select(value)  # form widgets only reach the script on submit
    next(button for button in app.button if button.label == "Score this case").click().run()
    assert not app.exception
    shown = metrics_of(app)["Probability of a lost-time injury"]
    assert shown.endswith("%")
    return float(shown.rstrip("%")) / 100


@pytest.fixture(scope="module")
def sample_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random base-table rows and the stored feature rows for the same positions."""
    import _common as common

    base = pd.read_parquet(
        PROCESSED_DIR / "base.parquet",
        columns=[
            *common.BATCH_REQUIRED,
            *common.BATCH_OPTIONAL,
            "CAL_YR",
            *common.HISTORY_COLUMNS,
            "type_at_time",
        ],
    )
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=list(FEATURE_COLUMNS))
    positions = np.sort(
        np.random.default_rng(RANDOM_STATE).choice(len(base), SAMPLE_ROWS, replace=False)
    )
    return base.iloc[positions].reset_index(drop=True), features.iloc[positions].reset_index(
        drop=True
    )


@pytest.fixture(scope="module", autouse=True)
def app_on_path() -> Iterator[None]:
    """Make ``_common`` importable, as the entrypoint does for the running app."""
    sys.path.insert(0, str(APP_DIR))
    yield
    sys.path.remove(str(APP_DIR))


def test_entrypoint_runs_and_shows_the_first_page() -> None:
    app = new_app().run()
    assert not app.exception
    assert app.title[0].value == "Data explorer"


@pytest.mark.parametrize(("page", "title"), PAGES.items())
def test_every_page_renders_without_exception(page: str, title: str) -> None:
    app = new_app().run()
    app.switch_page(page).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == title


def test_inference_form_returns_a_probability() -> None:
    probability = score_default_case()
    assert 0.0 <= probability <= 1.0


def test_inference_form_values_reach_the_model() -> None:
    """Amputation (nature 100) is lost time in every training row, so the score must rise."""
    sprain = score_default_case()
    amputation = score_default_case(f_NATURE_INJURY_CD="100")
    assert amputation > sprain


@pytest.mark.parametrize("model", ["logreg", "random_forest", "knn"])
def test_inference_scores_with_other_model_families(model: str) -> None:
    assert 0.0 <= score_default_case(model) <= 1.0


def test_inference_unknown_mine_warns_and_still_scores() -> None:
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    app.text_input(key="mine_id").set_value("9999999").run()
    assert any("No previous-year production record" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Score this case").click().run()
    assert not app.exception
    assert "Probability of a lost-time injury" in metrics_of(app)


def test_preprocessing_option_changes_the_transformed_shape() -> None:
    app = new_app().run()
    app.switch_page("pages/3_Preprocessing.py").run()
    default_shape = metrics_of(app)["Transformed shape"]
    app.selectbox(key="pp_encoder").select("ordinal").run()
    assert not app.exception
    shape = metrics_of(app)["Transformed shape"]
    assert shape.endswith(
        f"x {len(FEATURE_COLUMNS)}"
    )  # ordinal encoding keeps one column per input
    assert shape != default_shape
    assert metrics_of(app)["Missing values after"] == "0"


def test_training_page_runs_a_live_search() -> None:
    app = new_app().run()
    app.switch_page("pages/4_Model_Training.py").run()
    app.selectbox(key="mt_model").select("logreg")
    app.slider(key="mt_rows").set_value(2_000)
    app.button(key="mt_run").click().run()
    assert not app.exception
    assert "Best PR-AUC (CV mean)" in metrics_of(app)
    assert any("Best parameters" in markdown.value for markdown in app.markdown)


def test_mine_history_lookup_reproduces_the_training_join(
    sample_rows: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    import _common as common

    base, _ = sample_rows
    looked_up = common.with_history(base)
    for column in common.HISTORY_COLUMNS:
        expected = base[column].to_numpy(dtype=float)
        np.testing.assert_allclose(
            looked_up[column].to_numpy(dtype=float), expected, equal_nan=True, err_msg=column
        )


def test_batch_rows_give_the_stored_training_features(
    sample_rows: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Raw rows through ``prepare_batch`` and ``FeatureEngineer`` equal the stored feature rows.

    Every input is compared, including the 103(i) status: ``prepare_batch`` rebuilds it from the
    mine's recorded status and date with the same function as the training join.
    """
    import _common as common

    base, stored = sample_rows
    upload = base[[*common.BATCH_REQUIRED, *common.BATCH_OPTIONAL]].astype("string")
    upload["ACCIDENT_DT"] = base["ACCIDENT_DT"].dt.strftime("%Y-%m-%d")
    rebuilt = FeatureEngineer().transform(common.prepare_batch(upload))
    assert list(rebuilt.columns) == list(FEATURE_COLUMNS)
    different = []
    for column in FEATURE_COLUMNS:
        if column in NUMERIC_FEATURES:
            same = np.isclose(rebuilt[column], stored[column], equal_nan=True)
        else:
            same = rebuilt[column].astype(str).to_numpy() == stored[column].astype(str).to_numpy()
        if not np.all(same):
            different.append(column)
    assert different == []


def test_single_case_status_is_looked_up_from_the_mine(
    sample_rows: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The form default and ``raw_case_frame`` give the stored 103(i) status at all four levels."""
    import _common as common

    base, stored = sample_rows
    seen: set[str] = set()
    for position, level in enumerate(stored["status_103i_at_accident"].astype(str)):
        if level in seen:
            continue
        seen.add(level)
        row = base.iloc[position]
        raw = common.raw_case_frame({"MINE_ID": row["MINE_ID"], "ACCIDENT_DT": row["ACCIDENT_DT"]})
        assert raw["status_103i_at_accident"].iloc[0] == level
        assert common.mine_status_103i(row["MINE_ID"], row["ACCIDENT_DT"]) == level
    assert seen == set(common.STATUS_103I_LEVELS)


def test_live_grid_spans_the_full_grid_with_at_most_four_points() -> None:
    """Every model's live grid holds real grid points, first and last included, and no more than four."""
    import _common as common

    for name, spec in MODEL_ZOO.items():
        points = list(ParameterGrid(spec.param_grid))
        live = [
            {key: values[0] for key, values in point.items()} for point in common.live_grid(spec)
        ]
        assert 1 <= len(live) <= common.MAX_LIVE_CANDIDATES, name
        assert all(point in points for point in live), name
        assert live[0] == points[0] and live[-1] == points[-1], name
        assert len(live) == min(common.live_limit(spec), len(points)), name


def test_one_row_frame_works_with_feature_engineer_and_a_saved_model() -> None:
    import _common as common

    raw = common.raw_case_frame(
        {
            "MINE_ID": "0200024",
            "ACCIDENT_DT": "2025-06-02",
            "SUBUNIT_CD": "01",
            "NATURE_INJURY_CD": "330",
            "TOT_EXPER": 4.0,
        }
    )
    features = FeatureEngineer().transform(raw)
    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert not features[list(BINARY_FEATURES)].isna().any().any()
    probability = common.score_frame(common.load_pipeline(common.best_model()), raw)
    assert probability.shape == (1,)
    assert 0.0 <= probability[0] <= 1.0


def test_batch_upload_rejects_a_file_without_required_columns() -> None:
    import _common as common

    with pytest.raises(ValueError, match="lacks required columns"):
        common.prepare_batch(pd.DataFrame({"MINE_ID": ["0200024"]}))
