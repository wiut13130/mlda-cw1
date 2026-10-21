"""Smoke tests for the Streamlit app, run with ``streamlit.testing.v1.AppTest``.

The tests start the real entrypoint, visit every page, fill the inference form, change a
preprocessing option and run one live tuning job. They read the committed artefacts
(data/processed, models, reports) and are skipped when those are missing, as in a fresh clone
before ``uv run mlda-train``. Two further tests check that the inference path rebuilds the
training features: the lookup of mine history from MINE_ID and the accident date, and the
FeatureEngineer applied to raw rows, must agree with ``data/processed/features.parquet``. A last
group re-imports ``_common`` with MLDA_LOW_MEMORY set and unset to check the caps of the hosted app.
The Capacity policy tab of the Evaluation page and the workload-rule text of the About page have tests of their own.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from sklearn.model_selection import ParameterGrid
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from mlda.config import (
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    YEAR_MAX,
)
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
    """A well-formed id the reference data does not know is a warning, not an error: it is the new-mine fallback."""
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    app.text_input(key="mine_id").set_value("9999999").run()
    assert not app.error
    assert any(
        "unknown to the reference data" in warning.value and "Mine context" in warning.value
        for warning in app.warning
    )
    next(button for button in app.button if button.label == "Score this case").click().run()
    assert not app.exception
    assert "Probability of a lost-time injury" in metrics_of(app)


@pytest.mark.parametrize("entered", ["abc", "", "02-0024", "0200O24"])
def test_inference_malformed_mine_id_shows_an_error_and_no_result(entered: str) -> None:
    """Text is refused, not padded to "0000abc" and scored; an earlier result is hidden and the batch tab stays."""
    import _common as common

    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    # a result left over from an earlier submission, as after scoring a valid mine and then mistyping the id
    app.session_state["inf_case"] = common.raw_case_frame(
        {"MINE_ID": "0200024", "ACCIDENT_DT": "2025-06-02"}
    )
    app.text_input(key="mine_id").set_value(entered).run()
    assert not app.exception
    assert any("MINE_ID is a 7-digit number, for example 0200024" in error.value for error in app.error)
    assert not any(header.value.startswith("Result for mine") for header in app.subheader)
    assert not any(button.label == "Score this case" for button in app.button)
    assert not app.warning  # the unknown-mine warning is for well-formed ids only
    assert app.get("file_uploader")  # the Batch tab still renders: the page returns instead of calling st.stop()


def test_inference_known_mine_without_previous_year_record_gets_a_notice_not_a_warning() -> None:
    import _common as common

    profiles = common.mine_profiles()
    years = pd.read_parquet(common.MINE_YEAR_PATH, columns=["MINE_ID", "CALENDAR_YR"])
    # the default accident date is in YEAR_MAX, so the previous year is YEAR_MAX - 1
    produced = set(years.loc[years["CALENDAR_YR"] == YEAR_MAX - 1, "MINE_ID"])
    mine = next(mine_id for mine_id in profiles.index if mine_id not in produced)
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    app.text_input(key="mine_id").set_value(mine).run()
    assert not app.error
    assert not app.warning
    assert any(f"No previous-year production record for mine {mine}" in info.value for info in app.info)


def test_inference_mine_with_production_but_no_accidents_gets_a_profile_notice() -> None:
    """Most mines in the production table never had an accident row: exposure is found, the profile is not."""
    import _common as common

    years = pd.read_parquet(common.MINE_YEAR_PATH, columns=["MINE_ID", "CALENDAR_YR"])
    produced = sorted(set(years.loc[years["CALENDAR_YR"] == YEAR_MAX - 1, "MINE_ID"]))
    mine = next(mine_id for mine_id in produced if mine_id not in common.mine_profiles().index)
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    app.text_input(key="mine_id").set_value(mine).run()
    assert not app.error
    assert not app.warning
    assert any("its profile is not on file" in info.value for info in app.info)


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


def test_training_page_tunes_with_the_preprocessing_page_choice() -> None:
    app = new_app().run()
    app.switch_page("pages/3_Preprocessing.py").run()
    app.selectbox(key="pp_encoder").select("ordinal").run()
    app.switch_page("pages/4_Model_Training.py").run()
    app.selectbox(key="mt_model").select("logreg")
    app.slider(key="mt_rows").set_value(2_000)
    app.toggle(key="mt_use_pp").set_value(True).run()
    assert any("encoder = ordinal" in caption.value for caption in app.caption)
    app.button(key="mt_run").click().run()
    assert not app.exception
    change = metrics_of(app)["PR-AUC change from the Preprocessing page choice"]
    assert change != "+0.000"   # ordinal codes read as magnitudes change what logistic regression learns


@pytest.fixture
def reload_common(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str | None], ModuleType]]:
    """Re-import ``_common`` with MLDA_LOW_MEMORY set (or unset) and restore the original module afterwards.

    The caps are module constants read at import time, so the environment variable has to be set before
    the module is executed again. The pages import the constants from the module that is in ``sys.modules``.
    The cached results are cleared on every switch: their keys do not hold the mode, because a server
    process runs in one mode only, so a result computed in the other mode would be served as a hit.
    """
    import _common

    def load(value: str | None) -> ModuleType:
        if value is None:
            monkeypatch.delenv("MLDA_LOW_MEMORY", raising=False)
        else:
            monkeypatch.setenv("MLDA_LOW_MEMORY", value)
        reloaded = importlib.reload(_common)
        st.cache_data.clear()
        st.cache_resource.clear()
        return reloaded

    yield load
    monkeypatch.undo()  # puts the environment back before the module is executed one last time
    importlib.reload(_common)
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("TRUE", True), (" Yes ", True)]
    + [("", False), ("0", False), ("false", False), ("no", False), ("2", False)],
)
def test_low_memory_switch_accepts_only_the_documented_values(value: str, expected: bool) -> None:
    import _common as common

    assert common.low_memory_enabled({"MLDA_LOW_MEMORY": value}) is expected
    assert common.low_memory_enabled({}) is False


def test_training_page_in_low_memory_mode_shows_the_note_and_caps_the_slider(
    reload_common: Callable[[str | None], ModuleType],
) -> None:
    common = reload_common("1")
    assert (common.LOW_MEMORY, common.LIVE_N_JOBS, common.LIVE_CV_FOLDS) == (True, 1, 2)
    app = new_app().run()
    app.switch_page("pages/4_Model_Training.py").run()
    assert not app.exception
    assert any("low-memory mode" in caption.value for caption in app.caption)
    assert any("2-fold" in markdown.value for markdown in app.markdown)
    slider = app.slider(key="mt_rows")
    assert (slider.min, slider.max, slider.value) == (2_000, 5_000, 3_000)
    app.selectbox(key="mt_model").select("logreg")
    slider.set_value(2_000)
    app.button(key="mt_run").click().run()
    assert not app.exception
    assert "Best PR-AUC (CV mean)" in metrics_of(app)
    version = common.stamp(common.FEATURES_PATH, common.SPLIT_PATH)
    assert len(common.live_tuning("logreg", 2_000, RANDOM_STATE, version)["fold_scores"]) == 2


def test_training_page_keeps_the_full_caps_when_the_switch_is_off(
    reload_common: Callable[[str | None], ModuleType],
) -> None:
    common = reload_common(None)
    assert (common.LOW_MEMORY, common.LIVE_N_JOBS, common.LIVE_CV_FOLDS) == (False, 1, 3)
    app = new_app().run()
    app.switch_page("pages/4_Model_Training.py").run()
    assert not app.exception
    assert not any("low-memory mode" in caption.value for caption in app.caption)
    slider = app.slider(key="mt_rows")
    assert (slider.min, slider.max, slider.value) == (2_000, 20_000, 4_000)


def test_preprocessing_page_caps_the_sample_in_low_memory_mode(
    reload_common: Callable[[str | None], ModuleType],
) -> None:
    reload_common("1")
    app = new_app().run()
    app.switch_page("pages/3_Preprocessing.py").run()
    assert not app.exception
    assert app.slider(key="pp_rows").max == 5_000
    reload_common(None)
    app = new_app().run()
    app.switch_page("pages/3_Preprocessing.py").run()
    assert app.slider(key="pp_rows").max == 10_000


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


@pytest.mark.parametrize(
    ("entered", "expected"),
    [("0200024", "0200024"), ("200024", "0200024"), (" 200024 ", "0200024"), ("1", "0000001")]
    + [(text, None) for text in ("abc", "", "  ", "nan", "02-0024", "0200O24", "12345678", "20.0", "\u00b2")],
)
def test_mine_id_is_padded_only_when_it_is_all_digits(entered: str, expected: str | None) -> None:
    import _common as common

    assert common.normalise_mine_id(entered) == expected


@pytest.mark.parametrize("bad", ["abc", "", "12345678", "02-0024", pd.NA])
def test_batch_upload_refuses_a_malformed_mine_id_and_names_the_first_bad_row(
    sample_rows: tuple[pd.DataFrame, pd.DataFrame], bad: object
) -> None:
    import _common as common

    base, _ = sample_rows
    upload = base[[*common.BATCH_REQUIRED, *common.BATCH_OPTIONAL]].head(5).astype("string")
    upload["ACCIDENT_DT"] = base["ACCIDENT_DT"].head(5).dt.strftime("%Y-%m-%d")
    upload.loc[2, "MINE_ID"] = bad
    upload.loc[4, "MINE_ID"] = "also bad"
    with pytest.raises(ValueError, match=r"MINE_ID is a 7-digit number.*data row 3"):
        common.prepare_batch(upload)


def test_batch_upload_pads_mine_ids_that_lost_their_leading_zeros(
    sample_rows: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    import _common as common

    base, _ = sample_rows
    upload = base[[*common.BATCH_REQUIRED, *common.BATCH_OPTIONAL]].head(5).astype("string")
    upload["ACCIDENT_DT"] = base["ACCIDENT_DT"].head(5).dt.strftime("%Y-%m-%d")
    upload["MINE_ID"] = upload["MINE_ID"].str.lstrip("0")
    assert common.prepare_batch(upload)["MINE_ID"].tolist() == base["MINE_ID"].head(5).tolist()


def test_batch_upload_rejects_a_file_without_required_columns() -> None:
    import _common as common

    with pytest.raises(ValueError, match="lacks required columns"):
        common.prepare_batch(pd.DataFrame({"MINE_ID": ["0200024"]}))


FOLLOWUP_FILES = tuple(
    METRICS_DIR / name
    for name in (
        "followup_capacity_policy.csv",
        "followup_workload_rule.csv",
        "followup_open_calibration.csv",
        "followup_checks.json",
    )
)
needs_followup = pytest.mark.skipif(
    any(not path.exists() for path in FOLLOWUP_FILES),
    reason="follow-up artefacts missing: run `uv run python -m mlda.followup` first",
)
FOLLOWUP_COMMAND = "uv run python -m mlda.followup"


def tab_of(app: AppTest, label: str) -> Block:
    """The tab with the given label on the page that is open."""
    return next(tab for tab in app.tabs if tab.label == label)


@needs_followup
def test_evaluation_capacity_policy_tab_renders_the_stored_policy_tables() -> None:
    policy = pd.read_csv(METRICS_DIR / "followup_capacity_policy.csv", encoding="utf-8")
    app = new_app().run()
    app.switch_page("pages/6_Evaluation.py").run()
    assert not app.exception
    assert not app.error
    tab = tab_of(app, "Capacity policy")
    assert len(tab.dataframe) >= 2  # policy table, workload-rule table (and the calibration summary)
    for group, option in (("open_at_filing", None), ("all_test_rows", "All test reports")):
        if option:
            app.radio(key="ev_policy_group").set_value(option).run()
            assert not app.exception
            tab = tab_of(app, "Capacity policy")
        shown = tab.dataframe[0].value
        stored = policy[policy["group"] == group]
        assert len(shown) == len(stored)
        assert {"Model", "Policy", "Flag rate", "Recall", "Precision", "Specificity", "Cut-off"} <= set(
            shown.columns
        )
        assert np.allclose(shown["Flag rate"].to_numpy(dtype=float), stored["flag_rate"].to_numpy())
    assert [expander.label for expander in tab.expander] == ["Calibration on open reports"]


@needs_followup
def test_evaluation_threshold_explorer_points_to_the_capacity_policy_tab() -> None:
    policy = pd.read_csv(METRICS_DIR / "followup_capacity_policy.csv", encoding="utf-8")
    flagged = policy.loc[
        (policy["group"] == "open_at_filing")
        & (policy["model"] == "hist_gb")
        & (policy["policy"] == "capacity_threshold"),
        "flag_rate",
    ].iloc[0]
    app = new_app().run()
    app.switch_page("pages/6_Evaluation.py").run()
    notes = [caption.value for caption in tab_of(app, "Threshold explorer").caption]
    assert any(f"{flagged:.1%}" in note and "Capacity policy tab" in note for note in notes)


def test_evaluation_without_followup_files_points_to_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing follow-up file gives an info box with the command that creates it, not a crash."""
    import _common as common

    for loader in ("capacity_policy", "workload_rule", "open_calibration", "workload_checks"):
        monkeypatch.setattr(common, loader, lambda: None)
    app = new_app().run()
    app.switch_page("pages/6_Evaluation.py").run()
    assert not app.exception
    assert not app.error
    tab = tab_of(app, "Capacity policy")
    assert not tab.dataframe
    assert any(FOLLOWUP_COMMAND in info.value for info in tab.info)
    assert not any("Capacity policy tab" in caption.value for caption in app.caption)


def test_about_page_states_the_workload_rule_and_the_limits_of_the_probability() -> None:
    app = new_app().run()
    app.switch_page("pages/7_About.py").run()
    assert not app.exception
    text = " ".join(markdown.value for markdown in app.markdown)
    assert "workload rule" in text
    assert "retrospective" in text
    assert "ranking score" in text


WORKLOAD_RULE_CSV = METRICS_DIR / "followup_workload_rule.csv"
RULE_LABEL = "Workload rule"
CAPACITY_LABEL = "Capacity threshold"
needs_workload_rule = pytest.mark.skipif(
    not WORKLOAD_RULE_CSV.exists(),
    reason="follow-up artefacts missing: run `uv run python -m mlda.followup` first",
)


def stored_cutoff(model: str, policy: str = "top_75pct") -> float | None:
    """Cut-off of one model and policy read straight from the CSV, without the app's loader."""
    rule = pd.read_csv(WORKLOAD_RULE_CSV, encoding="utf-8")
    values = rule.loc[(rule["model"] == model) & (rule["policy"] == policy), "cutoff_oof"]
    return float(values.iloc[0]) if len(values) else None


def scored_inference_app() -> AppTest:
    """The inference page with the default case submitted."""
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    next(button for button in app.button if button.label == "Score this case").click().run()
    assert not app.exception
    return app


def batch_app() -> AppTest:
    """The inference page after the five-row batch template has been uploaded."""
    import _common as common

    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    template = common.template_csv().encode("utf-8")
    app.file_uploader(key="inf_upload").set_value(("batch_template.csv", template, "text/csv")).run()
    assert not app.exception
    assert not app.error
    return app


def shown_probability(app: AppTest) -> float:
    return float(metrics_of(app)["Probability of a lost-time injury"].rstrip("%")) / 100


@needs_workload_rule
def test_workload_cutoff_is_read_by_model_and_policy() -> None:
    import _common as common

    rule = pd.read_csv(WORKLOAD_RULE_CSV, encoding="utf-8")
    for model in rule["model"].unique():
        for policy in rule["policy"].unique():
            assert common.workload_cutoff(model, policy) == stored_cutoff(model, policy)
    assert common.workload_cutoff("hist_gb") == pytest.approx(0.3756, abs=1e-4)
    assert common.workload_cutoff("hist_gb", "top_50pct") > common.workload_cutoff("hist_gb")
    assert common.workload_cutoff("no_such_model") is None
    assert common.workload_cutoff("hist_gb", "no_such_policy") is None


def test_workload_cutoff_is_none_when_the_table_is_missing_or_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _common as common

    monkeypatch.setattr(common, "workload_rule", lambda: None)
    assert common.workload_cutoff("hist_gb") is None
    monkeypatch.setattr(
        common, "workload_rule", lambda: pd.DataFrame({"model": ["hist_gb"], "policy": ["top_75pct"]})
    )
    assert common.workload_cutoff("hist_gb") is None  # no cutoff_oof column


@needs_workload_rule
def test_inference_shows_the_workload_rule_card_that_matches_the_probability() -> None:
    import _common as common

    model = common.best_model()
    cutoff = stored_cutoff(model)
    if cutoff is None:
        pytest.skip(f"{model} has no top_75pct row in the workload-rule table")
    app = scored_inference_app()
    metrics = metrics_of(app)
    assert {RULE_LABEL, CAPACITY_LABEL, "At threshold 0.50"} <= set(metrics)
    assert not any(label.startswith("At the operating threshold") for label in metrics)
    assert metrics[RULE_LABEL] in {"Review", "Routine"}
    probability = shown_probability(app)
    if abs(probability - cutoff) > 0.0005:  # the card compares the unrounded probability
        assert metrics[RULE_LABEL] == ("Review" if probability >= cutoff else "Routine")
    card = next(metric for metric in app.metric if metric.label == RULE_LABEL)
    assert "top three in four of reports still open at filing" in card.help.lower()
    assert "cut-off from out-of-fold training scores" in card.help
    caption = next(c.value for c in app.caption if c.value.startswith("For a report with no return-to-work"))
    assert "ranking score" in caption and "about 0.12 below the observed rate" in caption
    assert caption.endswith("The cost note below applies to all filings.")


@needs_workload_rule
def test_inference_workload_card_is_highlighted_until_a_return_date_is_on_the_form() -> None:
    import _common as common

    if stored_cutoff(common.best_model()) is None:
        pytest.skip("the best model has no top_75pct row in the workload-rule table")
    app = scored_inference_app()
    assert app.selectbox(key="inf_dated").value == "No"
    card = next(metric for metric in app.metric if metric.label == RULE_LABEL)
    assert card.delta == "Applies to this report"
    assert not any("score is for information only" in info.value for info in app.info)
    app.selectbox(key="inf_dated").select("Yes").run()
    assert not app.exception
    card = next(metric for metric in app.metric if metric.label == RULE_LABEL)
    assert not card.delta
    assert any(
        "outcome is recorded on the form" in info.value and "for information only" in info.value
        for info in app.info
    )


def test_inference_cost_note_says_it_is_computed_on_all_filings() -> None:
    app = scored_inference_app()
    assert any("computed at the model's probability on all filings" in info.value for info in app.info)
    assert "all filings" in app.select_slider(key="inf_ratio").help


def test_inference_renders_when_the_workload_loader_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _common as common

    monkeypatch.setattr(common, "workload_rule", lambda: None)
    app = scored_inference_app()
    assert not app.error
    metrics = metrics_of(app)
    assert metrics[RULE_LABEL] == "not available"
    assert CAPACITY_LABEL in metrics and "At threshold 0.50" in metrics
    assert any(FOLLOWUP_COMMAND in caption.value for caption in app.caption)
    card = next(metric for metric in app.metric if metric.label == RULE_LABEL)
    assert FOLLOWUP_COMMAND in card.help
    assert not card.delta  # nothing is highlighted when there is no rule to apply
    assert any("break-even" in info.value for info in app.info)  # the cost note still renders


def test_inference_card_is_not_available_for_a_model_without_a_cutoff() -> None:
    """A saved model that is absent from the CSV (here logistic regression) gets the same fallback."""
    import _common as common

    if "logreg" not in common.ranked_models() or common.workload_cutoff("logreg") is not None:
        pytest.skip("logreg is not saved or already has a cut-off")
    app = new_app().run()
    app.switch_page("pages/5_Inference.py").run()
    app.selectbox(key="inf_model").select("logreg").run()
    next(button for button in app.button if button.label == "Score this case").click().run()
    assert not app.exception
    assert metrics_of(app)[RULE_LABEL] == "not available"


@needs_workload_rule
def test_batch_template_output_has_the_workload_rule_flag() -> None:
    import _common as common

    model = common.best_model()
    cutoff = stored_cutoff(model)
    if cutoff is None:
        pytest.skip(f"{model} has no top_75pct row in the workload-rule table")
    app = batch_app()
    table = app.dataframe[0].value
    assert list(table.columns[-4:]) == [
        "probability",
        "flag_operating_point",
        "flag_workload_rule",
        "flag_at_0_5",
    ]
    assert set(table["flag_workload_rule"]) <= {0, 1}
    assert (table["flag_workload_rule"] == (table["probability"] >= cutoff).astype(int)).all()
    metrics = metrics_of(app)
    assert metrics["Flagged under the workload rule"] == f"{int(table['flag_workload_rule'].sum()):,}"
    assert metrics["Rows scored"] == f"{len(table):,}"


def test_batch_output_without_a_cutoff_has_no_workload_column_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _common as common

    monkeypatch.setattr(common, "workload_rule", lambda: None)
    app = batch_app()
    table = app.dataframe[0].value
    assert "flag_workload_rule" not in table.columns
    assert {"flag_operating_point", "flag_at_0_5"} <= set(table.columns)
    assert metrics_of(app)["Flagged under the workload rule"] == "not available"
    assert any(FOLLOWUP_COMMAND in caption.value for caption in app.caption)
