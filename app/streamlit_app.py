"""Streamlit entrypoint: page configuration, navigation and the sidebar shared by every page.

Run it with ``uv run streamlit run app/streamlit_app.py``. The path shim below puts ``app/`` (for
``_common``) and ``src/`` (for ``mlda``) on ``sys.path``, so the app also starts on Streamlit
Community Cloud, where the package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for entry in (APP_DIR, APP_DIR.parent / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import streamlit as st  # noqa: E402
from _common import render_sidebar  # noqa: E402

st.set_page_config(
    page_title="Lost-time injury triage",
    page_icon=":material/engineering:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page(
        "pages/1_Data_Explorer.py",
        title="Data explorer",
        icon=":material/table_view:",
        default=True,
    ),
    st.Page("pages/2_EDA.py", title="Exploratory analysis", icon=":material/insights:"),
    st.Page("pages/3_Preprocessing.py", title="Preprocessing", icon=":material/tune:"),
    st.Page("pages/4_Model_Training.py", title="Model training", icon=":material/model_training:"),
    st.Page("pages/5_Inference.py", title="Inference", icon=":material/psychology:"),
    st.Page("pages/6_Evaluation.py", title="Evaluation", icon=":material/fact_check:"),
    st.Page("pages/7_About.py", title="About", icon=":material/info:"),
]

st.session_state["entrypoint_ran"] = True
render_sidebar()
current = st.navigation(PAGES)
if target := st.session_state.pop("return_to", None):  # a page opened directly came here first
    st.switch_page(target)
current.run()
