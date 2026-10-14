"""Page 7: purpose, provenance, reproduction commands, the run manifest and the limits of the model."""

from __future__ import annotations

import platform
from importlib.metadata import version

import pandas as pd
import psutil
import streamlit as st
from _common import (
    MSHA_LANDING_PAGE,
    ensure_entrypoint,
    model_label,
    run_manifest,
    snapshot_label,
    source_table,
)

from mlda.config import COST_FN_USD, COST_FP_USD, OPERATING_SPECIFICITY, SOURCES, STUDENT_ID
from mlda.persistence import read_manifest

ensure_entrypoint(__file__)

REPRODUCE = """\
# with uv (recommended): exact versions come from uv.lock
uv sync --locked
uv run mlda-download            # fetch the MSHA files and verify their SHA-256
uv run mlda-train               # rebuild data/processed, models/ and reports/ (add --quick for a short run)
uv run pytest -m "not slow"     # unit tests and the Streamlit smoke tests
uv run streamlit run app/streamlit_app.py

# with pip
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt && pip install -e .
streamlit run app/streamlit_app.py

# with Docker
docker build -t mlda-cw1 .
docker run --rm -p 8501:8501 mlda-cw1
"""

st.title("About this project")
purpose_tab, data_tab, reproduce_tab, manifest_tab, limits_tab = st.tabs(
    ["Purpose", "Data and licence", "Reproduce", "Run manifest", "Limits and ethics"]
)

with purpose_tab:
    st.markdown(
        "**Business question.** When an injury report (Form 7000-1) is filed at a US mine, can we predict "
        "whether the case will become a *lost-time injury*, meaning a fatality, a permanent disability or "
        "days away from work? The score is computed from the circumstances of the accident, the worker's "
        "experience and the mine's recent exposure and case history, and nothing that is recorded after "
        "the outcome is known."
    )
    st.markdown(
        "**Stakeholders.** The mine operator's safety officer completes the report. The workers' compensation "
        "insurer or self-insured operator uses the score to set a reserve and to assign a return-to-work "
        "coordinator early. MSHA district offices could use a sibling label for inspection planning."
    )
    st.markdown(
        f"**Decision rule.** Flagging an unnecessary case costs about USD {COST_FP_USD} of reviewer time and "
        f"missing a lost-time case about USD {COST_FN_USD:,} (an assumption, varied from 2:1 to 30:1 on the "
        f"Evaluation page). The operating point leaves {OPERATING_SPECIFICITY:.0%} of non-lost-time cases unflagged, "
        "which matches a review team that can read about three cases in four."
    )
    st.info(
        "The tool supports case review. It does not replace the judgement of a safety officer, adjuster or inspector."
    )

with data_tab:
    st.markdown(
        f"Source: MSHA Open Government Data, [{MSHA_LANDING_PAGE}]({MSHA_LANDING_PAGE}). Snapshot: {snapshot_label()}. "
        "Licence: works of the US federal government, public domain under 17 U.S.C. 105, so they may be copied, "
        "redistributed and analysed without restriction."
    )
    st.dataframe(
        source_table(), hide_index=True, column_config={"url": st.column_config.LinkColumn("url")}
    )
    checksums = pd.DataFrame(
        [
            {"file": s.filename, "bytes": s.size_bytes, "sha256": s.sha256, "url": s.url}
            for s in SOURCES.values()
        ]
    )
    st.markdown(
        "**Pinned files and checksums** (`mlda.config.SOURCES`; `uv run mlda-download` verifies them)"
    )
    st.dataframe(
        checksums, hide_index=True, column_config={"url": st.column_config.LinkColumn("url")}
    )

with reproduce_tab:
    st.code(REPRODUCE, language="bash")
    st.caption(
        "All randomness uses seed 42, the train and test row positions are stored in "
        "data/processed/split_indices.json, and uv.lock pins every package."
    )

with manifest_tab:
    manifest = run_manifest()
    environment = manifest.get("environment", {})
    if not manifest:
        st.info("No run manifest yet. Create one with `uv run mlda-train`.")
    else:
        left, middle, right = st.columns(3)
        left.metric("Best model", model_label(manifest["best_model"]), border=True)
        middle.metric(
            "Run type",
            "quick" if manifest.get("quick") else "full",
            help=f"{manifest.get('total_seconds', 0):.0f} s in total",
            border=True,
        )
        right.metric(
            "Git commit", str(environment.get("git_sha") or "not recorded")[:10], border=True
        )
        st.markdown("**Environment that produced the artefacts, and the one serving this page**")
        installed = {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "streamlit": version("streamlit"),
        }
        trained = {
            "python": environment.get("python_version"),
            "scikit-learn": environment.get("sklearn_version"),
            "numpy": environment.get("numpy_version"),
            "pandas": environment.get("pandas_version"),
            "streamlit": "not recorded",
        }
        st.dataframe(
            pd.DataFrame(
                {
                    "package": list(installed),
                    "training run": [trained[k] for k in installed],
                    "this app": list(installed.values()),
                }
            ),
            hide_index=True,
        )
        st.caption(
            f"Data checksum (SHA-256 over the per-file hashes): `{environment.get('data_checksum')}`"
        )
    models = read_manifest()
    if models:
        st.markdown("**Saved models** (`models/manifest.json`)")
        rows = [
            {
                "model": name,
                "estimator": e.get("estimator"),
                "size_kb": round(e.get("size_bytes", 0) / 1024, 1),
                "saved_at_utc": e.get("saved_at_utc"),
                "scikit-learn": e.get("sklearn_version"),
                "git_sha": str(e.get("git_sha"))[:10],
            }
            for name, e in models.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    rss = psutil.Process().memory_info().rss / 1e6
    st.metric(
        "Memory used by this app process",
        f"{rss:,.0f} MB",
        help="Resident set size measured with psutil at page load.",
        border=True,
    )

with limits_tab:
    st.markdown(
        "**Limitations.** The label is the degree-of-injury class that MSHA assigns when a case closes, and the "
        "separation of days away from restricted duty depends on whether an operator offers light work, so it "
        "reflects return-to-work practice as well as injury severity. Coal and metal mines differ strongly in that "
        "practice, which is why every metric is reported by sector. Mine history comes from the previous year and "
        "from cases filed before the accident; mines without a previous-year record are scored with missing "
        "exposure. Cost figures are assumptions, and the model was trained on United States reports from 2010 to "
        "2025, so it is a template for other countries rather than a calibrated tool."
    )
    st.markdown(
        "**Ethics.** The data are public and describe injured workers without names. Accident reports are filled "
        "in by operators, so under-reporting varies by mine and era, and a model trained on them inherits that "
        "bias. A score must not be used to deny a claim, to discipline a worker or to rank a mine without a human "
        "review of the case. Outputs on this site are estimates with error rates that are shown on the Evaluation page."
    )
    st.warning("Decision support only. A reviewer decides what happens to each case.")
    st.caption(
        f"Student ID {STUDENT_ID}. Coursework for 6COSC017C Machine Learning and Data Analytics."
    )
