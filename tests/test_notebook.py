"""Tests for the end-to-end notebook: inspect the committed file, execute a temporary copy, inspect the run.

The notebook is a graded artefact, so the test does what a marker does: open
``notebooks/01_end_to_end.ipynb``, run a copy top to bottom with a fresh kernel of this environment
(default mode, ``FULL_RUN`` unset), and check that no cell raised, that the run finished within
fifteen minutes, that the structure of the brief is there and that the executed file carries no
personal name. The committed ``notebooks/01_end_to_end.ipynb`` is never touched.
"""

from __future__ import annotations

import ast
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest

from mlda.config import MODELS_DIR, PROJECT_ROOT, RAW_DIR, SOURCES, STUDENT_ID

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_end_to_end.ipynb"
RUNTIME_LIMIT_SECONDS = 15 * 60
REQUIRED_HEADINGS = (
    "A. Load dataset",
    "B. Exploratory data analysis",
    "C. Data preparation",
    "D. Model training",
    "E. Model evaluation",
    "F. Deployment and reproducibility",
)
# Wording that the project style rules exclude from the notebook text.
BANNED_PHRASES = (
    chr(0x2014),  # em dash (written as a code point so that this file contains none)
    "delve",
    "leverage",
    "utilise",
    "foster",
    "crucial",
    "pivotal",
    "seamless",
    "in conclusion",
    "it is important to note",
    "highlighting the importance",
)



def _needs_data() -> None:
    missing = [s.extracted_path for s in SOURCES.values() if not s.extracted_path.exists() and s.member]
    if missing or not (MODELS_DIR / "manifest.json").exists() or not (RAW_DIR / "SNAPSHOT.json").exists():
        pytest.skip("the raw extracts or the mlda-train artefacts are absent (run mlda-download and mlda-train)")


@pytest.fixture(scope="module")
def generated() -> nbformat.NotebookNode:
    """The committed notebook as it is stored in the repository."""
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def executed(tmp_path_factory) -> tuple[nbformat.NotebookNode, float]:
    """Execute a temporary copy of the committed notebook with nbconvert, return (notebook, seconds)."""
    _needs_data()
    workdir = tmp_path_factory.mktemp("notebook")
    source = workdir / "01_end_to_end.ipynb"
    shutil.copyfile(NOTEBOOK, source)

    env = os.environ.copy()
    env.pop("FULL_RUN", None)
    env.pop("NOTEBOOK_MODELS", None)
    # The kernelspec starts "python" from PATH; put this interpreter first so the kernel is this environment.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    started = time.perf_counter()
    run = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--output", "executed.ipynb", "--output-dir", str(workdir),
            "--ExecutePreprocessor.timeout=1800", str(source),
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    seconds = time.perf_counter() - started
    assert run.returncode == 0, run.stderr[-4000:]
    return nbformat.read(workdir / "executed.ipynb", as_version=4), seconds


# --------------------------------------------------------------------------- structure (no execution)


def test_required_headings_in_order(generated) -> None:
    headings = [
        line[2:].strip()
        for cell in generated.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("# ")
    ]
    positions = [headings.index(h) for h in REQUIRED_HEADINGS]
    assert positions == sorted(positions), f"top-level headings out of order: {headings}"


def test_title_and_environment_cells_come_first(generated) -> None:
    title, environment = generated.cells[0], generated.cells[1]
    assert title.cell_type == "markdown" and environment.cell_type == "code"
    for needle in ("6COSC017C", "CW1", STUDENT_ID, "FULL_RUN", "SNAPSHOT.json"):
        assert needle in title.source, f"title cell lacks {needle!r}"
    assert "RANDOM_STATE" in environment.source and "platform.python_version()" in environment.source


def test_every_run_of_code_cells_follows_a_markdown_cell(generated) -> None:
    previous_kind = None
    for index, cell in enumerate(generated.cells):
        if cell.cell_type == "code" and previous_kind != "code":
            assert previous_kind == "markdown", f"cell {index} starts a code group without a markdown cell before it"
        previous_kind = cell.cell_type


def test_code_cells_are_valid_python(generated) -> None:
    for index, cell in enumerate(generated.cells):
        if cell.cell_type == "code":
            source = "\n".join("" if line.startswith("%") else line for line in cell.source.splitlines())
            ast.parse(source, filename=f"cell {index}")


def test_text_follows_the_style_rules(generated) -> None:
    prose = "\n".join(cell.source for cell in generated.cells if cell.cell_type == "markdown").lower()
    found = [phrase for phrase in BANNED_PHRASES if phrase in prose]
    assert not found, f"banned wording in the notebook text: {found}"


# --------------------------------------------------------------------------- execution


def test_no_cell_raises(executed) -> None:
    notebook, _ = executed
    problems = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                problems.append(f"cell {index}: {output.ename}: {output.evalue}")
            elif output.output_type == "stream" and output.name == "stderr" and "Traceback" in output.text:
                problems.append(f"cell {index}: traceback on stderr")
    assert not problems, problems


def test_every_code_cell_ran(executed) -> None:
    notebook, _ = executed
    unexecuted = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code" and c.execution_count is None]
    assert not unexecuted, f"cells without an execution count: {unexecuted}"


def test_runtime_is_under_fifteen_minutes(executed) -> None:
    _, seconds = executed
    assert seconds < RUNTIME_LIMIT_SECONDS, f"notebook took {seconds / 60:.1f} minutes"


def test_no_personal_name_in_the_executed_notebook(executed) -> None:
    notebook, _ = executed
    text = nbformat.writes(notebook).lower()
    # The notebook prints its project root; a clone under a user profile would otherwise fail on that path alone.
    root = str(PROJECT_ROOT).lower()
    for variant in (root, root.replace("\\", "\\\\"), root.replace("\\", "/")):
        text = text.replace(variant, "<project-root>")
    names = {getpass.getuser(), Path.home().name}
    for name in names:
        # Short or generic account names (CI runners) would match ordinary words, so only distinctive ones are checked.
        if len(name) >= 6 and name.lower() not in {"runner", "runneradmin", "ubuntu", "vagrant", "default"}:
            pattern = r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])"
            assert not re.search(pattern, text), "the executed notebook contains the account name of this machine"
    assert STUDENT_ID in text


def test_the_notebook_reports_the_reproducibility_checks(executed) -> None:
    notebook, _ = executed
    printed = "\n".join(
        output.text
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "stream" and output.name == "stdout"
    )
    assert "validate_base passed" in printed
    assert "split rebuilt from the seed equals the persisted split: True" in printed
    assert "none of the" in printed and "leakage or current-state columns" in printed
