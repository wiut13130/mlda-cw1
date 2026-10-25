"""The submission zip holds the deliverables only and refuses forbidden strings; --full snapshots the repository."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from mlda.submission import scan_forbidden


def test_scan_finds_forbidden_strings_in_text_docx_and_ipynb(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mlda.submission.PROJECT_ROOT", tmp_path)
    (tmp_path / "a.py").write_text("print('hello from Jane Example')\n", encoding="utf-8")
    nb = {"cells": [{"cell_type": "markdown", "source": ["written by jane example"]}], "metadata": {}}
    (tmp_path / "b.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "c.docx", "w") as archive:
        archive.writestr("docProps/core.xml", "<cp:creator>JANE EXAMPLE</cp:creator>")
        archive.writestr("word/document.xml", "<w:t>no names here</w:t>")
    (tmp_path / "d.png").write_bytes(b"\x89PNG")
    hits = scan_forbidden(sorted(tmp_path.iterdir()), ["Jane Example"])
    assert {file for file, _ in hits} == {"a.py", "b.ipynb", "c.docx"}


def test_scan_is_clean_without_matches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mlda.submission.PROJECT_ROOT", tmp_path)
    (tmp_path / "a.md").write_text("Student ID 13130 only\n", encoding="utf-8")
    assert scan_forbidden([tmp_path / "a.md"], ["Jane Example"]) == []
    assert scan_forbidden([tmp_path / "a.md"], []) == []


def make_deliverables(root: Path, monkeypatch, *, executed: bool = True) -> dict[str, bytes]:
    """The required deliverables in their repository places, with SOURCES pinned to the fake dataset files."""
    from mlda import config, submission

    (root / "report").mkdir(parents=True)
    with zipfile.ZipFile(root / submission.REPORT_DOCX, "w") as archive:
        archive.writestr("word/document.xml", "<w:t>Student ID 13130</w:t>")
    (root / "notebooks").mkdir()
    cell = {"cell_type": "code", "source": ["1 + 1"], "outputs": [], "execution_count": 1 if executed else None}
    (root / submission.NOTEBOOK).write_text(json.dumps({"cells": [cell], "metadata": {}}), encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    contents = {name: f"data of {name}".encode() for name in ("A.zip", "A_Definition_File.txt")}
    for name, data in contents.items():
        (root / "data" / "raw" / name).write_bytes(data)
    sources = {name: config.Source(name, hashlib.sha256(data).hexdigest(), len(data)) for name, data in contents.items()}
    monkeypatch.setattr(submission, "SOURCES", sources)
    return contents


def add_build_helpers(root: Path) -> None:
    """Files that helped to build the deliverables and must stay out of the submission zip."""
    helpers = (
        "report/build_report.py", "report/sections/01_introduction.md", "src/mlda/features.py",
        "tests/test_features.py", "notebooks/build_notebook.py", "reports/metrics/comparison.csv",
        "data/raw/SNAPSHOT.json", "data/processed/base.parquet", "README.md",
    )
    for rel in helpers:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("helper\n", encoding="utf-8")


def test_submission_zip_holds_the_deliverables_only(tmp_path: Path, monkeypatch, capsys) -> None:
    from mlda import submission

    root = tmp_path / "repo"
    make_deliverables(root, monkeypatch)
    add_build_helpers(root)
    out = submission.build_zip(tmp_path, [], root=root)
    assert out.name == "MLDA.CW1.13130.zip"
    with zipfile.ZipFile(out) as archive:
        assert sorted(archive.namelist()) == [
            "MLDA.CW1.13130/dataset/A.zip",
            "MLDA.CW1.13130/dataset/A_Definition_File.txt",
            "MLDA.CW1.13130/notebook/01_end_to_end.ipynb",
            "MLDA.CW1.13130/report/MLDA_CW1_13130_report.docx",
        ]
    assert "no report/MLDA_CW1_13130_report.pdf yet" in capsys.readouterr().out


def test_submission_zip_adds_the_pdf_and_the_notebook_export_when_present(tmp_path: Path, monkeypatch) -> None:
    from mlda import submission

    root = tmp_path / "repo"
    make_deliverables(root, monkeypatch)
    (root / submission.REPORT_PDF).write_bytes(b"%PDF-1.7")
    (root / submission.NOTEBOOK_HTML).write_text("<html></html>", encoding="utf-8")
    with zipfile.ZipFile(submission.build_zip(tmp_path, [], root=root)) as archive:
        names = set(archive.namelist())
    assert {"MLDA.CW1.13130/report/MLDA_CW1_13130_report.pdf", "MLDA.CW1.13130/notebook/01_end_to_end.html"} <= names
    assert len(names) == 6


def test_submission_zip_refuses_a_missing_deliverable(tmp_path: Path, monkeypatch) -> None:
    from mlda import submission

    root = tmp_path / "repo"
    make_deliverables(root, monkeypatch)
    (root / submission.NOTEBOOK).unlink()
    with pytest.raises(SystemExit) as refused:
        submission.build_zip(tmp_path, [], root=root)
    assert "notebooks/01_end_to_end.ipynb" in str(refused.value)
    assert not (tmp_path / "MLDA.CW1.13130.zip").exists()


def test_submission_zip_refuses_an_unexecuted_notebook_a_changed_dataset_and_a_forbidden_name(
    tmp_path: Path, monkeypatch
) -> None:
    from mlda import submission

    root = tmp_path / "repo"
    make_deliverables(root, monkeypatch, executed=False)
    with pytest.raises(SystemExit) as refused:
        submission.build_zip(tmp_path, [], root=root)
    assert "never ran" in str(refused.value)

    root = tmp_path / "repo2"
    make_deliverables(root, monkeypatch)
    (root / "data" / "raw" / "A.zip").write_bytes(b"a newer weekly refresh")
    with pytest.raises(SystemExit) as refused:
        submission.build_zip(tmp_path, [], root=root)
    assert "A.zip: SHA-256 differs" in str(refused.value)

    root = tmp_path / "repo3"
    make_deliverables(root, monkeypatch)
    with zipfile.ZipFile(root / submission.REPORT_DOCX, "w") as archive:
        archive.writestr("docProps/core.xml", "<cp:creator>Jane Example</cp:creator>")
    with pytest.raises(SystemExit) as refused:
        submission.build_zip(tmp_path, ["jane example"], root=root)
    assert "MLDA_CW1_13130_report.docx" in str(refused.value)
    assert not (tmp_path / "MLDA.CW1.13130.zip").exists()


def test_bundle_without_full_is_refused(tmp_path: Path, monkeypatch) -> None:
    from mlda import submission

    root = tmp_path / "repo"
    make_deliverables(root, monkeypatch)
    with pytest.raises(ValueError):
        submission.build_zip(tmp_path, [], root=root, bundle=True)


def test_repository_snapshot_refuses_forbidden(tmp_path: Path, monkeypatch) -> None:
    from mlda import submission

    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(submission, "tracked_files", lambda root=tmp_path: ["a.md"])
    (tmp_path / "a.md").write_text("report by Jane Example", encoding="utf-8")
    with pytest.raises(SystemExit):
        submission.build_zip(tmp_path, ["Jane Example"], root=tmp_path, full=True)
    out = submission.build_zip(tmp_path, [], root=tmp_path, full=True)
    assert out.name == "mlda-cw1-repository.zip"
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == ["mlda-cw1/a.md"]


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
AUTHOR = "student13130"


def run_git(repo: Path, *args: str, date: str = "2026-10-14T10:00:00") -> str:
    """Run git in ``repo`` with a fixed identity and author date, and return its output."""
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    identity = ["-c", f"user.name={AUTHOR}", "-c", "user.email=author@example.com", "-c", "commit.gpgsign=false"]
    result = subprocess.run(
        ["git", *identity, *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True, env=env
    )
    return result.stdout


def make_repository(path: Path, subjects: tuple[str, str] = ("first commit", "second commit")) -> Path:
    """A git repository with two commits on two different days."""
    path.mkdir()
    run_git(path, "init", "-q")
    (path / "a.md").write_text("version one\n", encoding="utf-8")
    run_git(path, "add", "a.md")
    run_git(path, "commit", "-q", "-m", subjects[0])
    (path / "a.md").write_text("version two\n", encoding="utf-8")
    (path / "b.py").write_text("print('b')\n", encoding="utf-8")
    run_git(path, "add", ".")
    run_git(path, "commit", "-q", "-m", subjects[1], date="2026-10-15T09:30:00")
    return path


@needs_git
def test_build_zip_adds_the_commit_history_and_a_bundle_that_clones(tmp_path: Path, capsys) -> None:
    from mlda import submission

    repo = make_repository(tmp_path / "repo")
    out = submission.build_zip(tmp_path, [], root=repo, full=True, bundle=True)
    unzipped = tmp_path / "unzipped"
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        history = archive.read("mlda-cw1/reports/commit_history.txt").decode("utf-8")
        archive.extract("mlda-cw1/repository.bundle", unzipped)
    assert sorted(names) == [
        "mlda-cw1/a.md",
        "mlda-cw1/b.py",
        "mlda-cw1/reports/commit_history.txt",
        "mlda-cw1/repository.bundle",
    ]
    assert "2 commits" in history.splitlines()[0]
    assert re.search(rf"^[0-9a-f]{{7,}} 2026-10-15 {AUTHOR} second commit$", history, re.MULTILINE)
    assert re.search(rf"^[0-9a-f]{{7,}} 2026-10-14 {AUTHOR} first commit$", history, re.MULTILINE)
    assert re.search(rf"^\s+2\t{AUTHOR}$", history, re.MULTILINE)  # git shortlog -s
    assert not (repo / "reports").exists()  # generated outside the working tree
    assert submission.RESTORE_COMMAND == "git clone repository.bundle mlda-cw1"
    assert submission.RESTORE_COMMAND in capsys.readouterr().out

    bundle = unzipped / "mlda-cw1" / "repository.bundle"
    clone = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(bundle), str(clone)], check=True, capture_output=True)
    assert run_git(clone, "rev-list", "--count", "HEAD").strip() == "2"
    assert run_git(clone, "log", "--format=%s").splitlines() == ["second commit", "first commit"]
    assert (clone / "a.md").read_text(encoding="utf-8") == "version two\n"


@needs_git
def test_build_zip_scans_the_commit_history_for_forbidden_strings(tmp_path: Path) -> None:
    from mlda import submission

    repo = make_repository(tmp_path / "repo", subjects=("first commit", "reviewed by Jane Example"))
    with pytest.raises(SystemExit) as refused:
        submission.build_zip(tmp_path, ["Jane Example"], root=repo, full=True)
    assert "reports/commit_history.txt" in str(refused.value)
    assert not (tmp_path / "mlda-cw1-repository.zip").exists()
    # the author name is part of the history text too
    with pytest.raises(SystemExit):
        submission.build_zip(tmp_path, [AUTHOR.upper()], root=repo, full=True)


@needs_git
def test_build_zip_replaces_a_committed_history_file_and_warns_about_uncommitted_changes(
    tmp_path: Path, capsys
) -> None:
    from mlda import submission

    repo = make_repository(tmp_path / "repo")
    (repo / "reports").mkdir()
    (repo / "reports" / "commit_history.txt").write_text("stale copy\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "third commit", date="2026-10-16T08:00:00")
    (repo / "a.md").write_text("edited after the last commit\n", encoding="utf-8")
    out = submission.build_zip(tmp_path, [], root=repo, full=True)
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist().count("mlda-cw1/reports/commit_history.txt") == 1
        history = archive.read("mlda-cw1/reports/commit_history.txt").decode("utf-8")
    assert "stale copy" not in history
    assert "3 commits" in history.splitlines()[0]
    assert "uncommitted changes" in capsys.readouterr().out
