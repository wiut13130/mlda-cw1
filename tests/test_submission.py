"""The submission packer must refuse to ship forbidden strings and must zip tracked files."""

import json
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


def test_build_zip_refuses_forbidden(tmp_path: Path, monkeypatch) -> None:
    from mlda import submission

    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(submission, "tracked_files", lambda root=tmp_path: ["a.md"])
    (tmp_path / "a.md").write_text("report by Jane Example", encoding="utf-8")
    with pytest.raises(SystemExit):
        submission.build_zip(tmp_path, ["Jane Example"], root=tmp_path)
    out = submission.build_zip(tmp_path, [], root=tmp_path)
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == ["mlda-cw1/a.md"]
