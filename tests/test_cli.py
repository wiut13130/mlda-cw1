"""Command-line entry points: argument parsing is checked with the heavy work stubbed out."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from mlda import cli
from mlda.models import MODEL_ZOO


def test_train_all_passes_quick_and_models_to_the_pipeline(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[dict[str, Any]] = []

    def stub(quick: bool = False, models: list[str] | None = None, **_: Any) -> dict[str, Any]:
        calls.append({"quick": quick, "models": models})
        return {"total_seconds": 1.5, "best_model": "hist_gb"}

    monkeypatch.setattr("mlda.pipeline.train_all", stub)
    monkeypatch.setattr(sys, "argv", ["mlda-train", "--quick", "--models", "logreg", "hist_gb"])
    cli.train_all()
    assert calls == [{"quick": True, "models": ["logreg", "hist_gb"]}]
    assert "finished in 1.5 s; best model hist_gb" in capsys.readouterr().out


def test_train_all_defaults_to_the_full_run_of_the_whole_zoo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def stub(quick: bool = False, models: list[str] | None = None, **_: Any) -> dict[str, Any]:
        calls.append({"quick": quick, "models": models})
        return {"total_seconds": 0.0, "best_model": "logreg"}

    monkeypatch.setattr("mlda.pipeline.train_all", stub)
    monkeypatch.setattr(sys, "argv", ["mlda-train"])
    cli.train_all()
    assert calls == [{"quick": False, "models": None}]


def test_train_all_rejects_a_model_outside_the_zoo(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["mlda-train", "--models", "xgboost"])
    with pytest.raises(SystemExit) as exit_info:
        cli.train_all()
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "invalid choice" in error and all(name in error for name in MODEL_ZOO)


def test_download_passes_force_to_the_downloader(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    seen: list[bool] = []

    def stub(force: bool = False) -> dict[str, str]:
        seen.append(force)
        return {"accidents": "data/raw/Accidents.txt"}

    monkeypatch.setattr("mlda.data.download.ensure_raw_files", stub)
    monkeypatch.setattr(sys, "argv", ["mlda-download", "--force"])
    cli.download()
    assert seen == [True]
    assert "accidents" in capsys.readouterr().out


def test_zip_submission_passes_its_options_to_the_packer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from mlda import submission

    calls: list[tuple] = []
    monkeypatch.setattr(
        submission, "build_zip", lambda out, forbid, full=False, bundle=False: calls.append((out, forbid, full, bundle))
    )
    monkeypatch.setattr(sys, "argv", ["mlda-zip", "--output-dir", str(tmp_path), "--forbid", "Jane", "--full", "--bundle"])
    cli.zip_submission()
    assert calls == [(tmp_path, ["Jane"], True, True)]


def test_zip_submission_refuses_bundle_without_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mlda-zip", "--bundle"])
    with pytest.raises(SystemExit) as stopped:
        cli.zip_submission()
    assert stopped.value.code == 2


def test_zip_submission_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlda import submission

    calls: list[tuple] = []
    monkeypatch.setattr(
        submission, "build_zip", lambda out, forbid, full=False, bundle=False: calls.append((out, forbid, full, bundle))
    )
    monkeypatch.setattr(sys, "argv", ["mlda-zip"])
    cli.zip_submission()
    assert calls == [(None, None, False, False)]
