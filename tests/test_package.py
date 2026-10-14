"""Smoke test: the package installs and reports a version."""

import mlda


def test_version_is_set() -> None:
    assert mlda.__version__.count(".") == 2
