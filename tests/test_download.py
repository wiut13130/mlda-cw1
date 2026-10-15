"""Checksum verification of the pinned MSHA snapshot (small files always; big files when present)."""

import pytest

from mlda.config import SOURCES
from mlda.data.download import download, sha256_of, verify

SMALL = ["accidents_codebook", "mines_codebook", "production_codebook"]
LARGE = ["accidents", "mines", "production"]


@pytest.mark.parametrize("name", SMALL)
def test_codebooks_match_pinned_hashes(name: str) -> None:
    source = SOURCES[name]
    assert source.path.exists(), f"{source.filename} is committed with the repository"
    assert sha256_of(source.path) == source.sha256


@pytest.mark.parametrize("name", LARGE)
def test_large_archives_match_when_present(name: str) -> None:
    source = SOURCES[name]
    if not source.path.exists():
        pytest.skip("archive not downloaded in this environment (run `uv run mlda-download`)")
    assert verify(source)
    # the archives are committed, the text members are not: extracting is part of the contract
    assert download(source) == source.extracted_path
    assert source.extracted_path.exists()


def test_sha256_helper_is_deterministic(tmp_path) -> None:
    file = tmp_path / "x.txt"
    file.write_bytes(b"mlda")
    assert sha256_of(file) == "99d1a7afa23361113c2f7e8d55f2faf9b0b3dce2745a89774e58db73b41c6b65"
