"""Download the pinned MSHA source files and verify them against recorded SHA-256 hashes.

The raw text extracts are several hundred megabytes, so they are not stored in git.
`ensure_raw_files()` makes the project reproducible from a clean clone: it fetches each
missing file, refuses anything whose checksum differs from the analysed snapshot, and
unzips the text members next to the archives.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

from mlda.config import RAW_DIR, SOURCES, Source

log = logging.getLogger(__name__)
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    """Stream a file through SHA-256 (files are too large to read in one go)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source: Source) -> bool:
    """True when the archive or text file exists and matches the pinned hash."""
    return source.path.exists() and sha256_of(source.path) == source.sha256


def download(source: Source, force: bool = False) -> Path:
    """Fetch one source into data/raw, verify its checksum and extract the zip member."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not verify(source):
        log.info("downloading %s (%.1f MB)", source.filename, source.size_bytes / 1e6)
        request = urllib.request.Request(source.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response, source.path.open("wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                out.write(chunk)
        actual = sha256_of(source.path)
        if actual != source.sha256:
            source.path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{source.filename}: checksum {actual[:12]} differs from the pinned snapshot "
                f"{source.sha256[:12]}. MSHA refreshes the file weekly; the analysis used the "
                "14 Oct 2026 version recorded in data/raw/SNAPSHOT.json."
            )
    if source.member and not source.extracted_path.exists():
        with zipfile.ZipFile(source.path) as archive:
            archive.extract(source.member, RAW_DIR)
    return source.extracted_path


def ensure_raw_files(names: tuple[str, ...] | None = None, force: bool = False) -> dict[str, Path]:
    """Download and verify every requested source (all of them by default)."""
    chosen = names or tuple(SOURCES)
    return {name: download(SOURCES[name], force=force) for name in chosen}
