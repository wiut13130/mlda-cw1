"""Build the intranet submission archive MLDA.CW1.<ID>.zip.

The brief asks for one zip with the report, the notebook and the dataset. This tool packs
every git-tracked file (code, data archives, processed tables, models, metrics, figures,
report sources) plus the rendered report and notebook exports, then scans the result for
strings that must not appear (the brief forbids the student's name anywhere). The forbidden
strings are supplied at run time through the MLDA_FORBIDDEN environment variable or the
--forbid option and are never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

from mlda.config import PROJECT_ROOT, STUDENT_ID

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv", ".cfg", ".ini", ".html", ".ipynb"}
EXTRA_FILES = (
    "report/MLDA_CW1_13130_report.docx",
    "report/MLDA_CW1_13130_report.pdf",
    "notebooks/01_end_to_end.html",
)
ALWAYS_EXCLUDE = (".github/", ".gitattributes", ".gitignore", ".dockerignore")


def tracked_files(root: Path = PROJECT_ROOT) -> list[str]:
    """Files under version control, relative to the repository root."""
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))


def _ipynb_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = "\n".join("".join(c.get("source", [])) for c in data.get("cells", []))
    return cells + "\n" + json.dumps(data.get("metadata", {}))


def scan_forbidden(paths: list[Path], forbidden: list[str]) -> list[tuple[str, str]]:
    """Return (file, matched string) pairs for every forbidden string found, case-insensitive."""
    if not forbidden:
        return []
    patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in forbidden if term.strip()]
    hits: list[tuple[str, str]] = []
    for path in paths:
        if path.suffix == ".docx":
            text = _docx_text(path)
        elif path.suffix == ".ipynb":
            text = _ipynb_text(path)
        elif path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        for pattern in patterns:
            if pattern.search(text) or pattern.search(str(path)):
                hits.append((str(path.relative_to(PROJECT_ROOT)), pattern.pattern))
    return hits


def build_zip(output_dir: Path | None = None, forbidden: list[str] | None = None, root: Path = PROJECT_ROOT) -> Path:
    """Create MLDA.CW1.<ID>.zip in output_dir (default: repository root) and return its path."""
    forbidden = forbidden if forbidden is not None else [t for t in os.environ.get("MLDA_FORBIDDEN", "").split(",") if t.strip()]
    members = [p for p in tracked_files(root) if not p.startswith(ALWAYS_EXCLUDE)]
    for extra in EXTRA_FILES:
        if (root / extra).exists() and extra not in members:
            members.append(extra)
    paths = [root / m for m in members if (root / m).is_file()]
    hits = scan_forbidden(paths, forbidden)
    if hits:
        listing = "\n".join(f"  {file}: '{term}'" for file, term in hits[:20])
        raise SystemExit(f"forbidden string found in {len(hits)} place(s); fix before packaging:\n{listing}")
    output_dir = output_dir or root
    output = output_dir / f"MLDA.CW1.{STUDENT_ID}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=f"mlda-cw1/{path.relative_to(root).as_posix()}")
    size_mb = output.stat().st_size / 1e6
    print(f"{output} ({size_mb:.1f} MB, {len(paths)} files)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--forbid", action="append", default=None, help="string that must not appear (repeatable)")
    args = parser.parse_args()
    build_zip(args.output_dir, args.forbid)


if __name__ == "__main__":
    main()
