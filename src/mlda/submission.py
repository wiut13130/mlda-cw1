"""Build the intranet submission archive MLDA.CW1.<ID>.zip.

The brief asks for one zip with the report, the notebook and the dataset. This tool packs
every git-tracked file (code, data archives, processed tables, models, metrics, figures,
report sources) plus the rendered report and notebook exports. When the code is a git
repository with commits it also adds the history: ``reports/commit_history.txt`` (one line
per commit, then the commits per author) and ``repository.bundle``, a ``git bundle`` of every
branch and tag, so that the marker can see the real commit dates without access to the hosting
service. The result is scanned for strings that must not appear (the brief forbids the
student's name anywhere). The forbidden strings are supplied at run time through the
MLDA_FORBIDDEN environment variable or the --forbid option and are never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
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
ARCHIVE_ROOT = "mlda-cw1"
HISTORY_NAME = "reports/commit_history.txt"
BUNDLE_NAME = "repository.bundle"
RESTORE_COMMAND = f"git clone {BUNDLE_NAME} {ARCHIVE_ROOT}"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``root``. Stdin is closed because ``git shortlog`` reads it when it is a pipe."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        check=True,
    )


def tracked_files(root: Path = PROJECT_ROOT) -> list[str]:
    """Files under version control, relative to the repository root."""
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_repository_root(root: Path) -> bool:
    """True when ``root`` is the top level of a git work tree that has at least one commit."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel").stdout.strip()
        _git(root, "rev-parse", "--verify", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return False
    return Path(top).resolve() == root.resolve()


def commit_history_text(root: Path) -> tuple[str, int]:
    """Every commit as ``hash date author subject``, then the commits per author; also the commit count.

    It covers all refs (``--all``), the same set that the bundle holds, so the text and the bundle agree.
    """
    log = _git(root, "log", "--all", "--format=%h %ad %an %s", "--date=short").stdout
    authors = _git(root, "shortlog", "-s", "--all").stdout
    n_commits = len(log.splitlines())
    text = (
        f"Commit history of the repository (student ID {STUDENT_ID}): {n_commits} commits.\n"
        "Each line: abbreviated hash, author date (YYYY-MM-DD), author, subject; most recent first.\n"
        "\n"
        f"{log}\n"
        "Commits per author (git shortlog -s):\n"
        f"{authors}"
    )
    return text, n_commits


def create_bundle(root: Path, target: Path) -> Path:
    """Write every branch, tag and HEAD of the repository to one ``git bundle`` file."""
    _git(root, "bundle", "create", str(target), "--all")
    return target


def _patterns(forbidden: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(re.escape(term), re.IGNORECASE) for term in forbidden if term.strip()]


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))


def _ipynb_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = "\n".join("".join(c.get("source", [])) for c in data.get("cells", []))
    return cells + "\n" + json.dumps(data.get("metadata", {}))


def scan_text(text: str, label: str, forbidden: list[str]) -> list[tuple[str, str]]:
    """Return (label, matched string) pairs for every forbidden string found in ``text``, case-insensitive."""
    return [(label, pattern.pattern) for pattern in _patterns(forbidden) if pattern.search(text)]


def scan_forbidden(paths: list[Path], forbidden: list[str], root: Path | None = None) -> list[tuple[str, str]]:
    """Return (file, matched string) pairs for every forbidden string found, case-insensitive."""
    if not forbidden:
        return []
    base = root if root is not None else PROJECT_ROOT
    patterns = _patterns(forbidden)
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
                hits.append((str(path.relative_to(base)), pattern.pattern))
    return hits


def build_zip(
    output_dir: Path | None = None,
    forbidden: list[str] | None = None,
    root: Path = PROJECT_ROOT,
    bundle: bool = False,
) -> Path:
    """Create MLDA.CW1.<ID>.zip in output_dir (default: repository root) and return its path.

    The commit history and the bundle are generated into a temporary folder and added to the archive;
    nothing is written into the working tree, so no stale copy can be committed. A ``root`` that is not
    a git repository with commits (for example a test folder) gets neither file. The bundle is optional
    (``bundle=True``) because it roughly doubles the archive; the hosted repository is the primary evidence.
    """
    forbidden = forbidden if forbidden is not None else [t for t in os.environ.get("MLDA_FORBIDDEN", "").split(",") if t.strip()]
    with_history = is_repository_root(root)
    generated = {HISTORY_NAME, BUNDLE_NAME} if with_history else set()
    members = [p for p in tracked_files(root) if not p.startswith(ALWAYS_EXCLUDE) and p not in generated]
    for extra in EXTRA_FILES:
        if (root / extra).exists() and extra not in members:
            members.append(extra)
    paths = [root / m for m in members if (root / m).is_file()]
    hits = scan_forbidden(paths, forbidden, root)
    history, n_commits = commit_history_text(root) if with_history else ("", 0)
    hits += scan_text(history, HISTORY_NAME, forbidden)
    if hits:
        listing = "\n".join(f"  {file}: '{term}'" for file, term in hits[:20])
        raise SystemExit(f"forbidden string found in {len(hits)} place(s); fix before packaging:\n{listing}")
    output_dir = output_dir or root
    output = output_dir / f"MLDA.CW1.{STUDENT_ID}.zip"
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").stdout.splitlines() if with_history else []
    with tempfile.TemporaryDirectory(prefix="mlda-zip-") as scratch, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, arcname=f"{ARCHIVE_ROOT}/{path.relative_to(root).as_posix()}")
        if with_history:
            history_file = Path(scratch) / "commit_history.txt"
            history_file.write_text(history, encoding="utf-8", newline="\n")
            archive.write(history_file, arcname=f"{ARCHIVE_ROOT}/{HISTORY_NAME}")
            bundle_mb = 0.0
            if bundle:
                bundle_file = create_bundle(root, Path(scratch) / "repository.bundle")
                # A bundle is a compressed pack already, so deflating it again only costs time.
                archive.write(bundle_file, arcname=f"{ARCHIVE_ROOT}/{BUNDLE_NAME}", compress_type=zipfile.ZIP_STORED)
                bundle_mb = bundle_file.stat().st_size / 1e6
    size_mb = output.stat().st_size / 1e6
    print(f"{output} ({size_mb:.1f} MB, {len(paths)} files)")
    if with_history:
        print(f"  {ARCHIVE_ROOT}/{HISTORY_NAME}: {n_commits} commits with their dates")
        if bundle:
            print(f"  {ARCHIVE_ROOT}/{BUNDLE_NAME}: every branch and tag ({bundle_mb:.1f} MB)")
            print(f"  To restore the repository with its history, unzip and run in the folder that holds {BUNDLE_NAME}:")
            print(f"    {RESTORE_COMMAND}")
        else:
            print("  repository bundle not included (pass --bundle to add it); the hosted repository holds the full history")
        if dirty:
            print(f"  warning: {len(dirty)} tracked file(s) have uncommitted changes; the zip holds them, the bundle does not")
    else:
        print("  not a git repository with commits: no commit history and no bundle were added")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--forbid", action="append", default=None, help="string that must not appear (repeatable)")
    parser.add_argument("--bundle", action="store_true", help="also include a git bundle of the whole repository")
    args = parser.parse_args()
    build_zip(args.output_dir, args.forbid, bundle=args.bundle)


if __name__ == "__main__":
    main()
