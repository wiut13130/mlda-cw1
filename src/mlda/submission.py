"""Build the intranet submission archive MLDA.CW1.<ID>.zip.

The brief asks for "a single zipped file containing your report, Jupyter Notebook and dataset".
By default this tool packs exactly those deliverables and nothing that only helped to build
them (no source code, tests, report sources, build scripts, models or metrics):

    MLDA.CW1.<ID>/report/    the report as .docx, and the .pdf exported from Word when it exists
    MLDA.CW1.<ID>/notebook/  the executed notebook, and its HTML export when it exists
    MLDA.CW1.<ID>/dataset/   the three MSHA archives and their three definition files

Before writing anything it checks that the required files exist, that every code cell of the
notebook has run, that each dataset file still matches the SHA-256 pinned in ``mlda.config``,
and that no forbidden string appears (the brief forbids the student's name anywhere). The
forbidden strings are supplied at run time through the MLDA_FORBIDDEN environment variable or
the --forbid option and are never written to disk.

``--full`` builds a different file, ``mlda-cw1-repository.zip``: a snapshot of every
git-tracked file plus ``reports/commit_history.txt`` and, with ``--bundle``, ``repository.bundle``
(a ``git bundle`` of every branch and tag). It is an archive of the repository, not the
submission, and its name keeps it from being uploaded by mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from mlda.config import PROJECT_ROOT, SOURCES, STUDENT_ID

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv", ".cfg", ".ini", ".html", ".ipynb"}

SUBMISSION_ROOT = f"MLDA.CW1.{STUDENT_ID}"
SUBMISSION_NAME = f"{SUBMISSION_ROOT}.zip"
REPORT_DOCX = f"report/MLDA_CW1_{STUDENT_ID}_report.docx"
REPORT_PDF = f"report/MLDA_CW1_{STUDENT_ID}_report.pdf"
NOTEBOOK = "notebooks/01_end_to_end.ipynb"
NOTEBOOK_HTML = "notebooks/01_end_to_end.html"

REPOSITORY_ROOT = "mlda-cw1"
REPOSITORY_NAME = "mlda-cw1-repository.zip"
ALWAYS_EXCLUDE = (".github/", ".gitattributes", ".gitignore", ".dockerignore")
HISTORY_NAME = "reports/commit_history.txt"
BUNDLE_NAME = "repository.bundle"
RESTORE_COMMAND = f"git clone {BUNDLE_NAME} {REPOSITORY_ROOT}"
EXTRA_FILES = (REPORT_DOCX, REPORT_PDF, NOTEBOOK_HTML)  # untracked outputs added to the --full snapshot


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
    outputs = "\n".join(json.dumps(c.get("outputs", [])) for c in data.get("cells", []) if c.get("cell_type") == "code")
    return cells + "\n" + outputs + "\n" + json.dumps(data.get("metadata", {}))


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


def _forbidden_from_environment(forbidden: list[str] | None) -> list[str]:
    if forbidden is not None:
        return forbidden
    return [t for t in os.environ.get("MLDA_FORBIDDEN", "").split(",") if t.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unexecuted_code_cells(notebook: Path) -> list[int]:
    """Indices of code cells that have no execution count, that is cells the saved notebook never ran."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    return [i for i, c in enumerate(data.get("cells", [])) if c.get("cell_type") == "code" and c.get("execution_count") is None]


def deliverables(root: Path = PROJECT_ROOT) -> list[tuple[str, str, bool]]:
    """(source path relative to ``root``, path inside the archive, required) for every deliverable."""
    items = [
        (REPORT_DOCX, f"{SUBMISSION_ROOT}/report/{Path(REPORT_DOCX).name}", True),
        (REPORT_PDF, f"{SUBMISSION_ROOT}/report/{Path(REPORT_PDF).name}", False),
        (NOTEBOOK, f"{SUBMISSION_ROOT}/notebook/{Path(NOTEBOOK).name}", True),
        (NOTEBOOK_HTML, f"{SUBMISSION_ROOT}/notebook/{Path(NOTEBOOK_HTML).name}", False),
    ]
    items += [(f"data/raw/{s.filename}", f"{SUBMISSION_ROOT}/dataset/{s.filename}", True) for s in SOURCES.values()]
    return items


def build_submission(output_dir: Path | None = None, forbidden: list[str] | None = None, root: Path = PROJECT_ROOT) -> Path:
    """Create MLDA.CW1.<ID>.zip with the report, the notebook and the dataset only, and return its path."""
    forbidden = _forbidden_from_environment(forbidden)
    items = deliverables(root)
    missing = [src for src, _, required in items if required and not (root / src).is_file()]
    if missing:
        raise SystemExit("required deliverable(s) missing; nothing was written:\n" + "\n".join(f"  {m}" for m in missing))
    present = [(root / src, arc) for src, arc, _ in items if (root / src).is_file()]
    problems: list[str] = []
    unexecuted = unexecuted_code_cells(root / NOTEBOOK)
    if unexecuted:
        problems.append(f"{NOTEBOOK}: {len(unexecuted)} code cell(s) never ran (first index {unexecuted[0]}); execute it before packaging")
    for source in SOURCES.values():
        if _sha256(root / "data" / "raw" / source.filename) != source.sha256:
            problems.append(f"data/raw/{source.filename}: SHA-256 differs from the pinned snapshot in mlda.config")
    hits = scan_forbidden([path for path, _ in present], forbidden, root)
    problems += [f"{file}: contains '{term}'" for file, term in hits]
    if problems:
        raise SystemExit("submission refused; nothing was written:\n" + "\n".join(f"  {p}" for p in problems[:20]))
    output = (output_dir or root) / SUBMISSION_NAME
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, arc in present:
            archive.write(path, arcname=arc)
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB, {len(present)} files)")
    for path, arc in present:
        print(f"  {arc}  ({path.stat().st_size / 1e6:.2f} MB)")
    if not (root / REPORT_PDF).is_file():
        print(f"  note: no {REPORT_PDF} yet; export it from Word and run this command again to include it")
    if is_repository_root(root):
        tracked = [src for src, _, _ in items if (root / src).is_file()]
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no", "--", *tracked).stdout.splitlines()
        if dirty:
            print(f"  warning: {len(dirty)} deliverable(s) differ from the last commit; commit them so the zip matches the repository")
    return output


def build_repository_snapshot(
    output_dir: Path | None = None,
    forbidden: list[str] | None = None,
    root: Path = PROJECT_ROOT,
    bundle: bool = False,
) -> Path:
    """Create mlda-cw1-repository.zip: every tracked file, the rendered outputs and the commit history.

    The commit history and the bundle are generated into a temporary folder and added to the archive;
    nothing is written into the working tree, so no stale copy can be committed. A ``root`` that is not
    a git repository with commits (for example a test folder) gets neither file. The bundle is optional
    (``bundle=True``) because it roughly doubles the archive; the hosted repository is the primary evidence.
    """
    forbidden = _forbidden_from_environment(forbidden)
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
    output = (output_dir or root) / REPOSITORY_NAME
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").stdout.splitlines() if with_history else []
    with tempfile.TemporaryDirectory(prefix="mlda-zip-") as scratch, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, arcname=f"{REPOSITORY_ROOT}/{path.relative_to(root).as_posix()}")
        bundle_mb = 0.0
        if with_history:
            history_file = Path(scratch) / "commit_history.txt"
            history_file.write_text(history, encoding="utf-8", newline="\n")
            archive.write(history_file, arcname=f"{REPOSITORY_ROOT}/{HISTORY_NAME}")
            if bundle:
                bundle_file = create_bundle(root, Path(scratch) / "repository.bundle")
                # A bundle is a compressed pack already, so deflating it again only costs time.
                archive.write(bundle_file, arcname=f"{REPOSITORY_ROOT}/{BUNDLE_NAME}", compress_type=zipfile.ZIP_STORED)
                bundle_mb = bundle_file.stat().st_size / 1e6
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB, {len(paths)} files); a repository archive, not the submission")
    if with_history:
        print(f"  {REPOSITORY_ROOT}/{HISTORY_NAME}: {n_commits} commits with their dates")
        if bundle:
            print(f"  {REPOSITORY_ROOT}/{BUNDLE_NAME}: every branch and tag ({bundle_mb:.1f} MB)")
            print(f"  To restore the repository with its history, unzip and run in the folder that holds {BUNDLE_NAME}:")
            print(f"    {RESTORE_COMMAND}")
        if dirty:
            print(f"  warning: {len(dirty)} tracked file(s) have uncommitted changes; the zip holds them, the bundle does not")
    else:
        print("  not a git repository with commits: no commit history and no bundle were added")
    return output


def build_zip(
    output_dir: Path | None = None,
    forbidden: list[str] | None = None,
    root: Path = PROJECT_ROOT,
    full: bool = False,
    bundle: bool = False,
) -> Path:
    """The submission zip (default), or with ``full=True`` the repository snapshot."""
    if full:
        return build_repository_snapshot(output_dir, forbidden, root, bundle=bundle)
    if bundle:
        raise ValueError("bundle=True needs full=True: the submission zip holds the deliverables only")
    return build_submission(output_dir, forbidden, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--forbid", action="append", default=None, help="string that must not appear (repeatable)")
    parser.add_argument("--full", action="store_true", help="build mlda-cw1-repository.zip (every tracked file) instead")
    parser.add_argument("--bundle", action="store_true", help="with --full: also include a git bundle of the repository")
    args = parser.parse_args()
    if args.bundle and not args.full:
        parser.error("--bundle needs --full: the submission zip holds the report, the notebook and the dataset only")
    build_zip(args.output_dir, args.forbid, full=args.full, bundle=args.bundle)


if __name__ == "__main__":
    main()
