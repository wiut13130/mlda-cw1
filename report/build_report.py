"""Build the coursework report (.docx) from Markdown section files.

Usage (from the repository root):
    uv run python report/build_report.py            # writes report/MLDA_CW1_13130_report.docx
    uv run python report/build_report.py --check    # prints word counts and source problems only (exit 1 if any)

Why a generator rather than a hand-edited document: every number in the report is produced by the
pipeline, so the text is kept in version control as Markdown (report/sections/*.md) and rendered
with the formatting rules of the brief (Times New Roman 12, single spacing, A4, student ID in the
header of every page, page numbers, automatic table of contents, Harvard references). Re-running
the pipeline and this script regenerates a consistent document.

Supported Markdown subset:
    # / ## / ### headings            paragraphs            - bullets   1. numbered items
    **bold** and *italic* inline    pipe tables (| a | b |) preceded by an optional
    'Table: caption text {#tab:key}' line
    ![Caption text](reports/figures/file.png){#fig:key width=6.3}   (the caption is also the alt text)
    {fig:key} / {tab:key} cross-references replaced by the assigned numbers
    \\pagebreak                        > quote (rendered as an indented paragraph)
    \\landscape / \\portrait          start a landscape or a portrait section (for a wide table)
Front matter (cover page, table of contents, lists of figures and tables, abbreviations) is generated
from report/report_meta.json; sections render in file-name order; files whose name starts with
"9" are treated as references (90_references.md) or appendices (9A_, 9B_, ...).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SECTIONS_DIR = REPORT_DIR / "sections"
META_PATH = REPORT_DIR / "report_meta.json"
FONT = "Times New Roman"
BODY_PT = 12
CAPTION_PT = 12
TABLE_PT = 11
MARGIN_CM = 2.54
TEXT_WIDTH_IN = 6.27  # A4 width 21.0 cm minus 2 x 2.54 cm margins

FIG_RE = re.compile(r"^!\[(?P<caption>.*?)\]\((?P<path>[^)]+)\)(?:\{(?P<attrs>[^}]*)\})?\s*$")
TAB_CAPTION_RE = re.compile(r"^Table:\s*(?P<caption>.*?)(?:\s*\{#tab:(?P<key>[\w-]+)\})?\s*$")
XREF_RE = re.compile(r"\{(fig|tab):([\w-]+)\}")
INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
PLACEHOLDER_RE = re.compile(r"\{\{[^{}]*\}\}")


@dataclass
class Counters:
    figures: dict[str, int] = field(default_factory=dict)
    tables: dict[str, int] = field(default_factory=dict)
    figure_captions: list[str] = field(default_factory=list)
    table_captions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    # True right after a page or section break, so that a level-1 heading does not add a second
    # break and leave a blank page (the front matter ends with a break before the first section).
    fresh_page: bool = False


# ----------------------------------------------------------------------------- helpers: OOXML


def add_field(paragraph, instruction: str, cached: str = "") -> None:
    """Insert a complex field (PAGE, NUMPAGES, SEQ, TOC) with a cached result Word can refresh."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    run._r.append(begin)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    run._r.append(instr)
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    run._r.append(separate)
    result = paragraph.add_run(cached)
    end_run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run._r.append(end)
    for r in (run, result, end_run):
        r.font.name = FONT
        r.font.size = Pt(BODY_PT)


def set_update_fields_on_open(document) -> None:
    settings = document.settings.element
    node = settings.find(qn("w:updateFields"))
    if node is None:
        node = OxmlElement("w:updateFields")
        settings.append(node)
    node.set(qn("w:val"), "true")


def set_cell_shading(cell, hex_fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_fill)
    properties.append(shading)


def keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


# ----------------------------------------------------------------------------- styles and layout


def configure_styles(document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for level, size in ((1, 16), (2, 14), (3, 12)):
        style = styles[f"Heading {level}"]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_before = Pt(18 if level == 1 else 12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption = styles["Caption"]
    caption.font.name = FONT
    caption.font.size = Pt(CAPTION_PT)
    caption.font.bold = False
    caption.font.italic = False
    caption.font.color.rgb = RGBColor(0, 0, 0)
    caption.paragraph_format.space_after = Pt(10)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = FONT
        style.font.size = Pt(BODY_PT)


def set_page_geometry(section, landscape: bool = False) -> None:
    """A4 in portrait or landscape orientation with 2.54 cm margins."""
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    long_side, short_side = Cm(29.7), Cm(21.0)
    section.page_width, section.page_height = (long_side, short_side) if landscape else (short_side, long_side)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(MARGIN_CM))


def start_section(document, landscape: bool, counters: Counters) -> None:
    """Start a new page in a new section; its header and footer are inherited from the first section."""
    section = document.add_section(WD_SECTION.NEW_PAGE)
    set_page_geometry(section, landscape)
    counters.fresh_page = True


def page_break(document, counters: Counters) -> None:
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    counters.fresh_page = True


def configure_page(document, student_id: str) -> None:
    section = document.sections[0]
    set_page_geometry(section, landscape=False)
    section.different_first_page_header_footer = False
    header = section.header.paragraphs[0]
    header.text = ""
    run = header.add_run(f"Student ID: {student_id}    6COSC017C Machine Learning and Data Analytics, CW1")
    run.font.name = FONT
    run.font.size = Pt(10)
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer = section.footer.paragraphs[0]
    footer.text = ""
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lead = footer.add_run("Page ")
    lead.font.name = FONT
    lead.font.size = Pt(10)
    add_field(footer, "PAGE", "1")
    mid = footer.add_run(" of ")
    mid.font.name = FONT
    mid.font.size = Pt(10)
    add_field(footer, "NUMPAGES", "1")


# ----------------------------------------------------------------------------- inline markdown


def add_inline(paragraph, text: str, counters: Counters, size: float | None = None) -> None:
    """Add text with **bold**, *italic*, `code` and {fig:key}/{tab:key} substitution."""
    text = XREF_RE.sub(lambda m: _resolve_xref(m, counters), text)
    for token in INLINE_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
        else:
            run = paragraph.add_run(token)
        if run.font.name is None:
            run.font.name = FONT
        run.font.size = Pt(size or BODY_PT)


def _resolve_xref(match: re.Match, counters: Counters) -> str:
    kind, key = match.group(1), match.group(2)
    table = counters.figures if kind == "fig" else counters.tables
    if key not in table:
        counters.problems.append(f"unresolved cross-reference {{{kind}:{key}}}")
        return f"{'Figure' if kind == 'fig' else 'Table'} ??"
    return f"{'Figure' if kind == 'fig' else 'Table'} {table[key]}"


def plain_text(text: str, counters: Counters) -> str:
    """Caption text with cross-references resolved and Markdown emphasis removed (for alt text)."""
    resolved = XREF_RE.sub(lambda m: _resolve_xref(m, counters), text)
    return re.sub(r"(\*\*|\*|`)", "", resolved)


def set_alt_text(picture, description: str) -> None:
    """Write the alternative text of an inline picture (Word: Format Picture, Alt Text)."""
    doc_pr = picture._inline.docPr
    doc_pr.set("descr", description)
    doc_pr.set("title", description.split(". ", 1)[0])


# ----------------------------------------------------------------------------- first pass: numbering


def collect_numbers(files: list[Path], counters: Counters) -> None:
    """Assign figure and table numbers in document order so cross-references can be resolved."""
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            fig = FIG_RE.match(line.strip())
            if fig:
                attrs = fig.group("attrs") or ""
                key_match = re.search(r"#fig:([\w-]+)", attrs)
                key = key_match.group(1) if key_match else f"auto{len(counters.figure_captions) + 1}"
                if key in counters.figures:
                    counters.problems.append(f"duplicate figure key fig:{key} ({path.name}:{line_no})")
                counters.figures[key] = len(counters.figure_captions) + 1
                counters.figure_captions.append(fig.group("caption"))
                continue
            tab = TAB_CAPTION_RE.match(line.strip())
            if tab:
                key = tab.group("key") or f"auto{len(counters.table_captions) + 1}"
                if key in counters.tables:
                    counters.problems.append(f"duplicate table key tab:{key} ({path.name}:{line_no})")
                counters.tables[key] = len(counters.table_captions) + 1
                counters.table_captions.append(tab.group("caption"))


def audit_sources(files: list[Path], meta: dict) -> list[str]:
    """Problems that rendering alone does not reveal, for the files in rendering order.

    A figure or table must be referenced before it appears and at least once; ``{{...}}``
    placeholders must be filled; cover-sheet values in report_meta.json must not be stubs.
    """
    problems: list[str] = []
    defined: dict[tuple[str, str], tuple[int, int, str]] = {}
    first_reference: dict[tuple[str, str], tuple[int, int, str]] = {}
    for file_index, path in enumerate(files):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            position = (file_index, line_no, path.name)
            for match in XREF_RE.finditer(line):
                first_reference.setdefault((match.group(1), match.group(2)), position)
            stripped = line.strip()
            fig = FIG_RE.match(stripped)
            key_match = re.search(r"#fig:([\w-]+)", fig.group("attrs") or "") if fig else None
            if key_match:
                defined.setdefault(("fig", key_match.group(1)), position)
            tab = TAB_CAPTION_RE.match(stripped)
            if tab and tab.group("key"):
                defined.setdefault(("tab", tab.group("key")), position)
            for match in PLACEHOLDER_RE.finditer(line):
                problems.append(f"unfilled placeholder {match.group(0)} ({path.name}:{line_no})")
    for (kind, key), (file_index, line_no, name) in defined.items():
        reference = first_reference.get((kind, key))
        if reference is None:
            problems.append(f"{kind}:{key} is never referenced ({name}:{line_no})")
        elif reference[:2] > (file_index, line_no):
            problems.append(
                f"{kind}:{key} is first referenced at {reference[2]}:{reference[1]}, after it appears at {name}:{line_no}"
            )
    for key, value in meta.items():
        if isinstance(value, str) and (PLACEHOLDER_RE.search(value) or value.lower().startswith("to be added")):
            problems.append(f"report_meta.json '{key}' is not filled: {value!r}")
    return problems


# ----------------------------------------------------------------------------- block rendering


def add_caption(document, label: str, number: int, text: str, counters: Counters) -> None:
    paragraph = document.add_paragraph(style="Caption")
    lead = paragraph.add_run(f"{label} ")
    lead.font.name = FONT
    lead.font.size = Pt(CAPTION_PT)
    add_field(paragraph, f"SEQ {label} \\* ARABIC", str(number))
    tail = paragraph.add_run(". ")
    tail.font.name = FONT
    tail.font.size = Pt(CAPTION_PT)
    add_inline(paragraph, text, counters, size=CAPTION_PT)


def render_table(document, header: list[str], rows: list[list[str]], counters: Counters) -> None:
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, title in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        add_inline(cell.paragraphs[0], title.strip(), counters, size=TABLE_PT)
        for run in cell.paragraphs[0].runs:
            run.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_cell_shading(cell, "E7E6E6")
    for row in rows:
        cells = table.add_row().cells
        for index in range(len(header)):
            value = row[index].strip() if index < len(row) else ""
            cells[index].text = ""
            add_inline(cells[index].paragraphs[0], value, counters, size=TABLE_PT)
            cells[index].paragraphs[0].paragraph_format.space_after = Pt(0)
            cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
    document.add_paragraph().paragraph_format.space_after = Pt(4)


def parse_table_block(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    def split(line: str) -> list[str]:
        return [c for c in line.strip().strip("|").split("|")]

    header = split(lines[0])
    body = [split(line) for line in lines[2:] if line.strip()]
    return header, body


def render_markdown(document, text: str, counters: Counters, figure_root: Path) -> int:
    """Render one Markdown file; return the number of body words (headings included, captions excluded)."""
    lines = text.splitlines()
    words = 0
    index = 0
    pending_table_caption: tuple[str, str | None] | None = None
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped == "\\pagebreak":
            page_break(document, counters)
            index += 1
            continue
        if stripped in ("\\landscape", "\\portrait"):
            start_section(document, stripped == "\\landscape", counters)
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            if level == 1 and not counters.fresh_page:
                page_break(document, counters)
            counters.fresh_page = False
            paragraph = document.add_heading(level=level)
            add_inline(paragraph, heading.group(2), counters, size={1: 16, 2: 14, 3: 12}[level])
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0, 0, 0)
            words += len(heading.group(2).split())
            index += 1
            continue
        fig = FIG_RE.match(stripped)
        if fig:
            attrs = fig.group("attrs") or ""
            key_match = re.search(r"#fig:([\w-]+)", attrs)
            width_match = re.search(r"width=([\d.]+)", attrs)
            width = min(float(width_match.group(1)), TEXT_WIDTH_IN) if width_match else TEXT_WIDTH_IN
            key = key_match.group(1) if key_match else None
            number = counters.figures.get(key) if key else None
            if number is None:
                number = counters.figure_captions.index(fig.group("caption")) + 1
            image_path = (figure_root / fig.group("path")).resolve()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            keep_with_next(paragraph)
            counters.fresh_page = False
            if image_path.exists():
                picture = paragraph.add_run().add_picture(str(image_path), width=Cm(width * 2.54))
                set_alt_text(picture, f"Figure {number}. {plain_text(fig.group('caption'), counters)}")
            else:
                counters.problems.append(f"missing figure file {image_path}")
                paragraph.add_run(f"[missing figure: {fig.group('path')}]")
            add_caption(document, "Figure", number, fig.group("caption"), counters)
            index += 1
            continue
        tab = TAB_CAPTION_RE.match(stripped)
        if tab:
            pending_table_caption = (tab.group("caption"), tab.group("key"))
            index += 1
            continue
        counters.fresh_page = False
        if stripped.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            header, rows = parse_table_block(block)
            if pending_table_caption:
                caption, key = pending_table_caption
                number = counters.tables.get(key) if key else counters.table_captions.index(caption) + 1
                caption_paragraph_added = True
                add_caption(document, "Table", number, caption, counters)
                keep_with_next(document.paragraphs[-1])
                pending_table_caption = None
            else:
                caption_paragraph_added = False
                counters.problems.append(f"table without caption near: {block[0][:60]}")
            render_table(document, header, rows, counters)
            _ = caption_paragraph_added
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if bullet or numbered:
            style = "List Bullet" if bullet else "List Number"
            content = (bullet or numbered).group(1)
            paragraph = document.add_paragraph(style=style)
            paragraph.paragraph_format.space_after = Pt(3)
            add_inline(paragraph, content, counters)
            words += len(content.split())
            index += 1
            continue
        if stripped.startswith(">"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(1.0)
            add_inline(paragraph, stripped.lstrip("> "), counters)
            words += len(stripped.split())
            index += 1
            continue
        # paragraph: join consecutive non-empty, non-special lines
        block = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if (not nxt or nxt.startswith(("#", "|", "![", "- ", "* ", ">", "Table:", "\\pagebreak", "\\landscape", "\\portrait"))
                    or re.match(r"^\d+[.)]\s", nxt)):
                break
            block.append(nxt)
            index += 1
        paragraph = document.add_paragraph()
        content = " ".join(block)
        add_inline(paragraph, content, counters)
        words += len(XREF_RE.sub("Figure 0", content).split())
    return words


# ----------------------------------------------------------------------------- front matter


def add_cover(document, meta: dict, word_count: int) -> None:
    def centred(text: str, size: int, bold: bool = False, space_after: int = 12) -> None:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(space_after)
        run = paragraph.add_run(text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold

    for _ in range(5):
        document.add_paragraph()
    centred(meta["university"], 14, space_after=6)
    centred(meta["school"], 12, space_after=36)
    centred(f"{meta['module_code']} {meta['module_title']}", 14, bold=True, space_after=6)
    centred(meta["assessment"], 13, space_after=36)
    centred(meta["title"], 18, bold=True, space_after=6)
    centred(meta["subtitle"], 13, space_after=48)
    for label, key in (
        ("Student ID", "student_id"),
        ("Marker", "marker"),
        ("Submission date", "submission_date"),
        ("Word count (executive summary and sections 1 to 5 with headings; tables, captions, references "
         "and appendices excluded)", None),
        ("Repository", "repo_url"),
        ("Deployed application", "app_url"),
    ):
        value = f"{word_count:,}" if key is None else meta.get(key, "")
        centred(f"{label}: {value}", 12, space_after=4)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_toc(document, counters: Counters) -> None:
    heading = document.add_paragraph()
    run = heading.add_run("Table of contents")
    run.bold = True
    run.font.size = Pt(16)
    run.font.name = FONT
    paragraph = document.add_paragraph()
    add_field(paragraph, 'TOC \\o "1-3" \\h \\z \\u',
              "Right-click and choose Update Field (or press F9) to fill the table of contents.")
    document.add_paragraph()
    for label, captions in (("List of figures", counters.figure_captions), ("List of tables", counters.table_captions)):
        head = document.add_paragraph()
        run = head.add_run(label)
        run.bold = True
        run.font.size = Pt(14)
        run.font.name = FONT
        for number, caption in enumerate(captions, start=1):
            item = document.add_paragraph()
            item.paragraph_format.space_after = Pt(2)
            item.paragraph_format.left_indent = Cm(0.8)
            item.paragraph_format.first_line_indent = Cm(-0.8)
            add_inline(item, f"{'Figure' if label.startswith('List of f') else 'Table'} {number}. {caption}", counters, size=CAPTION_PT)
        document.add_paragraph()
    page_break(document, counters)


def add_abbreviations(document, meta: dict, counters: Counters) -> None:
    items = meta.get("abbreviations", {})
    if not items:
        return
    heading = document.add_paragraph()
    run = heading.add_run("Abbreviations")
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = FONT
    table = document.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for abbreviation, meaning in items.items():
        cells = table.add_row().cells
        for cell, text in zip(cells, (abbreviation, meaning), strict=True):
            cell.text = ""
            add_inline(cell.paragraphs[0], text, counters, size=TABLE_PT)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
    page_break(document, counters)


# ----------------------------------------------------------------------------- main


def section_files() -> tuple[list[Path], list[Path], list[Path]]:
    files = sorted(SECTIONS_DIR.glob("*.md"))
    body = [f for f in files if not f.name.startswith("9")]
    references = [f for f in files if f.name.startswith("90")]
    appendices = [f for f in files if f.name.startswith("9") and f not in references]
    return body, references, appendices


def build(output: Path | None = None, check_only: bool = False) -> Path:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    body, references, appendices = section_files()
    if not body:
        raise SystemExit(f"no section files found in {SECTIONS_DIR}")
    counters = Counters()
    collect_numbers(body + appendices, counters)
    source_problems = counters.problems + audit_sources(body + references + appendices, meta)
    counters.problems = []

    # word count pass on a scratch document (cheap; the renderer is the single source of truth)
    scratch = Document()
    configure_styles(scratch)
    body_words = sum(render_markdown(scratch, f.read_text(encoding="utf-8"), counters, ROOT) for f in body)
    counters.problems.clear()

    if check_only:
        print(f"body words: {body_words:,}")
        problems = list(source_problems)
        for path in body + references + appendices:
            file_counters = Counters(
                figures=counters.figures, tables=counters.tables,
                figure_captions=counters.figure_captions, table_captions=counters.table_captions,
            )
            words = render_markdown(Document(), path.read_text(encoding="utf-8"), file_counters, ROOT)
            problems.extend(f"{problem} ({path.name})" for problem in file_counters.problems)
            if path in body:
                print(f"  {path.name:40} {words:>6,} words")
        print(f"figures: {len(counters.figure_captions)}, tables: {len(counters.table_captions)}")
        if problems:
            print(f"PROBLEMS ({len(problems)}):")
            for problem in dict.fromkeys(problems):
                print("  -", problem)
            raise SystemExit(1)
        print("no problems found")
        return META_PATH

    document = Document()
    configure_styles(document)
    configure_page(document, meta["student_id"])
    document.core_properties.author = meta["student_id"]
    document.core_properties.last_modified_by = meta["student_id"]
    document.core_properties.title = meta["title"]
    document.core_properties.subject = f"{meta['module_code']} {meta['assessment']}"
    document.core_properties.comments = ""
    document.core_properties.keywords = ""

    add_cover(document, meta, body_words)
    add_toc(document, counters)
    add_abbreviations(document, meta, counters)
    for path in body + references + appendices:
        render_markdown(document, path.read_text(encoding="utf-8"), counters, ROOT)
    set_update_fields_on_open(document)

    output = output or REPORT_DIR / f"MLDA_CW1_{meta['student_id']}_report.docx"
    document.save(output)
    print(f"written {output} ({output.stat().st_size / 1e6:.1f} MB); body words {body_words:,}; "
          f"figures {len(counters.figure_captions)}; tables {len(counters.table_captions)}")
    problems = source_problems + sorted(set(counters.problems))
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for problem in problems:
            print("  -", problem)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report word counts and cross-reference problems only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        build(args.output, args.check)
    except SystemExit as error:
        if isinstance(error.code, str):
            print(error, file=sys.stderr)
        raise
