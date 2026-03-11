"""Generate AL-01 Architecture PDF from the markdown source."""
from __future__ import annotations

import re
from fpdf import FPDF

INPUT = "docs/AL-01_Architecture.md"
OUTPUT = "docs/AL-01_Architecture.pdf"

# Unicode → Latin-1 safe replacements
_UNICODE_MAP = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u2192": "->",   # right arrow
    "\u2190": "<-",   # left arrow
    "\u2264": "<=",   # less-or-equal
    "\u2265": ">=",   # greater-or-equal
    "\u00b1": "+/-",  # plus-minus
    "\u2016": "||",   # double vertical
    "\u2225": "||",   # parallel
    "\u2081": "_1",   # subscript 1
    "\u2080": "_0",   # subscript 0
    "\u1d62": "_i",   # subscript i
    "\u2074": "^4",   # superscript 4
    "\u207b": "-",    # superscript minus
    "\u2074": "^4",   # superscript 4
    "\u00d7": "x",    # multiplication sign
    "\u03b4": "delta",  # delta
    "\u03a3": "SUM",    # sigma
    "\u2211": "SUM",    # n-ary summation
    "\u221e": "inf",  # infinity
    "\u2248": "~=",   # approximately
    "\U0001F480": "[skull]",  # skull emoji
    "\u2039": "<",    # single left angle quote
    "\u203a": ">",    # single right angle quote
}


def _sanitize(text: str) -> str:
    """Replace Unicode characters that Latin-1 fonts can't render."""
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    # Fallback: strip any remaining non-latin1 chars
    out = []
    for ch in text:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            out.append("?")
    return "".join(out)


class ArchPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=20)
        # Core fonts (built-in, no TTF needed)
        self.add_page()
        self.set_margins(20, 20, 20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "AL-01: Artificial Life System Architecture", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ---- writers ----
    def title_block(self, title: str, subtitle: str):
        self.ln(30)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 12, _sanitize(title), align="C")
        self.ln(4)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 8, _sanitize(subtitle), align="C")
        self.ln(20)

    def section_heading(self, text: str, level: int):
        sizes = {1: 18, 2: 14, 3: 12}
        sz = sizes.get(level, 11)
        self.ln(4 if level > 1 else 8)
        self.set_font("Helvetica", "B", sz)
        self.set_text_color(20, 60, 120)
        self.multi_cell(0, sz * 0.55, _sanitize(text))
        if level == 1:
            self.set_draw_color(20, 60, 120)
            self.set_line_width(0.5)
            y = self.get_y()
            self.line(20, y, self.w - 20, y)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(1)

    def bold_text(self, label: str, rest: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.cell(self.get_string_width(_sanitize(label)) + 1, 5.5, _sanitize(label))
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, _sanitize(rest))
        self.ln(1)

    def code_block(self, text: str):
        self.set_font("Courier", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(240, 240, 240)
        x = self.get_x()
        for line in text.split("\n"):
            self.cell(0, 5, "  " + _sanitize(line), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        # Use a dash instead of unicode bullet (built-in fonts)
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(0.5)

    def table(self, headers: list[str], rows: list[list[str]]):
        n = len(headers)
        usable = self.w - 40  # margins
        col_w = usable / n

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_w, 7, _sanitize(h), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if self.get_y() > self.h - 25:
                self.add_page()
            self.set_fill_color(245, 248, 255) if fill else self.set_fill_color(255, 255, 255)
            max_h = 7
            for cell_text in row:
                # Estimate needed height
                lines_needed = max(1, len(cell_text) / (col_w / 2))
                max_h = max(max_h, int(lines_needed) * 5)
            max_h = min(max_h, 20)  # cap
            for cell_text in row:
                self.cell(col_w, max_h, _sanitize(cell_text[:60]), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


def parse_and_render(pdf: ArchPDF, md_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    title_done = False
    in_table = False
    table_headers: list[str] = []
    table_rows: list[list[str]] = []
    in_code = False
    code_buf: list[str] = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if table_headers and table_rows:
            pdf.table(table_headers, table_rows)
        in_table = False
        table_headers = []
        table_rows = []

    def flush_code():
        nonlocal in_code, code_buf
        if code_buf:
            pdf.code_block("\n".join(code_buf))
        in_code = False
        code_buf = []

    while i < len(lines):
        line = lines[i].rstrip("\n")
        raw = line

        # Title page
        if not title_done and line.startswith("# "):
            title = line.lstrip("# ").strip()
            subtitle = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("**"):
                i += 1
                subtitle = lines[i].strip().strip("*")
            pdf.title_block(title, subtitle)
            title_done = True
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            if in_table:
                flush_table()
            if in_code:
                flush_code()
            i += 1
            continue

        # Code block (indented with 4 spaces)
        if line.startswith("    ") and not line.strip().startswith("|") and not line.strip().startswith("-"):
            if in_table:
                flush_table()
            if not in_code:
                in_code = True
                code_buf = []
            code_buf.append(line[4:])
            i += 1
            continue
        elif in_code:
            flush_code()

        # Table
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Separator row
            if all(re.match(r"^[-:]+$", c) for c in cells):
                i += 1
                continue
            if not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            pdf.section_heading(m.group(2).strip(), level)
            i += 1
            continue

        # Bullet
        if line.strip().startswith("- "):
            text = line.strip()[2:]
            # Handle **bold** prefix in bullets
            bm = re.match(r"\*\*(.+?)\*\*\s*(.*)", text)
            if bm:
                pdf.bullet(f"{bm.group(1)} {bm.group(2)}")
            else:
                pdf.bullet(text)
            i += 1
            continue

        # Numbered list
        nm = re.match(r"^\d+\.\s+(.*)", line.strip())
        if nm:
            pdf.bullet(nm.group(1))
            i += 1
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Bold-prefixed paragraph
        bm = re.match(r"^\*\*(.+?)\*\*\s*(.*)", line.strip())
        if bm:
            rest = bm.group(2)
            # Collect continuation lines
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith("#") and not lines[i + 1].startswith("|") and not lines[i + 1].startswith("- ") and not re.match(r"^\d+\.", lines[i + 1].strip()) and not lines[i + 1].startswith("    "):
                i += 1
                rest += " " + lines[i].strip()
            pdf.bold_text(bm.group(1) + " ", rest)
            i += 1
            continue

        # Regular paragraph text
        para = line.strip()
        while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith("#") and not lines[i + 1].startswith("|") and not lines[i + 1].startswith("- ") and not re.match(r"^\d+\.", lines[i + 1].strip()) and not lines[i + 1].startswith("    ") and not lines[i + 1].strip().startswith("**"):
            i += 1
            para += " " + lines[i].strip()
        # Strip remaining markdown formatting
        para = re.sub(r"\*\*(.+?)\*\*", r"\1", para)
        para = re.sub(r"\*(.+?)\*", r"\1", para)
        para = re.sub(r"`(.+?)`", r"\1", para)
        pdf.body_text(para)
        i += 1

    if in_table:
        flush_table()
    if in_code:
        flush_code()


def main():
    pdf = ArchPDF()
    pdf.alias_nb_pages()
    parse_and_render(pdf, INPUT)
    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    main()
