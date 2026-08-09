"""Build the publication PDF from paper.md and checked-in experiment evidence."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "paper.md"
OUT = ROOT / "output" / "pdf" / "quarm-paper.pdf"
FIGURES = ROOT / "artifacts" / "figures"
RESULTS = ROOT / "artifacts" / "results"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#D55E00")
INK = colors.HexColor("#20242A")
MUTED = colors.HexColor("#5B6470")
PALE = colors.HexColor("#F2F6F9")
RULE = colors.HexColor("#CBD6DF")


def register_fonts() -> None:
    paths = {
        "DVSerif": font_manager.findfont("DejaVu Serif"),
        "DVSerif-Bold": font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold")),
        "DVSerif-Italic": font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", style="italic")),
        "DVSans": font_manager.findfont("DejaVu Sans"),
        "DVSans-Bold": font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")),
        "DVMono": font_manager.findfont("DejaVu Sans Mono"),
    }
    for name, path in paths.items():
        pdfmetrics.registerFont(TTFont(name, path))


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=draw_page))

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"H1", "H2", "H3"}:
            level = {"H1": 0, "H2": 1, "H3": 2}[flowable.style.name]
            text = flowable.getPlainText()
            key = f"section-{self.page}-{abs(hash(text))}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = letter
    if doc.page > 1:
        canvas.setFont("DVSans-Bold", 7.2)
        canvas.setFillColor(NAVY)
        canvas.drawString(doc.leftMargin, height - 0.43 * inch, "QUARM | COUNTERFACTUAL DATA-QUALITY EVIDENCE")
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(1.0)
        canvas.line(doc.leftMargin, height - 0.49 * inch, width - doc.rightMargin, height - 0.49 * inch)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.46 * inch, width - doc.rightMargin, 0.46 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("DVSans", 7)
    canvas.drawString(doc.leftMargin, 0.28 * inch, "Anonymous research artifact | 7 August 2026")
    canvas.drawRightString(width - doc.rightMargin, 0.28 * inch, str(doc.page))
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="DVSerif",
        fontSize=8.7,
        leading=11.6,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "title": ParagraphStyle("Title", parent=body, fontName="DVSans-Bold", fontSize=24, leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=body, fontName="DVSans", fontSize=9, leading=12, textColor=MUTED, spaceAfter=14),
        "abstract_label": ParagraphStyle("AbstractLabel", parent=body, fontName="DVSans-Bold", fontSize=9.5, leading=12, textColor=BLUE, spaceBefore=3, spaceAfter=4),
        "abstract": ParagraphStyle("Abstract", parent=body, fontSize=8.5, leading=11.3, leftIndent=14, rightIndent=14, borderColor=RULE, borderWidth=0.6, borderPadding=9, backColor=PALE, spaceAfter=8),
        "H1": ParagraphStyle("H1", parent=body, fontName="DVSans-Bold", fontSize=15, leading=18, textColor=NAVY, spaceBefore=15, spaceAfter=7, keepWithNext=True),
        "H2": ParagraphStyle("H2", parent=body, fontName="DVSans-Bold", fontSize=11.2, leading=14, textColor=BLUE, spaceBefore=11, spaceAfter=5, keepWithNext=True),
        "H3": ParagraphStyle("H3", parent=body, fontName="DVSans-Bold", fontSize=9.4, leading=12, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "body": body,
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=2, firstLineIndent=0, spaceAfter=2),
        "caption": ParagraphStyle("Caption", parent=body, fontName="DVSans", fontSize=7.4, leading=9.5, textColor=MUTED, alignment=TA_LEFT, spaceBefore=4, spaceAfter=10),
        "equation": ParagraphStyle("Equation", parent=body, fontName="DVSerif-Italic", fontSize=9.3, leading=13, alignment=TA_CENTER, leftIndent=18, rightIndent=18, borderColor=RULE, borderWidth=0.5, borderPadding=7, backColor=colors.HexColor("#FAFBFC"), spaceBefore=5, spaceAfter=8),
        "table_caption": ParagraphStyle("TableCaption", parent=body, fontName="DVSans", fontSize=7.5, leading=9.5, textColor=MUTED, spaceAfter=4),
        "reference": ParagraphStyle("Reference", parent=body, fontSize=7.4, leading=9.4, leftIndent=14, firstLineIndent=-14, alignment=TA_LEFT, spaceAfter=3),
        "toc_title": ParagraphStyle("TOCTitle", parent=body, fontName="DVSans-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6),
    }


def inline(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="DVMono">\1</font>', text)
    return text


def image_flowable(filename: str, caption: str, style) -> KeepTogether:
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    max_width = 7.05 * inch
    max_height = 4.25 * inch
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([Spacer(1, 6), image, Paragraph(inline(caption), style)])


def styled_table(data, widths, *, font_size=6.5, repeat_rows=1):
    table = Table(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "DVSans-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "DVSans"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def main_results_table(st):
    frame = pd.read_csv(RESULTS / "study_summary.csv")
    data = [["Dataset / model", "Clean", "Full", "Debt", "Top channel", "P(top)", "Pair ratio"]]
    for row in frame.itertuples():
        data.append(
            [
                row.setting.replace("__", " / ").replace("synthetic_nonlinear", "synthetic"),
                f"{row.clean_loss:.3f}",
                f"{row.fully_corrupted_loss:.3f}",
                f"{row.total_debt:+.3f}",
                row.top_channel.replace("_", " "),
                f"{row.top_probability:.2f}",
                f"{row.median_paired_unpaired_width_ratio:.2f}",
            ]
        )
    caption = Paragraph("Table 1. Main results at 15% prevalence. Pair ratio is paired/unpaired interval width; lower is better.", st["table_caption"])
    table = styled_table(data, [1.75 * inch, 0.48 * inch, 0.48 * inch, 0.48 * inch, 1.05 * inch, 0.48 * inch, 0.62 * inch])
    return [caption, table, Spacer(1, 8)]


def attribution_detail_table(st):
    frame = pd.read_csv(RESULTS / "attributions.csv")
    data = [["Setting", "Channel", "Debt", "95% interval", "P(harm)", "P(top)"]]
    for row in frame.itertuples():
        data.append(
            [
                row.setting.replace("__", "/").replace("synthetic_nonlinear", "synthetic"),
                row.channel.replace("_", " "),
                f"{row.debt:+.4f}",
                f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]",
                f"{row.probability_harmful:.3f}",
                f"{row.rank_1_probability:.3f}",
            ]
        )
    caption = Paragraph("Table 2. Complete mechanism attributions from 2,000 paired bootstrap draws.", st["table_caption"])
    table = styled_table(data, [1.65 * inch, 1.0 * inch, 0.55 * inch, 1.2 * inch, 0.6 * inch, 0.55 * inch], font_size=5.9)
    return [caption, table]


def parse_markdown(st):
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_kind = None
    in_abstract = False
    first_h1 = True

    def flush_paragraph():
        nonlocal paragraph, in_abstract
        if not paragraph:
            return
        text = " ".join(piece.strip() for piece in paragraph)
        style = st["abstract"] if in_abstract else (st["reference"] if re.match(r"^\[\d+\]", text) else st["body"])
        story.append(Paragraph(inline(text), style))
        paragraph = []
        in_abstract = False

    def flush_list():
        nonlocal list_items, list_kind
        if not list_items:
            return
        items = [ListItem(Paragraph(inline(item), st["bullet"]), leftIndent=12) for item in list_items]
        story.append(
            ListFlowable(
                items,
                bulletType="1" if list_kind == "number" else "bullet",
                start="1",
                leftIndent=20,
                bulletFontName="DVSans",
                bulletFontSize=7.5,
                spaceAfter=5,
            )
        )
        list_items, list_kind = [], None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            flush_paragraph()
            flush_list()
            continue
        fig_match = re.fullmatch(r"\[\[FIGURE:([^|]+)\|(.+)\]\]", line)
        eq_match = re.fullmatch(r"\[\[EQUATION:(.+)\]\]", line)
        table_match = re.fullmatch(r"\[\[TABLE:([^]]+)\]\]", line)
        if fig_match or eq_match or table_match:
            flush_paragraph()
            flush_list()
            if fig_match:
                story.append(image_flowable(fig_match.group(1), fig_match.group(2), st["caption"]))
            elif eq_match:
                equation = eq_match.group(1).replace("\\\\", "\\")
                story.append(Paragraph(inline(equation), st["equation"]))
            elif table_match.group(1) == "main_results":
                story.extend(main_results_table(st))
            elif table_match.group(1) == "attribution_detail":
                story.extend(attribution_detail_table(st))
            continue
        if line.startswith("# "):
            flush_paragraph()
            flush_list()
            title = line[2:]
            if first_h1:
                story.append(Spacer(1, 0.28 * inch))
                story.append(HRFlowable(width="18%", thickness=4, color=BLUE, hAlign="LEFT", spaceAfter=14))
                story.append(Paragraph(inline(title), st["title"]))
                first_h1 = False
            else:
                story.append(Paragraph(inline(title), st["H1"]))
            continue
        if line.startswith("## "):
            flush_paragraph()
            flush_list()
            heading = line[3:]
            if heading == "Abstract":
                story.append(Paragraph("ABSTRACT", st["abstract_label"]))
                in_abstract = True
            elif heading == "References":
                story.append(PageBreak())
                story.append(Paragraph(inline(heading), st["H1"]))
            else:
                story.append(Paragraph(inline(heading), st["H1"]))
            continue
        if line.startswith("### "):
            flush_paragraph()
            flush_list()
            story.append(Paragraph(inline(line[4:]), st["H2"]))
            continue
        if re.match(r"^- ", line):
            flush_paragraph()
            if list_kind not in {None, "bullet"}:
                flush_list()
            list_kind = "bullet"
            list_items.append(line[2:])
            continue
        numbered = re.match(r"^\d+\. (.+)", line)
        if numbered:
            flush_paragraph()
            if list_kind not in {None, "number"}:
                flush_list()
            list_kind = "number"
            list_items.append(numbered.group(1))
            continue
        if first_h1 is False and len(story) == 3:
            story.append(Paragraph(inline(line), st["subtitle"]))
            continue
        paragraph.append(line)
    flush_paragraph()
    flush_list()
    return story


def main() -> int:
    register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for required in [
        RESULTS / "study_summary.csv",
        RESULTS / "attributions.csv",
        FIGURES / "figure1_framework.png",
        FIGURES / "figure2_attributions.png",
        FIGURES / "figure3_interactions.png",
        FIGURES / "figure4_dose_response.png",
    ]:
        if not required.exists():
            raise FileNotFoundError(f"required evidence missing: {required}")
    doc = PaperDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="QuaRM: Data Quality as a Counterfactual Risk Surface",
        author="Anonymous Authors",
        subject="Data quality, counterfactual evaluation, and Shapley attribution",
        creator="QuaRM reproducible paper builder",
    )
    doc.multiBuild(parse_markdown(styles()))
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
