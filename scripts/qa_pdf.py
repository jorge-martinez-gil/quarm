"""Render every paper page and create contact sheets for visual QA."""

from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "quarm-paper.pdf"
OUT = ROOT / "tmp" / "pdfs"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(PDF)
    rendered = []
    for index, page in enumerate(document):
        image = page.render(scale=1.45).to_pil().convert("RGB")
        path = OUT / f"quarm-page-{index + 1:02d}.png"
        image.save(path)
        rendered.append(image)

    # Two readable seven-page sheets: 2 columns x 4 rows, with page labels.
    thumb_w, thumb_h, gutter, label_h = 444, 575, 18, 22
    for sheet_index, start in enumerate(range(0, len(rendered), 7), 1):
        group = rendered[start : start + 7]
        canvas = Image.new("RGB", (2 * thumb_w + 3 * gutter, 4 * (thumb_h + label_h) + 5 * gutter), "#D9E1E7")
        draw = ImageDraw.Draw(canvas)
        for offset, image in enumerate(group):
            row, col = divmod(offset, 2)
            x = gutter + col * (thumb_w + gutter)
            y = gutter + row * (thumb_h + label_h + gutter)
            thumb = image.copy()
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
            draw.text((x + 4, y + thumb_h + 3), f"Page {start + offset + 1}", fill="#16324F")
        canvas.save(OUT / f"quarm-contact-{sheet_index}.png")

    reader = PdfReader(PDF)
    with pdfplumber.open(PDF) as pdf:
        out_of_bounds = []
        for page_number, page in enumerate(pdf.pages, 1):
            for char in page.chars:
                if char["x0"] < -0.1 or char["x1"] > page.width + 0.1 or char["top"] < -0.1 or char["bottom"] > page.height + 0.1:
                    out_of_bounds.append((page_number, char.get("text", "")))
        print(f"pages={len(reader.pages)} rendered={len(rendered)} out_of_bounds={len(out_of_bounds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
