"""Render and mechanically inspect the ICDE submission PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default=str(ROOT / "output" / "pdf" / "main.pdf"))
    args = parser.parse_args()
    pdf_path = Path(args.pdf).resolve()
    out = ROOT / "tmp" / "icde_pdf"
    out.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(pdf_path)
    pages = []
    for index, page in enumerate(document):
        image = page.render(scale=1.75).to_pil().convert("RGB")
        path = out / f"icde-page-{index + 1:02d}.png"
        image.save(path)
        pages.append(image)

    thumb_w, thumb_h, gutter, label_h = 425, 550, 18, 22
    for sheet_index, start in enumerate(range(0, len(pages), 6), 1):
        group = pages[start : start + 6]
        canvas = Image.new("RGB", (2 * thumb_w + 3 * gutter, 3 * (thumb_h + label_h) + 4 * gutter), "#D9E1E7")
        draw = ImageDraw.Draw(canvas)
        for offset, page in enumerate(group):
            row, col = divmod(offset, 2)
            x = gutter + col * (thumb_w + gutter)
            y = gutter + row * (thumb_h + label_h + gutter)
            thumb = page.copy()
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
            draw.text((x + 4, y + thumb_h + 3), f"Page {start + offset + 1}", fill="#16324F")
        canvas.save(out / f"icde-contact-{sheet_index}.png")

    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    out_of_bounds = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for char in page.chars:
                if char["x0"] < -0.1 or char["x1"] > page.width + 0.1 or char["top"] < -0.1 or char["bottom"] > page.height + 0.1:
                    out_of_bounds.append((page_number, char.get("text", "")))
    normalized_text = " ".join(text.split())
    print(
        f"pages={len(reader.pages)} rendered={len(pages)} out_of_bounds={len(out_of_bounds)} "
        f"references={'References' in text or 'REFERENCES' in text} ai_ack={'ACKNOWLEDGMENT OF AI-GENERATED CONTENT' in text.upper()} "
        f"author_placeholder={'AUTHOR NAMES REQUIRED' in text} artifact_placeholder={'PUBLIC-ARTIFACT' in normalized_text or 'public repository/DOI required' in normalized_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
