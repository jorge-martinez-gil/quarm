"""Audit scientific evidence and ICDE-format invariants for the release package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "icde2027"
MAIN_PDF = ROOT / "output" / "pdf" / "quarm-icde2027-submission.pdf"
SUPPLEMENT_PDF = ROOT / "output" / "pdf" / "quarm-icde2027-supplement.pdf"

NYC_HASHES = {
    ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet": "9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a",
    ROOT / "data" / "raw" / "taxi_zone_lookup.csv": "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path, *, main: bool) -> dict:
    if not path.exists():
        return {"exists": False, "errors": [f"missing {path}"]}
    reader = PdfReader(path)
    page_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_text)
    page_sizes = []
    fonts = {}
    for page in reader.pages:
        page_sizes.append((float(page.mediabox.width), float(page.mediabox.height)))
        resources = page.get("/Resources")
        if resources and resources.get("/Font"):
            for name, reference in resources["/Font"].get_object().items():
                fonts[str(name)] = reference.get_object()
    unembedded_fonts = []
    for name, font in fonts.items():
        descriptor = None
        descendants = font.get("/DescendantFonts")
        if descendants:
            descriptor = descendants[0].get_object().get("/FontDescriptor")
        else:
            descriptor = font.get("/FontDescriptor")
        descriptor = descriptor.get_object() if descriptor else None
        if not descriptor or not any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
            unembedded_fonts.append(f"{name}:{font.get('/BaseFont')}")
    out_of_bounds = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for char in page.chars:
                if char["x0"] < -0.1 or char["x1"] > page.width + 0.1 or char["top"] < -0.1 or char["bottom"] > page.height + 0.1:
                    out_of_bounds.append((page_number, char.get("text", "")))
    reference_page = next((index + 1 for index, value in enumerate(page_text) if "References" in value or "REFERENCES" in value), None)
    content_pages = None
    if reference_page is not None:
        value = page_text[reference_page - 1]
        marker = value.find("References") if "References" in value else value.find("REFERENCES")
        content_pages = reference_page - 1 if marker < 120 else reference_page
    errors = []
    if any(abs(width - 612) > 1 or abs(height - 792) > 1 for width, height in page_sizes):
        errors.append("non-US-Letter page detected")
    if out_of_bounds:
        errors.append(f"{len(out_of_bounds)} glyphs outside page bounds")
    if unembedded_fonts:
        errors.append(f"unembedded fonts: {', '.join(unembedded_fonts)}")
    if "??" in text:
        errors.append("unresolved cross-reference marker")
    if main and content_pages is not None and content_pages > 12:
        errors.append(f"content page limit exceeded: {content_pages}")
    if main and ("Appendix" in text or "APPENDIX" in text):
        errors.append("appendix detected in main paper")
    if main and reference_page is None:
        errors.append("reference section not detected")
    if "ACKNOWLEDGMENT OF AI-GENERATED CONTENT" not in text.upper():
        errors.append("AI-generated-content acknowledgment not detected")
    blockers = []
    normalized_text = " ".join(text.split())
    if "AUTHOR NAMES REQUIRED" in text:
        blockers.append("author metadata placeholder remains")
    if "public repository/DOI required" in normalized_text or "PUBLIC REPOSITORY/DOI REQUIRED" in normalized_text.upper():
        blockers.append("public artifact URL placeholder remains")
    return {
        "exists": True,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "pages": len(reader.pages),
        "content_pages_before_references": content_pages,
        "letter_pages": all(abs(width - 612) <= 1 and abs(height - 792) <= 1 for width, height in page_sizes),
        "out_of_bounds": len(out_of_bounds),
        "fonts_embedded": not unembedded_fonts,
        "font_count": len(fonts),
        "errors": errors,
        "blockers": blockers,
    }


def audit_manifests() -> tuple[list[dict], list[str]]:
    paths = [
        ROOT / "artifacts" / "results" / "manifest.json",
        ROOT / "artifacts" / "icde_results" / "manifest.json",
        ROOT / "artifacts" / "icde_results" / "nyc_manifest.json",
        ROOT / "artifacts" / "icde_results" / "sampling_scalability_m12_manifest.json",
        ROOT / "artifacts" / "icde_results" / "paired_design_ablation_manifest.json",
    ]
    records, errors = [], []
    for manifest_path in paths:
        if not manifest_path.exists():
            errors.append(f"missing manifest: {manifest_path.relative_to(ROOT)}")
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = []
        for relative, expected in data.get("files", {}).items():
            target = ROOT / relative
            if not target.exists():
                mismatches.append(f"missing {relative}")
            elif sha256(target) != expected:
                mismatches.append(f"hash mismatch {relative}")
        records.append({"manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "mismatches": mismatches})
        errors.extend(mismatches)
    return records, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    checks = []
    errors = []
    blockers = []

    if args.run_tests:
        completed = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "-q"], cwd=ROOT)
        checks.append({"tests_exit_code": completed.returncode})
        if completed.returncode:
            errors.append("test suite failed")

    main_pdf = inspect_pdf(MAIN_PDF, main=True)
    supplement_pdf = inspect_pdf(SUPPLEMENT_PDF, main=False)
    errors.extend(main_pdf.get("errors", []))
    errors.extend(supplement_pdf.get("errors", []))
    blockers.extend(main_pdf.get("blockers", []))
    blockers.extend(supplement_pdf.get("blockers", []))

    data_hashes = []
    for path, expected in NYC_HASHES.items():
        if path.exists():
            actual = sha256(path)
            match = actual == expected
            data_hashes.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "expected": expected, "actual": actual, "match": match})
            if not match:
                errors.append(f"NYC hash mismatch: {path.name}")
        else:
            data_hashes.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "expected": expected, "present": False})

    manifests, manifest_errors = audit_manifests()
    errors.extend(manifest_errors)
    metadata = (SUBMISSION / "author_metadata.yaml").read_text(encoding="utf-8")
    if "REQUIRED" in metadata:
        blockers.append("author_metadata.yaml remains incomplete")
    scope_length = len((SUBMISSION / "cmt_scope_statement.txt").read_text(encoding="utf-8"))
    if scope_length > 1000:
        errors.append(f"CMT scope statement exceeds 1000 characters: {scope_length}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if errors else ("blocked_on_submission_metadata" if blockers else "ready"),
        "main_pdf": main_pdf,
        "supplement_pdf": supplement_pdf,
        "nyc_inputs": data_hashes,
        "manifests": manifests,
        "checks": checks,
        "cmt_scope_characters": scope_length,
        "errors": sorted(set(errors)),
        "submission_blockers": sorted(set(blockers)),
    }
    out = SUBMISSION / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
