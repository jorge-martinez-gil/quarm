"""Fail closed if checked-in evidence, claims, or the final PDF are inconsistent."""

from __future__ import annotations

import json
import re
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PDF = ROOT / "output" / "pdf" / "quarm-paper.pdf"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        require(path.exists(), f"manifest file missing: {relative}")
        require(digest(path) == expected, f"hash mismatch: {relative}")

    summary = pd.read_csv(RESULTS / "study_summary.csv")
    observations = pd.read_csv(RESULTS / "coalition_observations.csv")
    attributions = pd.read_csv(RESULTS / "attributions.csv")
    interactions = pd.read_csv(RESULTS / "interactions.csv")
    require(len(summary) == 8, "expected eight settings")
    require((observations.groupby("setting").size() == 192).all(), "incomplete 12 x 16 design")
    require(summary["efficiency_residual"].abs().max() < 1e-12, "allocation does not exactly account")
    require(int((summary["total_debt"] > 0).sum()) == 7, "positive-debt setting count changed")
    require(int((summary["shapley_choice"] == summary["oracle_choice"]).sum()) == 7, "oracle-match count changed")
    require(abs(summary["shapley_realized_repair_gain"].mean() - 0.01564200478903069) < 1e-12, "mean repair gain changed")
    require(abs(summary["uniform_expected_repair_gain"].mean() - 0.0047486458040462) < 1e-12, "uniform gain changed")
    require(abs(summary["median_paired_unpaired_width_ratio"].median() - 0.8504086689524784) < 1e-12, "pair ratio changed")
    require(abs(interactions["interaction"].abs().max() - 0.0203189300411522) < 1e-12, "maximum interaction changed")
    counterexample = attributions[(attributions["setting"] == "synthetic_nonlinear__logistic") & (attributions["channel"] == "feature_noise")]
    require(len(counterexample) == 1 and counterexample.iloc[0]["ci_high"] < 0, "negative-debt counterexample missing")

    reader = PdfReader(PDF)
    require(len(reader.pages) == 13, "unexpected paper page count")
    text = "".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = re.sub(r"\s+", " ", text)
    require("PLACEHOLDER" not in text and "TODO" not in text, "paper contains placeholder text")
    require("QuaRM: Data Quality as a Counterfactual Risk Surface" in normalized_text, "paper title missing")
    with pdfplumber.open(PDF) as pdf:
        out_of_bounds = [
            (number, char.get("text", ""))
            for number, page in enumerate(pdf.pages, 1)
            for char in page.chars
            if char["x0"] < -0.1
            or char["x1"] > page.width + 0.1
            or char["top"] < -0.1
            or char["bottom"] > page.height + 0.1
        ]
    require(not out_of_bounds, f"out-of-bounds PDF glyphs: {out_of_bounds[:3]}")
    print("artifact audit: PASS")
    print("8 settings | 1,536 main fits | 512 dose fits | exact accounting | 13-page PDF")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"artifact audit: FAIL: {exc}", file=sys.stderr)
        raise
