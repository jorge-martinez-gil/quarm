# ICDE 2027 final submission checklist

## Blocking metadata

- [ ] Replace the visible author placeholder in `paper/icde2027/main.tex`.
- [ ] Enter every author's legal/preferred publication name, affiliation, email, country, and ORCID in `author_metadata.yaml`.
- [ ] Replace author placeholders in `CITATION.cff` and `pyproject.toml` before publishing the artifact.
- [ ] Confirm author order and the final title with every author.
- [ ] Collect CMT conflicts and institutional domains for every author.
- [ ] Create a public, archival artifact release and replace the artifact placeholder with its immutable URL/DOI.
- [ ] Have every listed author independently review the code, reproduce the evidence, verify the citations, and approve the AI acknowledgment.

## Scientific freeze

- [ ] Re-run `pytest` with zero failures.
- [ ] Re-run `scripts/make_icde_paper_assets.py` from checked-in CSV evidence.
- [ ] Rebuild both PDFs from a clean output directory.
- [ ] Run `scripts/audit_icde_submission.py` and archive its JSON report.
- [ ] Confirm all manuscript numbers against `artifacts/icde_results` and `artifacts/results`.
- [ ] Confirm the NYC source hashes match the pinned manifest.
- [ ] Confirm no statement treats the NYC snapshot as verified clean; the estimand is susceptibility.
- [ ] Confirm no claim treats Shapley accounting as a universally optimal repair policy.
- [ ] Confirm the public artifact reproduces the paper without private files.

## Format and policy

- [ ] Use `IEEEtran` conference format on US Letter pages.
- [ ] Confirm at most 12 content pages before references.
- [ ] Confirm there is no appendix in the submission PDF.
- [ ] Confirm author names/affiliations are visible (single-blind).
- [ ] Confirm ORCIDs are included for all authors.
- [ ] Confirm references are complete and every citation resolves.
- [ ] Confirm the AI-generated-content acknowledgment identifies OpenAI Codex and all affected components.
- [ ] Confirm fonts are embedded, figures are readable at 100%, and no content crosses page bounds.
- [ ] Confirm PDF accessibility metadata and title are set if the final IEEE tooling requires them.

## CMT entry

- [ ] Paste `abstract.txt` and compare it character-for-character with the PDF abstract.
- [ ] Paste `cmt_scope_statement.txt` (under 1,000 characters).
- [ ] Choose topics covering data preparation/cleaning/integration, data quality/curation/provenance/workflows, and benchmarking/testing.
- [ ] Upload the submission PDF and supplemental PDF.
- [ ] Enter the public artifact URL in the supplemental-material field.
- [ ] Verify author list, title, abstract, topics, conflicts, and files in the final CMT preview.
- [ ] Download the submitted PDF from CMT and visually compare it with the local audited PDF.

## Before the deadline

- [ ] Submit well before 11 November 2026, 17:00 Pacific Time.
- [ ] Save the CMT confirmation, submission ID, final PDFs, artifact release ID, hashes, and audit report.
