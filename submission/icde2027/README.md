# ICDE 2027 submission package

Target: regular research track, Round 2, IEEE ICDE 2027.

Official constraints verified on 7 August 2026:

- Conference: Copenhagen, Denmark, 17–21 May 2027.
- Round 2 deadline: 11 November 2026, 17:00 Pacific Time.
- Review model: single-blind; authors and affiliations must be visible.
- Format: IEEE conference format, at most 12 content pages. References and the AI-generated-content acknowledgment are excluded from that limit.
- Appendices are not allowed in the paper.
- Every author must supply an ORCID.
- Title and author list require exceptional care because post-acceptance changes are restricted.
- A public supplemental-artifact URL is expected; supplemental material is visible to reviewers.
- AI-generated content must be acknowledged with the system and scope of use.

Primary sources:

- https://icde2027.github.io/cf-research-papers.html
- https://icde2027.github.io/submission-guidelines.html
- https://icde2027.github.io/important-dates.html

## Package status

The scientific manuscript, code, experiments, figures, official-format build, evidence manifests, and reviewer-facing materials are prepared. Submission is intentionally blocked on real author metadata and a public archival artifact URL; these cannot be invented.

Final PDF target: `../../output/pdf/quarm-icde2027-submission.pdf`

Supplement target: `../../output/pdf/quarm-icde2027-supplement.pdf`

## Build

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_tectonic.ps1
.venv\Scripts\python.exe scripts\make_icde_paper_assets.py
$env:TECTONIC_CACHE_DIR = (Resolve-Path 'tools\tectonic').Path + '\cache'
tools\tectonic\tectonic.exe -X compile paper\icde2027\main.tex --outdir output\pdf
tools\tectonic\tectonic.exe -X compile paper\icde2027\supplement.tex --outdir output\pdf
.venv\Scripts\python.exe scripts\qa_icde_pdf.py output\pdf\quarm-icde2027-submission.pdf
```

Use `SUBMISSION_CHECKLIST.md` for the final CMT handoff.
