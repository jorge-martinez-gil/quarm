# QuaRM

QuaRM measures data quality as *workload-conditioned quality debt*: the change in bounded analytical risk caused by a specified set of data-defect mechanisms. It evaluates mechanism coalitions under paired randomness, attributes aggregate debt with exact or sampled Shapley values, exposes interactions, separates accounting from current-state repair, and reports repeat-level uncertainty.

The central design choice is deliberate: QuaRM does **not** pretend that completeness, label integrity, feature accuracy, and duplication are exchangeable percentages. It asks what each defect costs in the model and decision context that will actually consume the data.

## What is included

- Deterministic, composable ML corruption channels for missing cells, feature noise, target noise, and duplicates.
- Six relational mechanisms for missing measures, value noise, orphan foreign keys, stale timestamps, dimension drift, and duplicate facts.
- Eight analytical DuckDB queries over controlled star schemas and an official NYC TLC snapshot loader.
- Exact full-factorial risk-surface estimation with paired train/test splits and channel-specific random streams.
- Exact Shapley quality-debt attribution and pairwise Shapley interactions.
- Cached permutation-Shapley estimation with standard errors, coalition coverage, and a simultaneous finite-sample radius.
- Full-context repair marginals kept distinct from average Shapley accounting.
- Nonparametric paired bootstrap intervals and ranking probabilities.
- A CSV-oriented CLI, ML confirmation study, tests, figures, raw observations, provenance manifests, and an IEEEtran ICDE 2027 submission package.
- Honest boundaries: results are conditional on the chosen task, learner, clean evaluation set, channels, and severities.

## Reproduce

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --extra dev --extra paper
.venv\Scripts\python -m pytest
.venv\Scripts\python.exe experiments\run_study.py --repeats 12
.venv\Scripts\python.exe experiments\run_icde_study.py
.venv\Scripts\python.exe scripts\fetch_public_data.py
.venv\Scripts\python.exe experiments\run_icde_real.py
.venv\Scripts\python.exe experiments\run_sampling_scalability.py
.venv\Scripts\python.exe experiments\analyze_paired_design.py
powershell -ExecutionPolicy Bypass -File scripts\build_icde_submission.ps1
```

The checked-in evidence was generated with fixed master seeds and can be regenerated from source. The manifests in `artifacts/results` and `artifacts/icde_results` record versions, parameters, runtimes, source hashes, and SHA-256 evidence hashes. Official NYC source files are downloaded on demand and are not redistributed.

## Audit a CSV

```powershell
.venv\Scripts\quarm audit data.csv --target outcome --task classification `
  --model logistic --severity 0.15 --repeats 12 --output audit.json
```

The output includes coalition risks, debt attributions, confidence intervals, interaction indices, and the assumptions needed to interpret them.

## Scientific contract

QuaRM estimates a counterfactual quantity, not an intrinsic universal grade. A quality-debt estimate is meaningful only relative to:

1. a trusted evaluation distribution;
2. a learner and loss;
3. explicit, auditable defect mechanisms;
4. a severity policy; and
5. the randomness represented by the repeated paired design.

If no trusted test distribution exists, QuaRM can still be used as a stress test, but the result must be called *susceptibility*, not observed quality debt. See the paper for identifiability, concentration, and limitations.

## Repository map

```text
src/quarm/                 library, CLI, relational backend, and sampling
tests/                     invariants and reproducibility tests
experiments/               ML, relational, real-data, scale, and ablation studies
artifacts/results/         ML evidence and provenance
artifacts/icde_results/    relational evidence and provenance
artifacts/icde_figures/    publication figures
paper/icde2027/            IEEEtran paper and supplement sources
submission/icde2027/       CMT metadata, checklist, rebuttal, and audit report
output/pdf/                rendered submission PDFs
scripts/                   data fetch, build, asset generation, QA, and audit
```

## License

MIT. The scikit-learn toy datasets retain their upstream terms and are loaded from the installed package; no dataset copies are redistributed here.
