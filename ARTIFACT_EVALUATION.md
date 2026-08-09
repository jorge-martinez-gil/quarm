# QuaRM artifact evaluation

This document maps every central empirical claim to an executable source and a machine-readable result. It is intentionally narrower than the paper: passing the artifact checks verifies computational integrity, not the external validity of the defect models.

## Quick evaluation

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m compileall -q src experiments scripts
.venv\Scripts\quarm.exe --help
powershell -ExecutionPolicy Bypass -File scripts\build_icde_submission.ps1
```

Expected time is under one minute after dependencies are installed. Full regeneration takes about 25 minutes on the recorded CPU environment:

```powershell
.venv\Scripts\python.exe experiments\run_study.py --repeats 12 --dose-repeats 8 --bootstrap 2000
.venv\Scripts\python.exe experiments\run_icde_study.py
.venv\Scripts\python.exe scripts\fetch_public_data.py
.venv\Scripts\python.exe experiments\run_icde_real.py
.venv\Scripts\python.exe experiments\run_sampling_scalability.py
.venv\Scripts\python.exe experiments\analyze_paired_design.py
powershell -ExecutionPolicy Bypass -File scripts\build_icde_submission.ps1
```

## Claim-evidence map

| Claim | Evidence | Independent check |
|---|---|---|
| Every relational population evaluates all 64 coalitions for 12 paired repeats | `relational_observations.csv`, `nyc_observations.csv` | 768 unique `(repeat, coalition)` rows per population |
| Shapley accounting exactly closes aggregate debt | `relational_summary.csv`, `nyc_summary.csv`, `study_summary.csv` | every stored `efficiency_residual` is zero |
| Equal 10% prevalence produces population-specific accounts | `relational_attributions.csv`, `nyc_attributions.csv` | compare top mechanisms and intervals |
| Query priorities reverse across joins, rollups, counts, and averages | `query_attributions.csv`, `nyc_query_attributions.csv` | group by query and select maximum debt |
| Interactions reach 0.0590 on SQL and 0.0203 on ML | relational/ML `interactions.csv` | maximum absolute `interaction` |
| Shapley-first repair is not the full-context oracle in three generated populations | `relational_summary.csv` | compare `shapley_choice` and `oracle_choice` |
| Shapley-first removal increases current loss in Uniform and Seasonal | `relational_summary.csv` | `shapley_gain < 0` |
| NYC dimension drift accounts for 0.0956 [0.0845, 0.1073] | `nyc_attributions.csv` | inspect point and bootstrap interval |
| At 12 mechanisms, 64 permutations achieve 95.8% top-one recovery while evaluating 12.8% of the surface | `sampling_scalability_m12.csv` | group 400 trials at `permutations=64` |
| Pairing can narrow or widen intervals | `paired_design_ablation.csv` | ratios range from 0.10 to 3.66 |
| Exact 1M-fact evaluation completes in 97.22 seconds | `scale.csv` | inspect the 1,000,000-row record |
| Seven of eight ML settings have positive debt; a signed negative case is retained | `study_summary.csv`, `attributions.csv` | count positive totals and inspect synthetic-logistic |
| The final PDFs are US Letter, in bounds, and hash-audited | `submission/icde2027/audit_report.json` | inspect `errors`, page geometry, and hashes |

## Expected non-claims

- The study does not claim that the bundled synthetic channels reproduce all real defects.
- It does not claim universal superiority over singleton stress testing or that the largest Shapley account is the best current repair.
- It does not identify latent observed quality without a trusted reference.
- It does not turn predictive utility into semantic, fairness, privacy, or legal correctness.

## Result files

- `artifacts/results/coalition_observations.csv`: repeat-level bounded losses.
- `artifacts/results/attributions.csv`: exact debt allocations and paired bootstrap summaries.
- `artifacts/results/interactions.csv`: pairwise Shapley interaction indices.
- `artifacts/results/study_summary.csv`: setting-level claims and baselines.
- `artifacts/results/dose_response.csv`: severity-response allocations.
- `artifacts/results/manifest.json`: ML environment, protocol, runtime, and evidence hashes.
- `artifacts/icde_results/relational_observations.csv`: relational repeat-level bounded losses.
- `artifacts/icde_results/relational_attributions.csv`: relational exact allocations and uncertainty.
- `artifacts/icde_results/relational_interactions.csv`: relational interaction indices.
- `artifacts/icde_results/query_attributions.csv`: per-query allocations.
- `artifacts/icde_results/nyc_*.csv`: official-snapshot evidence.
- `artifacts/icde_results/sampling_scalability_m12.csv`: 12-mechanism approximation trials.
- `artifacts/icde_results/scale.csv`: end-to-end exact row-scaling results.
- `artifacts/icde_results/paired_design_ablation.csv`: paired-versus-independent interval widths.
- `submission/icde2027/audit_report.json`: final PDF, source-data, and evidence audit.
