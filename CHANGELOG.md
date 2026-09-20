# Changelog

## 0.2.0 - 2026-08-09

- Report singleton effects, current-state repair gains with paired bootstrap intervals, and normalized Banzhaf values (with residual) directly from `attribute_quality_debt`.
- Vectorize the exact Shapley and Banzhaf allocations behind precomputed weight matrices; every bootstrap allocation is now one matrix product with unchanged point estimates.
- Add reversed-pair (antithetic) permutation sampling with valid pair-based standard errors and a configurable failure probability for the Hoeffding radius.
- Add declared nonnegative query-weight policies to the relational workload risk (equal weights remain the default and reproduce prior numbers exactly).
- Expand the test suite from 23 to 30 tests, including an executable version of the manuscript's accounting/intervention counterexample.
- Fold the supplement's evidence tables (complete relational allocations, ML confirmation study, sampling budgets, mechanism contract) into the main ICDE paper; the supplement becomes a lean artifact guide.
- Surface the new estimands in the CLI audit payload and update the AI-generated-content acknowledgment.

## 0.1.0 - 2026-08-07

- Introduce the counterfactual quality-risk surface and exact mechanism-level Shapley debt.
- Add pairwise interaction indices, paired bootstrap intervals, ranking probabilities, and a bounded-loss concentration radius.
- Add four deterministic corruption channels with coalition-invariant random streams.
- Add a DuckDB relational backend with eight analytical queries and six referential, temporal, dimensional, value, missingness, and duplication mechanisms.
- Add exact workload surfaces for three 100K-fact populations and a hash-pinned 100K-row official NYC TLC snapshot.
- Distinguish Shapley accounting from full-context repair and expose the empirical decision gap.
- Add normalized Banzhaf comparison, cached permutation Shapley with finite-sample control, a 12-mechanism scalability study, and a paired-design ablation.
- Add CSV auditing, offline benchmarks, 23 tests, evidence manifests, an IEEEtran ICDE 2027 paper, supplementary material, and a render-and-audit submission workflow.
