# QuaRM: Data Quality as a Counterfactual Risk Surface

Anonymous submission | Research artifact version 0.1.0 | 7 August 2026

## Abstract

Data-quality dashboards commonly place heterogeneous defect rates beside one another, although an equal prevalence of missing values, corrupted labels, noisy features, and duplicates need not have equal consequences. A universal scalar score is therefore easy to compute and hard to defend. We introduce **QuaRM**, a reference-based framework that defines *task-conditioned quality debt* as the clean-test risk added by explicit defect mechanisms. QuaRM estimates the complete counterfactual risk surface over defect coalitions using paired randomness, allocates aggregate debt with exact Shapley values, measures non-additive interactions, and quantifies uncertainty by resampling whole experimental repeats. The framework includes an identifiability boundary: without a trusted reference and evaluation distribution, the same procedure estimates susceptibility to additional defects, not the latent quality of observed data.

We provide an open, deterministic Python implementation and a controlled study spanning four tabular datasets, classification and regression, and linear and tree learners. At equal 15% prevalence, the most harmful channel depends on the dataset-model pair: target noise leads in three of eight settings, feature noise in three, missingness in two, and duplicates in none. QuaRM's first repair choice matches the full-context oracle in seven settings and captures 1.56 risk percentage points on average, compared with 0.47 for a uniformly chosen repair; a singleton-effect baseline also matches seven, showing that this compact benchmark establishes accounting and diagnosis rather than universal selection superiority. Pairwise interactions reach 2.03 risk points. Paired resampling yields 15% narrower intervals at the median, but not in every setting, exactly as the covariance analysis predicts. A deliberately preserved negative-debt case shows that a nominal defect can regularize a misspecified learner. QuaRM replaces an unqualified score with an auditable scientific claim: quality relative to a reference, use, mechanism, severity, and uncertainty.

**Keywords:** data quality; data-centric AI; counterfactual evaluation; Shapley attribution; data defects; uncertainty; reproducibility

## 1. Introduction

The phrase "data quality" carries an appealing promise: inspect a dataset, compute a score, and decide whether the data are good. The promise is usually false unless the intended use is specified. Data consumers have long treated quality as multidimensional and contextual [1]. Modern systems operationalize useful checks for missingness, constraints, drift, and anomalies [2-4]. Yet a monitoring panel still leaves the consequential question unanswered: which defect matters for the model or decision that will consume the data?

Consider four training tables, each with 15% defective units. In one, cells are missing; in another, numeric cells are perturbed; in a third, targets are corrupted; in a fourth, rows are duplicated. A prevalence-only score makes these tables look comparable. They are not. Median imputation can absorb missingness, a robust learner can tolerate feature noise, labels can determine the decision boundary, and random duplicates can be almost inert. The order can reverse under another learner, task, severity, or population. Collapsing prevalence into a weighted sum merely relocates the scientific problem into unvalidated weights.

The alternative is to define quality through counterfactual consequences. Let a trusted reference dataset be acted on by explicit defect mechanisms. For every coalition of mechanisms, train the intended learner under paired randomness and evaluate on a clean reference distribution. The resulting set function is a *risk surface*. Its difference between the fully corrupted and clean coalitions is aggregate quality debt. Its marginal changes reveal context dependence. A Shapley allocation distributes that debt across mechanisms while satisfying exact accounting, symmetry, dummy, and additivity properties [12]. Pairwise Shapley interaction indices expose deviations from additive thinking.

QuaRM realizes this design as both a mathematical object and an executable protocol. It makes seven contributions:

1. **A reference-based estimand.** Quality debt is defined as added task risk under a declared learner, loss, test distribution, defect family, and severity policy.
2. **A complete mechanism-level risk surface.** QuaRM evaluates all 2^m coalitions for modest m, preserving interactions that one-defect-at-a-time stress tests omit.
3. **Exact accounting with uncertainty.** Shapley debt allocations sum exactly to aggregate debt; paired bootstrap intervals and top-rank probabilities preserve repeat-level dependence.
4. **Deterministic intervention semantics.** Channel-specific cryptographic seeds are invariant to coalition membership, and a canonical composition order makes every counterfactual replayable.
5. **An explicit identifiability boundary.** Without a trusted reference, latent observed debt cannot generally be identified. QuaRM labels the estimand susceptibility in that regime.
6. **Theory matched to implementation.** We give allocation, variance, and finite-sample results and test their computational invariants.
7. **A reproducible artifact.** The package includes a CSV CLI, benchmark registry, tests, raw coalition observations, figures, hashes, environment metadata, and a paper built from those evidence files.

[[FIGURE:figure1_framework.png|Figure 1. QuaRM is an experimental contract, not a universal grade. Every output remains conditional on its reference, task, mechanisms, and randomization policy.]]

The result is not a claim that Shapley values magically reveal intrinsic quality. The players here are defect *mechanisms*, not individual records, and the value function is a bounded reference risk. This distinction separates QuaRM from data valuation while borrowing the only part of cooperative game theory that we need: principled allocation of a known aggregate across interacting causes.

## 2. Related Work and the Missing Estimand

### 2.1 Quality dimensions and automated checks

Wang and Strong's consumer-grounded study established that data quality extends beyond accuracy to contextual, representational, and accessibility dimensions [1]. Contemporary validation systems turn parts of this taxonomy into executable constraints. HoloClean combines integrity constraints, external information, and statistical signals in probabilistic repairs [2]. Raha learns to combine error detectors with limited user labels [3]. Large-scale verification systems such as Deequ derive and check data constraints in production pipelines [4]. These systems answer whether values violate a model, rule, or learned regularity and, in some cases, how to repair them.

QuaRM asks a different downstream question. A violation count does not identify the loss attributable to that violation under a consuming task. We therefore take detectors and corruption models as inputs rather than competitors. A production integration can map observed violations to calibrated mechanisms, then use QuaRM to prioritize which class of issue deserves remediation.

### 2.2 Data cleaning and downstream utility

Data cleaning research increasingly evaluates predictive consequences. HoloClean reports repair accuracy and downstream behavior [2], while conformal data cleaning calibrates error detection and assesses downstream gains over tabular tasks [11]. This trajectory is important: task performance provides a consequential criterion that row-level detection metrics cannot. However, interventions are often assessed one method or one error family at a time. That design cannot allocate the loss of a dataset containing several interacting defect families.

QuaRM fills this narrow gap with a full-factorial mechanism design. It does not propose a new repair algorithm. It estimates the response surface on which repair policies can be compared and audited.

### 2.3 Data valuation

Data Shapley allocates model utility to training records [5], with efficient approximations [6], distributional extensions [7], and alternatives designed for noisy utility evaluation [8]. OpenDataVal shows that no valuation algorithm is uniformly best across downstream data-selection tasks [9]. Recent theory also demonstrates that Shapley-based selection can be no better than random for unconstrained utility functions [10]. Those findings motivate restraint.

QuaRM differs along three axes. First, its players are named defect mechanisms, usually four to ten, rather than thousands or millions of records. Exact enumeration is therefore feasible and approximation error is unnecessary. Second, the value is risk under controlled interventions, not the contribution of a record already present. Third, efficiency has an operational interpretation: the allocated mechanism debts must sum to the measured difference between the clean and jointly corrupted references. QuaRM does not claim that the largest Shapley component always yields the best next repair. Full-context marginal gain answers that narrower decision when every current defect is known. Shapley answers the broader accounting question over all repair orders.

### 2.4 The missing estimand

The literature supplies rich taxonomies, detectors, repairs, and record valuations. What is often missing is a fully declared estimand connecting a *family of defects* to a *task loss* under *co-occurrence*. QuaRM's central argument is that a data-quality number without this contract is not merely incomplete; it is ambiguous. The same observed prevalence can induce different loss, and the same observed dataset can be consistent with different latent clean references. Sections 3 and 4 formalize the claim.

## 3. Counterfactual Quality Debt

### 3.1 Setup

Let D* be a trusted reference training sample and P* a trusted evaluation distribution. Let A_omega be a learning algorithm whose stochastic state, including the split and model seed, is omega. Let M = {1,...,m} index defect channels. Channel j is a stochastic operator C_j(theta_j, xi_j) with severity theta_j and channel-specific randomness xi_j. A fixed canonical order defines the composition C_S for coalition S subset M. The loss ell lies in [0,1].

The coalition risk is

[[EQUATION:R(S) = E_{omega,xi,(x,y)~P*} [ ell( A_omega(C_S(D*))(x), y ) ].]]

The risk surface is the collection {R(S): S subset M}. The quality debt of coalition S is Q(S) = R(S) - R(empty). The aggregate debt at the declared severities is Q(M). Negative debt is allowed. It means the specified intervention improved the specified learner's reference risk; it does not make the corrupted records intrinsically correct.

The distinction between *defect existence* and *task harm* is essential. A channel can have high prevalence and negligible debt, or low prevalence and large debt. A channel can also change sign across contexts because regularization, redundancy, imputation, class geometry, and model inductive bias mediate the effect.

### 3.2 Exact Shapley debt

For mechanism j, define

[[EQUATION:phi_j = sum_{S subset M\\{j}} |S|! (m-|S|-1)! / m! * [ R(S union {j}) - R(S) ].]]

Thus phi_j is the average marginal risk of adding channel j over every possible predecessor coalition, with each repair or corruption order weighted uniformly. It is not a singleton effect R({j})-R(empty), nor the full-context repair gain R(M)-R(M\{j}). Those are useful but answer different questions.

**Proposition 1 (exact and unique accounting).** For a fixed risk surface, the vector phi is the unique allocation satisfying efficiency, symmetry, dummy, and additivity. In particular, sum_j phi_j = R(M)-R(empty).

*Proof sketch.* Apply Shapley's 1953 characterization [12] to the cooperative game v(S)=R(S)-R(empty). The constant shift leaves all marginals unchanged. Efficiency gives the displayed identity. The implementation enumerates every coalition and tests the residual numerically; no Monte Carlo Shapley approximation is used.

### 3.3 Interactions

One-defect-at-a-time analysis assumes, usually without saying so, that joint harm is additive. For channels i and j, the discrete second difference in context S is

[[EQUATION:Delta_ij(S) = R(S union {i,j}) - R(S union {i}) - R(S union {j}) + R(S).]]

QuaRM reports the Shapley interaction index, a factorially weighted average of Delta_ij(S) over all background coalitions. Positive interaction denotes super-additive joint risk; negative interaction denotes attenuation. An interaction is evidence about this mechanism composition and learner, not a universal causal relation between semantic dimensions such as completeness and accuracy.

### 3.4 Paired estimation

For repeat r, QuaRM draws one split and one model seed omega_r. Every coalition uses that same omega_r. Each channel obtains a seed derived by SHA-256 from the master seed, repeat, and channel name, so its random choices do not change merely because another channel enters the coalition. The estimate R_hat(S) is the mean bounded loss over repeats.

**Proposition 2 (variance of paired marginals).** For a marginal comparison between S union {j} and S, the paired estimator has variance

[[EQUATION:Var(L_{S+j} - L_S) / n = [ Var(L_{S+j}) + Var(L_S) - 2 Cov(L_{S+j},L_S) ] / n.]]

Whenever the within-repeat covariance is nonnegative, pairing is no worse than independent evaluation and is strictly better when covariance is positive.

This condition matters. Pairing is not guaranteed to shrink every interval; a channel can interact with split difficulty so that covariance is weak or negative. Our experiments report the paired-to-unpaired width ratio rather than assuming a benefit.

### 3.5 Uncertainty and finite-sample control

QuaRM bootstraps whole repeats. Resampling cells or coalition rows independently would destroy the pairing that defines the estimator. For each bootstrap sample, it recomputes coalition means and the exact allocation. The output includes percentile intervals, Pr(phi_j > 0), and the probability that j ranks first. These are uncertainty summaries, not posterior probabilities about a universal truth [13].

**Proposition 3 (simultaneous bounded-loss guarantee).** Suppose repeats are independent and every coalition loss lies in [0,1]. With probability at least 1-delta, every channel satisfies

[[EQUATION:| phi_hat_j - phi_j | <= 2 sqrt( log(2^(m+1)/delta) / (2n) ).]]

*Proof sketch.* Hoeffding's inequality and a union bound over 2^m coalition means imply a uniform error epsilon with probability at least 1-delta. Every Shapley value is a convex combination of marginal differences; if both coalition values are within epsilon, every marginal is within 2 epsilon. The bound is intentionally conservative. Bootstrap intervals are more informative in the reported small study, while the bound documents how repeat count and channel count enter worst-case control.

### 3.6 Identifiability boundary

If D* and P* are trusted, quality debt is a well-defined experimental estimand. If only an observed dataset D is available, latent debt is generally not identifiable.

**Proposition 4 (no-reference non-identifiability).** There exist two reference-mechanism pairs (D*_1,C_1) and (D*_2,C_2) that induce the same observed D but different clean risks R_1(empty) and R_2(empty). Therefore observed D alone cannot determine R(M)-R(empty).

*Construction.* Let D be any sample on which two clean labelings yield different decision boundaries. In world one, D is clean. In world two, D is the output of a label channel applied to a different clean labeling. The observed table is identical, while the clean-reference risks differ. No statistic of D can distinguish the worlds without assumptions or external evidence.

Accordingly, when QuaRM begins from an unverified dataset and adds synthetic defects, the result is *susceptibility*. Calling it measured latent quality debt would be a category error. A trusted adjudicated subset, prior clean snapshot, simulator, or external reference can upgrade the claim.

## 4. Algorithm and Software Contract

### 4.1 Full-factorial protocol

For m channels, each repeat evaluates 2^m coalitions. Within a repeat: (1) split the reference once; (2) keep the test set untouched; (3) apply each coalition to the training partition using subset-invariant channel streams; (4) fit the same learner configuration; and (5) record bounded test loss and realized intervention diagnostics. Across repeats, aggregate risks, compute exact attributions and interactions, and resample repeats for uncertainty.

Exact enumeration is a feature for small mechanism vocabularies and a limitation for large ones. The package rejects more than ten channels by default, making the exponential cost visible. Fractional factorial designs or permutation sampling are possible extensions, but they introduce approximation choices that this first artifact avoids.

### 4.2 Intervention semantics

The bundled channels represent four common training-data defects:

- **Missing cells:** replace exactly round(theta times number of cells) uniformly sampled feature cells with missing values.
- **Feature noise:** select the same number of cells and add Gaussian noise with standard deviation three times the empirical feature scale.
- **Target noise:** replace selected classification labels with a uniformly chosen alternative class; for regression, add Gaussian noise with standard deviation twice the target scale.
- **Duplicate rows:** append exactly round(theta times number of rows) sampled row copies.

Feature-scale estimates are protected against zero variance. Severity zero changes nothing. Every operator copies its inputs. Duplicates execute last because they change row count; this priority is part of the declared mechanism. These are controlled stress channels, not claims that real errors are missing completely at random, Gaussian, symmetric, or uniformly duplicated.

### 4.3 Losses and preprocessing

Classification uses error rate. Regression uses clipped absolute error divided by the test-set interquartile range, with standard deviation and one as fallbacks. Both lie in [0,1]. Median imputation and standardization are fit only on the corrupted training partition. This choice represents a realistic consuming pipeline and ensures the missingness channel tests a system that can execute rather than crash.

### 4.4 Artifact design

The Python API separates corruptions, evaluation, attribution, benchmark registration, serialization, and command-line concerns. The CSV command emits raw observations, coalition means, attributions, interactions, configuration, and an interpretation warning in one JSON record. Fourteen tests cover exact corruption counts, non-mutation, composition priority, subset-invariant random streams, additive and interacting games, Shapley efficiency, deterministic bootstrap, full-factorial completeness, bounded loss, end-to-end serialization, and repeated-run equality.

The experiment writes raw CSV files and a manifest containing package versions, platform, parameters, runtime, design semantics, and SHA-256 hashes. Figures and the paper read those files rather than copied numbers. This structure follows the broader view that data-science evidence includes code, execution environment, and data transformations, not only a final model [14].

## 5. Experimental Design

### 5.1 Questions

We ask four bounded questions.

- **RQ1:** At equal prevalence, does downstream debt differ by defect mechanism and setting?
- **RQ2:** Are joint defect effects materially non-additive?
- **RQ3:** Does a Shapley-first repair recover full-context remediation value, and how does it compare with singleton and uniform choices?
- **RQ4:** Does paired evaluation reduce uncertainty in practice, and where does it fail to do so?

These questions evaluate the estimand and protocol. They do not establish superiority over record-level data valuation, error detection, or repair systems, which solve different problems.

### 5.2 Data and learners

We use three packaged reference datasets and one pinned synthetic generator. Breast Cancer Wisconsin has 569 rows, 30 features, and a binary target. Wine has 178 rows, 13 features, and three classes. Diabetes has 442 rows, 10 features, and a continuous progression target. The synthetic classification set has 900 rows, 18 features, three clusters per class, 28% minority prevalence, moderate separation, and a fixed generator seed.

Each classification dataset is evaluated with logistic regression and a 100-tree random forest. Diabetes uses ridge regression and a 100-tree random forest. Hyperparameters are fixed before evaluation; there is no test-set tuning. The two families deliberately vary inductive bias rather than chase peak benchmark accuracy.

### 5.3 Protocol

The main study fixes every channel at theta=0.15, evaluates all 16 coalitions, and repeats the paired 70/30 train-test experiment 12 times. We use 2,000 paired bootstrap replicates. The dose-response study evaluates theta in {0.05, 0.10, 0.20, 0.30} with eight repeats on synthetic classification and logistic regression. A fixed master seed governs all runs.

The full-context repair oracle chooses argmax_j R(M)-R(M\{j}). The Shapley policy chooses the largest phi_j. The singleton policy chooses the largest R({j})-R(empty). Uniform repair is the mean full-context marginal across channels. Because the oracle uses the exact target decision, it is an upper bound for a one-step policy in this retrospective evaluation.

### 5.4 Evaluation discipline

We report effect sizes and uncertainty rather than a grid of uncorrected null-hypothesis tests. Bootstrap intervals are descriptive across split/model repeats. Dataset-model pairs are not treated as an independent random sample from all applications, so we do not use a significance test across eight settings to claim population-wide superiority. The conservative simultaneous radius from Proposition 3 is included in the machine-readable summary; with 12 repeats it is too wide to be decision-useful, which honestly exposes the cost of distribution-free worst-case guarantees.

## 6. Results

### 6.1 Equal prevalence is not equal harm

[[TABLE:main_results]]

Seven of eight settings have positive aggregate debt. Median debt is 2.75 risk percentage points. The exception, synthetic classification with logistic regression, improves by 1.33 points under all four channels. Inspecting its allocation shows feature noise at -1.76 points with a 95% interval [-2.36,-1.16]. The learner is linear while the generator is not; moderate feature perturbation behaves like regularization for this fixed pipeline. A metric that forces nonnegative quality penalties would conceal this behavior.

At the identical 15% prevalence, target noise is the top mechanism in three settings, feature noise in three, and missingness in two; duplicates are never top. The setting dependence is not a nuisance around a universal ordering. It is the empirical object that a contextual quality measure should reveal.

[[FIGURE:figure2_attributions.png|Figure 2. Exact mechanism-level debt allocation across eight dataset-model settings. A one-point value is one percentage point of bounded clean-test risk. Intervals resample complete paired repeats.]]

The strongest allocations are target noise for Wine-logistic at 4.40 points [2.78, 6.06], feature noise for Diabetes-ridge at 3.63 [2.77, 4.41], and feature noise for synthetic-forest at 3.32 [2.68, 3.99]. In contrast, duplicate-row intervals overlap zero in most settings. These results should not be generalized into a rule that duplicates are harmless: the operator duplicates random rows. Duplicating a subgroup, leakage-prone observations, or mislabeled rows defines a different channel and could be severe.

### 6.2 Interactions are visible and consequential

The largest absolute pairwise interaction is 2.03 risk points for feature and target noise under Wine-logistic. The sign is negative, so their joint effect is attenuated relative to contextual additivity. Breast Cancer-logistic shows -1.42 points for the same pair, while synthetic-logistic shows +0.58. Interaction sign therefore changes with the setting.

[[FIGURE:figure3_interactions.png|Figure 3. Pairwise Shapley interaction indices. Blue negative values indicate attenuation; red positive values indicate super-additive harm. Zeros would support an additive checklist model.]]

These values explain why a sum of isolated defect penalties is not generally an accounting identity. The interactions do not prove physical causality between defect types: they summarize the response of the complete pipeline under the declared composition.

### 6.3 Repair prioritization

QuaRM's top mechanism matches the full-context oracle in seven of eight settings. Its mean realized repair gain is 1.564 risk points, compared with 0.475 for a uniform choice and 1.572 for the oracle. The only miss is synthetic-logistic, where all-channel debt is negative and target repair is marginally preferable while missingness has the largest average Shapley component.

The singleton policy also matches the oracle in seven settings and attains exactly the same mean gain as QuaRM here. This negative result is informative. On this compact benchmark, mechanism ordering is mostly stable enough that one-at-a-time stress tests select the same first repair. QuaRM's demonstrated advantages are exact aggregate accounting, interactions, and uncertainty over all contexts - not a universal selection win. Larger mechanism vocabularies, targeted rather than random corruptions, and sequential repair budgets are necessary tests for a selection claim.

### 6.4 Pairing and uncertainty

The median paired-to-unpaired 95% interval width ratio across settings is 0.85, a 15% reduction. Ratios range from 0.23 for Diabetes-ridge to 1.13 for synthetic-forest. Thus pairing is highly effective where split difficulty moves coalition losses together, neutral in some classification settings, and mildly counterproductive in two synthetic settings. This is consistent with Proposition 2 and contradicts any unconditional variance-reduction claim.

Rank probabilities distinguish clear from ambiguous priorities. Target noise ranks first with probability 0.998 for Wine-logistic; feature noise does so with probability 1.000 for Diabetes-ridge and 0.991 for synthetic-forest. Breast Cancer-forest is less settled: missingness ranks first with probability 0.683. Reporting only point ranks would erase that distinction.

### 6.5 Dose-response and non-monotonicity

[[FIGURE:figure4_dose_response.png|Figure 4. Dose-response allocation for the intentionally misspecified synthetic-logistic setting. Feature noise acts as a model-specific credit over the studied range; target noise changes sign.]]

The dose experiment rejects a linear conversion from prevalence to debt. Feature noise changes from an uncertain -0.46 points at 5% to -2.64 points at 30%. Target noise moves from negative at 10% to +2.10 points at 30%. Missingness becomes clearly harmful only at 30%. These curves describe a fixed learner and reference, not the intrinsic desirability of corruption. Their purpose is precisely to expose thresholding, regularization, and interaction effects that a single prevalence score suppresses.

## 7. Discussion

### 7.1 What QuaRM changes

QuaRM changes the unit of a quality claim. Instead of "this table is 92% high quality," the output says, for example: "relative to this trusted reference, at these channel severities, under this pipeline and clean-test distribution, feature noise accounts for 3.63 bounded-risk points [2.77,4.41]." The second statement is longer because it carries the assumptions required to make it falsifiable.

This contract supports three uses. In a controlled benchmark, it measures the harm of known injected defects. With a trusted earlier snapshot or adjudicated sample, it attributes measured degradation between references. Without either, it stress-tests susceptibility and helps design monitoring priorities. The label attached to the output must follow the evidence regime.

### 7.2 Relation to detection and repair

QuaRM can consume rather than replace detector outputs. Suppose Raha, HoloClean, or domain rules identify candidate error families [2,3]. A calibration layer can define mechanisms that reproduce their location, conditionality, and severity. QuaRM then asks how those families influence downstream risk jointly. Repair systems can be evaluated as interventions that move the dataset across the estimated surface.

The current channels are intentionally simple and inspectable. Production mechanisms should reflect missing-not-at-random processes, systematic labeler confusions, subgroup duplication, temporal staleness, schema drift, referential violations, and leakage. A realistic channel model is part of the scientific claim, not an implementation detail.

### 7.3 Why Shapley, and why not only Shapley

Shapley values are appropriate here because m is small, the full game is observed, and exact accounting matters. Data Banzhaf is compelling when noisy utility estimates and large player sets make robustness central [8]. Full-context marginal gain is the direct answer for one repair when the current coalition is known. Singleton effects are inexpensive stress tests. QuaRM reports enough of the risk surface to compute each, avoiding the mistake of treating one value concept as universally optimal [9,10].

### 7.4 Negative debt is evidence, not embarrassment

Calling an intervention a defect encodes semantic knowledge: a label was wrong, a cell was missing, or a row was copied. Its predictive effect can still be negative. Noise can regularize; duplicates can reweight useful regions; missingness can suppress unstable features. QuaRM retains negative allocations because task utility and semantic correctness are distinct axes. Governance should not preserve a false label merely because one model benefited from it. Rather, the negative value warns that a task-only criterion is insufficient for legal, ethical, or semantic quality.

## 8. Limitations and Threats to Validity

**Reference dependence.** A corrupted, shifted, or unrepresentative test set invalidates the debt interpretation. No method can repair this with algebra alone. Multiple reference populations and subgroup-specific losses should be evaluated when deployment is heterogeneous.

**Mechanism misspecification.** Uniform missingness, Gaussian noise, random label replacement, and random duplication are controlled probes. Real defects are often conditional and adversarial. Conclusions transfer only to the extent that channel models reproduce the relevant failure process.

**Composition order.** Some channels do not commute. QuaRM fixes and records an order, with duplication last. Changing order changes the game and can change allocations. An extension could treat order itself as a random variable or report order sensitivity.

**Exponential cost.** Exact evaluation requires 2^m learner fits per repeat. This is practical for four mechanisms and expensive for ten. Sparse interaction assumptions, fractional factorial designs, and adaptive coalition sampling require separate approximation-error analyses.

**Small empirical scope.** The study contains four small tabular datasets and two learner families. It is sufficient to test implementation invariants and demonstrate contextual reversals, but not to support a field-wide ranking or best-paper-level empirical claim. Image, text, temporal, relational, and large production datasets remain future work.

**Uncertainty scope.** Bootstrap intervals represent variation over declared splits and stochastic states. They do not cover uncertainty about the channel family, severity calibration, reference validity, or deployment shift. The worst-case bound is conservative and does not solve those modeling uncertainties.

**Decision scope.** A Shapley allocation is not automatically an optimal repair policy. Repair cost, feasibility, fairness, compliance, and sequential interactions belong in the decision objective. Our oracle comparison considers one uncosted repair only.

**Task utility versus semantic quality.** A low or negative predictive debt does not certify correctness, fairness, consent, provenance, timeliness, or legal fitness. QuaRM is one consequential lens within a broader governance process.

## 9. Reproducibility and Artifact Audit

The repository exposes one command for tests, one for the complete study, and one for the paper. The checked-in manifest records Python, NumPy, pandas, scikit-learn, and Matplotlib versions; the operating platform; the exact seeds and repeat counts; runtime; loss semantics; pairing policy; and SHA-256 hashes for every result table and figure. Raw coalition observations allow independent recomputation of every allocation.

The artifact passed 14 tests. The main experiment executed 2,048 coalition-level fits: 8 settings times 12 repeats times 16 coalitions, plus 512 dose-response fits. Aggregate Shapley efficiency residual is exactly zero to stored floating-point precision in every setting. The study completes on CPU in roughly 4.5 minutes in the recorded environment. No network access is required after dependency installation, and packaged reference datasets are loaded from scikit-learn without redistributing copies.

Reproduction cannot guarantee identical forest results across every architecture and future library version. The manifest therefore treats versions as evidence, and tolerances rather than file identity should be used across platforms. Within the recorded environment, the unit suite verifies repeated-run equality.

## 10. Conclusion

Data quality is not a property that becomes scientific by attaching a percentage sign. It is a relationship among data, an intended use, a reference world, defect mechanisms, and uncertainty. QuaRM makes that relationship executable. Its risk surface measures joint consequences; exact Shapley allocation reconciles parts with the whole; interactions expose non-additivity; paired repeats reduce avoidable noise when covariance permits; and the identifiability boundary prevents a stress test from masquerading as latent truth.

The study's most important result is not that one defect always dominates. No defect does. Equal prevalence yields different debts across learners and datasets, and a nominal defect can even improve a misspecified predictor. Those are not failures of the metric. They are failures of the idea that a universal data-quality score could have been valid in the first place.

Future work should build realistic conditional channels from detector evidence, extend the design to distribution and temporal defects, optimize cost-aware repair sequences, evaluate subgroup risks, and develop adaptive coalition sampling with honest approximation bounds. The artifact establishes a rigorous foundation and, equally important, records exactly where that foundation ends.

## References

[1] R. Y. Wang and D. M. Strong. Beyond Accuracy: What Data Quality Means to Data Consumers. *Journal of Management Information Systems*, 12(4):5-33, 1996. doi:10.1080/07421222.1996.11518099.

[2] T. Rekatsinas, X. Chu, I. F. Ilyas, and C. Re. HoloClean: Holistic Data Repairs with Probabilistic Inference. *PVLDB*, 10(11):1190-1201, 2017. doi:10.14778/3137628.3137631.

[3] M. Mahdavi, Z. Abedjan, R. C. Fernandez, S. Madden, M. Ouzzani, M. Stonebraker, and N. Tang. Raha: A Configuration-Free Error Detection System. *SIGMOD*, pages 865-882, 2019. doi:10.1145/3299869.3324956.

[4] S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, and A. Grafberger. Automating Large-Scale Data Quality Verification. *PVLDB*, 11(12):1781-1794, 2018. doi:10.14778/3229863.3229867.

[5] A. Ghorbani and J. Zou. Data Shapley: Equitable Valuation of Data for Machine Learning. *ICML*, pages 2242-2251, 2019.

[6] R. Jia, D. Dao, B. Wang, et al. Towards Efficient Data Valuation Based on the Shapley Value. *AISTATS*, pages 1167-1176, 2019.

[7] A. Ghorbani, M. Kim, and J. Zou. A Distributional Framework for Data Valuation. *ICML*, pages 3535-3544, 2020.

[8] J. T. Wang and R. Jia. Data Banzhaf: A Robust Data Valuation Framework for Machine Learning. *AISTATS*, pages 6388-6421, 2023.

[9] K. F. Jiang, W. Liang, J. Zou, and Y. Kwon. OpenDataVal: A Unified Benchmark for Data Valuation. *NeurIPS*, 36, 2023.

[10] J. T. Wang, T. Yang, J. Zou, Y. Kwon, and R. Jia. Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits. *ICML*, pages 52033-52063, 2024.

[11] S. Jager and F. Biessmann. From Data Imputation to Data Cleaning: Automated Cleaning of Tabular Data Improves Downstream Predictive Performance. *AISTATS*, pages 3394-3402, 2024.

[12] L. S. Shapley. A Value for n-Person Games. *Contributions to the Theory of Games*, 2:307-317, 1953.

[13] B. Efron and R. J. Tibshirani. *An Introduction to the Bootstrap*. Chapman and Hall/CRC, 1993.

[14] D. Donoho. 50 Years of Data Science. *Journal of Computational and Graphical Statistics*, 26(4):745-766, 2017.

## Appendix A. Algorithm

**Algorithm 1: Exact paired quality-risk surface**

Input: reference D*, evaluation distribution P*, channels M, severities theta, learner A, repeats n.

1. For repeat r = 1,...,n, draw one train-test split and model seed omega_r.
2. For every coalition S subset M:
   a. derive each channel seed from (master seed, r, channel name);
   b. apply channels in canonical order to the same clean training partition;
   c. fit A with model seed omega_r;
   d. evaluate bounded loss on the untouched test partition.
3. Average repeat losses to obtain R_hat(S).
4. Enumerate exact Shapley debt and pairwise interaction indices.
5. Bootstrap complete repeats, recomputing steps 3-4 for every draw.
6. Return raw observations, risk surface, allocations, interactions, intervals, ranks, and assumptions.

Time is O(n 2^m T_A), where T_A is one learner fit. Stored coalition statistics require O(n 2^m) space; streaming can reduce this when raw audit records are not needed.

## Appendix B. Interpretation Checklist

Before using a QuaRM number, answer:

1. What dataset or simulator is trusted as D*?
2. Why does the test distribution represent the intended population?
3. Which learner, preprocessing, and bounded loss define utility?
4. What real process justifies each defect channel?
5. How was severity calibrated, and is a response curve needed?
6. Does channel order matter?
7. What randomness do repeats represent?
8. Are the intervals narrow enough for the decision?
9. Is Shapley accounting, full-context repair gain, or a cost-aware policy the actual target?
10. Which semantic, fairness, privacy, provenance, and legal dimensions remain outside task loss?

If questions 1-5 cannot be answered, report susceptibility, not quality debt.

## Appendix C. Additional Numerical Results

[[TABLE:attribution_detail]]

All values are proportions of bounded loss. The repository CSV files retain more precision than the rendered table. Probability harmful is the fraction of paired bootstrap draws with positive allocation. Probability rank 1 is the fraction in which the channel has the largest allocation.
