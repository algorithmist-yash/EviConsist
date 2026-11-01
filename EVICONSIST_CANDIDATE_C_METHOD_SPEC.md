# Candidate C: conservative multinomial confidence-set projection

Version C-v1, 18 October 2026. Prospective mathematical/implementation specification, not an amendment approving SAP v1 or an experiment. Candidates A/B, historical reports, code, thresholds and manuscript remain unchanged. Labels are assistant-generated operational silver, not independently validated human gold.

## Population, estimands and assumptions

One eligible target from each independently grouped source, canonical task assigned before labels/predictions; independent, identically distributed draws within each of Summary, QA and Data2txt, conditional on the eligible population. Counts n_t are fixed or conditioned on. Repeated targets/related sources are not independent observations. Merely renaming source IDs does not establish this assumption. Fixed trained models and validation-selected thresholds must be independent of the evaluation sources. This interval conditions on training/threshold selection and does not cover annotation validity, population shift or residual source dependence.

Order the eight cells by j=4Y+2E+C. Let p_tj denote each task's joint probability, with sum_j p_tj=1. Always use w_t=1/3, irrespective of task sample sizes. Define

- T_C=sum_t,j w_t Y C p_tj; D_C=sum_t,j w_t C p_tj.
- T_E=sum_t,j w_t Y E p_tj; D_E=sum_t,j w_t E p_tj.
- D_Y=sum_t,j w_t Y p_tj; N_R=sum_t,j w_t Y(C−E)p_tj.
- ΔP=T_C/D_C−T_E/D_E; ΔR=N_R/D_Y.

The immutable decision is lower(ΔP)>0 AND lower(ΔR)>−0.02. +0.03 precision is a planning alternative only. The same three task weights and eligible resolved-target population are retained. No pseudocounts, variance floors, artificial disagreements, outcome deletion or task reweighting.

## Confidence allocation fixed before results

Allocate total region failure α_R=1/40=0.025 equally across all 24 cells and both tails: a=1/1920 per tail. Each marginal interval has coverage at least 1−1/960. For count k among n, the Clopper–Pearson limits are L=0 for k=0, otherwise Beta^{-1}(a;k,n−k+1); U=1 for k=n, otherwise Beta^{-1}(1−a;k+1,n−k). Intersect the 24 bounds with the three simplexes.

Each cell count has a binomial marginal even though counts within its task are dependent. A union bound, requiring no independence between the 24 intervals, gives P(all true cell probabilities in the region)≥0.975 under the stated sampling model. Deterministic projection of that event gives simultaneous coverage of both contrasts at least 0.975. Thus either projected lower bound exceeds its true value with probability at most 0.025, and the joint rule has type-I error at most 0.025 under the union null. This is deliberately stronger coverage than nominal two-sided 95%; report the limits as conservative two-sided 95% limits obtained from a 97.5% region, not equal-tailed or exact-95%-coverage contrast intervals.

This is a coverage argument, not a claim that simulations prove coverage. The confidence region remains random: a true parameter need not lie inside every region drawn in a simulation. Deterministic inclusion tests use specified fixtures whose parameter is known to be inside the constructed region; random noncoverage is retained and measured.

## Numerical enclosure and certified optimization

Beta quantiles are starting guesses only. Round endpoints outward to a fixed rational grid of 2^-40, then verify their defining binomial-tail inequalities using finite nonnegative sums with directed Decimal rounding at 60 digits. An uncertified endpoint stops the calculation; do not silently use an approximate endpoint. Outward enlargement preserves the coverage argument. Verify that the empirical vector is feasible. An absent task, infeasible region or invalid input is reported explicitly.

Required denominators must be strictly positive everywhere in the region. Compute their exact rational extrema over each bounded simplex by subset-sum formulas, then task-weight them. If D_C or D_E can reach zero, precision is non-estimable; if D_Y can reach zero, recall is non-estimable. Either prevents joint success. No zero-denominator substitutions.

Recall projection is linear-fractional programming with the Charnes–Cooper substitution z=p/D_Y, t=1/D_Y: sum_j z_tj=t, L_tj t≤z_tj≤U_tj t, D_Y(z)=1. Bound z and t using certified denominator extrema.

Precision needs global optimization. Partition r=D_C/D_E, whose positive finite global bounds follow from denominator extrema. In a node [a,b], substitute z=p/D_C, t=1/D_C; then D_C(z)=1 and a D_E(z)≤1≤b D_E(z). Since T_E(z)≥0,

T_C(z)−b T_E(z) ≤ ΔP ≤ T_C(z)−a T_E(z).

Solve the two bounding linear programs and bisect the ratio interval. These bounds enclose the objective on the entire node; their union covers the entire original region. Feasible probability-vector witnesses bound the attainable extrema from the opposite side. Stop only when both extremum gaps are ≤0.002, or a fixed 512-node / 60-second precision-projection cap is reached. Return all valid outer bounds with their gaps; a capped/uncertified projection is not a passed optimization check or an adopted method. This tolerance is absolute probability difference and is numerical enclosure error, not statistical coverage slack. Decision endpoints use outer bounds, never witness values.

LP solver status alone is not a certificate. For a minimization with finite box l≤x≤u, Ax≤b, Hx=d, take solver-proposed duals y≤0 and free v. Exact rational weak duality yields

lower = b'y+d'v + sum_i min((c−A'y−H'v)_i l_i, (c−A'y−H'v)_i u_i).

Any y≤0 and v gives a valid lower bound, even with dual residuals. Evaluate this expression with Fraction arithmetic; negate for upper bounds. Store duals, model hashes and exact rational lower bounds for replay. LP witnesses are normalized and mixed toward a feasible empirical anchor until all rational simplex/box constraints hold. A floating infeasibility status cannot discard a node: require a positive certified phase-I slack lower bound, otherwise stop as uncertified. Final floats are rounded outward from exact rational endpoints. Failures remain explicit.

## Frozen small checks and feasibility work

See EVICONSIST_CANDIDATE_C_LOCAL_CHECK_PROTOCOL_v1.json, frozen before code execution. Only deterministic fixtures, small exact enumerations and timer checks are authorized in this phase. No random screening, response generation, detector scoring, model fitting, confirmation sampling, Mac work or 40,000-cell validation.

Hand checks: binomial n=1 endpoints; a conditional 2×2 table embedded in eight cells; point-simplex projections; direct task-weighted ratios with unequal task n. Edge checks: zero disagreements, no flags, no positives, absent task, k=0/n, incompatible boxes, degenerate simplexes, duplicated source units, and invalid outcomes. Nonempty valid regions must not shrink to zero solely because a disagreement cell was unobserved. Test LP certificates against an independently enumerated small polytope and verify witness feasibility/projection inclusion.

Feasibility profiles are deterministic count tables at n=40/task (the proposed 120-source planning-assessment partition), 200/task (all 600 proposed development groups, not an evaluation population to pool across partitions), and 2,000/task (illustrative larger population). Also check zero disagreements and sparse outcomes at n=40/task. Record actual widths, gaps, estimability, runtime, seeds (none for deterministic cases), versions and hashes. Do not treat these designs as selected cohorts or power estimates.

## Prospective later-screening proposal; not authorized here

A separately frozen candidate-specific protocol will be required after review. Proposed fixed design: 8 independent-one-target-per-group populations × 200 outer draws, seed 20261018, equal and heterogeneous task prevalences, recall/precision boundary and joint corner, sparse flags/positives and zero-disagreement populations; include sample sizes before execution. Retain every failure in denominators. Correlated multi-target scenarios are assumption-violation diagnostics, not valid multinomial coverage tests.

Report all cell-region/contrast noncoverage, false-success, estimability, widths, certification gaps and runtime with exact binomial Monte Carlo intervals. At 200 draws and a 0.025 rate, Monte Carlo standard error is approximately 0.011; this cannot certify tight calibration. No requirement that a finite-simulation upper limit lie below 0.025 is used as proof. A one-sided exact binomial test for excess over 0.025 at familywise 0.01 across the 8 prespecified populations is a debugging stop, not mathematical validation. Reject/revise on assumption mismatch, failed proof, any false numerical certificate or reproducible optimization failure. All revisions precede new results; no large simulation follows automatically.

## Decision conventions

Ready for prespecified screening only if all mathematical/implementation checks pass, all ordinary feasibility fixtures obtain certified extrema within tolerance/caps, and the limitations are explicit. Non-estimable edge fixtures must correctly prevent success. Excessively wide intervals are quantified without narrowing the method or changing the immutable research question; practical feasibility and full power remain separate decisions. Any failed ordinary fixture or uncertified numerical result means requires revision; a false coverage argument or irreparable design mismatch means rejected.

## Primary references

[Clopper & Pearson (1934), Biometrika 26, 404–413](https://doi.org/10.1093/biomet/26.4.404): binomial confidence limits, not this project's entire projection procedure.

[Charnes & Cooper (1962), Programming with linear fractional functionals](https://doi.org/10.1002/nav.3800090303): linear-fractional transformation.

[Leisenring, Alonzo & Pepe (2000)](https://pubmed.ncbi.nlm.nih.gov/10877288/): paired predictive-value comparisons; not automatically applicable to this task-standardized estimand.

The coverage proof and global-bound construction above are project-specific derivations requiring review; neither citation is presented as external approval.
