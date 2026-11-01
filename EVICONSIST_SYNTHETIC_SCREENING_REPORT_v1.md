# EviConsist synthetic screening report v1

**16 October 2026. Status: screening complete and independently audited. Both statistical candidates fail screening; neither is adopted or eligible for the frozen full validation. Detector experiments remain blocked.**

## Study integrity and scope

The returned Mac archive contains **48,000 unique synthetic studies**, exactly **2,000 in each of 24 scenarios**, with all three method attempts paired on the same saved outer inputs. The numerical code, scenario table, seed rules, original lock and transfer manifest match the frozen transfer. ZIP CRC, all 47 internal return hashes, SQLite integrity, input content hashes and per-task diagnostics pass. No uploaded source code was executed during the local audit.

Archive SHA-256: `b85aae30c304a364a2af9795695fb858d78d39c86606eac59fbff1d9ef6cc742`. No external matching `.sha256` sidecar was supplied. This is a locally computed digest; verification additionally used the original trusted transfer and all internal hashes. The archive and extracted return are preserved under `results/synthetic_screening_mac_return_audit_v1_2026_10_16`.

The local audit recalculated **all 48,000 Candidate B intervals**, independently rebuilt all 72 scenario/method summary rows and regenerated **96** outcome-independent sampled outer datasets. A/reference were replayed at one hash-selected index in every scenario with the prescribed 9,999-resample setting and original failure guards (undefined originals stop before resampling), matching within absolute/relative tolerance 1e-12. A first exhaustive seed audit was stopped after its completed first scenario because of cost; no full-regeneration claim is made. All saved input hashes/diagnostics were checked. Exact replay indices and audit scope are recorded. These checks are not a second screening or a change to its criteria.

The frozen design remains 300 true source groups, equal task-mixture ratios, one worker, Candidate A studentized source bootstrap, Candidate B cluster influence-function t interval and the failed percentile reference. A/reference use 9,999 resamples with chunk ≤64 when their original quantities are defined; original undefined cases return non-estimable under the frozen rules. All 48,000 outer studies remain in every rejection denominator. No failed study was discarded or replaced.

## Screening decision

| Candidate | Estimable intervals / 48,000 | Undefined intervals | Rejected cells / 24 | Decision |
|---|---:|---:|---:|---|
| A | 20,369 | 27,631 | 19 | Rejected; no full validation |
| B | 40,783 | 7,217 | 13 | Rejected; no full validation |

Candidate A has **20,414** studies with at least one undefined required studentized replicate, plus **7,217** undefined original ratio/variance/df cases. B shares those **7,217** original non-estimability cases. These are retained outcomes, not zeros substituted for precision or missing studies. Low apparent false-positive rates in cells with poor estimability are not evidence of a useful procedure.

Two prespecified cells show error-control failures independently of the usability failures:

| Scenario and candidate | False-positive joint claims | Rate | Wilson 95% interval | Frozen stop rule |
|---|---:|---:|---:|---|
| precision_boundary_H6, A | 85/2,000 | 4.25% | 3.45–5.23% | Lower bound >2.5%; fails |
| recall_boundary_H6, B | 158/2,000 | 7.90% | 6.80–9.16% | Lower bound >2.5%; fails |
| precision_boundary_H6, B | 71/2,000 | 3.55% | 2.82–4.45% | Lower bound >2.5%; fails |

These are **false-positive statistical success claims under synthetic nulls**, not detector false alarms. In the joint-null H6 cell, B has only 0.95% joint successes but **7.05% recall-component rejections**; checking only the corner joint rate would hide a component failure. All relevant null components were checked under the frozen screening rules.

Sparse observed disagreements and the strict any-undefined-resample rule make A frequently unusable. B prevents the original zero-SE collapse by declining some inferences, but that also violates support requirements; its H6 false-positive inflation remains even with high estimability. This supports rejecting the fixed candidates, not guessing that a different SE correction or larger cohort would repair them. Detailed causal attribution of every new failure is not established by this audit.

## All prespecified cells

Availability is the fraction with a defined two-endpoint interval; joint rates retain undefined studies as non-success. Cell-level absence of a detected stop is not proof of calibration. A method must satisfy every required cell.

| Scenario | A availability | A joint rate | A cell | B availability | B joint rate | B cell |
|---|---:|---:|---|---:|---:|---|
| recall_boundary_H1 | 0.00% | 0.00% | Fail | 92.50% | 0.00% | Fail |
| recall_boundary_H2 | 88.25% | 1.15% | Fail | 100.00% | 1.30% | No stop detected |
| recall_boundary_H3 | 0.00% | 0.00% | Fail | 78.00% | 0.00% | Fail |
| recall_boundary_H4 | 0.00% | 0.00% | Fail | 89.80% | 0.00% | Fail |
| recall_boundary_H5 | 0.00% | 0.00% | Fail | 4.80% | 0.00% | Fail |
| recall_boundary_H6 | 13.50% | 0.00% | Fail | 99.80% | 7.90% | Fail |
| recall_boundary_H7 | 99.80% | 1.55% | Fail | 100.00% | 1.05% | No stop detected |
| recall_boundary_H8 | 100.00% | 1.60% | No stop detected | 100.00% | 1.40% | No stop detected |
| precision_boundary_H1 | 8.65% | 2.10% | Fail | 99.65% | 2.60% | No stop detected |
| precision_boundary_H2 | 97.95% | 2.40% | No stop detected | 100.00% | 2.20% | No stop detected |
| precision_boundary_H3 | 0.50% | 0.25% | Fail | 97.70% | 2.90% | No stop detected |
| precision_boundary_H4 | 9.90% | 2.10% | Fail | 100.00% | 2.75% | No stop detected |
| precision_boundary_H5 | 0.00% | 0.00% | Fail | 10.50% | 0.00% | Fail |
| precision_boundary_H6 | 96.05% | 4.25% | Fail | 100.00% | 3.55% | Fail |
| precision_boundary_H7 | 100.00% | 2.80% | No stop detected | 100.00% | 2.55% | No stop detected |
| precision_boundary_H8 | 100.00% | 2.85% | No stop detected | 100.00% | 2.75% | No stop detected |
| joint_null_corner_H1 | 0.00% | 0.00% | Fail | 91.50% | 0.00% | Fail |
| joint_null_corner_H2 | 88.80% | 0.35% | Fail | 100.00% | 0.45% | Fail |
| joint_null_corner_H3 | 0.00% | 0.00% | Fail | 77.90% | 0.00% | Fail |
| joint_null_corner_H4 | 0.00% | 0.00% | Fail | 91.40% | 0.00% | Fail |
| joint_null_corner_H5 | 0.00% | 0.00% | Fail | 5.70% | 0.00% | Fail |
| joint_null_corner_H6 | 15.30% | 0.00% | Fail | 99.90% | 0.95% | Fail |
| joint_null_corner_H7 | 99.75% | 0.45% | Fail | 100.00% | 0.45% | No stop detected |
| joint_null_corner_H8 | 100.00% | 0.30% | No stop detected | 100.00% | 0.30% | No stop detected |

## Resource and provenance records

The return records **5.05 hours** screening wall time, peak process memory **106.48 MiB**, and an **8-hour / 512-MiB** budget. The saved runtime reports macOS-27.0.1-arm64-arm-64bit, Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1, with numerical thread environment variables set to one. These are this run's observed/reported resources, not a controlled Windows-versus-Mac speed comparison. No GPU/model inference was involved.

Machine-readable audited evidence includes `RECOMPUTED_SCREENING_CELLS.json` (joint/marginal rates and intervals, coverage, conditional widths/SE/df, estimability, undefined reasons, disagreement/support distributions, runtime/memory), `DETERMINISTIC_INTERVAL_REPLAYS.json`, `WINDOWS_RETURN_AUDIT.json` and `ARCHIVE_AUDIT.json`. The returned SQLite checkpoint, original summary, logs, budget, runtime and locks remain unchanged. Width and estimable-case coverage summaries are explicitly conditional; unconditional summaries do not omit failures. No non-vacuity alternative study was run because both methods already fail mandatory null-cell screening; no approval can follow from that omission.

## Next decision

**Do not run the 40,000-study-per-cell validation for these failed candidates.** Preserve the results and develop/review a separately versioned statistical-method proposal before changing an interval, source count, support rule or acceptance criterion. The present revision and original SAP remain unchanged. No candidate is selected by favorable precision, power or partial results.

Development generation, detector scoring, E/S/C fitting, final sample-size calculation and confirmation sampling remain blocked. Source overlap/exposure, upstream provenance/licensing and the later experimental resource gates also remain open. The manuscript has not been changed to claim effectiveness. This screening tests statistical machinery only; future detector metrics concern assistant-generated operational silver labels, not independently validated human factuality.
