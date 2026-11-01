# Candidate C engineering correction v3 — 21 October 2026

The user requests fixes after two recorded stops. This revision preserves the statistical region, thresholds, seeds, .002 precision gap, 512-node/60-second precision budget, 120-second worker ceiling and 512 MiB worker peak-RSS ceiling. No new random datasets are used to select this correction.

## Mathematical correction

The existing exact E/C permutation invariance guard is moved inside branch selection rather than applied only after a two-sided search exhausted its budget. For an invariant region and antisymmetric contrast, the infimum equals minus the supremum. Every feasible witness has a feasible reflected witness with exactly negated contrast. A globally certified supremum enclosure therefore certifies both ends; search may prioritize only the upper unresolved gap. The partition still covers the entire positive denominator-ratio domain and children are removed only under the existing certificates. Positive symmetric ratio intervals use an interior geometric split represented by an exact rational; floating computation proposes a split only. The exact interior guard and all rational LP certificates remain mandatory. Asymmetric regions retain the previous arithmetic split and two-sided priority. Neither tolerance nor node/time limits increases.

Saved Colab recall_boundary_H5 index 11 is used as a deterministic regression, never as a new Monte Carlo observation. New precision gap 0.0019383188140931402 at 47 nodes; original failure remains historical. A witness-only polishing attempt did not pass and is preserved separately; it is not part of the production fix.

## Runtime correction

The Windows virtual-environment executable can be a launcher whose PID is not the actual numerical interpreter. Launch sys._base_executable directly, with isolated/no-site startup and explicit paths to the same pinned environment's packages. The monitored PID is then the actual single-process numerical worker. No sitecustomize or PYTHONPATH hooks are loaded. Tests import the pinned NumPy/SciPy inside that process and verify nontrivial measured RSS. Persist separate memory_limit, wall_time_limit and child_exit reasons, observed elapsed/RSS and worker PID. Kill and wait for that directly launched worker on interruption or limits. No other worker processes are created by this adapter.

## Checkpoint transition

Use a new v3 checkpoint directory. Preserve v2 files and STOP markers unchanged in their original locations and backups. Accept only the explicitly listed v2 parent package lock for previously certified records, keeping those records byte-identical. Save original stopped case records and stop/state files under historical_v2 in the v3 checkpoint. Derive a separately identified v3 replacement for a saved failed case from its saved counts, with parent record hash and historical failure metadata; retain the original failure and all denominator accounting. If the correction still fails, stop. A resource-killed case without a committed result may resume the identical original seed under corrected runtime supervision after migration, retaining the original resource failure log. This is a documented engineering transition, not an untouched original-version confirmation run.

Carry forward cumulative resource expenditure. Run no extra seeds or additional Monte Carlo indices. No duplicate or failed record is silently dropped. Mixed numerical implementations must be disclosed in the final report; original certification failures and post-revision results are separate diagnostics. All method approval and practical-usefulness gates remain open.
