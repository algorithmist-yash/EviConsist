# Current reporting supplement — 16 October 2026

The original evidence-only and frozen verifier comparison are completed component studies. The technical consistency pilot is complete only for generation/scoring; structural offsets and assistant semantics are separate checks. New human gold, human agreement, larger-cohort models and combined-method confirmation are not complete. Report assistant taxonomy and semantic decisions as provisional, and preserve uncertainty/quality exclusions. Do not reuse inspected test sources for central method selection or confuse forward time with wall time. ANNOTATION_POLICY.md, CURRENT_RESEARCH_STATUS.md and the cohort plan supersede historical pending-status statements below. Prespecified historical measurement definitions/results stay unchanged.

# Status update — 15 October 2026

Completed results, calibration/risk plots and intervals are available in results/completed_baseline_2026_10_14. The 36-case focused review is in results/validation_diagnosis_2026_10_14. No detector change has been made.

# Frozen baseline reporting protocol

The completed canonical experiment is results/completed_baseline_2026_10_14. results/full is the preserved Windows handoff, not the current completed run. Preserve the original inference code, locks and configuration as the frozen baseline; create separately versioned code and outputs for any new detector. Completed source snapshots and hashes are retained in the canonical directory. All 17,617 selected responses have been scored; measured full-data performance is now reported.

The separate baseline_supplement.py reads completed predictions without refitting models or changing thresholds. Reliability/risk–coverage plots, prevalence comparisons and the reproducible validation sample are already available. The original packet retains blank classification fields for provenance; the 36-case focused review is separately recorded in results/validation_diagnosis_2026_10_14/reviewed_validation_cases.jsonl. Independent adjudication remains outstanding. The command below is for deliberate regeneration only, not a required next step:

```powershell
.\.venv\Scripts\python.exe baseline_supplement.py --directory results/completed_baseline_2026_10_14
```

The reporting supplement has completed. No waiting process or inference is active in this Windows workspace. Do not resume the historical Windows checkpoint for this completed experiment. Future experiments require separate output directories.

## Review categories

- Retrieval failure: relevant support or contradiction exists in supplied context but is absent from selected windows.
- Insufficient context: supplied context does not settle the claim; do not equate this with factual falsity.
- False NLI contradiction: selected evidence supports or is compatible with the claim, but NLI predicts contradiction.
- Segmentation error: splitting loses needed attribution, qualifiers or referents.
- Compound claim: a sentence merges supported and unsupported facts.
- Annotation ambiguity: the label policy or annotator interpretation is unclear.
- Other or no apparent error: explain in notes, including for correct decisions.

Use the complete supplied context, raw annotations including implicit_true, and per-window NLI probabilities. Review validation for future development; held-out test examples in the experiment report are descriptive only. Do not tune the frozen experiment after reviewing test errors. A later baseline version needs a new directory and a documented change.

## Gates for the controlled experiment

After reviewing the complete evidence report, plan a 100-prompt same-model feasibility study. Generate five answers per prompt with one fixed model revision, prompt, context and decoding settings; save independent seeds. Designate target answers in advance and annotate them separately. Do not transfer RAGTruth labels. Group all related samples by source across splits.

Compare evidence-only, consistency-only, combined logistic regression, combined without contradiction and calibrated combined on the same labelled controlled test set. Fit models on training, choose thresholds on validation, and use separate calibration data or a documented cross-fitting protocol when calibration and threshold selection share limited validation data. Use paired source-grouped bootstrap comparisons. Do not assume fusion improves performance.

The workspace manuscript paper/main.tex reports measured full-baseline performance and exploratory validation findings while retaining proposal status for the controlled consistency/fusion study. The verified PROBE/TRIVIA+ references remain. No combined-system, calibration-benefit or adaptive-efficiency result is claimed.
