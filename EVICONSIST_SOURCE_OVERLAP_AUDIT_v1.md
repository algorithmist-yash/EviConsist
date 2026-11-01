# EviConsist source-overlap case audit v1

As of 18 October 2026. The four flagged cases are resolved at the available-record level. This is a retrospective limitation audit; original splits, labels, predictions, thresholds, models and metrics remain unchanged. The fresh confirmation frame and exhaustive underlying-article independence gate are not cleared.

## Verified evidence and affected evaluation units

| Case | Exact relationship | Eligible historical units | Consequence |
|---|---|---|---|
| QA roast | 15112 test passage 1 = 15419 train passage 3; 90 normalized words; whole contexts/questions differ | Test 13158–13163 (6); train 14988,14989,14990,14992,14993 (5) | Evidence crosses train/test despite distinct source IDs |
| QA ribs | 15203 train passage 2 = 15317 test passage 1; 86 normalized words; whole contexts/questions differ | Train 13698–13703 (6); test 14376–14381 (6) | Evidence crosses train/test despite distinct source IDs |
| Summary duplicate | 12084 train and 12085 validation have byte-identical source and prompt | Train 3702–3707 (6); validation 3708–3713 (6) | Validation contains the same article/prompt as training |
| Business/pilot identity | 14041 and original-pilot source 14218 share Broad Street Oyster Company, 418 State St, Santa Barbara, CA | Both historical train; 7038–7043 and 8046–8051 (12) | Same-business exposure must propagate to related source records |

There are 48 response records and 47 good-quality eligible records across the eight sources. Response 14991 is not good quality and was excluded by the original rule; it is retained in the audit inventory. All 47 eligible responses have saved original predictions. The 12 affected test responses occur in both models' frozen paired predictions with matching IDs and operational labels.

The two shared QA passages are byte-identical after trimming their numbered blocks, not merely semantically similar. Their normalized fingerprints are recorded in CASE_EVIDENCE.json, with passage numbers and character block boundaries in the authoritative source strings. Normalization is Unicode NFKC, case-folding and whitespace collapse; it was used only for matching, never to change source evidence or labels.

The saved original retrieval windows include at least 20 consecutive normalized words from the shared passage for all 23 eligible QA responses in these groups. The alternative's test cache confirms the same diagnostic for all 12 affected test responses. The row-level file records each matching claim/window index, matching-word length and window hash. This traces retrieval exposure, not whether a claim is supported, whether its NLI judgment is correct, or whether overlap caused a particular prediction.

## Impact on the historical conclusions

The QA matches involve 2 of 450 test source IDs (0.44%) and 12 of 2,675 eligible test responses (0.45%). Thus source-ID disjointness did not ensure passage independence. Both models are affected on the same units; a paired comparison does not erase this dependence. Its direction or size cannot be inferred from overlap counts. The pretrained NLI models were not fine-tuned on these RAGTruth passages; the downstream classifier was fitted on numerical verification features. These observations do not establish text memorization or invalidate every historical result.

The Summary duplicate contributes 6 of 2,878 validation responses (0.21%). In addition, train response 3707 and validation response 3713 are byte-identical outputs from llama-2-70b-chat, both with no positive operational span. This is response-level duplication as well as article duplication. Validation performance and threshold selection therefore used some dependent material. The limitation applies to the original validation-based operating-point decisions, including the original 0.29 threshold and matched-recall comparisons; it does not quantify how the selected thresholds would differ after deduplication.

No exclusions, refits, threshold searches or revised test metrics were performed. Any future deduplicated sensitivity analysis must be separately versioned and identified as post hoc. The published frozen results remain a historical comparison on their recorded fold, with this dependence limitation attached; they are not fresh confirmation of the combined method.

The business case is a strong operational identity match but lacks an upstream business ID. Record 14041 has reviews dated 17–18 January 2022 and explicit hours; 14218 has reviews dated 19–29 November 2020, null hours and explicit ambience attributes. Categories and reviews differ. These are related snapshots, not interchangeable evidence. Do not transfer opening hours, ambience values, labels or span judgments between them. Neither is a held-out historical test record, so this pair does not itself establish historical test leakage. Exposure through the 12-source pilot excludes both from future unexposed cohorts.

## Future grouping decision

Before any future split, combine exact whole-source matches, exact shared QA passages, conservative name/address business identities, and prior near-duplicate edges into transitive connected components. Keep every target and its comparison samples inside the component's single partition. If a component touches a previously inspected test or annotation pilot, exclude the entire component from an unexposed cohort. Development use must follow its separately approved exposure rules.

FUTURE_RELATED_SOURCE_GROUPS.json contains the deterministic component inventory, edge reasons and test/pilot exposure flags; it selects no cohort. In addition to the four reviewed cases, it retains same-fold exact-passage groups 14426/15181/15272 and 14325/15545, and earlier token-set near-duplicate edges. The first touches the 120-source pilot, so 14426 inherits that exposure. Existing availability counts must not be treated as newly approved sample sizes.

## Scope and remaining gate

Case-level review is complete for the four requested flags. Existing exact/passage/proxy and token-set screens are finite diagnostics, not a corpus-wide semantic containment audit. Underlying MARCO passage URLs, CNN/DM article IDs, recent-news publisher/URLs and Yelp business IDs are absent from these local records. A new confirmation pool still needs provider IDs, passage/chunk containment checks, article/entity matching and an exposure register before sampling. Missing identity evidence is unresolved, not proof of independence.

Evidence: results/source_overlap_provenance_audit_v1_2026_10_18/CASE_EVIDENCE.json, AFFECTED_EVALUATION_UNITS.jsonl, FUTURE_RELATED_SOURCE_GROUPS.json, FROZEN_INPUT_HASHES.json and AUDIT_CHECKS.json. The audit streams saved predictions and reads the alternative cache in SQLite read-only mode; no model is loaded. Input hashes match before and after the audit and at publication.
