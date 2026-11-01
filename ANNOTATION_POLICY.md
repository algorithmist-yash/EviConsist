# Source-grounded annotation policy v1 — 16 October 2026

Scope: newly generated target responses only. Historical RAGTruth labels are not inherited. Annotation is against the supplied source/question/structured record, not external world knowledge. Freeze this policy before adopting labels; preserve proposals, adjudication and adopted gold separately. No model scores, consistency samples, split outcomes or desired class counts may guide labels.

## Claim decisions

- **Supported:** the source entails the substance of the claim, allowing ordinary paraphrase and modest context-preserving summarization. Lack of identical wording is not an error.
- **Unsupported:** the available source does not adequately warrant a material assertion, including a new attribution, causal relation, frequency quantifier or relationship. Absence of a word alone is insufficient to establish this decision.
- **Contradiction:** a source proposition or explicit structured value conflicts with the same entity/property/relation asserted by the target. Map the proposal label `contradicted` to span kind `contradiction`. Do not turn every missing detail into a contradiction.
- **Uncertain:** evidence, interpretation, attribution or schema semantics do not permit a defensible supported/unsupported/contradiction decision. Preserve alternatives, rationale, and needed adjudication. Do not coerce uncertainty into a binary label.

Structured booleans: an explicit `false` is treated as a source-level negative for the named attribute, not a universal real-world truth. A null/missing value is unknown, not false. Different properties and synonyms require semantic review: `intimate=false` does not automatically settle `cozy`. Review text and structured attributes are both evidence; a genuine conflict between them needs adjudication rather than an automatic hierarchy. An undocumented hours sentinel such as `0:0-0:0` must not silently be decoded as closed, 24-hour opening or a normal daily schedule. Attribute-source reliability remains a limitation.

Preserve quantifier, attribution and agent/beneficiary scope. An author's reaction is not automatically a report of fans' reaction; one reviewer intending to return is not evidence of customer-wide habitual return. A recommendation to visit another venue does not itself measure proximity. A statement that a venue benefits an organization does not establish that charities benefit the venue. If competing reasonable readings remain, mark uncertain even when other claims make the response positive.

## Refusal and response quality

`incorrect_refusal` is a separate quality status. Source 12253 refuses although the passage explicitly identifies Suzanne “Crazy Eyes” Warren as having serious mental illness. Do not create a fictitious factual hallucination span or silently mark it gold negative. For the planned primary **adequate-response factuality** cohort, quality-exclude this target and report the exclusion; retain the full record and an independent refusal-quality endpoint. This is an explicit new-target policy, not a revision of historical RAGTruth labels. Appropriate refusal for an incomplete/unanswerable question can be an adequate negative, as in source 12311.

Other quality values: adequate, truncated, ambiguous, incorrect_refusal. Review source completeness and target identity. Never regenerate or replace individual targets to achieve desired labels or better detector results. If quality exclusions affect source allocation or class coverage, document that before fitting and never selectively move targets between folds.

## Response decision and gold adoption

1. Review the entire target explicitly. An empty positive-span list with unreviewed status is **unlabelled**, not negative.
2. For adequate targets, one accepted unsupported/contradiction span supports an assistant-proposed positive response decision. Preserve unresolved additional claims and flag localization incompleteness. No accepted positive plus any unresolved material claim means response decision uncertain. A fully reviewed adequate response with no positive or unresolved material claim supports a negative decision.
3. An assistant may produce semantic review aids and proposed decisions with `reviewer_type=assistant`. Such records are not independent human annotations or validated gold.
4. Human reviewers confirm/reject each claim using source citations, exact zero-based half-open offsets, text, rationale and confidence. Save independent initial reviews before adjudication. Record disagreements and agreement **before** consensus; assistant-versus-assistant agreement is not human inter-annotator agreement.
5. Only explicitly adopted human decisions populate a separate final annotation set. Record real reviewer identities, timestamps, provenance, adjudicator, policy version/hash, source/target hashes, quality decision and exclusions. Gold is 0/1 only for eligible resolved targets; unresolved and excluded cases retain null. Do not use the 12-target assistant review as training gold.
6. Validate exact target identity, offsets, kind mapping, coverage, complete-review flags and provenance. Keep the original `annotations.json` unchanged as an auditable unlabelled snapshot. No training or scaling execution until annotation stability is established; cohort planning can proceed independently.

The existing historical `consistency_pilot.py fit` gate is a prototype and accepts a named single reviewer; it does not prove a reviewer is human. Do not bypass this policy by copying assistant proposals into that file. A future training entry point must enforce the adopted provenance/eligibility manifest and use the larger cohort, not the 12-target technical pilot.

## Evaluation safeguards

Keep all target/comparison responses for each source together. Exact/near-duplicate source groups must not cross partitions. Fit all three classifiers on the same adopted adequate targets; validation selects thresholds and any calibration without access to confirmatory test. Freeze feature and analysis specifications before that test. The inspected 450 RAGTruth test sources are no longer untouched confirmation for the combined method.
