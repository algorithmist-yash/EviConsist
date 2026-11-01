# EviConsist upstream provenance and licensing review v1

Reviewed 18 October 2026. The available-evidence review is complete; the permission/provenance gate remains unresolved. No provider was contacted, terms accepted, data removed or material published. This records verified documents and missing applicability evidence, not an inference that existing work is unlawful or automatically authorized.

## Acquisition lineage and source coverage

The local ParticleMedia/RAGTruth checkout is commit c103204b9ce28d6bbad859304bf30de72b8ed8fe. Source/response bytes and local license are hashed in the audit. Its local MIT notice covers software and associated documentation; do not use that alone to infer permission for all underlying third-party passages. The [official repository](https://github.com/ParticleMedia/RAGTruth) is the immediate distribution source.

The [RAGTruth paper](https://aclanthology.org/2024.acl-long.585.pdf) describes QA from MS MARCO, business data from Yelp and summarization material from CNN/DailyMail plus a recent-news platform. It does not provide the missing per-record rights/identity crosswalk needed here. Local source counts are 628 CNN/DM, 315 Recent News, 989 MARCO and 1,033 Yelp, totaling 2,965. SOURCE_PROVENANCE_REGISTER.jsonl covers every local source without copying its passage/review text.

| Collection | Verified external evidence | Applicability gap and action before approving new collection/redistribution |
|---|---|---|
| CNN/DM | The [dataset card](https://huggingface.co/datasets/abisee/cnn_dailymail/blob/e2e3760cc40b92105a93592dcaeb92b512637aad/README.md) distinguishes versions and states Apache-2.0 for v1.0.0; [original collection code](https://github.com/google-deepmind/rc-data) obtains publisher articles | RAGTruth's precise release/article IDs are not mapped locally. Establish release and article URL/ID crosswalk and underlying-text use basis; code licensing alone does not settle publisher-text rights |
| Recent News | The paper identifies a news-platform collection; local records retain text but no publisher/URL provenance fields | Recover publisher, URL, date and applicable permission/use basis for each article. No blanket license is established |
| MARCO | [Official terms](https://microsoft.github.io/msmarco/) limit the intended use to noncommercial research and expressly do not extend underlying intellectual-property rights | Identify the source release and passage URLs/IDs, applicable use record and basis for any raw-text redistribution. Availability for research does not establish ownership of underlying pages |

## Yelp agreement and unresolved applicability

The [official Yelp-hosted agreement](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/dc1cabe7cb95/assets/vendor/Dataset_User_Agreement.pdf) is dated 16 February 2021. Sections 1/3 restrict use to the stated academic/noncommercial purposes; section 3 requires submission of academic findings before public release. Sections 4A/D/H restrict data display, third-party use and transfer; 4E includes an academic-disclosure exception. Section 10 specifies a twelve-month term. These conditions require review alongside the actual accepted agreement, effective date, dataset release and institutional use record. Those records are absent locally. This PDF is evidence of published terms, not proof that its version governs this particular RAGTruth-derived snapshot or that this project has a current permission grant. Do not treat a repository license as superseding it. Future raw-review transfer or publication must resolve the applicable conditions first.

## Model provenance is a separate record

The pinned [Qwen generator license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) is Apache-2.0. The pinned [alternative NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/6c749ce3425cd33b46d187e45b92bbf96ee12ec7/README.md) declares Apache-2.0 and SNLI/MultiNLI training. Retain the model revisions, notices and software environment in later manifests. Model licensing does not clear source passages or prove that pretrained models never encountered a source article. No raw-weight redistribution is prepared here.

## Gate disposition

Evidence register: complete for the four local collections and their missing fields. Gate: requires revision/blocked by missing release, rights and identity records. Recover the existing acquisition/acceptance records and provider crosswalks; if insufficient, seek an attributable institutional/provider decision or choose a separately licensed source frame through a versioned protocol. Do not substitute a different dataset silently. Keep raw source/review text out of a public evidence bundle until the applicable rights are established. This review neither alters prior experiments nor supplies the still-missing fresh confirmation frame.
