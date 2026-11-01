# EviConsist

**Evidence Verification for Hallucination Detection: A Controlled Verifier Study on RAGTruth**

Implementation, manuscript source, original diagrams/plots and aggregate results for a controlled evidence-only NLI verifier comparison. The main completed experiment compares small and base DeBERTa cross-encoders with fixed retrieval, segmentation, label mapping, features and classifier-training protocol.

## Measured test results

Both systems use the same 2,675 eligible responses from 450 source IDs, with thresholds independently selected on validation for at least 90% recall.

| Metric | Small | Base |
|---|---:|---:|
| Precision | 0.3834 | 0.4087 |
| Recall | 0.8736 | 0.8915 |
| F1 | 0.5329 | 0.5605 |
| AUROC | 0.6426 | 0.7016 |
| Average precision | 0.4257 | 0.4847 |
| False positives | 1,256 | 1,153 |

Task-specific trade-offs and source-overlap limitations matter. These are benchmark point estimates, not deployment guarantees. The consistency-only and combined EviConsist effectiveness experiments are **not complete**. Synthetic statistical-engineering checks do not establish detector effectiveness or inferential approval.

## Manuscript and original assets

- [Main IEEE-style LaTeX source](paper/main.tex)
- [Original figures and provenance](paper/figures/README.md)
- [Original measured tables](paper/tables/README.md)
- [Verified aggregate results](paper/reproducibility/VERIFIED_PAPER_RESULTS.json)
- [Submission handoff](paper/SUBMISSION_HANDOFF.md)
- [Public release scope and limitations](docs/PUBLIC_REPOSITORY_SCOPE.md)

Compile `paper/main.tex` with an IEEEtran-enabled pdfLaTeX installation. The bibliography is inline; figure paths and table inputs are relative to `paper/`. All manuscript citations, DOI/URL links and internal equation/figure/table references are clickable. No figures/tables are copied from other papers. AI assistance is disclosed in the manuscript.

## Code

- `baseline.py`, `full_baseline.py`: dataset audit, sentence proxies, TF–IDF evidence retrieval and original verifier.
- `verifier_comparison.py`, `final_verifier_test.py`: controlled alternative-verifier and frozen comparison code.
- `experiment_report.py`, `baseline_supplement.py`: evaluation and descriptive diagnostics.
- `deadline_paper_analysis_2026_11_01.py`: checks measured results and generates plots from locally retained predictions.
- `paper/generate_original_figures.py`: original architecture, confusion, task and sensitivity artwork; aggregate-data fallback works without private archives.
- `consistency_pilot.py`, `silver_labels.py`: preliminary consistency/annotation workflow, not a completed combined detector.
- `eviconsist_candidate_*`, `candidate_c_*`: experimental statistical and numerical-engineering work; historical failures are preserved in accompanying protocol documentation.

Historical runners use dated local artifact paths. This repository is not a claim that every experiment reruns without obtaining its data and preparing those inputs. In particular, the final test runner intentionally refuses to overwrite an existing completed evaluation.

## Setup and data

Use Python and install the pinned dependencies:

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.lock.txt
```

Obtain [RAGTruth from its official repository](https://github.com/ParticleMedia/RAGTruth), observe upstream terms, and place its dataset files under `data/RAGTruth/dataset/`. The audited dataset commit and input checksums are recorded in `paper/reproducibility/dataset_audit.json`. Raw data, model weights, prediction archives, SQLite caches and cloud checkpoint ZIPs are excluded from Git.

Core checks (the source-group test needs the dataset):

```sh
python -m unittest test_baseline test_supplement
```

Regenerate aggregate-data artwork:

```sh
python paper/generate_original_figures.py
```

Prediction-level PR/reliability/risk curves require the locally retained prediction files and full directory structure referenced by `deadline_paper_analysis_2026_11_01.py`; those archives are not distributed here.

## Reproducibility limitations

The benchmark label is positive when a response has at least one annotation with `implicit_true == false`; only `quality == good` responses are included. Source IDs are separated across folds, but later audits found content overlap between distinct IDs. Post-hoc exclusions do not prove full independence. Archived percentile-bootstrap intervals condition on fitted models/thresholds and are descriptive rather than validated simultaneous guarantees.

Aggregate data is supplied for checking tables and figures. No public raw-data release, new human annotation, combined-model improvement, publication acceptance or IEEE Xplore inclusion is claimed.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the project. Cite RAGTruth and the pretrained verifier resources separately when using them. The manuscript is an unpublished research draft; it has no assigned DOI or accepted venue. No project code license has yet been selected by the authors. Dependencies and upstream data retain their own terms.
