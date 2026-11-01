# Controlled consistency pilot — 16 October 2026

The component comparison is complete. Freeze the alternative evidence verifier, preprocessing and feature definitions; do not revisit the evaluated RAGTruth test fold for detector selection. Frozen evidence-study artifacts and intervals remain in PROJECT_STATUS.json and the returned final-test directory.

## Immediate scope

Run on the friend's Apple Silicon Mac using a local generator. Start with 12 distinct RAGTruth **development-training sources**, four per task (Summary/QA/Data2txt), chosen by a deterministic source hash among source payloads <=6,000 characters. Each task contributes three pilot-training sources and one pilot-validation source. These are already-seen development sources, not fresh confirmatory sources. Short-source selection is a feasibility restriction and limits representativeness.

One target and five comparison responses per source: 72 independent sampling calls, 12 new targets to annotate. All six use exactly the same messages, generator/revision and stochastic decoding distribution; sample index 0 is the target, never included in its own comparison features. Seeds are independently derived from source ID and sample index. Different seeds need not produce different texts; preserve duplicates and measure sample diversity rather than replacing them.

Generator: Qwen/Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; MPS float16, eager attention, temperature 0.8, top-p 0.95, top-k 0, repetition penalty 1, one sequence/one beam, maximum 256 generated tokens. Native Python 3.12 and pinned requirements.mac.txt. No API billing. Check all templated prompts <=3,840 input tokens before generation; no silent truncation or model/device fallback. A different configuration requires a distinct experiment directory and documented reason. Fix settings before viewing new labels. Seeds improve traceability but do not guarantee bitwise replay across devices or library changes. Record full resolved generation settings, package versions, prompt hashes, token counts, finish reasons and times. No generator weights are downloaded on Windows.

## Consistency signal

SelfCheckGPT-inspired NLI sampling baseline, **not an exact reproduction**: https://aclanthology.org/2023.emnlp-main.557/ . Target sentence proxies retain character offsets. Each comparison response is divided into overlapping 120-word windows, stride 80. For each target sentence and each of five samples, take the maximum entailment and maximum contradiction across that sample's windows. The two maxima may come from different windows; preserve this limitation in diagnostics. Pair order is premise=comparison window, hypothesis=target sentence. The raw supplied source is never passed to consistency NLI.

NLI scorer is the pinned alternative `cross-encoder/nli-deberta-v3-base` at `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, CPU float32, four threads, batches of 16, 512 tokens, premise-only truncation. Record truncation counts; full samples remain cached. Its published label order is contradiction, entailment, neutral. Sharing the NLI scorer between evidence and consistency components may correlate their mistakes; it does not establish independent errors or zero computational cost.

For each sentence: disagreement = 1 - mean entailment across samples; contradiction = mean of per-sample maximum contradiction. Response features: mean and maximum sentence disagreement, mean and maximum sentence contradiction, and one minus the proportion of sentence/sample entailment scores >=0.5. This is a prespecified feature threshold, not a calibrated truth probability. Existing sentences() are sentence proxies, not atomic claims. Retain individual scores, costs and cached pairs.

## Matched evidence and combined models

Score these **same newly generated targets**, using full_baseline.prepare() unchanged (TF-IDF top-3, 120/80 evidence windows, original sentence splitter) and the same alternative NLI. Evidence features remain [max unsupported, max contradiction, mean unsupported]. No historical response feature or label is substituted. Consistency-only uses five features; evidence-only uses three; combined concatenates the eight. Removing all consistency features and refitting is exactly the evidence-only ablation.

For an eventual matched labelled dataset, fit all three downstream LogisticRegression(random_state=42) models on the same new training targets and select operating thresholds on the same validation sources, maximizing precision subject to recall >=90% (highest threshold breaks precision ties). Freeze the existing component-study classifiers as historical/transfer references, not as unfairly differently trained competitors to a newly fitted fusion classifier. Pilot fit is diagnostic only; nine training and three validation sources cannot establish superiority, robust calibration or uncertainty. A one-class fold must halt fitting; never fabricate positives or select replacement sources based on labels. Do not report pilot metrics as a confirmatory test.

## New annotation and progression gates

The annotation packet exposes source/question and target text but hides comparison samples and detector scores. Gold initially null, never inherited from RAGTruth. Human reviewers mark unsupported or contradicted spans with exact offsets/text and source-based rationale. Record reviewer identities and adjudication; an empty span list is negative **only after explicit complete review**. Refusal/truncation/ambiguity require documented quality review. Seek two independent human annotations with adjudication before confirmatory claims; the prototype permits a named single reviewer but that limitation must be reported. Consistency comparison samples need no gold labels unless they become evaluation targets.

Do not treat the assistant's old taxonomy or an NLI model's predictions as validated labels for these new responses. Automatic labels require an independently validated labelling procedure, which has not been provided. The prototype refuses training with incomplete annotations, identity mismatch, one-class folds or length-limited samples. Resolve technical pilot failures before any scale-up, without selectively resampling failures to improve metrics.

After this 12-source technical pilot, review generation health, annotation feasibility and costs. Only then preregister a larger development pilot (e.g. 100 sources) with justified sample size/class coverage. Plan fresh external or untouched sources for final confirmation of A/B/C; the original 450 test sources are already inspected and cannot be called fresh. Audit exact and near-duplicate source overlap against **all** RAGTruth sources and prior reviewed material before freezing the new holdout. A newly generated target on an old source does not create a fresh source holdout. No confirmatory dataset or sample-size justification is claimed yet.

Final analysis should prespecify paired source-bootstrap intervals, calibration bins/Brier, risk among accepted low-risk targets versus coverage, and complete generation+scoring costs. Avoid multiple inconsistent definitions of selective risk. Keep thresholds fixed during bootstrap and disclose selection/training uncertainty exclusions. Define a recall acceptability margin before confirmatory results if making noninferiority claims. Do not make a first-ever signal-combination claim; scope novelty as controlled complementarity with matched labels/baselines, uncertainty and cost.

## Mac commands

In a copy of the project with the prepared `results/consistency_pilot_2026_10_16` lock and sources:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.mac.txt
.venv/bin/python -m unittest test_consistency_pilot.py
caffeinate -i .venv/bin/python -u consistency_pilot.py generate
caffeinate -i .venv/bin/python -u consistency_pilot.py score
.venv/bin/python consistency_pilot.py annotations
```

Review and annotate `annotations.json` manually. Do not run `init` again on the Mac or edit locked inputs. After complete human annotations and technical checks, `fit` produces explicitly development-only diagnostics. Do not run the historical verifier runners. Keep generation and scoring sequential so both models do not occupy memory together. The model downloads occur on the Mac at the pinned revisions. Return the pilot lock, sources, full SQLite cache, generation/config/runtime files, feature records, blank or completed annotation file, score-cost records, logs and actual environment—not a replacement manuscript. No generation or annotation has yet been completed by this implementation.

References for implementation: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , https://huggingface.co/docs/transformers/main/en/chat_templating , https://huggingface.co/docs/transformers/main/en/main_classes/text_generation .
