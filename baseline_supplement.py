"""Post-process frozen baseline predictions; never refit or change thresholds."""
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
CATEGORIES=['retrieval_failure','insufficient_context','false_nli_contradiction','segmentation_error','compound_claim','annotation_ambiguity','other','no_apparent_error']

def reliability(rows,bins=10):
    p=np.array([o['probability'] for o in rows]); y=np.array([o['gold'] for o in rows])
    membership=np.minimum((p*bins).astype(int),bins-1)
    return [dict(bin=i,lower=i/bins,upper=(i+1)/bins,n=int((membership==i).sum()),mean_probability=float(p[membership==i].mean()),positive_fraction=float(y[membership==i].mean())) for i in range(bins) if np.any(membership==i)]

def completed(directory):
    try:
        return json.loads((directory/'progress.json').read_text()).get('status')=='complete'
    except (FileNotFoundError,json.JSONDecodeError):
        # The inference process may be replacing the progress file during this read.
        return False

def build(directory):
    import matplotlib
    os.environ['MPLCONFIGDIR']=str(directory/'mpl-cache')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((directory/'metrics.json').read_text())
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf8').splitlines()]
    groups={f:[o for o in rows if o['fold']==f] for f in ['train','validation','test']}
    comparisons={}
    curves={}
    for f,g in groups.items():
        prevalence=float(np.mean([o['gold'] for o in g])); ap=report['folds'][f]['metrics']['auprc']
        comparisons[f]=dict(n=len(g),positive_prevalence=prevalence,average_precision=ap,ap_minus_prevalence=ap-prevalence,ap_over_prevalence=ap/prevalence if prevalence else None)
        curves[f]=reliability(g)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for f,color in [('validation','#2166ac'),('test','#b2182b')]:
        points=curves[f]; axes[0].plot([p['mean_probability'] for p in points],[p['positive_fraction'] for p in points],marker='o',label=f.title(),color=color)
    axes[0].plot([0,1],[0,1],color='gray',linestyle='--')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Reliability: 10 equal-width bins'); axes[0].legend()
    rc=report['risk_coverage']; axes[1].plot([p['coverage'] for p in rc],[p['accepted_hallucination_rate'] for p in rc],color='#2166ac')
    axes[1].axhline(comparisons['test']['positive_prevalence'],linestyle='--',color='gray',label='Test prevalence')
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Coverage (lowest risk accepted)',ylabel='Hallucination rate among accepted',title='Test risk–coverage'); axes[1].legend()
    fig.savefig(directory/'calibration_and_risk.png',dpi=180); plt.close(fig)
    # Choose review cases from validation only, including correct decisions.
    rng=np.random.default_rng(42); sample=[]
    for task in sorted({o['task_type'] for o in groups['validation']}):
        for name,gold,pred in [('FP',0,1),('FN',1,0),('TP',1,1),('TN',0,0)]:
            candidates=[o for o in groups['validation'] if o['task_type']==task and o['gold']==gold and o['prediction']==pred]
            for idx in rng.choice(len(candidates),size=min(3,len(candidates)),replace=False):
                sample.append(dict(review_stratum=f'{task}/{name}',**candidates[idx]))
    raw={o['id']:o for o in map(json.loads,(ROOT/'data/RAGTruth/dataset/response.jsonl').read_text(encoding='utf8').splitlines())}
    sources={o['source_id']:o for o in map(json.loads,(ROOT/'data/RAGTruth/dataset/source_info.jsonl').read_text(encoding='utf8').splitlines())}
    for o in sample:
        o['raw_annotations']=raw[o['id']]['labels']; o['full_source_context']=sources[o['source_id']]['source_info']; o['review_category']=''; o['review_notes']=''
    (directory/'validation_review.jsonl').write_text('\n'.join(json.dumps(o,ensure_ascii=False) for o in sample),encoding='utf8')
    with (directory/'validation_review.csv').open('w',encoding='utf8',newline='') as stream:
        fields=['id','source_id','review_stratum','probability','gold','prediction','review_category','review_notes']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows({key:o[key] for key in fields} for o in sample)
    review=['# Validation review packet','','Seed 42; up to three examples from each task × FP/FN/TP/TN stratum. Correct predictions are included. This is a descriptive purposive sample, not an estimate of error-category frequency. Review category fields are blank until manual inspection.','','Allowed categories: '+', '.join(CATEGORIES)+'.','','Use full source context and raw annotations in validation_review.jsonl to distinguish retrieval failure from insufficient context. Neutral NLI alone does not prove hallucination. Tag compound claims or segmentation issues separately where needed. Do not use held-out test errors for baseline tuning.','']
    for o in sample:
        review.extend([f'## Response {o["id"]}: {o["review_stratum"]}','',f'Source {o["source_id"]}; probability {o["probability"]:.4f}; gold {o["gold"]}; prediction {o["prediction"]}.','',raw[o['id']]['response'],'','### Sentence evidence',''])
        for c in o['claims']:
            review.extend([f'- [{c["start"]}:{c["end"]}] {c["text"]}',f'  Gold overlap {c["gold"]}; unsupported {c["unsupported"]:.4f}; contradiction {c["contradiction"]:.4f}.'])
            for passage,probs in zip(c['evidence'],c['nli_probabilities']):
                review.extend([f'  Evidence (entailment {probs["entailment"]:.4f}, contradiction {probs["contradiction"]:.4f}): {passage}'])
        review.extend(['','Category: ______  Notes: ______',''])
    (directory/'VALIDATION_REVIEW.md').write_text('\n'.join(review),encoding='utf8')
    (directory/'supplement.json').write_text(json.dumps(dict(prevalence_comparison=comparisons,reliability_bins=curves,review_cases=len(sample),review_seed=42,manual_review_status='not yet classified',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256((directory/'predictions.jsonl').read_bytes()).hexdigest()),indent=2))
    lines=['# Evidence baseline reporting supplement','','This analysis uses the frozen fitted model, predictions and validation thresholds. No refitting or recalibration is performed.','','| Fold | Positive prevalence | Average precision | AP minus prevalence |','|---|---:|---:|---:|']
    for f,d in comparisons.items(): lines.append(f'| {f} | {d["positive_prevalence"]:.4f} | {d["average_precision"]:.4f} | {d["ap_minus_prevalence"]:.4f} |')
    lines.extend(['','Prevalence provides the population no-skill PR reference. AP is sklearn average precision; it is not trapezoidal PR area. Finite-sample random rankings can differ from prevalence.','','![Reliability and risk–coverage](calibration_and_risk.png)','','Reliability curves are descriptive, use 10 equal-width bins and omit empty bins. They do not establish calibration from a small pilot. ECE and Brier remain in metrics.json. Selective risk here means the gold hallucination prevalence among accepted low-risk responses, not classification error.','',f'The validation review packet contains {len(sample)} cases with blank manual classification fields. Review both errors and correct decisions before proposing a new evidence verifier. Keep this baseline frozen; any later version requires a new experiment directory.','', 'No consistency, combined, calibrated-fusion, or adaptive-verification result is claimed.'])
    (directory/'REPORT_SUPPLEMENT.md').write_text('\n'.join(lines),encoding='utf8')
    print(f'Supplement complete: {directory}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--directory',default='results/full'); parser.add_argument('--wait',action='store_true'); args=parser.parse_args()
    directory=ROOT/args.directory
    if args.wait:
        print('Waiting for complete baseline predictions; no partial dataset is evaluated.',flush=True)
        while not completed(directory):
            time.sleep(10)
    build(directory)
