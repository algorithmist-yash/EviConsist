"""Evaluation and report creation after training/validation selection is complete."""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from baseline import ROOT, fold, spans, metrics

def select_threshold(y,p):
    candidates=np.linspace(0,1,101)
    scores=[precision_recall_fscore_support(y,p>=t,average='binary',zero_division=0)[2] for t in candidates]
    return float(candidates[np.argmax(scores)])

def diagnostic(rows,threshold):
    y=np.array([o['gold'] for o in rows]); p=np.array([o['probability'] for o in rows])
    return dict(n=len(rows),gold_positive=int(y.sum()),predicted_positive=int((p>=threshold).sum()),probability_min=float(p.min()),probability_max=float(p.max()),probability_quantiles=np.quantile(p,[0,.1,.25,.5,.75,.9,1]).tolist(),confusion_matrix_tn_fp_fn_tp=confusion_matrix(y,p>=threshold,labels=[0,1]).tolist(),metrics=metrics(y,p,threshold))

def bootstrap(rows,threshold,repeats=500):
    # Resample source clusters with replacement; preserve all responses in each cluster.
    grouped={}
    for i,row in enumerate(rows): grouped.setdefault(row['source_id'],[]).append(i)
    groups=[np.array(v,dtype=int) for v in grouped.values()]
    y=np.array([o['gold'] for o in rows]); p=np.array([o['probability'] for o in rows])
    rng=np.random.default_rng(42); estimates=[]
    for _ in range(repeats):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))<2: continue
        estimates.append(metrics(y[indices],p[indices],threshold))
    return dict(method='source-cluster percentile bootstrap',source_clusters=len(groups),seed=42,requested_replicates=repeats,valid_replicates=len(estimates),intervals={key:np.quantile([e[key] for e in estimates],[.025,.975]).tolist() for key in estimates[0]} if estimates else {})

def generate(directory,selected,predictions):
    missing={r['id'] for r in selected}-set(predictions)
    if missing: raise ValueError(f'Incomplete extraction: {len(missing)} responses remain; no full report produced.')
    outputs=[]
    for row in selected:
        o=predictions[row['id']]
        o['gold']=int(bool(spans(row)))
        for c in o['claims']:
            c['gold']=int(any(c['start']<a['end'] and c['end']>a['start'] for a in spans(row)))
        outputs.append(o)
    groups={f:[o for o in outputs if o['fold']==f] for f in ['train','validation','test']}
    train,val,test=(groups[f] for f in ['train','validation','test'])
    clf=LogisticRegression(random_state=42).fit([o['features'] for o in train],[o['gold'] for o in train])
    for group in groups.values():
        for row,p in zip(group,clf.predict_proba([o['features'] for o in group])[:,1]): row['probability']=float(p)
    threshold=select_threshold([o['gold'] for o in val],np.array([o['probability'] for o in val]))
    vc=[c for o in val for c in o['claims']]
    sentence_threshold=select_threshold([c['gold'] for c in vc],np.array([c['unsupported'] for c in vc]))
    tc=[dict(gold=c['gold'],probability=c['unsupported'],source_id=o['source_id']) for o in test for c in o['claims']]
    for o in outputs:
        o['prediction']=int(o['probability']>=threshold)
        for c in o['claims']: c['prediction']=int(c['unsupported']>=sentence_threshold)
    config=json.loads((directory/'config.json').read_text())
    report=dict(status='full' if config['limit_per_fold']==0 else 'pilot',response_threshold=threshold,sentence_threshold=sentence_threshold,folds={f:diagnostic(g,threshold) for f,g in groups.items()},sentence_proxy_test=diagnostic(tc,sentence_threshold),response_test_ci=bootstrap(test,threshold),sentence_test_ci=bootstrap(tc,sentence_threshold),logistic_coefficients=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),subgroups={key:{value:diagnostic([o for o in test if o[key]==value],threshold) for value in sorted({o[key] for o in test})} for key in ['task_type','model']})
    accepted=sorted(test,key=lambda o:o['probability'])
    report['risk_coverage']=[dict(coverage=k/len(test),accepted_hallucination_rate=float(np.mean([o['gold'] for o in accepted[:k]]))) for k in np.unique(np.linspace(1,len(test),min(100,len(test))).astype(int))]
    report['cost']=json.loads((directory/'progress.json').read_text())
    (directory/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (directory/'predictions.jsonl').write_text('\n'.join(json.dumps(o) for o in outputs),encoding='utf8')
    with (directory/'predictions.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','source_id','fold','task_type','model','gold','prediction','probability'])
        writer.writeheader(); writer.writerows({k:o[k] for k in writer.fieldnames} for o in outputs)
    (directory/'fusion_model.json').write_text(json.dumps(dict(coefficients=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),classes=clf.classes_.tolist(),threshold=threshold,feature_names=config['features']),indent=2))
    errors={kind:sorted([o for o in test if o['gold']==gold and o['prediction']==pred],key=lambda o:o['probability'],reverse=bool(pred))[:5] for kind,gold,pred in [('false_positives',0,1),('false_negatives',1,0)]}
    (directory/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(f,group) in zip(axes,groups.items()):
        for label,color in [(0,'#2166ac'),(1,'#b2182b')]:
            ax.hist([o['probability'] for o in group if o['gold']==label],bins=np.linspace(0,1,26),alpha=.55,label=f'Gold {label}',color=color)
        ax.axvline(threshold,color='black',linestyle='--',label=f'Threshold {threshold:.2f}')
        ax.set(title=f'{f.title()} (n={len(group):,})',xlabel='Predicted hallucination probability',ylabel='Responses',xlim=(0,1)); ax.legend(fontsize=8)
    fig.savefig(directory/'probability_distribution.png',dpi=180); plt.close(fig)
    cm=np.array(report['folds']['test']['confusion_matrix_tn_fp_fn_tp'])
    fig,ax=plt.subplots(figsize=(4,4),layout='constrained'); ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',fontsize=18,color='white' if v>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Prediction',ylabel='Gold',title='Test response confusion matrix')
    fig.savefig(directory/'confusion_matrix.png',dpi=180); plt.close(fig)
    lines=['# Evidence-only experiment report','',f'Status: {report["status"]}. All selected good-quality responses were scored. Official test sources are held out. Model and threshold selection use training and validation only.','',f'Model: `{config["model"]}`; revision `{config["revision"]}`. The operational label excludes `implicit_true=true`. Sentence overlap labels are proxies, not atomic claims or exact spans.','', '## Response detection','', '| Fold | N | Gold positives | Predicted positives | AUROC | Average precision | Precision | Recall | F1 | Brier | ECE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f,d in report['folds'].items():
        m=d['metrics']; lines.append(f'| {f} | {d["n"]} | {d["gold_positive"]} | {d["predicted_positive"]} | '+ ' | '.join(f'{m[k]:.4f}' if m[k] is not None else 'NA' for k in ['auroc','auprc','precision','recall','f1','brier','ece_10_bins'])+' |')
    lines.extend(['',f'Validation-selected threshold: {threshold:.2f}. Confusion matrix is [[TN, FP], [FN, TP]]: `{cm.tolist()}`. AUPRC is reported as sklearn average precision, not trapezoidal PR area.','', '![Probability distributions](probability_distribution.png)','', '![Confusion matrix](confusion_matrix.png)','', '## Source-grouped uncertainty','', '| Test metric | 95% interval |','|---|---|'])
    for k,(lo,hi) in report['response_test_ci']['intervals'].items(): lines.append(f'| {k} | [{lo:.4f}, {hi:.4f}] |')
    lines.extend(['','500 source-cluster bootstrap draws with seed 42; the fitted model and validation threshold are held fixed. These intervals measure test-source sampling uncertainty, not training/model-selection uncertainty.','', '## Sentence localization and cost','',f'Sentence-proxy test metrics: `{report["sentence_proxy_test"]["metrics"]}`. Its threshold {sentence_threshold:.2f} is also chosen on validation only. This is not span-level F1.','',f'Inference counters: `{report["cost"]}`. Wall time includes preprocessing and model inference but excludes initial checkpoint loading. This CPU-only run makes no paid API calls.','', '## Errors',''])
    lines.insert(lines.index('## Sentence localization and cost'),f'Test source clusters: {report["response_test_ci"]["source_clusters"]}; valid response bootstrap replicates: {report["response_test_ci"]["valid_replicates"]}/500. A tiny pilot cannot support reliable uncertainty estimates even when clustering is correct.\n')
    for kind,examples in errors.items():
        lines.extend([f'### {kind.replace("_"," ").title()}',''])
        if not examples: lines.append('No errors of this type at the frozen threshold.')
        for o in examples:
            lines.append(f'- Response {o["id"]}, source {o["source_id"]}, {o["task_type"]}, {o["model"]}: probability {o["probability"]:.4f}.')
            c=max(o['claims'],key=lambda c:c['unsupported']) if kind=='false_positives' else next((c for c in o['claims'] if c['gold']),o['claims'][0])
            lines.append('  '+json.dumps(c['text'],ensure_ascii=False)+f' (unsupported={c["unsupported"]:.4f}, contradiction={c["contradiction"]:.4f}).')
        lines.append('')
    lines.extend(['## Interpretation and reproducibility','', 'Performance must be assessed from measured metrics; no benefit from consistency or calibration is asserted. Max entailment and max contradiction can originate from different windows, so both may be high. Structured data flattening, sentence segmentation, missing evidence, and NLI domain mismatch remain limitations. Error examples are selected after final evaluation for descriptive analysis; they do not alter the frozen threshold.','', 'Reproduce with the command in run_command.txt. config.json freezes the data, code and environment hashes; fusion_model.json records fitted coefficients; features.sqlite persists inference; predictions.jsonl includes per-window probabilities and evidence. requirements.lock.txt at project root captures installed packages.','', 'The manuscript remains a proposal. Same-model sampling, calibrated fusion, ablations and adaptive verification are outside this baseline report.'])
    (directory/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    progress=report['cost']; progress['status']='complete'; (directory/'progress.json').write_text(json.dumps(progress,indent=2))
    print(f'Completed report: {directory / "EXPERIMENT_REPORT.md"}',flush=True)
