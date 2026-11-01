"""Recompute manuscript tables/figures from saved predictions; no fitting or inference."""
import os
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'results/submission_preparation_2026_11_01';O.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(O/'mpl-cache')
import json,hashlib,collections,shutil
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,precision_recall_fscore_support,brier_score_loss,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=R/'results/final_test_mac_return_2026_10_16/results/final_verifier_test_2026_10_16'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x]
sources={s['source_id']:s for s in lines(R/'data/RAGTruth/dataset/source_info.jsonl')}
raw={r['id']:r for r in lines(R/'data/RAGTruth/dataset/response.jsonl')}
rows=lines(BASE/'paired_test_predictions.jsonl');old=load(BASE/'test_metrics.json')
assert len(rows)==2675 and len({r['id'] for r in rows})==2675
for row in rows:
    rr=raw[row['id']];assert rr['source_id']==row['source_id'] and rr['quality']=='good' and rr['split']=='test'
    assert row['gold']==int(any(s['implicit_true'] is False for s in rr['labels']))
TH={k:old[v]['threshold'] for k,v in [('baseline','baseline_matched'),('alternative','alternative_matched')]}
def metrics(rs,k):
    y=np.array([r['gold'] for r in rs]);p=np.array([r[k+'_probability'] for r in rs]);pred=p>=TH[k]
    pr,re,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    b=np.minimum((p*10).astype(int),9)
    return dict(n=len(rs),sources=len({r['source_id'] for r in rs}),positives=int(sum(y)),prevalence=float(np.mean(y)),
        precision=float(pr),recall=float(re),f1=float(f),auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),
        ece10=float(sum(np.mean(b==i)*abs(np.mean(y[b==i])-np.mean(p[b==i])) for i in range(10) if np.any(b==i))),
        confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())
allmetrics={k:metrics(rows,k) for k in TH}
for k,v in [('baseline','baseline_matched'),('alternative','alternative_matched')]:
    for metric in ['precision','recall','f1','auroc','average_precision','brier']:
        assert abs(allmetrics[k][metric]-old[v][metric])<1e-12
task={t:{k:metrics([r for r in rows if sources[r['source_id']]['task_type']==t],k) for k in TH} for t in ['Summary','QA','Data2txt']}
# Conservative flagged components, not a claim of exhaustive deduplication.
flags=load(R/'results/candidate_c_statistical_validation_design_v1_2026_10_18/SOURCE_EXPOSURE_FOLLOWUP.json')['verified_positive_flags']
edges=[tuple(x['source_ids']) for x in flags]
edges += [(x['a'],x['b']) for x in load(R/'results/source_overlap_provenance_audit_v1_2026_10_18/FUTURE_RELATED_SOURCE_GROUPS.json')['edges']]
parent={s:s for s in sources}
def find(s):
    while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
    return s
for a,b in edges:parent[find(a)]=find(b)
exposed={find(r['source_id']) for r in raw.values() if r['split']!='test'}
excluded={r['source_id'] for r in rows if find(r['source_id']) in exposed}
qa_excluded={'15112','15317'}
sens={name:{k:metrics([r for r in rows if r['source_id'] not in ids],k) for k in TH}
    for name,ids in [('remove_two_QA_sources',qa_excluded),('remove_flagged_crossfold_components',excluded)]}
report=dict(date='2026-11-01',scope='post-hoc saved-prediction analysis; no new model inference, fitting, threshold choice or independent confirmation',
    thresholds=TH,overall=allmetrics,task_results=task,sensitivity=sens,excluded_flagged_test_sources=sorted(excluded),
    flagged_pair_count=len(flags),crossfold_flag_count=sum(x['crosses_historical_folds'] for x in flags),
    historical_paired_percentile_bootstrap=old['paired_source_bootstrap'],
    input_sha256={str(p.relative_to(R)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [BASE/'paired_test_predictions.jsonl',BASE/'test_metrics.json',R/'data/RAGTruth/dataset/source_info.jsonl',R/'data/RAGTruth/dataset/response.jsonl']})
(O/'VERIFIED_PAPER_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf8')
# Figure is entirely descriptive on frozen test predictions.
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(7.1,2.25),layout='constrained')
colors=['#2166ac','#b35806'];labels=['Small verifier','Base verifier']
y=np.array([r['gold'] for r in rows])
for (k,color,label) in zip(TH,colors,labels):
    p=np.array([r[k+'_probability'] for r in rows]);pr,rc,_=precision_recall_curve(y,p)
    axs[0].plot(rc,pr,color=color,label=label,lw=1.3)
    bins=np.minimum((p*10).astype(int),9);xx=[];yy=[]
    for i in range(10):
        if np.any(bins==i):xx.append(np.mean(p[bins==i]));yy.append(np.mean(y[bins==i]))
    axs[1].plot(xx,yy,'o-',ms=3,color=color,label=label,lw=1)
    order=np.argsort(p,kind='stable');risk=np.cumsum(y[order])/np.arange(1,len(y)+1)
    axs[2].plot(np.arange(1,len(y)+1)/len(y),risk,color=color,label=label,lw=1.3)
axs[0].axhline(y.mean(),color='grey',ls=':',lw=.9,label='Test prevalence')
axs[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));axs[0].legend(fontsize=6,loc='upper right')
axs[1].plot([0,1],[0,1],ls=':',color='grey',lw=.9);axs[1].set(xlabel='Mean predicted risk',ylabel='Observed positive rate',xlim=(0,1),ylim=(0,1))
axs[2].set(xlabel='Accepted coverage',ylabel='Positive rate among accepted',xlim=(0,1),ylim=(0,1))
for ax in axs:ax.grid(alpha=.15)
assets=R/'paper/figures';assets.mkdir(exist_ok=True)
fig.savefig(assets/'verifier_diagnostics.pdf');fig.savefig(assets/'verifier_diagnostics.svg');fig.savefig(assets/'verifier_diagnostics.png',dpi=240);fig.savefig(O/'verifier_diagnostics.png',dpi=180);plt.close(fig)
backup=O/'main_before_revision.tex'
if not backup.exists():shutil.copy2(R/'paper/main.tex',backup)
print(json.dumps(dict(overall=allmetrics,task_results=task,sensitivity=sens,excluded_flagged_test_sources=sorted(excluded)),indent=2))
