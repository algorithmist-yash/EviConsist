"""Source-grouped RAGTruth audit and sentence-level evidence-only NLI baseline."""
import argparse, collections, hashlib, json, re, subprocess, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, brier_score_loss

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/RAGTruth/dataset'
OUT = ROOT / 'results'

def read(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding='utf8').splitlines()]

def spans(row):
    return [a for a in row['labels'] if a['implicit_true'] is False]

def fold(row):
    if row['split'] == 'test': return 'test'
    return 'validation' if int(hashlib.sha256(row['source_id'].encode()).hexdigest(), 16) % 5 == 0 else 'train'

def sentences(text):
    # Sentence proxies, not atomic claims. Preserve original character offsets.
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S.*?(?:[.!?](?=\s|$)|\n|$)', text)]

def evidence(source):
    info = source['source_info']
    if source['task_type'] == 'QA': return info['passages']
    if isinstance(info, str): return info
    def flatten(value, prefix=''):
        if isinstance(value, dict):
            return '\n'.join(flatten(v, prefix + k + ': ') for k,v in value.items())
        if isinstance(value, list): return '\n'.join(flatten(v, prefix) for v in value)
        return prefix + json.dumps(value, ensure_ascii=False)
    return flatten(info)

def audit():
    rows, sources = read('response.jsonl'), read('source_info.jsonl')
    ids = {s['source_id'] for s in sources}
    train = {r['source_id'] for r in rows if r['split']=='train'}
    test = {r['source_id'] for r in rows if r['split']=='test'}
    invalid = [r['id'] for r in rows for a in r['labels'] if not (0 <= a['start'] < a['end'] <= len(r['response'])) or r['response'][a['start']:a['end']] != a['text']]
    report = dict(responses=len(rows), sources=len(sources), official_splits=dict(collections.Counter(r['split'] for r in rows)), quality=dict(collections.Counter(r['quality'] for r in rows)), tasks=dict(collections.Counter(s['task_type'] for s in sources)), implicit_true_spans=sum(a['implicit_true'] for r in rows for a in r['labels']), positive_responses=sum(bool(spans(r)) for r in rows), source_overlap=sorted(train & test), missing_sources=sorted({r['source_id'] for r in rows}-ids), invalid_span_response_ids=invalid, dataset_commit=subprocess.check_output(['git','-C',str(DATA.parent),'rev-parse','HEAD'],text=True).strip(), file_sha256={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['response.jsonl','source_info.jsonl']})
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(report, indent=2),encoding='utf8')
    (OUT/'split_manifest.jsonl').write_text('\n'.join(json.dumps(dict(id=r['id'],source_id=r['source_id'],fold=fold(r),quality=r['quality'])) for r in rows),encoding='utf8')
    assert not report['source_overlap'] and not report['missing_sources']
    print(json.dumps(report,indent=2))

def metrics(y,p,t):
    y,p=np.asarray(y),np.asarray(p)
    pr,rc,f,_=precision_recall_fscore_support(y,p>=t,average='binary',zero_division=0)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(np.mean(y[bins==b])-np.mean(p[bins==b])) for b in range(10) if np.any(bins==b))
    return dict(auroc=roc_auc_score(y,p) if len(set(y))>1 else None, auprc=average_precision_score(y,p),precision=pr,recall=rc,f1=f,brier=brier_score_loss(y,p),ece_10_bins=float(ece))

def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,revision=args.revision).eval()
    labels={v.lower():int(k) for k,v in model.config.id2label.items()}
    if args.model == 'cross-encoder/nli-deberta-v3-small' and set(labels)=={'label_0','label_1','label_2'}:
        labels={'contradiction':0,'entailment':1,'neutral':2}  # Published model card mapping.
    if not all(k in labels for k in ['entailment','contradiction','neutral']): raise ValueError(f'Explicit NLI label mapping required: {labels}')
    rows=read('response.jsonl'); sources={s['source_id']:s for s in read('source_info.jsonl')}
    selected=[]
    for f in ['train','validation','test']:
        group=[r for r in rows if fold(r)==f and r['quality']=='good']
        selected.extend(group if args.limit==0 else group[:args.limit])
    started=time.perf_counter(); outputs=[]
    for idx,r in enumerate(selected):
        claims=sentences(r['response']); context=evidence(sources[r['source_id']])
        # Small overlapping word windows prevent silent whole-context truncation.
        words=context.split(); passages=[' '.join(words[i:i+120]) for i in range(0,len(words),80)] or ['']
        vectorizer=TfidfVectorizer().fit(passages+[c[2] for c in claims])
        pv=vectorizer.transform(passages); records=[]
        for start,end,claim in claims:
            similarities=(pv @ vectorizer.transform([claim]).T).toarray().ravel()
            chosen=np.argsort(-similarities,kind='stable')[:args.top_k]
            inputs=tokenizer([passages[i] for i in chosen],[claim]*len(chosen),padding=True,truncation='only_first',max_length=512,return_tensors='pt')
            with torch.inference_mode(): probs=model(**inputs).logits.softmax(-1).numpy()
            ent=float(probs[:,labels['entailment']].max()); con=float(probs[:,labels['contradiction']].max())
            records.append(dict(start=start,end=end,text=claim,entailment=ent,contradiction=con,unsupported=1-ent,retrieval_similarity=float(similarities[chosen[0]]),gold=int(any(start<a['end'] and end>a['start'] for a in spans(r))),evidence=[passages[i] for i in chosen]))
        if not records: raise ValueError(f'No claims: {r["id"]}')
        features=[max(c['unsupported'] for c in records),max(c['contradiction'] for c in records),np.mean([c['unsupported'] for c in records])]
        outputs.append(dict(id=r['id'],source_id=r['source_id'],fold=fold(r),gold=int(bool(spans(r))),features=features,claims=records))
        if idx%10==0: print(f'Scored {idx+1}/{len(selected)}',flush=True)
    train=[o for o in outputs if o['fold']=='train']; val=[o for o in outputs if o['fold']=='validation']; test=[o for o in outputs if o['fold']=='test']
    clf=LogisticRegression(random_state=42).fit([o['features'] for o in train],[o['gold'] for o in train])
    vp=clf.predict_proba([o['features'] for o in val])[:,1]; vy=np.array([o['gold'] for o in val])
    threshold=max(np.linspace(0,1,101),key=lambda t:metrics(vy,vp,t)['f1'])
    for o in outputs: o['probability']=float(clf.predict_proba([o['features']])[0,1])
    tp=np.array([o['probability'] for o in test]); ty=np.array([o['gold'] for o in test])
    vc=[c for o in val for c in o['claims']]; tc=[c for o in test for c in o['claims']]
    cy=np.array([c['gold'] for c in vc]); cp=np.array([c['unsupported'] for c in vc])
    ct=max(np.linspace(0,1,101),key=lambda t:metrics(cy,cp,t)['f1'])
    report=dict(status='pilot' if args.limit else 'full',limit_per_fold=args.limit,model=args.model,requested_revision=args.revision,resolved_revision=getattr(model.config,'_commit_hash',None),top_k=args.top_k,threshold=float(threshold),response_test=metrics(ty,tp,threshold),seconds=time.perf_counter()-started,logistic_coefficients=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),note='Sentence proxies; no atomic claim or exact span F1 claimed. No calibration or consistency experiment yet.')
    report['sentence_threshold']=float(ct)
    report['sentence_proxy_test']=metrics(np.array([c['gold'] for c in tc]),np.array([c['unsupported'] for c in tc]),ct)
    order=np.argsort(tp)
    report['risk_coverage']=[dict(coverage=k/len(test),accepted_hallucination_rate=float(ty[order[:k]].mean())) for k in range(1,len(test)+1)]
    OUT.mkdir(exist_ok=True)
    (OUT/'evidence_predictions.jsonl').write_text('\n'.join(json.dumps(o) for o in outputs),encoding='utf8')
    (OUT/'evidence_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['audit','run']); parser.add_argument('--limit',type=int,default=30,help='Responses per fold; 0 for all'); parser.add_argument('--model',default='cross-encoder/nli-deberta-v3-small'); parser.add_argument('--revision',default='fa2804872c3b4bd748f38c0185cc85775361e735'); parser.add_argument('--top-k',type=int,default=3)
    args=parser.parse_args()
    audit() if args.command=='audit' else run(args)

