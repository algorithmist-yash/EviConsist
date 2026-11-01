"""Locked test-only continuation; no training, threshold search, or candidate selection."""
import hashlib, json, os, sqlite3, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from verifier_comparison import metrics
from baseline import read, fold, spans

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/final_verifier_test_2026_10_16'
BASE=ROOT/'results/completed_baseline_2026_10_14'
ALT=ROOT/'results/mac_return_2026_10_16/results/verifier_base_2026_10_15'
MODEL='cross-encoder/nli-deberta-v3-base'
REVISION='6c749ce3425cd33b46d187e45b92bbf96ee12ec7'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def load_rows():
    rows=[]
    with (BASE/'predictions.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['fold']=='test': rows.append(row)
    return rows

def prepare():
    from sklearn.linear_model import LogisticRegression
    from verifier_comparison import matched_threshold
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'LOCK.json').exists(): raise ValueError('Lock already exists; do not replace it')
    rows=load_rows(); raw={r['id']:r for r in read('response.jsonl') if r['quality']=='good' and fold(r)=='test'}
    assert len(rows)==len(raw)==2675 and len({r['id'] for r in rows})==2675
    assert {r['id'] for r in rows}==set(raw)
    assert sum(r['gold'] for r in rows)==894
    for r in rows: assert r['source_id']==raw[r['id']]['source_id'] and r['gold']==int(bool(spans(raw[r['id']])))
    # Recover and save the original fitted classifier without changing its protocol.
    training=[]; validation=[]
    with (BASE/'predictions.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['fold']=='train': training.append(r)
            elif r['fold']=='validation': validation.append(r)
    assert {r['source_id'] for r in rows}.isdisjoint({r['source_id'] for r in training+validation})
    lr=LogisticRegression(random_state=42).fit([r['features'] for r in training],[r['gold'] for r in training])
    assert np.allclose(lr.predict_proba([r['features'] for r in rows])[:,1],[r['probability'] for r in rows],atol=1e-10,rtol=0)
    bt=matched_threshold(np.array([r['gold'] for r in validation]),np.array([r['probability'] for r in validation]))
    am=json.loads((ALT/'validation_metrics.json').read_text()); ac=json.loads((ALT/'config.json').read_text())
    assert ac['model']==MODEL and ac['revision']==REVISION
    inputs=[BASE/'config.json',BASE/'metrics.json',BASE/'predictions.jsonl',ALT/'config.json',ALT/'validation_metrics.json',ALT/'validation_predictions.jsonl',ALT/'runner_snapshot.py',ALT/'features.sqlite',ROOT/'data/RAGTruth/dataset/response.jsonl',ROOT/'data/RAGTruth/dataset/source_info.jsonl',ROOT/'verifier_comparison.py',Path(__file__)]
    checkpoint=ROOT/'models/hub/models--cross-encoder--nli-deberta-v3-base/snapshots'/REVISION
    inputs.extend(p for p in checkpoint.iterdir() if p.is_file())
    lock=dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),decision='Evaluate exactly this validation-selected alternative once; no test-based tuning or winner selection',model=MODEL,revision=REVISION,baseline_revision='fa2804872c3b4bd748f38c0185cc85775361e735',baseline=dict(coefficients=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),matched_threshold=bt,historical_threshold=.29),alternative=dict(coefficients=am['coefficients'][0],intercept=am['intercept'][0],matched_threshold=am['validation']['threshold']),test_count=2675,test_positive=894,test_sources=len({r['source_id'] for r in rows}),device='cpu',dtype='float32',threads=4,batch_size=16,max_tokens=512,truncation='only_first',label_order=['contradiction','entailment','neutral'],bootstrap=dict(replicates=1000,seed=42,unit='source_id',thresholds='fixed validation thresholds',interval='paired percentile 95%; conditional on fitted models/thresholds'),files={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    for r in rows:
        assert np.isfinite(r['features']).all()
    (OUT/'test_ids_labels.json').write_text(json.dumps([dict(id=r['id'],source_id=r['source_id'],gold=r['gold']) for r in rows],indent=2))
    (OUT/'LOCK.json').write_text(json.dumps(lock,indent=2))
    (OUT/'LOCK.sha256').write_text(sha(OUT/'LOCK.json')+'\n')
    print('Locked 2675 test IDs, 894 positives, 450 sources; thresholds:',bt,am['validation']['threshold'],flush=True)

def run():
    lock=json.loads((OUT/'LOCK.json').read_text())
    assert sha(OUT/'LOCK.json')==(OUT/'LOCK.sha256').read_text().strip()
    for name,h in lock['files'].items(): assert sha(ROOT/name)==h,'Frozen file changed: '+name
    if (OUT/'test_metrics.json').exists(): raise ValueError('Final evaluation already exists; refusing rerun')
    rows=load_rows(); assert [dict(id=r['id'],source_id=r['source_id'],gold=r['gold']) for r in rows]==json.loads((OUT/'test_ids_labels.json').read_text())
    os.environ['HF_HOME']=str(ROOT/'models'); os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4); torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,local_files_only=True).eval()
    assert model.config.num_labels==3
    (OUT/'environment.txt').write_text(__import__('subprocess').check_output([sys.executable,'-m','pip','freeze'],text=True))
    (OUT/'runtime.json').write_text(json.dumps(dict(platform=__import__('platform').platform(),torch=torch.__version__,device='cpu',dtype='float32'),indent=2))
    db=sqlite3.connect(OUT/'features.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, record TEXT)'); db.execute('CREATE TABLE IF NOT EXISTS pairs (hash TEXT PRIMARY KEY, probabilities TEXT)')
    db.execute('ATTACH DATABASE ? AS development',(str(ALT/'features.sqlite'),))
    done={x[0] for x in db.execute('SELECT id FROM responses')}; pending=[r for r in rows if r['id'] not in done]
    previous=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
    start=time.perf_counter(); n=0; pairs_run=0; truncations=0
    for offset in range(0,len(pending),8):
        group=pending[offset:offset+8]; cache={}; unique={}
        for r in group:
            for c in r['claims']:
                for e in c['evidence']:
                    pair=(e,c['text']); key=hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest()
                    if key in cache or key in unique: continue
                    hit=db.execute('SELECT probabilities FROM pairs WHERE hash=?',(key,)).fetchone()
                    if hit is None: hit=db.execute('SELECT probabilities FROM development.pairs WHERE hash=?',(key,)).fetchone()
                    if hit: cache[key]=json.loads(hit[0])
                    else: unique[key]=pair
        order=sorted(unique,key=lambda k:sum(map(len,unique[k])))
        for b in range(0,len(order),16):
            keys=order[b:b+16]; premises=[unique[k][0] for k in keys]; hypotheses=[unique[k][1] for k in keys]
            truncations+=sum(len(v)>512 for v in tokenizer(premises,hypotheses,truncation=False)['input_ids'])
            inputs=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=512,return_tensors='pt')
            with torch.inference_mode(): probs=model(**inputs).logits.softmax(-1).numpy().tolist()
            pairs_run+=len(keys)
            for key,p in zip(keys,probs): cache[key]=p; db.execute('INSERT INTO pairs VALUES (?,?)',(key,json.dumps(p)))
        for r in group:
            claims=[]
            for c in r['claims']:
                probs=[cache[hashlib.sha256(json.dumps((e,c['text']),ensure_ascii=False).encode()).hexdigest()] for e in c['evidence']]
                claims.append(dict(start=c['start'],end=c['end'],text=c['text'],evidence=c['evidence'],window_indices=c['window_indices'],nli_probabilities=probs,unsupported=1-max(p[1] for p in probs),contradiction=max(p[0] for p in probs)))
            features=[max(c['unsupported'] for c in claims),max(c['contradiction'] for c in claims),float(np.mean([c['unsupported'] for c in claims]))]
            probability=float(expit(np.dot(features,lock['alternative']['coefficients'])+lock['alternative']['intercept']))
            record=dict(id=r['id'],source_id=r['source_id'],gold=r['gold'],fold='test',claims=claims,features=features,probability=probability)
            db.execute('INSERT INTO responses VALUES (?,?)',(r['id'],json.dumps(record)))
        db.commit(); n+=len(group); elapsed=time.perf_counter()-start
        progress=dict(status='test_inference_running',pid=os.getpid(),completed=len(done)+n,total=2675,inference_seconds=previous.get('inference_seconds',0)+elapsed,nli_pairs=previous.get('nli_pairs',0)+pairs_run,truncated_pairs=previous.get('truncated_pairs',0)+truncations,eta_hours=(len(pending)-n)*elapsed/n/3600)
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
    result={x[0]:json.loads(x[1]) for x in db.execute('SELECT id,record FROM responses')}; db.close(); assert len(result)==2675
    y=np.array([r['gold'] for r in rows]); bp=np.array([r['probability'] for r in rows]); ap=np.array([result[r['id']]['probability'] for r in rows]); assert np.isfinite(ap).all()
    bt=lock['baseline']['matched_threshold']; at=lock['alternative']['matched_threshold']
    paired=[dict(id=r['id'],source_id=r['source_id'],gold=r['gold'],baseline_probability=float(b),alternative_probability=float(a)) for r,b,a in zip(rows,bp,ap)]
    (OUT/'paired_test_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in paired))
    summary=dict(baseline_matched=metrics(y,bp,bt),alternative_matched=metrics(y,ap,at),baseline_historical=metrics(y,bp,.29),cost=progress,timing_caveat='Alternative test CPU-only continuation; baseline elapsed spans mixed hardware and folds, so not a controlled speed comparison')
    clusters={}
    for i,r in enumerate(rows): clusters.setdefault(r['source_id'],[]).append(i)
    names=list(clusters); rng=np.random.default_rng(42); differences={k:[] for k in ['precision','recall','auroc','average_precision','brier','ece10','f1']}
    for _ in range(1000):
        ids=np.concatenate([clusters[s] for s in rng.choice(names,len(names),replace=True)])
        if len(np.unique(y[ids]))<2: continue
        b=metrics(y[ids],bp[ids],bt); a=metrics(y[ids],ap[ids],at)
        for k in differences: differences[k].append(a[k]-b[k])
    summary['paired_source_bootstrap']=dict(replicates=len(differences['precision']),source_clusters=len(names),seed=42,alternative_minus_baseline_95ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in differences.items()},limitation='Conditional on fitted models and validation thresholds; no training/threshold-selection uncertainty')
    # Selective risk is classification error among retained most-confident responses.
    with (OUT/'risk_coverage.json').open('w') as f:
        curves={}
        for name,p,t in [('baseline',bp,bt),('alternative',ap,at)]:
            order=np.argsort(-np.abs(p-.5),kind='stable'); errors=((p>=t)!=y)[order]; curves[name]=[dict(coverage=float(c),retained=int(np.ceil(c*len(y))),classification_error=float(errors[:int(np.ceil(c*len(y)))].mean())) for c in np.linspace(.1,1,10)]
        json.dump(dict(confidence='abs(probability-0.5); classification at locked threshold; stable ties',curves=curves),f,indent=2)
    (OUT/'test_metrics.json').write_text(json.dumps(summary,indent=2))
    progress['status']='final_test_complete'; (OUT/'progress.json').write_text(json.dumps(progress,indent=2))
    print('Final test metrics and paired source bootstrap saved.',flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
