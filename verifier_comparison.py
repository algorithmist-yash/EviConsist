"""Resumable verifier-only development experiment; never evaluates test examples."""
import hashlib, json, os, sqlite3, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'results/completed_baseline_2026_10_14/predictions.jsonl'
OUT = ROOT/'results/verifier_base_2026_10_15'
MODEL = 'cross-encoder/nli-deberta-v3-base'
REVISION = '6c749ce3425cd33b46d187e45b92bbf96ee12ec7'

def matched_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    eligible = np.flatnonzero(recall[:-1] >= .90)
    # Maximum precision subject to >=90% recall; highest threshold breaks ties.
    return float(thresholds[max(eligible, key=lambda i: (precision[i], thresholds[i]))])

def metrics(y, p, t):
    precision, recall, f1, _ = precision_recall_fscore_support(y, p >= t, average='binary', zero_division=0)
    ece = 0.
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if mask.any(): ece += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return dict(threshold=t, n=len(y), auroc=roc_auc_score(y,p), average_precision=average_precision_score(y,p), precision=precision, recall=recall, f1=f1, confusion_matrix=confusion_matrix(y,p>=t).tolist(), brier=brier_score_loss(y,p), ece10=ece)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(model=MODEL, revision=REVISION, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), folds=['train','validation'], threads=4, batch_size=16, max_length=512, truncation='only_first', label_order=['contradiction','entailment','neutral'], training='LogisticRegression(random_state=42), defaults', threshold_rule='maximum precision at validation recall >=0.90; highest threshold breaks ties', original_threshold=.29)
    config_path = OUT/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config: raise ValueError('Changed experiment configuration')
    config_path.write_text(json.dumps(config,indent=2))
    (OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    with SOURCE.open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['fold'] in config['folds']: rows.append(row)
    assert len(rows)==14942 and sum(r['fold']=='validation' for r in rows)==2878
    val=[r for r in rows if r['fold']=='validation']
    y=np.array([r['gold'] for r in val]); p=np.array([r['probability'] for r in val])
    (OUT/'baseline_validation.json').write_text(json.dumps(dict(original=metrics(y,p,.29), matched_recall=metrics(y,p,matched_threshold(y,p))),indent=2))
    os.environ['HF_HOME']=str(ROOT/'models')
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    torch.set_num_threads(4); torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION).eval()
    assert model.config.num_labels==3
    db=sqlite3.connect(OUT/'features.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, record TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (hash TEXT PRIMARY KEY, probabilities TEXT)')
    done={r[0] for r in db.execute('SELECT id FROM responses')}
    pending=[r for r in rows if r['id'] not in done]
    old=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
    start=time.perf_counter(); count=0; pairs_run=0; truncated=0
    for offset in range(0,len(pending),8):
        group=pending[offset:offset+8]; cache={}; unique={}
        for r in group:
            for c in r['claims']:
                for evidence in c['evidence']:
                    pair=(evidence,c['text']); key=hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest()
                    hit=db.execute('SELECT probabilities FROM pairs WHERE hash=?',(key,)).fetchone()
                    if hit: cache[key]=json.loads(hit[0])
                    else: unique[key]=pair
        order=sorted(unique,key=lambda k:sum(map(len,unique[k])))
        for b in range(0,len(order),16):
            keys=order[b:b+16]; premises=[unique[k][0] for k in keys]; hypotheses=[unique[k][1] for k in keys]
            truncated+=sum(len(x)>512 for x in tokenizer(premises,hypotheses,truncation=False)['input_ids'])
            inputs=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=512,return_tensors='pt')
            with torch.inference_mode(): probabilities=model(**inputs).logits.softmax(-1).numpy().tolist()
            pairs_run+=len(keys)
            for key, probs in zip(keys,probabilities):
                cache[key]=probs; db.execute('INSERT INTO pairs VALUES (?,?)',(key,json.dumps(probs)))
        for r in group:
            claims=[]
            for c in r['claims']:
                probs=[cache[hashlib.sha256(json.dumps((e,c['text']),ensure_ascii=False).encode()).hexdigest()] for e in c['evidence']]
                claims.append(dict(start=c['start'],end=c['end'],text=c['text'],evidence=c['evidence'],window_indices=c['window_indices'],nli_probabilities=probs,unsupported=1-max(p[1] for p in probs),contradiction=max(p[0] for p in probs)))
            record=dict(id=r['id'],source_id=r['source_id'],fold=r['fold'],gold=r['gold'],claims=claims,features=[max(c['unsupported'] for c in claims),max(c['contradiction'] for c in claims),float(np.mean([c['unsupported'] for c in claims]))])
            db.execute('INSERT INTO responses VALUES (?,?)',(r['id'],json.dumps(record)))
        db.commit(); count+=len(group); elapsed=time.perf_counter()-start
        progress=dict(status='running',pid=os.getpid(),completed=len(done)+count,total=len(rows),inference_seconds=old.get('inference_seconds',0)+elapsed,nli_pairs=old.get('nli_pairs',0)+pairs_run,truncated_pairs=old.get('truncated_pairs',0)+truncated,eta_hours=(len(pending)-count)*elapsed/count/3600)
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
    records=[json.loads(r[0]) for r in db.execute('SELECT record FROM responses')]; db.close()
    train=[r for r in records if r['fold']=='train']; val=[r for r in records if r['fold']=='validation']
    lr=LogisticRegression(random_state=42).fit([r['features'] for r in train],[r['gold'] for r in train])
    p=lr.predict_proba([r['features'] for r in val])[:,1]; y=np.array([r['gold'] for r in val]); t=matched_threshold(y,p)
    result=dict(validation=metrics(y,p,t),coefficients=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),test_status='not evaluated')
    (OUT/'validation_metrics.json').write_text(json.dumps(result,indent=2))
    with (OUT/'validation_predictions.jsonl').open('w') as f:
        for r,prob in zip(val,p): f.write(json.dumps(dict(id=r['id'],source_id=r['source_id'],gold=r['gold'],probability=float(prob)))+'\n')
    progress=json.loads((OUT/'progress.json').read_text()); progress['status']='validation_complete_test_held_out'; (OUT/'progress.json').write_text(json.dumps(progress,indent=2))

if __name__=='__main__': main()
