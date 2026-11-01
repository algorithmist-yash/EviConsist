"""Resumable batched execution of the frozen evidence-only baseline.

Inference never reads labels. They are attached only during evaluation.
"""
import argparse, hashlib, json, os, sqlite3, subprocess, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from baseline import ROOT, DATA, read, fold, sentences, evidence, spans

REVISION='fa2804872c3b4bd748f38c0185cc85775361e735'
MODEL='cross-encoder/nli-deberta-v3-small'

def prepare(row, source, top_k):
    claims=sentences(row['response'])
    if not claims: raise ValueError(f'No sentences in response {row["id"]}')
    words=evidence(source).split()
    passages=[' '.join(words[i:i+120]) for i in range(0,len(words),80)] or ['']
    vectorizer=TfidfVectorizer().fit(passages+[c[2] for c in claims])
    similarities=(vectorizer.transform([c[2] for c in claims]) @ vectorizer.transform(passages).T).toarray()
    records=[]
    for (start,end,text),scores in zip(claims,similarities):
        chosen=np.argsort(-scores,kind='stable')[:top_k]
        records.append(dict(start=start,end=end,text=text,evidence=[passages[i] for i in chosen],retrieval_similarity=float(scores[chosen[0]]),window_indices=chosen.tolist()))
    return records

def load_predictions(directory):
    db=sqlite3.connect(directory/'features.sqlite')
    records={key:json.loads(value) for key,value in db.execute('SELECT id, record FROM responses')}
    db.close()
    return records

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/full')
    parser.add_argument('--limit',type=int,default=0,help='0 means all good-quality responses')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    directory=ROOT/args.output; directory.mkdir(parents=True,exist_ok=True)
    if not args.report_only:
        (directory/'process_id.txt').write_text(str(os.getpid()),encoding='utf8')
    rows=read('response.jsonl'); sources={s['source_id']:s for s in read('source_info.jsonl')}
    selected=[]
    for f in ['train','validation','test']:
        group=[r for r in rows if fold(r)==f and r['quality']=='good']
        selected.extend(group if not args.limit else group[:args.limit])
    if args.report_only:
        from experiment_report import generate
        generate(directory,selected,load_predictions(directory)); return
    config=dict(model=MODEL,revision=REVISION,top_k=args.top_k,batch_size=args.batch_size,threads=args.threads,limit_per_fold=args.limit,seed=42,quality='good',validation='sha256(source_id) modulo 5 = 0 within official train',evidence_window_words=120,evidence_stride_words=80,max_tokens=512,features=['max_unsupported','max_contradiction','mean_unsupported'],threshold='maximize validation response F1 on 0.00..1.00 at step 0.01; lowest threshold breaks ties',dataset_sha256={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['response.jsonl','source_info.jsonl']},code_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['baseline.py','full_baseline.py','experiment_report.py']},environment_sha256=hashlib.sha256((ROOT/'requirements.lock.txt').read_bytes()).hexdigest())
    config_path=directory/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Configuration changed; use a new output directory to prevent stale cache reuse.')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf8')
    snapshot=directory/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for filename in ['baseline.py','full_baseline.py','experiment_report.py','requirements.lock.txt']:
        (snapshot/filename).write_bytes((ROOT/filename).read_bytes())
    (directory/'run_command.txt').write_text(f'.\\.venv\\Scripts\\python.exe -X utf8 full_baseline.py --output {args.output} --limit {args.limit} --batch-size {args.batch_size} --threads {args.threads} --top-k {args.top_k}\n',encoding='utf8')
    os.environ['MPLCONFIGDIR']=str(directory/'mpl-cache')
    os.environ['HF_HOME']=str(ROOT/'models')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(args.threads); torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,local_files_only=True).eval()
    # Model card explicitly publishes contradiction, entailment, neutral in this order.
    label_names={str(v).lower():int(k) for k,v in model.config.id2label.items()}
    if set(label_names)=={'label_0','label_1','label_2'}: label_names={'contradiction':0,'entailment':1,'neutral':2}
    if set(label_names)!={'contradiction','entailment','neutral'}: raise ValueError(label_names)
    db=sqlite3.connect(directory/'features.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (hash TEXT PRIMARY KEY, probabilities TEXT NOT NULL)')
    done={r[0] for r in db.execute('SELECT id FROM responses')}
    pending=[r for r in selected if r['id'] not in done]
    start=time.perf_counter(); processed=0; model_pairs=0; truncations=0
    previous=json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else {}
    for offset in range(0,len(pending),8):
        group=pending[offset:offset+8]; records=[]; pairs=[]; locations=[]
        for i,row in enumerate(group):
            records.append(prepare(row,sources[row['source_id']],args.top_k))
            for j,c in enumerate(records[-1]):
                for k,premise in enumerate(c['evidence']):
                    pairs.append((premise,c['text'])); locations.append((i,j,k))
                c['nli_probabilities']=[None]*len(c['evidence'])
        # Group similar lengths to reduce padding; original locations retain order.
        keys=[hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest() for pair in pairs]
        cached={}
        for key in set(keys):
            hit=db.execute('SELECT probabilities FROM pairs WHERE hash=?',(key,)).fetchone()
            if hit: cached[key]=json.loads(hit[0])
        unique={}
        for i,key in enumerate(keys):
            if key not in cached: unique.setdefault(key,i)
        order=sorted(unique.values(),key=lambda i:len(pairs[i][0])+len(pairs[i][1]))
        for batch_start in range(0,len(order),args.batch_size):
            indices=order[batch_start:batch_start+args.batch_size]
            premises=[pairs[i][0] for i in indices]; hypotheses=[pairs[i][1] for i in indices]
            untruncated=tokenizer(premises,hypotheses,add_special_tokens=True,truncation=False)['input_ids']
            truncations+=sum(len(ids)>512 for ids in untruncated)
            inputs=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=512,return_tensors='pt')
            with torch.inference_mode(): scores=model(**inputs).logits.softmax(-1).numpy()
            model_pairs+=len(indices)
            for index,score in zip(indices,scores):
                probabilities={name:float(score[pos]) for name,pos in label_names.items()}
                cached[keys[index]]=probabilities
                db.execute('INSERT INTO pairs VALUES (?,?)',(keys[index],json.dumps(probabilities)))
        for index,key in enumerate(keys):
            i,j,k=locations[index]
            records[i][j]['nli_probabilities'][k]=cached[key]
        for row,claims in zip(group,records):
            for c in claims:
                c['entailment']=max(p['entailment'] for p in c['nli_probabilities'])
                c['contradiction']=max(p['contradiction'] for p in c['nli_probabilities'])
                c['unsupported']=1-c['entailment']
            result=dict(id=row['id'],source_id=row['source_id'],fold=fold(row),model=row['model'],task_type=sources[row['source_id']]['task_type'],claims=claims,features=[max(c['unsupported'] for c in claims),max(c['contradiction'] for c in claims),float(np.mean([c['unsupported'] for c in claims]))])
            db.execute('INSERT INTO responses VALUES (?,?)',(row['id'],json.dumps(result)))
        db.commit(); processed+=len(group)
        elapsed=time.perf_counter()-start
        progress=dict(status='running',completed=len(done)+processed,total=len(selected),seconds_this_session=elapsed,total_inference_seconds=previous.get('total_inference_seconds',0)+elapsed,nli_pairs=previous.get('nli_pairs',0)+model_pairs,truncated_pairs=previous.get('truncated_pairs',0)+truncations,estimated_remaining_hours=(len(pending)-processed)*elapsed/max(processed,1)/3600)
        (directory/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
        print(f'{len(done)+processed}/{len(selected)} responses; {elapsed:.0f}s this session; ETA {progress["estimated_remaining_hours"]:.2f} h',flush=True)
    db.close()
    progress=json.loads((directory/'progress.json').read_text())
    progress['status']='features_complete'; (directory/'progress.json').write_text(json.dumps(progress,indent=2))
    from experiment_report import generate
    generate(directory,selected,load_predictions(directory))

if __name__=='__main__': main()
