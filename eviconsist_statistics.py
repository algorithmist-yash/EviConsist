"""SAP-v1 primary paired ratios and planning simulation; no detector dependencies.

Input rows: group, task, y, e, c. Flags are locked binary predictions.
Repeated outer group draws receive new instance identities for inner bootstrap.
"""
import math
import numpy as np
TASKS=('Summary','QA','Data2txt')

def validate(rows):
    seen={}
    for r in rows:
        if r['task'] not in TASKS or any(r[k] not in (0,1) for k in ('y','e','c')): raise ValueError('Invalid task/binary endpoint')
        if r['group'] in seen and seen[r['group']]!=r['task']: raise ValueError('Cross-task group must be canonicalized')
        seen[r['group']]=r['task']

def difference(rows):
    counts={t:sum(r['task']==t for r in rows) for t in TASKS}
    if not all(counts.values()): return None
    answer=[]
    for m in ('e','c'):
        tp=fp=fn=0.
        for r in rows:
            w=1/(3*counts[r['task']]); y=r['y']; z=r[m]
            tp+=w*y*z; fp+=w*(1-y)*z; fn+=w*y*(1-z)
        if tp+fp==0 or tp+fn==0: return None
        answer.append(np.array([tp/(tp+fp),tp/(tp+fn)]))
    return answer[1]-answer[0]

def grouped(rows):
    validate(rows); result={t:{} for t in TASKS}
    for r in rows: result[r['task']].setdefault(str(r['group']),[]).append(r)
    return {t:[g[k] for k in sorted(g)] for t,g in result.items()}

def draw(groups,rng,counts=None):
    output=[]
    for t in TASKS:
        g=groups[t]; n=len(g) if counts is None else counts[t]
        if not g or n<1: return []
        for instance,index in enumerate(rng.integers(0,len(g),n)):
            output.extend(dict(r,group=f'{t}:{instance}') for r in g[index])
    return output

def intervals(rows,replicates=9999,seed=20261016,rng=None):
    if replicates<1: raise ValueError('Positive replicate count required')
    groups=grouped(rows); point=difference(rows)
    if point is None: return dict(estimable=False,undefined_original=True,undefined_replicates=0,lower=None,upper=None)
    rng=np.random.Generator(np.random.PCG64(seed)) if rng is None else rng
    # Aggregate whole-group sufficient counts, then draw paired group indices.
    total=np.zeros((replicates,5))
    for t in TASKS:
        a=np.array([[len(g),sum(r['y'] for r in g),sum(r['y']*r['e'] for r in g),sum(r['e'] for r in g),sum(r['y']*r['c'] for r in g),sum(r['c'] for r in g)] for g in groups[t]],dtype=float)
        sampled=a[rng.integers(0,len(a),size=(replicates,len(a)))].sum(axis=1)
        total+=sampled[:,1:]/sampled[:,0,None]/3
    defined=(total[:,0]>0)&(total[:,2]>0)&(total[:,4]>0)
    bad=int((~defined).sum())
    if bad: return dict(estimable=False,undefined_original=False,undefined_replicates=bad,lower=None,upper=None)
    values=np.column_stack((total[:,3]/total[:,4]-total[:,1]/total[:,2],(total[:,3]-total[:,1])/total[:,0]))
    limits=np.quantile(values,[.025,.975],axis=0,method='linear')
    return dict(estimable=True,point=point.tolist(),undefined_replicates=0,lower=limits[0].tolist(),upper=limits[1].tolist())

def success(lower): return lower is not None and lower[0]>0 and lower[1]>-.02

def wilson(k,n):
    z=1.959963984540054; p=k/n; a=1+z*z/n
    center=(p+z*z/(2*n))/a; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [center-half,center+half]

def select_threshold(rows,score_key,target_recall=.90):
    """Validation-only weighted precision maximization; highest threshold tie."""
    n={t:sum(r['task']==t for r in rows) for t in TASKS}
    if not all(n.values()): raise ValueError('Missing task')
    scores=np.array([r[score_key] for r in rows],dtype=float)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite score')
    w=np.array([1/(3*n[r['task']]) for r in rows]); y=np.array([r['y'] for r in rows]); positives=(w*y).sum()
    if positives==0: raise ValueError('No positives')
    choices=[]
    for t in np.unique(scores):
        z=scores>=t; tp=(w*y*z).sum(); flags=(w*z).sum(); recall=tp/positives
        if recall>=target_recall: choices.append((tp/flags,float(t),float(recall)))
    p,t,r=max(choices)
    return dict(threshold=t,precision=float(p),recall=r)

def power(rows,counts,outer=2000,inner=1999,seed=20261017,theta=None):
    """theta=None: empirical effect; otherwise SAP local location shift."""
    groups=grouped(rows); reference=difference(rows)
    if reference is None: raise ValueError('Undefined planning reference')
    rng=np.random.Generator(np.random.PCG64(seed)); joint=0; marginal=np.zeros(2,dtype=int); bad=0
    for _ in range(outer):
        sampled=draw(groups,rng,counts); result=intervals(sampled,inner,rng=rng)
        if not result['estimable']: bad+=1; continue
        low=np.array(result['lower'])
        if theta is not None: low+=np.asarray(theta)-reference
        passed=low>np.array([0.,-.02]); marginal+=passed; joint+=int(passed.all())
    p=joint/outer
    return dict(joint_power=p,marginal_pass=(marginal/outer).tolist(),undefined=bad,outer=outer,inner=inner,mcse=math.sqrt(p*(1-p)/outer),wilson95=wilson(joint,outer),seed=seed,method='empirical' if theta is None else 'local location-shift approximation',theta=theta)
