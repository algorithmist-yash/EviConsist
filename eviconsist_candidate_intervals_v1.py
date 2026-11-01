"""Revision-v1 candidate implementation only; NOT approved for study inference.

Columns per group: n, positives, TP_E, flags_E, TP_C, flags_C.
Source groups canonicalized to one task; all rows within a group travel together.
"""
import numpy as np
from scipy.stats import t as student_t

TASKS=('Summary','QA','Data2txt')

def aggregate(rows):
    by={t:{} for t in TASKS}; seen={}
    for r in rows:
        task=r['task'];g=str(r['group'])
        if task not in by or any(r[k] not in (0,1) for k in ('y','e','c')): raise ValueError('Invalid task or binary flag')
        if g in seen and seen[g]!=task: raise ValueError('Cross-task group')
        seen[g]=task;by[task].setdefault(g,np.zeros(6))
        y,e,c=r['y'],r['e'],r['c'];by[task][g]+=np.array([1,y,y*e,e,y*c,c])
    return [np.array([by[t][g] for g in sorted(by[t])],dtype=float).reshape(-1,6) for t in TASKS]

def estimate(groups):
    """Batched weighted-ratio estimate and prespecified linearized variance.

    Each task array has shape (..., groups_in_task, 6); leading axes agree.
    """
    if len(groups)!=3: raise ValueError('Three tasks required')
    if any(a.shape[-1]!=6 or not np.isfinite(a).all() or (a<0).any() for a in groups): raise ValueError('Invalid sufficient counts')
    if any(a.shape[-2]<2 for a in groups): return None
    totals=[a.sum(axis=-2) for a in groups]
    if any((s[...,0]<=0).any() for s in totals): return None
    means=sum(s[...,1:]/s[...,0,None]/3 for s in totals)
    # rows: E precision, E recall, C precision, C recall.
    numerator=means[...,[1,1,3,3]]; denominator=means[...,[2,0,4,0]]
    with np.errstate(divide='ignore',invalid='ignore'):
        ratios=numerator/denominator
        contribution=[];vt=[]
        for a,s in zip(groups,totals):
            u=a[...,[2,2,4,4]]-ratios[...,None,:]*a[...,[3,1,5,1]]
            centered=u-a[...,0,None]/s[...,0,None,None]*u.sum(axis=-2)[...,None,:]
            h=centered/(3*s[...,0,None,None]*denominator[...,None,:])
            delta_h=h[...,2:]-h[...,:2]
            contribution.append(delta_h)
            g=a.shape[-2];vt.append(g/(g-1)*(delta_h**2).sum(axis=-2))
        variance=sum(vt);dfden=sum(v**2/(a.shape[-2]-1) for a,v in zip(groups,vt));df=variance**2/dfden
    d=ratios[...,2:]-ratios[...,:2];se=np.sqrt(variance)
    valid=np.isfinite(d).all(axis=-1)&np.isfinite(se).all(axis=-1)&(se>0).all(axis=-1)&np.isfinite(df).all(axis=-1)&(df>0).all(axis=-1)&(denominator>0).all(axis=-1)
    return dict(point=d,se=se,variance=variance,df=df,valid=valid,contributions=contribution)

def failure(reason,**extra): return dict(estimable=False,reason=reason,lower=None,upper=None,**extra)

def candidate_b(rows):
    result=estimate(aggregate(rows))
    if result is None or not bool(result['valid']): return failure('undefined_original_ratio_variance_or_df')
    q=student_t.ppf(.975,result['df']);d=result['point'];s=result['se']
    return dict(estimable=True,point=d.tolist(),se=s.tolist(),df=result['df'].tolist(),lower=(d-q*s).tolist(),upper=(d+q*s).tolist())

def bootstrap_indices(groups,keys,index):
    # Independent per-replicate keys make batching mathematically irrelevant.
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261020,spawn_key=tuple(keys)+(index,))))
    return [rng.integers(0,len(a),size=len(a)) for a in groups]

def candidate_a(rows,replicates=9999,keys=(0,0,0,0),chunk_size=64,allow_full_fixture=False):
    if replicates<1 or chunk_size<1 or (chunk_size>64 and not allow_full_fixture): raise ValueError('Positive count; production chunk <=64')
    groups=aggregate(rows);base=estimate(groups)
    if base is None or not bool(base['valid']): return failure('undefined_original_ratio_variance_or_df',undefined_replicates=0)
    pivots=np.empty((replicates,2));undefined=0
    for start in range(0,replicates,chunk_size):
        stop=min(start+chunk_size,replicates)
        indices=[bootstrap_indices(groups,keys,i) for i in range(start,stop)]
        sampled=[np.stack([a[ix[t]] for ix in indices]) for t,a in enumerate(groups)]
        result=estimate(sampled)
        if result is None: undefined+=stop-start;continue
        valid=result['valid'];undefined+=int((~valid).sum())
        with np.errstate(divide='ignore',invalid='ignore'): pivots[start:stop]=(result['point']-base['point'])/result['se']
    if undefined: return failure('undefined_required_studentized_replicate',undefined_replicates=undefined,replicates=replicates)
    q=np.quantile(pivots,[.025,.975],axis=0,method='linear');d=base['point'];s=base['se']
    return dict(estimable=True,point=d.tolist(),se=s.tolist(),lower=(d-q[1]*s).tolist(),upper=(d-q[0]*s).tolist(),undefined_replicates=0,replicates=replicates)
