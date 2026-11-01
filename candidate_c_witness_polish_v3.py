"""Deterministic exact feasible-witness sharpening; outer certificates are unchanged."""
from fractions import Fraction as F
import eviconsist_candidate_c_v1_1 as c
from candidate_c_assumption_optimizer_audit_v1 import exchange,invariant

def oracle(region,g):
    q=region.lower.copy()
    for task in range(3):
        ids=list(range(8*task,8*task+8));remaining=1-sum(q[k] for k in ids)
        for k in sorted(ids,key=lambda k:(g[k],-k),reverse=True):
            add=min(remaining,region.upper[k]-q[k]);q[k]+=add;remaining-=add
        assert remaining==0
    assert region.feasible(q)
    return q

def improve(region,points,iterations=32):
    """No optimization package, randomness, new LPs or uncertified floating witnesses."""
    candidates=[p.copy() for p in points if region.feasible(p)]
    if invariant(region):candidates += [exchange(p) for p in candidates.copy()]
    best=max(candidates,key=lambda p:c.contrasts(p)[0]);initial=c.contrasts(best)[0]
    for _ in range(iterations):
        dc,de=c.dot(c.DC,best),c.dot(c.DE,best)
        if not dc or not de:break
        tc,te=c.dot(c.TC,best),c.dot(c.TE,best)
        gradient=[a/dc-tc*b/dc**2-d/de+te*e/de**2 for a,b,d,e in zip(c.TC,c.DC,c.TE,c.DE)]
        q=oracle(region,gradient)
        choices=[best]+[[p+t*(v-p) for p,v in zip(best,q)] for t in [F(1),F(1,2),F(1,4),F(1,8),F(1,16)]]
        assert all(region.feasible(p) for p in choices)
        updated=max(choices,key=lambda p:c.contrasts(p)[0])
        if c.contrasts(updated)[0]<=c.contrasts(best)[0]:break
        best=updated
    return best,dict(initial_exact=str(initial),improved_exact=str(c.contrasts(best)[0]),
        iterations_limit=iterations,feasible=region.feasible(best))

def sharpen_saved(result):
    """Review an immutable saved result. No re-solving or synthetic data generation."""
    from candidate_c_bounded_wrapper_v1 import guarded_intersection
    region=c.from_counts(result['counts'])
    if not invariant(region) or not result.get('symmetry',{}).get('applied'):
        return None,dict(applicable=False,reason='Exact symmetry certificate required')
    points=[[F(q) for q in p] for p in result['symmetry']['witnesses']]
    best,audit=improve(region,points)
    updated,certificate=guarded_intersection(region,result['base_precision_projection'],points+[best])
    return updated,dict(applicable=True,polish=audit,certificate=certificate,
        original_status=result['status'],original_gap=result['precision']['gap'],
        outer_bounds_unchanged=(updated['lower_exact']==result['precision']['lower_exact'] and updated['upper_exact']==result['precision']['upper_exact']))
