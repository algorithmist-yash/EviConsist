"""Separate symmetry guard/certificate wrapper; frozen Candidate C code is untouched."""
from fractions import Fraction as F
import time
import eviconsist_candidate_c_v1_1 as c
from candidate_c_assumption_optimizer_audit_v1 import Recorder,exchange,invariant
from run_candidate_c_local_checks_v1_1 import replay_certificate

VERSION='bounded-C-symmetry-v1'

def guarded_intersection(region,base,points):
    if not invariant(region):
        return None,dict(guard_satisfied=False,applied=False,reason='Exact cell-bound permutation invariance fails')
    if 'lower_exact' not in base:
        return None,dict(guard_satisfied=True,applied=False,reason='No certified global enclosure available')
    if min(region.denominator_range(c.DC)[0],region.denominator_range(c.DE)[0])<=0:
        return None,dict(guard_satisfied=True,applied=False,reason='Required precision denominator may vanish')
    left,right=F(base['lower_exact']),F(base['upper_exact'])
    lo=max(left,-right);hi=min(right,-left)
    assert lo<=hi
    candidates=[]
    for p in points:
        assert region.feasible(p) and region.feasible(exchange(p))
        assert c.contrasts(exchange(p))==tuple(-q if q is not None else None for q in c.contrasts(p))
        candidates.extend([p,exchange(p)])
    values=[c.contrasts(p)[0] for p in candidates]
    assert values and all(v is not None for v in values)
    lower_witness=min(values);upper_witness=max(values)
    gap=max(lower_witness-lo,hi-upper_witness)
    assert gap>=0
    result=dict(base)
    result.update(lower_exact=str(lo),upper_exact=str(hi),lower=c.outward(lo,True),upper=c.outward(hi,False),
        gap=c.outward(gap,False),witness_min=float(lower_witness),witness_max=float(upper_witness),
        status='certified' if gap<=F(1,500) else 'requires_revision',
        reason='Exact symmetry certificate meets original tolerance' if gap<=F(1,500) else 'Symmetry intersection remains outside original tolerance')
    certificate=dict(guard_satisfied=True,applied=True,permutation=[0,2,1,3,4,6,5,7],
        lower_exact=str(lo),upper_exact=str(hi),gap_exact=str(gap),
        witnesses=[[str(q) for q in p] for p in candidates],
        proof='Exact region invariance plus antisymmetric contrast; intersection of global outer enclosures')
    return result,certificate

def evaluate(counts,truep):
    start=time.perf_counter();base=c.from_counts(counts)
    positive_flags=min(base.denominator_range(c.DC)[0],base.denominator_range(c.DE)[0])>0
    region=Recorder(base.lower,base.upper,base.anchor,counts) if positive_flags else base
    audit=c.Audit()
    recall=c.recall_projection(region,audit)
    precision=c.precision_projection(region,audit,max_nodes=512,max_seconds=60)
    original=dict(precision)
    points=[region.min_point,region.max_point] if positive_flags else []
    updated,certificate=guarded_intersection(region,precision,points)
    if updated is not None:precision=updated
    for cert in audit.certificates:replay_certificate(region,cert)
    status='certified' if precision['status']==recall['status']=='certified' else 'non_estimable' if 'non_estimable' in (precision['status'],recall['status']) else 'requires_revision'
    success=status=='certified' and F(precision['lower_exact'])>0 and F(recall['lower_exact'])>F(-1,50)
    truth=c.contrasts(truep);inside=region.feasible(truep)
    containment={};defects=[]
    for i,(name,endpoint) in enumerate([('precision',precision),('recall',recall)]):
        certified=endpoint.get('status')=='certified'
        containment[name]=bool(certified and F(endpoint['lower_exact'])<=truth[i]<=F(endpoint['upper_exact']))
        if certified:
            assert endpoint['gap']<=.002
            if inside and not containment[name]:defects.append(name+' excludes truth despite true-vector membership')
    return dict(wrapper_version=VERSION,status=status,counts=counts,base_precision_projection=original,
        precision=precision,recall=recall,symmetry=certificate,joint_success=success,
        true_probability_vector_in_region=inside,exact_truth=list(map(str,truth)),interval_contains_truth=containment,
        denominator_ranges={k:list(map(str,region.denominator_range(v))) for k,v in [('C_flags',c.DC),('E_flags',c.DE),('Y_positive',c.DY)]},
        hard_defects=defects,lp_certificates=audit.certificates,certificates_replayed=len(audit.certificates),
        lp_calls=audit.lp_calls,seconds=time.perf_counter()-start)

def scientific(r):
    def endpoint(k):return {n:v for n,v in r.get(k,{}).items() if n!='seconds'}
    return dict(status=r['status'],precision=endpoint('precision'),recall=endpoint('recall'),
        symmetry=r.get('symmetry'),joint_success=r.get('joint_success'),
        certificate_models=[x['model_sha256'] for x in r.get('lp_certificates',[])])
