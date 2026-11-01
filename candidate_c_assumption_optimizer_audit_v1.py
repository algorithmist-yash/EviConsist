"""Deterministic source-unit and symmetry-certificate audit of one saved case."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from collections import Counter,defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time
import zlib
from threadpoolctl import threadpool_limits
import eviconsist_candidate_c_v1_1 as c
from run_candidate_c_local_checks_v1_1 import replay_certificate

ROOT=Path(__file__).resolve().parent
PRE=ROOT/'results/candidate_c_screening_preflight_v1_2026_10_18'
OUT=ROOT/'results/candidate_c_assumption_optimizer_audit_v1_2026_10_18'
PERM=(0,2,1,3,4,6,5,7)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def exchange(p):return [p[8*t+PERM[j]] for t in range(3) for j in range(8)]
def invariant(region):return exchange(region.lower)==region.lower and exchange(region.upper)==region.upper

class Recorder(c.Region):
    def __post_init__(self):
        super().__post_init__()
        self.witness_min=self.witness_max=c.contrasts(self.anchor)[0]
        self.min_point=self.max_point=self.anchor.copy()
        self.witness_calls=0

    def repair_witness(self,z):
        p=super().repair_witness(z);value=c.contrasts(p)[0];self.witness_calls+=1
        if value<self.witness_min:self.witness_min=value;self.min_point=p
        if value>self.witness_max:self.witness_max=value;self.max_point=p
        return p

def freeze():
    OUT.mkdir(parents=True,exist_ok=False)
    names=['EVICONSIST_CANDIDATE_C_ASSUMPTION_OPTIMIZER_AUDIT_PLAN_v1.json','candidate_c_assumption_optimizer_audit_v1.py',
        'EVICONSIST_CANDIDATE_C_METHOD_SPEC.md','EVICONSIST_CANDIDATE_C_SCREENING_PROTOCOL_v1.md',
        'EVICONSIST_CANDIDATE_C_FEASIBILITY_REPORT_v1.md','EVICONSIST_CANDIDATE_C_SCREENING_DECISION_v1.md',
        'EVICONSIST_STATISTICAL_ANALYSIS_PLAN_v1.md','EVICONSIST_STATISTICAL_METHOD_DECISION_MEMO_v1.md',
        'EVICONSIST_SYNTHETIC_SCREENING_REPORT_v1.md','eviconsist_candidate_c_v1_1.py','eviconsist_candidate_c_v1.py',
        'eviconsist_screening_scenarios_v1.py','eviconsist_candidate_intervals_v1.py','candidate_interval_reference_v1.py','paper/main.tex']
    paths=[ROOT/n for n in names]+[PRE/n for n in ('SCENARIOS.json','LOCK.json','PREFLIGHT.sqlite','PREFLIGHT_REPORT.json','DELIVERABLE_MANIFEST.json')]
    save('LOCK.json',dict(date='2026-10-18',status='frozen before deterministic replay/sharpening result',
        python=platform.python_version(),random_draws=0,files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},
        plan=load(ROOT/names[0])))
    (OUT/'LOCK.sha256').write_text(sha(OUT/'LOCK.json')+'\n',encoding='utf8')
    print('Audit plan and code frozen. No simulation or optimization executed.')

def locked():
    assert sha(OUT/'LOCK.json')==(OUT/'LOCK.sha256').read_text().strip()
    lock=load(OUT/'LOCK.json')
    for name,h in lock['files'].items():assert sha(ROOT/name)==h,name
    return lock

def run():
    lock=locked();start=time.perf_counter()
    db=sqlite3.connect((PRE/'PREFLIGHT.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    matrix=[];fixture=None
    for sid,blob,payload in db.execute('select scenario,rows_zlib,record from studies order by scenario'):
        record=json.loads(payload);rows=json.loads(zlib.decompress(blob))
        if sid=='recall_boundary_H5':fixture=record
        if record['compatible']:continue
        counts={}
        for task in c.TASKS:
            taskrows=[r for r in rows if r['task']==task]
            groups=Counter(r['group'] for r in taskrows)
            counts[task]=dict(response_rows=len(taskrows),true_groups=len(groups),
                nominal_sources=len({r['child_source'] for r in taskrows}),group_size_frequency=dict(sorted(Counter(groups.values()).items())))
        matrix.append(dict(scenario=sid,rows=len(rows),true_groups=len({r['group'] for r in rows}),
            nominal_sources=len({r['child_source'] for r in rows}),per_task=counts,
            inference_status=record['status'],coverage_proof_applicable=False))
    db.close();assert fixture is not None and len(matrix)==9
    save('NINE_SCENARIO_MATRIX.json',matrix)
    counts=fixture['counts'];base=c.from_counts(counts)
    assert invariant(base)
    denominators={name:base.denominator_range(coef) for name,coef in [('C_flags',c.DC),('E_flags',c.DE),('positive_label',c.DY)]}
    assert all(lo>0 for lo,hi in denominators.values())
    # A fixed negative control checks that we do not infer detector interchangeability.
    asymmetric=[row.copy() for row in counts];asymmetric[0][0]-=1;asymmetric[0][1]+=1
    assert not invariant(c.from_counts(asymmetric))
    for p in (base.anchor,exchange(base.anchor)):
        assert base.feasible(exchange(p))
        assert c.contrasts(exchange(p))==tuple(-q for q in c.contrasts(p))
    for cert in fixture['certificates']:replay_certificate(base,cert)
    region=Recorder(base.lower,base.upper,base.anchor,counts)
    audit=c.Audit()
    with threadpool_limits(limits=1):
        original_replay=c.precision_projection(region,audit,max_nodes=512,max_seconds=60)
    for cert in audit.certificates:replay_certificate(region,cert)
    # Sharpen the saved envelope using a proved exact set symmetry, not a local optimum.
    old=fixture['precision'];left,right=F(old['lower_exact']),F(old['upper_exact'])
    sharpened_left=max(left,-right);sharpened_right=min(right,-left)
    points=[region.min_point,region.max_point,exchange(region.min_point),exchange(region.max_point)]
    for p in points:
        assert region.feasible(p)
        assert c.contrasts(exchange(p))==tuple(-q for q in c.contrasts(p))
    values=[c.contrasts(p)[0] for p in points]
    feasible_min,feasible_max=min(values),max(values)
    gap=max(feasible_min-sharpened_left,sharpened_right-feasible_max)
    assert gap>=0
    certified=gap<=F(1,500)
    symmetry=dict(status='certified_to_original_tolerance' if certified else 'unresolved',
        reason='Region invariant under E/C permutation; contrast antisymmetric; intersect [L,U] with [-U,-L]',
        lower_exact=str(sharpened_left),upper_exact=str(sharpened_right),
        lower=c.outward(sharpened_left,True),upper=c.outward(sharpened_right,False),
        extremum_gap_exact=str(gap),extremum_gap=c.outward(gap,False),required_gap=0.002,
        feasible_min_exact=str(feasible_min),feasible_max_exact=str(feasible_max),
        feasible_witnesses=[[str(q) for q in p] for p in points],permutation=list(PERM),
        symmetry_guard=True,asymmetric_negative_control_rejected=True)
    save('SPARSE_SYMMETRY_CERTIFICATE.json',symmetry)
    # Hand example: group A one cell-7 response, group B four cell-5 and four cell-2.
    group_a=[F(0)]*8;group_a[7]=F(1)
    group_b=[F(0)]*8;group_b[5]=group_b[2]=F(4)
    response_joint=[(a+b)/9 for a,b in zip(group_a,group_b)]*3
    group_joint=[(a+b/8)/2 for a,b in zip(group_a,group_b)]*3
    response_metrics=c.contrasts(response_joint);group_metrics=c.contrasts(group_joint)
    assert response_metrics==(F(4,5),F(4,5))
    assert group_metrics==(F(1,3),F(1,3))
    save('CLUSTER_WEIGHTING_COUNTEREXAMPLE.json',dict(group_sizes=[1,8],group_counts=[list(map(str,group_a)),list(map(str,group_b))],
        response_weighted_differences=list(map(str,response_metrics)),equal_group_weighted_differences=list(map(str,group_metrics))))
    save('SPARSE_REPRODUCTION.json',dict(counts=counts,original_saved_projection=old,deterministic_replay=original_replay,
        scientific_endpoints_reproduced=all(original_replay.get(k)==old.get(k) for k in ('lower_exact','upper_exact','gap','nodes','status')),
        denominators={k:list(map(str,v)) for k,v in denominators.items()},
        saved_certificates_replayed=len(fixture['certificates']),new_precision_certificates_replayed=len(audit.certificates),
        precision_lp_calls=audit.lp_calls,exact_feasible_witness_calls=region.witness_calls,
        original_node_cap=512,original_time_cap_seconds=60,criterion_changed=False,
        precision_certificates=audit.certificates,recall=fixture['recall']))
    elapsed=time.perf_counter()-start
    assert elapsed<lock['plan']['audit_total_budget_seconds']
    locked()
    save('AUDIT_CHECKS.json',dict(random_draws=0,new_response_rows=0,scenario_matrix_rows=9,
        symmetry_certificate_status=symmetry['status'],exact_gap=symmetry['extremum_gap'],
        elapsed_seconds=elapsed,historical_hashes_unchanged=True,group_weighting_example_verified=True,
        recommendation_eligibility='independent-unit prospective option only; actual source frame still unapproved'))
    print(json.dumps(dict(symmetry_status=symmetry['status'],bounds=[symmetry['lower'],symmetry['upper']],gap=symmetry['extremum_gap'],
        precision_replay_status=original_replay['status'],original_endpoints_reproduced=all(original_replay.get(k)==old.get(k) for k in ('lower_exact','upper_exact','nodes','status')),
        elapsed_seconds=elapsed),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:run()
