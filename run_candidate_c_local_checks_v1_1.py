"""Five prospective deterministic feasibility fixtures; never random screening."""
import contextlib
from fractions import Fraction as F
import hashlib
import io
import json
import platform
from pathlib import Path
import time
import unittest
import numpy as np
import scipy
import eviconsist_candidate_c_v1_1 as c
import test_candidate_c_v1_1 as tests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/candidate_c_local_v1_1_2026_10_18'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def replay_certificate(region,cert):
    context=cert['context']
    phase_one=isinstance(context,dict) and 'phase_one_for' in context
    if phase_one: context=context['phase_one_for']
    if isinstance(context,str):
        A,b,H,d,lo,hi=c.transformed(region,c.DY)
        obj=(c.NR if context=='recall_min' else [-q for q in c.NR])+[F(0)]
    else:
        a,bound=map(F,context['ratio_interval'])
        A,b,H,d,lo,hi=c.transformed(region,c.DC)
        A.append([a*q for q in c.DE]+[F(0)]);b.append(F(1))
        A.append([-bound*q for q in c.DE]+[F(0)]);b.append(F(-1))
        obj=([tc-bound*te for tc,te in zip(c.TC,c.TE)] if context['precision']=='min' else [-tc+a*te for tc,te in zip(c.TC,c.TE)])+[F(0)]
    if phase_one: obj,A,b,H,d,lo,hi=c.phase_one_model(A,b,H,d,lo,hi)
    model=dict(c=list(map(str,obj)),A=[[str(q) for q in r] for r in A],b=list(map(str,b)),H=[[str(q) for q in r] for r in H],
        d=list(map(str,d)),lower=list(map(str,lo)),upper=list(map(str,hi)))
    assert hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()==cert['model_sha256']
    bound=c.exact_dual_lower(obj,A,b,H,d,lo,hi,list(map(F,cert['inequality_duals'])),list(map(F,cert['equality_duals'])))
    assert str(bound)==cert['lower_exact']

def deterministic_truth_checks(region,result):
    # Predetermined perturbations around the empirical anchor, preserving each simplex.
    points=[region.anchor]
    for task in range(3):
        for i in range(8):
            for j in range(8):
                if i==j:continue
                a,b=8*task+i,8*task+j
                delta=min(region.upper[a]-region.anchor[a],region.anchor[b]-region.lower[b])/2
                if delta<=0:continue
                p=region.anchor.copy();p[a]+=delta;p[b]-=delta
                assert region.feasible(p);points.append(p)
    for p in points:
        truth=tests.direct(p)
        for k,key in enumerate(('precision','recall')):
            endpoint=result.get(key,{})
            if 'lower_exact' in endpoint:
                assert F(endpoint['lower_exact'])<=truth[k]<=F(endpoint['upper_exact']), (key,truth,endpoint)
    return len(points)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    protocol=json.loads((ROOT/'EVICONSIST_CANDIDATE_C_LOCAL_CHECK_PROTOCOL_v1.json').read_text(encoding='utf8'))
    frozen=[ROOT/'EVICONSIST_STATISTICAL_ANALYSIS_PLAN_v1.md',ROOT/'EVICONSIST_STATISTICAL_METHOD_REVISION_v1.md',
        ROOT/'EVICONSIST_SYNTHETIC_SCREENING_REPORT_v1.md',ROOT/'EVICONSIST_STATISTICAL_METHOD_DECISION_MEMO_v1.md',
        ROOT/'eviconsist_candidate_intervals_v1.py',ROOT/'candidate_interval_reference_v1.py',ROOT/'eviconsist_statistics.py',ROOT/'paper/main.tex']
    frozen_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in frozen}
    inputs=['EVICONSIST_CANDIDATE_C_METHOD_SPEC.md','EVICONSIST_CANDIDATE_C_LOCAL_CHECK_PROTOCOL_v1.json',
        'eviconsist_candidate_c_v1_1.py','test_candidate_c_v1_1.py','run_candidate_c_local_checks_v1_1.py',
        'EVICONSIST_CANDIDATE_C_IMPLEMENTATION_NOTE_v1_1.md']
    hashes={name:sha(ROOT/name) for name in inputs}
    save('PREFLIGHT_FREEZE.json',dict(input_sha256=hashes,historical_input_sha256=frozen_hashes,
        protocol=protocol,seeds=None,random_simulations=0,created_before_fixture_results=True,
        environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)))
    log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    testresult=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/'DETERMINISTIC_TEST_LOG.txt').write_text(log.getvalue(),encoding='utf8')
    save('TEST_SUMMARY.json',dict(tests=testresult.testsRun,failures=len(testresult.failures),errors=len(testresult.errors),passed=testresult.wasSuccessful()))
    if not testresult.wasSuccessful():
        save('DECISION.json',dict(decision='requires revision',reason='deterministic test failure; no feasibility expansion'))
        return
    proportions=list(map(F,protocol['ordinary_fixture_proportions']))
    fixtures=[(f'ordinary_n{n}_per_task',[[int(n*p) for p in proportions]]*3) for n in protocol['feasibility_counts_per_task']]
    fixtures += [('zero_disagreement_n40_per_task',[protocol['zero_disagreement_counts_per_task']]*3),
                 ('sparse_n40_per_task',[protocol['sparse_counts_per_task']]*3)]
    results=[]
    for name,counts in fixtures:
        if time.perf_counter()-start>protocol['local_total_runtime_cap_seconds']:
            results.append(dict(fixture=name,status='not_run_budget_exhausted',counts=counts,joint_success=False));continue
        print('Starting '+name,flush=True)
        r,audit=c.evaluate(counts,max_nodes=protocol['precision_max_nodes'],max_seconds=protocol['precision_max_seconds'])
        r['fixture']=name
        region=c.from_counts(counts)
        for cert in audit.certificates:replay_certificate(region,cert)
        r['replayed_lp_certificates']=len(audit.certificates)
        r['deterministic_true_vectors_checked']=deterministic_truth_checks(region,r)
        save(name+'_CERTIFICATES.json',dict(fixture=name,counts=counts,certificates=audit.certificates))
        save(name+'_RESULT.json',r);results.append(r)
        print(json.dumps({k:v for k,v in r.items() if k not in ('counts',)}),flush=True)
    unchanged={name:sha(ROOT/name)==h for name,h in frozen_hashes.items()}
    assert all(unchanged.values())
    assert all(sha(ROOT/name)==h for name,h in hashes.items())
    passed=all(r['status']=='certified' for r in results)
    decision=dict(decision='ready for prespecified screening' if passed else 'requires revision',
        meaning='Local implementation/feasibility gate only; no statistical adoption or detector effectiveness claim.',
        fixtures=len(results),certified=sum(r['status']=='certified' for r in results),
        seconds=time.perf_counter()-start,random_outer_studies=0,models_run=False,frozen_historical_unchanged=unchanged,
        total_replayed_lp_certificates=sum(r.get('replayed_lp_certificates',0) for r in results),
        total_deterministic_true_vectors_checked=sum(r.get('deterministic_true_vectors_checked',0) for r in results),
        limitations=['Independent-source sampling assumptions remain unestablished for any future confirmation frame.',
                    'Wide intervals may make +0.03 precision and -0.02 recall requirements impractical.',
                    'No random screening, power analysis or independent external method approval has occurred.'])
    save('FEASIBILITY_RESULTS.json',results);save('DECISION.json',decision)
    print(json.dumps(decision),flush=True)

if __name__=='__main__':main()
