import itertools,unittest
import numpy as np
from scipy.stats import t
import eviconsist_candidate_intervals_v1 as candidate
import candidate_interval_reference_v1 as reference

def fixture(repeats=1,unequal=False):
    rows=[]
    for task in candidate.TASKS:
        for repeat in range(repeats):
            for i,(y,e,c) in enumerate(itertools.product((0,1),repeat=3)):
                rows.extend(dict(group=f'{task}-{repeat:03}-{i}',task=task,y=y,e=e,c=c) for _ in range(1+i%3 if unequal else 1))
    return rows

class CandidateChecks(unittest.TestCase):
    def test_hand_variance(self):
        r=candidate.estimate(candidate.aggregate(fixture()))
        np.testing.assert_allclose(r['point'],[0,0],atol=1e-15)
        np.testing.assert_allclose(r['variance'],[1/42,1/21],rtol=1e-13)
        np.testing.assert_allclose(r['df'],[21,21],rtol=1e-13)
    def test_independent_reference(self):
        for unequal in (False,True):
            rows=fixture(2,unequal);r=candidate.estimate(candidate.aggregate(rows));ref=reference.scalar_variance(rows)
            np.testing.assert_allclose(r['point'],reference.contrast(rows),atol=2e-14)
            for key in ('variance','se','df'): np.testing.assert_allclose(r[key],ref[key],rtol=1e-12)
    def test_unequal_task_sizes_and_group_sizes(self):
        rows=fixture(3,True);rows=[r for r in rows if r['task']!='QA' or '-000-' in r['group']]
        rr=reference.scalar_variance(rows);out=candidate.estimate(candidate.aggregate(rows))
        np.testing.assert_allclose(out['point'],reference.contrast(rows),atol=2e-14)
        for key in ('variance','se','df'):np.testing.assert_allclose(out[key],rr[key],rtol=1e-12)
    def test_finite_difference_all_group_contributions(self):
        rows=fixture(2,True);arrays=candidate.aggregate(rows);r=candidate.estimate(arrays);eps=1e-5
        for index,task in enumerate(candidate.TASKS):
            ids=sorted({x['group'] for x in rows if x['task']==task})
            for j,g in enumerate(ids):
                plus=np.array(reference.contrast(rows,{g:1+eps}));minus=np.array(reference.contrast(rows,{g:1-eps}))
                np.testing.assert_allclose((plus-minus)/(2*eps),r['contributions'][index][j],rtol=1e-7,atol=1e-10)
    def test_contribution_centering(self):
        r=candidate.estimate(candidate.aggregate(fixture(2,True)))
        for h in r['contributions']: np.testing.assert_allclose(h.sum(axis=0),0,atol=1e-15)
    def test_candidate_b_interval_formula(self):
        rows=fixture(2,True);rr=reference.scalar_variance(rows);d=np.array(reference.contrast(rows));out=candidate.candidate_b(rows)
        q=t.ppf(.975,rr['df']);np.testing.assert_allclose(out['lower'],d-q*np.array(rr['se']),rtol=1e-12)
    def test_candidate_a_independent_row_reference(self):
        rows=fixture(4,True);arrays=candidate.aggregate(rows);base=np.array(reference.contrast(rows));se=np.array(reference.scalar_variance(rows)['se']);pivots=[]
        by={task:[] for task in candidate.TASKS}
        for task in candidate.TASKS:
            for g in sorted({r['group'] for r in rows if r['task']==task}):by[task].append([r for r in rows if r['group']==g])
        for index in range(31):
            indices=candidate.bootstrap_indices(arrays,(0,0,0,0),index);sampled=[]
            for task,ix in zip(candidate.TASKS,indices):
                for instance,original in enumerate(ix):sampled.extend(dict(r,group=f'{task}:{instance}') for r in by[task][original])
            d=np.array(reference.contrast(sampled));s=np.array(reference.scalar_variance(sampled)['se']);pivots.append((d-base)/s)
        q=np.quantile(pivots,[.025,.975],axis=0,method='linear');out=candidate.candidate_a(rows,31)
        self.assertTrue(out['estimable']);np.testing.assert_allclose(out['lower'],base-q[1]*se,rtol=1e-12,atol=1e-14);np.testing.assert_allclose(out['upper'],base-q[0]*se,rtol=1e-12,atol=1e-14)
    def test_full_chunk_equivalence(self):
        rows=fixture(8,True);full=candidate.candidate_a(rows,replicates=127,chunk_size=127,allow_full_fixture=True)
        self.assertTrue(full['estimable'])
        for chunk in (1,7,64): self.assertEqual(full,candidate.candidate_a(rows,127,chunk_size=chunk))
    def test_batch_bound(self):
        with self.assertRaises(ValueError): candidate.candidate_a(fixture(),5,chunk_size=65)
    def test_original_guards(self):
        transformations=[lambda r:[x for x in r if x['task']!='QA'],lambda r:[dict(x,y=0) for x in r],lambda r:[dict(x,e=0) for x in r],lambda r:[dict(x,c=0) for x in r],lambda r:[dict(x,c=x['e']) for x in r],lambda r:[dict(x,group=x['task']) for x in r]]
        for transform in transformations:
            rows=transform(fixture())
            self.assertFalse(candidate.candidate_b(rows)['estimable']);self.assertFalse(candidate.candidate_a(rows,19)['estimable'])
    def test_bad_schema(self):
        for value in (float('nan'),float('inf'),.5):
            rows=fixture();rows[0]['c']=value
            with self.assertRaises(ValueError): candidate.candidate_b(rows)
        rows=fixture();rows[-1]['group']=rows[0]['group']
        with self.assertRaises(ValueError): candidate.candidate_a(rows,9)
    def test_undefined_replicate_gate(self):
        # Original has finite variances; resampling very small cells can erase them.
        r=candidate.candidate_a(fixture(),999,chunk_size=7)
        self.assertFalse(r['estimable']);self.assertGreater(r['undefined_replicates'],0);self.assertIsNone(r['lower'])
    def test_duplicate_group_instances(self):
        a=candidate.aggregate(fixture(2));indices=candidate.bootstrap_indices(a,(0,0,0,0),1)
        sampled=[x[ix] for x,ix in zip(a,indices)]
        for x,y in zip(a,sampled):self.assertEqual(x.shape,y.shape)
        self.assertTrue(any(len(set(ix))<len(ix) for ix in indices))
    def test_seed_replay(self):
        rows=fixture(3);self.assertEqual(candidate.candidate_a(rows,31),candidate.candidate_a(rows,31))
    def test_strict_success_boundaries(self):
        from eviconsist_statistics import success
        self.assertFalse(success([0,0]));self.assertFalse(success([.01,-.02]));self.assertTrue(success([.01,-.019]))

if __name__=='__main__':unittest.main()
