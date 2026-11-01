import unittest
import numpy as np
from eviconsist_statistics import TASKS,difference,draw,grouped,intervals,power,success,select_threshold

def rows(n=20):
    return [dict(group=f'{t}-{i}',task=t,y=int(i%2==0),e=1,c=1,score=.5) for t in TASKS for i in range(n)]

class StatisticalTests(unittest.TestCase):
    def test_identical_paired_zero(self):
        result=intervals(rows(),199)
        self.assertEqual(result['lower'],[0.,0.]); self.assertFalse(success(result['lower']))
    def test_strict_boundaries(self):
        self.assertFalse(success([0.,0.])); self.assertFalse(success([.03,-.02])); self.assertTrue(success([.001,-.019]))
    def test_task_standardization(self):
        r=rows(); a=difference(r)
        r+= [dict(x,group=x['group']+'copy') for x in r if x['task']=='QA']
        np.testing.assert_allclose(a,difference(r))
    def test_known_nonzero_ratios_and_vector_bootstrap(self):
        r=[]
        for t in TASKS:
            for i in range(10):
                r.extend([dict(group=f'{t}-{i}',task=t,y=1,e=1,c=1),dict(group=f'{t}-{i}',task=t,y=0,e=1,c=0)])
        np.testing.assert_allclose(difference(r),[.5,0])
        result=intervals(r,199)
        np.testing.assert_allclose(result['lower'],[.5,0]); np.testing.assert_allclose(result['upper'],[.5,0])
    def test_task_weights_are_ratios_not_average_ratios(self):
        r=[dict(group=t+str(i),task=t,y=y,e=e,c=c) for t,items in [('Summary',[(1,1,1),(0,1,0)]),('QA',[(1,1,1)]),('Data2txt',[(1,1,1),(0,1,1),(0,1,1)])] for i,(y,e,c) in enumerate(items)]
        # E precision=(.5+1+1/3)/3; C flags=(.5+1+1)/3.
        np.testing.assert_allclose(difference(r),[(11/18)/(5/6)-11/18,0])
    def test_undefined_original(self):
        r=rows()
        for x in r: x['e']=0
        self.assertFalse(intervals(r,19)['estimable'])
    def test_undefined_replicates_not_dropped(self):
        r=rows(2)
        result=intervals(r,999,seed=3)
        self.assertFalse(result['estimable']); self.assertGreater(result['undefined_replicates'],0); self.assertIsNone(result['lower'])
    def test_missing_task(self): self.assertIsNone(difference([r for r in rows() if r['task']!='QA']))
    def test_cross_task_group_rejected(self):
        r=rows(); r[-1]['group']=r[0]['group']
        with self.assertRaises(ValueError): grouped(r)
    def test_resampled_instances_distinct_and_clustered(self):
        r=rows(2); r.append(dict(r[0]))
        sampled=draw(grouped(r),np.random.default_rng(2),{t:10 for t in TASKS})
        self.assertEqual(len(grouped(sampled)['Summary']),10)
        self.assertTrue(all(len(g) in (1,2) for g in grouped(sampled)['Summary']))
    def test_repeatability(self): self.assertEqual(intervals(rows(),99),intervals(rows(),99))
    def test_threshold(self):
        r=rows()
        for x in r: x['score']=.8 if x['y'] else .2
        self.assertEqual(select_threshold(r,'score')['threshold'],.8)
    def test_synthetic_power_identity(self):
        p=power(rows(),{t:20 for t in TASKS},outer=10,inner=19)
        self.assertEqual(p['joint_power'],0.)
if __name__=='__main__': unittest.main()
