import unittest
from baseline import spans, fold, sentences, read, metrics

class DataProtocolTests(unittest.TestCase):
    def test_implicit_true_is_excluded(self):
        self.assertEqual(spans({'labels':[{'implicit_true':True},{'implicit_true':False}]}),[{'implicit_true':False}])

    def test_no_source_leakage(self):
        groups={}
        for row in read('response.jsonl'):
            groups.setdefault(row['source_id'],set()).add(fold(row))
        self.assertTrue(all(len(v)==1 for v in groups.values()))

    def test_offsets(self):
        text='First claim.\nSecond claim! Last fragment'
        parts=sentences(text)
        self.assertEqual(len(parts),3)
        self.assertTrue(all(text[a:b]==c for a,b,c in parts))

    def test_ece_includes_probability_one(self):
        self.assertAlmostEqual(metrics([0,1],[1.,0.],.5)['ece_10_bins'],1.)

    def test_threshold_uses_validation_only(self):
        import numpy as np
        from experiment_report import select_threshold
        self.assertAlmostEqual(select_threshold([0,1],np.array([.2,.8])),.21)

if __name__=='__main__': unittest.main()
