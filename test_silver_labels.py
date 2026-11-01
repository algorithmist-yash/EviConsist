import copy,json,unittest
from pathlib import Path
from silver_labels import validate,summary,select_training_records
from silver_batch_pilot import similarity,canonical

class SilverLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).resolve().parent/'results/silver_reference_v2_2026_10_16/silver_reference.jsonl'
        cls.records=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
    def test_frozen_reference_counts_and_all_null_gold(self):
        s=summary(self.records)
        self.assertEqual(s['decision_counts'],dict(pending=0,positive=5,negative=5,uncertain=1,quality_excluded=1))
        self.assertEqual((s['positive_span_candidates'],s['uncertain_claim_candidates']),(13,4))
        self.assertEqual(s['gold_nonnull'],0)
    def test_default_gold_mode_refuses_silver(self):
        with self.assertRaises(ValueError): select_training_records(self.records)
    def test_explicit_silver_mode_still_refuses_pilot_reference(self):
        with self.assertRaises(ValueError): select_training_records(self.records,label_mode='silver',cohort_design_locked=True)
    def test_research_role_needs_locked_design_and_eligibility(self):
        rows=copy.deepcopy(self.records)
        for r in rows: r['dataset_role']='research_development'
        with self.assertRaises(ValueError): select_training_records(rows,label_mode='silver')
        eligible=select_training_records(rows,label_mode='silver',cohort_design_locked=True)
        self.assertEqual(len(eligible),10); self.assertTrue(all(r['gold'] is None for r in eligible))
    def test_cannot_convert_to_gold_or_human_validation(self):
        r=copy.deepcopy(self.records[0]); r['gold']=0
        with self.assertRaises(AssertionError): validate(r)
        r=copy.deepcopy(self.records[0]); r['human_review_complete']=True
        with self.assertRaises(AssertionError): validate(r)
    def test_offsets_and_source_citations_are_checked(self):
        r=copy.deepcopy(next(r for r in self.records if r['spans'])); r['spans'][0]['start']+=1
        with self.assertRaises(AssertionError): validate(r)
        r=copy.deepcopy(next(r for r in self.records if r['spans'])); r['spans'][0]['source_evidence'][0]['quote']='A fabricated quotation absent from source'
        with self.assertRaises(AssertionError): validate(r)
    def test_wrong_refusal_cannot_be_negative(self):
        r=copy.deepcopy(next(r for r in self.records if r['source_id']=='12253')); r['decision']='negative'; r['silver_label']=0
        with self.assertRaises(AssertionError): validate(r)
    def test_near_duplicate_normalization(self):
        self.assertEqual(similarity(canonical({'text':'SAME  source'}),canonical({'text':'same source'})),1.)
        self.assertLess(similarity('alpha beta gamma','different terms only'),.85)

if __name__=='__main__': unittest.main()
