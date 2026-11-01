import copy,json,unittest
from pathlib import Path
from annotation_checks import validate_review,require_human_training_labels

class AnnotationChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review=json.loads((Path(__file__).resolve().parent/'results/annotation_adjudication_2026_10_16/annotations.assistant_adjudicated.json').read_text(encoding='utf8'))
    def test_actual_review_offsets_schema_and_decisions(self): self.assertTrue(validate_review(self.review))
    def test_assistant_review_cannot_train(self):
        with self.assertRaises(ValueError): require_human_training_labels(self.review['records'])
    def test_incorrect_refusal_is_not_negative(self):
        r=next(r for r in self.review['records'] if r['source_id']=='12253')
        self.assertEqual(r['quality_proposal'],'incorrect_refusal'); self.assertIsNone(r['response_label_proposal']); self.assertFalse(r['accepted_span_proposals'])
    def test_uncertain_claim_is_not_forced_negative(self):
        r=next(r for r in self.review['records'] if r['source_id']=='15793')
        self.assertEqual(r['response_decision'],'uncertain'); self.assertIsNone(r['response_label_proposal'])
    def test_corrupted_offset_rejected(self):
        r=copy.deepcopy(self.review); r['records'][0]['claim_decisions'][0]['start']+=1
        with self.assertRaises(AssertionError): validate_review(r)
    def test_adopted_adequate_negative_and_quality_exclusion(self):
        adequate=dict(reviewer_type='human',human_review_complete=True,human_adjudication_complete=True,annotators=['Reviewer 1'],quality='adequate',gold=0,response_decision='negative')
        refusal=dict(adequate,quality='incorrect_refusal',gold=None,response_decision='quality_excluded')
        self.assertEqual(require_human_training_labels([adequate,refusal]),[adequate])
        refusal['gold']=0
        with self.assertRaises(ValueError): require_human_training_labels([adequate,refusal])

if __name__=='__main__': unittest.main()
