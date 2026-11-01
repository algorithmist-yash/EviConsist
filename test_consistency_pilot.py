import sqlite3, unittest, json, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from consistency_pilot import seed, generation_record, collect, aggregate_consistency, validate_annotations, messages

class ConsistencyPilotTests(unittest.TestCase):
    def test_unicode_locked_sources_round_trip(self):
        import consistency_pilot as pilot
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory); sources=[dict(source_id='1',text='Curly quotes: \u201csource\u201d')]
            pilot.dump(root/'sources.json',sources)
            lock=dict(source_manifest_sha256=pilot.sha(root/'sources.json'),files={})
            pilot.dump(root/'LOCK.json',lock); (root/'LOCK.sha256').write_text(pilot.sha(root/'LOCK.json'))
            with patch.object(pilot,'OUT',root),patch.object(pilot,'ROOT',root):
                self.assertEqual(pilot.locked()[1],sources)

    def test_sample_agreement_and_contradiction_are_separate(self):
        features=aggregate_consistency([[.9,.9],[.1,.1]],[[.05,.05],[.8,.8]])
        np.testing.assert_allclose(features,[.5,.9,.425,.8,.5])
        # Permuting independently drawn comparison responses does not change features.
        np.testing.assert_allclose(aggregate_consistency([[.9,.1]],[[.05,.8]]),aggregate_consistency([[.1,.9]],[[.8,.05]]))

    def test_partial_group_cannot_be_scored(self):
        db=sqlite3.connect(':memory:'); db.execute('create table generations(source_id text,sample_index integer,record text)')
        r=generation_record('source',0,'A fact.',3,True,1.)
        db.execute('insert into generations values (?,?,?)',('source',0,json.dumps(r)))
        with self.assertRaises(ValueError): collect(db,[dict(source_id='source')],5)

    def test_target_identity_and_annotation_gate(self):
        r=generation_record('source',0,'A fact.',3,True,1.)
        groups={'source':{0:r}}; features=[dict(source_id='source',target_sha256=r['text_sha256'],generated_length_flags=[])]
        a=dict(source_id='source',reviewed=False,adjudicated=False,annotators=[],quality=None,gold=None,spans=[],target='A fact.',target_sha256=r['text_sha256'])
        with self.assertRaises(ValueError): validate_annotations([a],features,groups)
        a.update(reviewed=True,adjudicated=True,annotators=['human-reviewer'],quality='adequate',gold=0)
        self.assertEqual(validate_annotations([a],features,groups)['source']['gold'],0)
        a['gold']=1
        with self.assertRaises(AssertionError): validate_annotations([a],features,groups)
        a['spans']=[dict(start=0,end=6,text='A fact',kind='unsupported',rationale='Absent from supplied source')]
        validate_annotations([a],features,groups)
        a['target']='Different response.'
        with self.assertRaises(AssertionError): validate_annotations([a],features,groups)

    def test_target_and_comparison_seeds_differ(self):
        self.assertEqual(seed('a',0),seed('a',0))
        self.assertEqual(len({seed('a',i) for i in range(6)}),6)
        self.assertNotEqual(seed('a',0),seed('b',0))

    def test_question_is_in_generator_prompt(self):
        prompt=messages(dict(task_type='QA',source_info=dict(question='Which city?',passages='The venue is in Paris.')))
        self.assertIn('Which city?',prompt[1]['content'])
        self.assertIn('Paris',prompt[1]['content'])

if __name__=='__main__': unittest.main()
