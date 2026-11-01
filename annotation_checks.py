"""Validate semantic review structure and refuse assistant-only training labels."""
import hashlib

def validate_review(payload):
    records=payload['records']; assert len(records)==12 and len({r['source_id'] for r in records})==12
    assert payload['reviewer_type']=='assistant' and payload['not_gold'] is True
    for r in records:
        assert r['gold'] is None and not r['human_review_complete'] and not r['human_adjudication_complete']
        assert hashlib.sha256(r['target'].encode()).hexdigest()==r['target_sha256']
        for d in r['claim_decisions']:
            assert 0<=d['start']<d['end']<=len(r['target'])
            assert r['target'][d['start']:d['end']]==d['text']
            assert d['decision'] in ['unsupported','contradiction','uncertain','supported']
            assert d['span_kind']==(d['decision'] if d['decision'] in ['unsupported','contradiction'] else None)
            assert d['rationale'] and d['evidence']
        positive=[d for d in r['claim_decisions'] if d['span_kind'] is not None]
        uncertain=[d for d in r['claim_decisions'] if d['decision']=='uncertain']
        assert len(positive)==len(r['accepted_span_proposals'])
        assert len(uncertain)==r['unresolved_claim_count']
        if r['response_decision']=='positive': assert positive and r['response_label_proposal']==1
        elif r['response_decision']=='negative': assert not positive and not uncertain and r['response_label_proposal']==0
        elif r['response_decision']=='uncertain': assert not positive and uncertain and r['response_label_proposal'] is None
        elif r['response_decision']=='quality_excluded': assert r['quality_proposal']!='adequate' and r['response_label_proposal'] is None
        else: raise AssertionError('Unknown response decision')
    return True

def require_human_training_labels(records):
    """Prerequisite check for a future larger-cohort training entry point."""
    eligible=[]
    for r in records:
        if r.get('reviewer_type')!='human' or not r.get('human_review_complete') or not r.get('human_adjudication_complete'):
            raise ValueError('Assistant/proposed/unreviewed decisions are not adopted human training gold')
        if not r.get('annotators') or any(str(a).lower().startswith(('assistant','ai','model')) for a in r['annotators']):
            raise ValueError('Record genuine human reviewer identities')
        if r.get('quality')!='adequate':
            if r.get('gold') is not None: raise ValueError('Quality-excluded target must not become a factuality negative')
            continue
        if r.get('response_decision')=='uncertain':
            if r.get('gold') is not None: raise ValueError('Uncertain response must retain null')
            continue
        if r.get('gold') not in [0,1] or r.get('gold') is None: raise ValueError('Resolved adequate target lacks binary gold')
        eligible.append(r)
    if not eligible: raise ValueError('No adopted eligible targets')
    return eligible
