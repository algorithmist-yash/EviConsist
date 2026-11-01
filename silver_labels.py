"""Explicit silver-label representation. No model fitting and no gold conversion."""
import hashlib,json

SCHEMA_VERSION='eviconsist-silver-v1'
def text_hash(text): return hashlib.sha256(text.encode('utf8')).hexdigest()
def source_hash(source): return text_hash(json.dumps(source,ensure_ascii=False,sort_keys=True,separators=(',',':')))
def resolve(source,path):
    for key in path.strip('/').split('/') if path.strip('/') else []:
        key=key.replace('~1','/').replace('~0','~')
        source=source[int(key)] if isinstance(source,list) else source[key]
    return source

def validate(record):
    assert record['schema_version']==SCHEMA_VERSION and record['label_origin']=='assistant_generated'
    assert record['gold'] is None and record['human_review_complete'] is False and record['human_adjudication_complete'] is False and record['independent_human_validation'] is False
    assert record['target_sha256']==text_hash(record['target'])
    assert record['source_sha256']==source_hash(record['source'])
    assert record['policy_version']=='silver-v1' and len(record['policy_sha256'])==64
    assert record['dataset_role'] in ['reference_only','annotation_policy_pilot','research_development','confirmatory_test']
    assert record['annotation_status'] in ['pending','completed']
    assert record['decision'] in ['pending','positive','negative','uncertain','quality_excluded']
    assert record['quality'] in [None,'adequate','incorrect_refusal','truncated','source_ambiguous','other_exclusion']
    assert record['silver_label'] in [None,0,1]
    for field,kinds in [('spans',['unsupported','contradiction']),('uncertain_claims',['uncertain'])]:
        for span in record[field]:
            assert 0<=span['start']<span['end']<=len(record['target']) and record['target'][span['start']:span['end']]==span['text']
            assert span['kind'] in kinds and span['rationale'].strip() and span['confidence'] in ['high','medium-high','medium','medium-low','low']
            assert span['source_evidence']
            for ref in span['source_evidence']:
                actual=resolve(record['source'],ref['pointer'])
                if ref.get('quote') is not None: assert isinstance(actual,str) and ref['quote'] in actual
                else: assert 'value' in ref and ref['value']==actual
    if record['annotation_status']=='pending':
        assert record['decision']=='pending' and record['silver_label'] is None and not record['full_target_review_complete'] and not record['spans'] and not record['uncertain_claims']
    else:
        assert record['full_target_review_complete'] and record['reviewer_id'] and record['decision_rationale'].strip()
        if record['decision']=='positive': assert record['quality']=='adequate' and record['silver_label']==1 and record['spans']
        elif record['decision']=='negative': assert record['quality']=='adequate' and record['silver_label']==0 and not record['spans'] and not record['uncertain_claims']
        elif record['decision']=='uncertain': assert record['quality']=='adequate' and record['silver_label'] is None and record['uncertain_claims'] and not record['spans']
        elif record['decision']=='quality_excluded': assert record['quality'] not in [None,'adequate'] and record['silver_label'] is None
        else: raise AssertionError('Completed record cannot remain pending')
    assert record['localization_complete']==(record['annotation_status']=='completed' and not record['uncertain_claims'])
    effort=record['review_effort_seconds']; assert effort is None or effort>=0
    return True

def select_training_records(records,label_mode='gold',cohort_design_locked=False):
    """Future explicit silver-data gate; deliberately excludes both policy pilots."""
    if label_mode!='silver': raise ValueError('Assistant silver records cannot enter human-gold mode; explicitly select label_mode=silver')
    if not cohort_design_locked: raise ValueError('Lock the larger research cohort/analysis before training')
    eligible=[]
    for r in records:
        validate(r)
        if r['dataset_role']!='research_development': raise ValueError('Reference and annotation-policy pilot records are not research training data')
        if r['annotation_status']!='completed': raise ValueError('Unreviewed targets present')
        if r['decision'] in ['positive','negative']: eligible.append(r)
    return eligible

def summary(records):
    for r in records: validate(r)
    completed=[r for r in records if r['annotation_status']=='completed']
    resolved=[r for r in completed if r['decision'] in ['positive','negative']]
    adequate=[r for r in completed if r['quality']=='adequate']
    positive=sum(r['decision']=='positive' for r in completed); uncertain=sum(r['decision']=='uncertain' for r in completed)
    times=[r['review_effort_seconds'] for r in completed if r['review_effort_seconds'] is not None]
    return dict(total=len(records),completed=len(completed),pending=len(records)-len(completed),decision_counts={k:sum(r['decision']==k for r in records) for k in ['pending','positive','negative','uncertain','quality_excluded']},targets_with_any_uncertain_claim=sum(bool(r['uncertain_claims']) for r in completed),target_uncertainty_rate=sum(bool(r['uncertain_claims']) for r in completed)/len(completed) if completed else None,quality_exclusion_rate=sum(r['decision']=='quality_excluded' for r in completed)/len(completed) if completed else None,resolved_silver_positive_rate=positive/len(resolved) if resolved else None,adequate_silver_positive_bounds=[positive/len(adequate),(positive+uncertain)/len(adequate)] if adequate else None,positive_span_candidates=sum(len(r['spans']) for r in records),uncertain_claim_candidates=sum(len(r['uncertain_claims']) for r in records),review_effort_seconds_recorded=sum(times) if times else None,completed_targets_missing_effort=len(completed)-len(times),gold_nonnull=0,interpretation='Operational silver labels; not independent factuality or human reliability')
