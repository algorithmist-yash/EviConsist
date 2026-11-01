"""Add sentence localization and selective risk diagnostics to saved predictions."""
import json
import numpy as np
from baseline import OUT, metrics

rows=[json.loads(line) for line in (OUT/'evidence_predictions.jsonl').read_text(encoding='utf8').splitlines()]
report=json.loads((OUT/'evidence_metrics.json').read_text(encoding='utf8'))
val=[c for o in rows if o['fold']=='validation' for c in o['claims']]
test=[c for o in rows if o['fold']=='test' for c in o['claims']]
y=np.array([c['gold'] for c in val]); p=np.array([c['unsupported'] for c in val])
t=max(np.linspace(0,1,101),key=lambda t:metrics(y,p,t)['f1'])
report['sentence_threshold']=float(t)
report['sentence_proxy_test']=metrics(np.array([c['gold'] for c in test]),np.array([c['unsupported'] for c in test]),t)
accepted=sorted([o for o in rows if o['fold']=='test'],key=lambda o:o['probability'])
report['risk_coverage']=[dict(coverage=k/len(accepted),accepted_hallucination_rate=float(np.mean([o['gold'] for o in accepted[:k]]))) for k in range(1,len(accepted)+1)]
(OUT/'evidence_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
