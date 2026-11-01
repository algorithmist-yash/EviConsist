"""Independent scalar/row-level reference; never imports candidate estimators."""
import math
from collections import defaultdict

TASKS=('Summary','QA','Data2txt')

def ratios(rows,weights=None):
    weights={} if weights is None else weights
    output=[]
    for model in ('e','c'):
        a=b=p=0.
        for task in TASKS:
            rr=[r for r in rows if r['task']==task]
            n=sum(weights.get(str(r['group']),1.) for r in rr)
            if not n: raise ValueError('Missing task')
            for r in rr:
                w=weights.get(str(r['group']),1.)/(3*n)
                a+=w*r['y']*r[model];b+=w*r[model];p+=w*r['y']
        if not b or not p: raise ValueError('Undefined ratio')
        output.append([a/b,a/p])
    return output

def contrast(rows,weights=None):
    r=ratios(rows,weights)
    return [r[1][j]-r[0][j] for j in range(2)]

def scalar_variance(rows):
    r=ratios(rows);by=defaultdict(list)
    for row in rows: by[(row['task'],str(row['group']))].append(row)
    h={};v=[0.,0.];vt={}
    for task in TASKS:
        rr=[x for x in rows if x['task']==task];groups=sorted(g for t,g in by if t==task);n=len(rr)
        for endpoint in (0,1):
            derivatives=[]
            for model_index,model in enumerate(('e','c')):
                risk=r[model_index][endpoint]
                B=sum(sum(x[model] if endpoint==0 else x['y'] for x in rows if x['task']==t)/len([x for x in rows if x['task']==t])/3 for t in TASKS)
                all_u=sum(x['y']*x[model]-risk*(x[model] if endpoint==0 else x['y']) for x in rr)
                dd=[]
                for g in groups:
                    gg=by[(task,g)];u=sum(x['y']*x[model]-risk*(x[model] if endpoint==0 else x['y']) for x in gg)
                    dd.append((u-len(gg)/n*all_u)/(3*n*B))
                derivatives.append(dd)
            hh=[c-e for e,c in zip(*derivatives)];h[(task,endpoint)]=hh
            value=len(groups)/(len(groups)-1)*sum(x*x for x in hh);v[endpoint]+=value;vt[(task,endpoint)]=value
    df=[v[j]**2/sum(vt[(t,j)]**2/(len({str(x['group']) for x in rows if x['task']==t})-1) for t in TASKS) for j in range(2)]
    return dict(variance=v,se=[math.sqrt(x) for x in v],df=df,contributions=h)
