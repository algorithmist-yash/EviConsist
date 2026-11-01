"""Candidate C-v1.2: exact symmetry-aware branch selection; historical v1.1 preserved.

No corpus, detector, historical inference candidate or model imports.
"""
from dataclasses import dataclass, field
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

TASKS=('Summary','QA','Data2txt')
TAIL=F(1,1920)
BITS=40
Z=F(0)
ONE=F(1)
WEIGHT=F(1,3)
CELLS=[(j//4,(j//2)%2,j%2) for j in range(8)]
TC=[WEIGHT*y*c for _ in TASKS for y,e,c in CELLS]
TE=[WEIGHT*y*e for _ in TASKS for y,e,c in CELLS]
DC=[WEIGHT*c for _ in TASKS for y,e,c in CELLS]
DE=[WEIGHT*e for _ in TASKS for y,e,c in CELLS]
DY=[WEIGHT*y for _ in TASKS for y,e,c in CELLS]
NR=[WEIGHT*y*(c-e) for _ in TASKS for y,e,c in CELLS]

class InvalidDesign(ValueError): pass
class NumericalFailure(RuntimeError): pass
class CertifiedEmptyNode(RuntimeError): pass

def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Z)

def outward(x,lower):
    x=F(x); v=float(x)
    if (lower and F(v)>x) or (not lower and F(v)<x):
        v=math.nextafter(v,-math.inf if lower else math.inf)
    return v

def decimal_point(q,rounding):
    with localcontext() as ctx:
        ctx.prec=60; ctx.rounding=rounding
        return Decimal(q.numerator)/Decimal(q.denominator)

def finite_tail_enclosure(n,k,p,upper_tail):
    """Enclose P[X>=k] or P[X<=k] by outward finite binomial sums."""
    p=F(p)
    if p==0 or p==1:
        x=n if p==1 else 0
        v=Decimal(int(x>=k if upper_tail else x<=k))
        return v,v
    complement=False
    if upper_tail:
        if k>n//2: a,b=k,n
        else: a,b=0,k-1; complement=True
    else:
        if k<n//2: a,b=0,k
        else: a,b=k+1,n; complement=True
    bounds=[]
    for rounding in (ROUND_FLOOR,ROUND_CEILING):
        with localcontext() as ctx:
            ctx.prec=60; ctx.rounding=rounding
            pp=decimal_point(p,rounding); qq=decimal_point(1-p,rounding)
            # Recurrence denominator needs opposite outward direction.
            qdiv=decimal_point(1-p,ROUND_CEILING if rounding==ROUND_FLOOR else ROUND_FLOOR)
            total=Decimal(0)
            if a<=b:
                term=Decimal(math.comb(n,a))*(pp**a)*(qq**(n-a))
                total=term
                for j in range(a,b):
                    term=(term*Decimal(n-j)*pp)/(Decimal(j+1)*qdiv)
                    total+=term
            bounds.append(total)
    lo,hi=bounds
    if complement:
        with localcontext() as ctx:
            ctx.prec=60; ctx.rounding=ROUND_FLOOR
            clo=Decimal(1)-hi
        with localcontext() as ctx:
            ctx.prec=60; ctx.rounding=ROUND_CEILING
            chi=Decimal(1)-lo
        lo,hi=clo,chi
    return max(Decimal(0),lo),min(Decimal(1),hi)

@lru_cache(maxsize=4096)
def cp_bounds(k,n):
    if type(n) is not int or type(k) is not int or n<=0 or not 0<=k<=n:
        raise InvalidDesign('CP requires integer 0 <= k <= n and n > 0')
    scale=2**BITS
    lo=Z if k==0 else F(math.floor(float(beta.ppf(float(TAIL),k,n-k+1))*scale),scale)
    hi=ONE if k==n else F(math.ceil(float(beta.isf(float(TAIL),k+1,n-k))*scale),scale)
    alpha_lo=decimal_point(TAIL,ROUND_FLOOR)
    for side in ('lower','upper'):
        for attempt in range(16):
            endpoint=lo if side=='lower' else hi
            if (side=='lower' and endpoint==0) or (side=='upper' and endpoint==1): break
            _,bound=finite_tail_enclosure(n,k,endpoint,side=='lower')
            if bound<=alpha_lo: break
            if side=='lower': lo=max(Z,lo-F(2**attempt,scale))
            else: hi=min(ONE,hi+F(2**attempt,scale))
        else: raise NumericalFailure('CP tail certification failed')
    if not lo<=F(k,n)<=hi: raise NumericalFailure('Empirical vector outside CP bounds')
    return lo,hi

@dataclass
class Region:
    lower: list
    upper: list
    anchor: list
    counts: list|None=None

    def __post_init__(self):
        self.lower=list(map(F,self.lower)); self.upper=list(map(F,self.upper)); self.anchor=list(map(F,self.anchor))
        if not len(self.lower)==len(self.upper)==len(self.anchor)==24:
            raise InvalidDesign('Three tasks and eight categories are required')
        for t in range(3):
            sl=slice(8*t,8*t+8)
            if any(not 0<=l<=u<=1 for l,u in zip(self.lower[sl],self.upper[sl])):
                raise InvalidDesign('Invalid probability bounds')
            if sum(self.lower[sl])>1 or sum(self.upper[sl])<1:
                raise InvalidDesign('Empty bounded simplex')
        if not self.feasible(self.anchor): raise InvalidDesign('Anchor is not feasible')

    def feasible(self,p):
        return len(p)==24 and all(l<=x<=u for l,x,u in zip(self.lower,p,self.upper)) and all(sum(p[8*t:8*t+8])==1 for t in range(3))

    def denominator_range(self,coeff):
        # Coefficients are equal positive weights on a cell subset in each task.
        lows=[]; highs=[]
        for t in range(3):
            chosen=[i for i in range(8*t,8*t+8) if coeff[i]]
            other=[i for i in range(8*t,8*t+8) if not coeff[i]]
            lows.append(WEIGHT*max(sum(self.lower[i] for i in chosen),1-sum(self.upper[i] for i in other)))
            highs.append(WEIGHT*min(sum(self.upper[i] for i in chosen),1-sum(self.lower[i] for i in other)))
        return sum(lows,Z),sum(highs,Z)

    def repair_witness(self,z):
        # LP witnesses are only hints; certify feasibility using exact rationals.
        cand=[]
        for t in range(3):
            vals=[max(Z,F(float(x))) for x in z[8*t:8*t+8]]
            total=sum(vals)
            cand.extend([x/total for x in vals] if total else self.anchor[8*t:8*t+8])
        lam=ONE
        for a,x,l,u in zip(self.anchor,cand,self.lower,self.upper):
            if x<a: lam=min(lam,(a-l)/(a-x))
            elif x>a: lam=min(lam,(u-a)/(x-a))
        lam=max(Z,lam)*F(999999,1000000)
        p=[a+lam*(x-a) for a,x in zip(self.anchor,cand)]
        if not self.feasible(p): raise NumericalFailure('Exact witness repair failed')
        return p

def from_counts(counts):
    if len(counts)!=3: raise InvalidDesign('Absent task; weights cannot be redistributed')
    lower=[];upper=[];anchor=[]
    for row in counts:
        if len(row)!=8 or any(type(x) is not int or x<0 for x in row) or sum(row)==0:
            raise InvalidDesign('Each task needs eight nonnegative integer counts and n > 0')
        n=sum(row)
        for k in row:
            l,u=cp_bounds(k,n);lower.append(l);upper.append(u);anchor.append(F(k,n))
    return Region(lower,upper,anchor,counts)

def count_records(records):
    counts=[[0]*8 for _ in TASKS];seen=set()
    for r in records:
        if r['group'] in seen: raise InvalidDesign('Multiple targets from one independent group')
        seen.add(r['group'])
        if r['task'] not in TASKS or any(type(r[k]) is not int or r[k] not in (0,1) for k in ('y','e','c')):
            raise InvalidDesign('Invalid task or binary outcome')
        counts[TASKS.index(r['task'])][4*r['y']+2*r['e']+r['c']]+=1
    return counts

def contrasts(p):
    dc,de,dy=dot(DC,p),dot(DE,p),dot(DY,p)
    return (dot(TC,p)/dc-dot(TE,p)/de if dc and de else None,
            dot(NR,p)/dy if dy else None)

def exact_dual_lower(c,A,b,H,d,lower,upper,y,v):
    """Independent of solver optimality, primal feasibility or dual stationarity."""
    y=[min(Z,F(q)) for q in y];v=list(map(F,v))
    residual=[F(c[i])-sum((y[j]*A[j][i] for j in range(len(A))),Z)-sum((v[j]*H[j][i] for j in range(len(H))),Z) for i in range(len(c))]
    return dot(y,b)+dot(v,d)+sum((min(r*l,r*u) for r,l,u in zip(residual,lower,upper)),Z)

@dataclass
class Audit:
    certificates: list=field(default_factory=list)
    lp_calls: int=0

def phase_one_model(A,b,H,d,lower,upper):
    aa=[list(row)+[-ONE] for row in A];bb=list(b)
    for row,rhs in zip(H,d):
        aa.append(list(row)+[-ONE]);bb.append(rhs)
        aa.append([-q for q in row]+[-ONE]);bb.append(-rhs)
    radius=[max(abs(l),abs(u)) for l,u in zip(lower,upper)]
    cap=ONE+max([abs(rhs)+dot([abs(q) for q in row],radius) for row,rhs in zip(list(A)+list(H),list(b)+list(d))]+[Z])
    return [Z]*len(lower)+[ONE],aa,bb,[],[],list(lower)+[Z],list(upper)+[cap]

def linear_bound(c,A,b,H,d,lower,upper,audit,context,allow_phase_one=True):
    audit.lp_calls+=1
    result=linprog(np.array(c,dtype=float),A_ub=np.array(A,dtype=float) if A else None,b_ub=np.array(b,dtype=float) if A else None,
        A_eq=np.array(H,dtype=float) if H else None,b_eq=np.array(d,dtype=float) if H else None,
        bounds=list(zip(map(float,lower),map(float,upper))),method='highs',options={'time_limit':5.0})
    if result.status==2 and allow_phase_one:
        phase=phase_one_model(A,b,H,d,lower,upper)
        proof,_=linear_bound(*phase,audit,{'phase_one_for':context},allow_phase_one=False)
        if proof>0:
            raise CertifiedEmptyNode('Strictly positive exact phase-I lower bound')
        raise NumericalFailure('Floating infeasibility not confirmed by an exact positive phase-I certificate')
    if not result.success:
        raise NumericalFailure(f'LP status {result.status}: no optimality claim or node deletion permitted')
    y=[F(float(q)) for q in result.ineqlin.marginals] if A else []
    v=[F(float(q)) for q in result.eqlin.marginals] if H else []
    bound=exact_dual_lower(c,A,b,H,d,lower,upper,y,v)
    model=dict(c=list(map(str,c)),A=[[str(q) for q in r] for r in A],b=list(map(str,b)),H=[[str(q) for q in r] for r in H],
               d=list(map(str,d)),lower=list(map(str,lower)),upper=list(map(str,upper)))
    audit.certificates.append(dict(context=context,model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
        lower_exact=str(bound),inequality_duals=list(map(str,y)),equality_duals=list(map(str,v))))
    return bound,result.x

def transformed(region,denom):
    low,high=region.denominator_range(denom)
    if low<=0: return None
    A=[];b=[];H=[];d=[]
    for i,(l,u) in enumerate(zip(region.lower,region.upper)):
        row=[Z]*25;row[i]=-ONE;row[-1]=l;A.append(row);b.append(Z)
        row=[Z]*25;row[i]=ONE;row[-1]=-u;A.append(row);b.append(Z)
    for t in range(3):
        row=[Z]*25
        for i in range(8*t,8*t+8): row[i]=ONE
        row[-1]=-ONE;H.append(row);d.append(Z)
    H.append(list(denom)+[Z]);d.append(ONE)
    lower=[Z]*24+[ONE/high];upper=[u/low for u in region.upper]+[ONE/low]
    return A,b,H,d,lower,upper

def recall_projection(region,audit,tol=F(1,500)):
    domain=transformed(region,DY)
    if domain is None: return dict(status='non_estimable',reason='positive-label denominator can be zero')
    A,b,H,d,lo,hi=domain
    lower,x=linear_bound(NR+[Z],A,b,H,d,lo,hi,audit,'recall_min')
    neg_upper,y=linear_bound([-q for q in NR]+[Z],A,b,H,d,lo,hi,audit,'recall_max')
    p=region.repair_witness(x[:24]);q=region.repair_witness(y[:24])
    witness_lo=min(contrasts(p)[1],contrasts(q)[1],contrasts(region.anchor)[1])
    witness_hi=max(contrasts(p)[1],contrasts(q)[1],contrasts(region.anchor)[1])
    lower=max(-ONE,lower);upper=min(ONE,-neg_upper)
    gap=max(witness_lo-lower,upper-witness_hi)
    return dict(status='certified' if gap<=tol else 'requires_revision',lower=outward(lower,True),upper=outward(upper,False),
        lower_exact=str(lower),upper_exact=str(upper),gap=outward(gap,False),witness_min=float(witness_lo),witness_max=float(witness_hi))

def precision_projection(region,audit,tol=F(1,500),max_nodes=512,max_seconds=60):
    dc_lo,dc_hi=region.denominator_range(DC); de_lo,de_hi=region.denominator_range(DE)
    if min(dc_lo,de_lo)<=0:
        return dict(status='non_estimable',reason='model-flag denominator can be zero')
    domain=transformed(region,DC);A,b,H,d,lo,hi=domain
    start=time.perf_counter(); evaluated=0
    permutation=[8*t+j for t in range(3) for j in (0,2,1,3,4,6,5,7)]
    symmetric=all(region.lower[k]==region.lower[j] and region.upper[k]==region.upper[j] for k,j in enumerate(permutation))
    anchor=contrasts(region.anchor)[0]; witness_lo=anchor;witness_hi=anchor
    def node(a,bound):
        nonlocal evaluated,witness_lo,witness_hi
        evaluated+=1
        aa=[list(row) for row in A];bb=list(b)
        aa.append([a*q for q in DE]+[Z]);bb.append(ONE)
        aa.append([-bound*q for q in DE]+[Z]);bb.append(-ONE)
        c=[tc-bound*te for tc,te in zip(TC,TE)]+[Z]
        lb,x=linear_bound(c,aa,bb,H,d,lo,hi,audit,{'precision':'min','ratio_interval':[str(a),str(bound)]})
        c=[-tc+a*te for tc,te in zip(TC,TE)]+[Z]
        nub,y=linear_bound(c,aa,bb,H,d,lo,hi,audit,{'precision':'max','ratio_interval':[str(a),str(bound)]})
        for z in (x,y):
            val=contrasts(region.repair_witness(z[:24]))[0]
            witness_lo=min(witness_lo,val);witness_hi=max(witness_hi,val)
            if symmetric:
                witness_hi=max(witness_hi,-witness_lo);witness_lo=-witness_hi
        return (a,bound,max(-ONE,lb),min(ONE,-nub))
    nodes=[];failure=None
    try: nodes=[node(dc_lo/de_hi,dc_hi/de_lo)]
    except CertifiedEmptyNode: failure='Initial region unexpectedly empty despite feasible anchor'
    except NumericalFailure as exc: failure=str(exc)
    if not nodes:
        return dict(status='requires_revision',reason=failure,lower=-1.0,upper=1.0,nodes=evaluated,seconds=time.perf_counter()-start)
    while True:
        global_lo=min([witness_lo]+[n[2] for n in nodes]);global_hi=max([witness_hi]+[n[3] for n in nodes])
        if symmetric:global_lo=-global_hi
        gap=max(witness_lo-global_lo,global_hi-witness_hi)
        if gap<=tol: status='certified';break
        if evaluated+2>max_nodes or time.perf_counter()-start>=max_seconds:
            status='requires_revision';failure='fixed node/time budget reached';break
        k=max(range(len(nodes)),key=lambda i:(nodes[i][3]-witness_hi) if symmetric else max(witness_lo-nodes[i][2],nodes[i][3]-witness_hi))
        parent=nodes[k]
        split_lo,split_hi=parent[0],parent[1]
        mid=F(math.sqrt(float(split_lo)*float(split_hi))) if symmetric and split_lo>0 else (split_lo+split_hi)/2
        if not split_lo<mid<split_hi:mid=(split_lo+split_hi)/2
        try:
            children=[]
            for a,bound in ((parent[0],mid),(mid,parent[1])):
                try: children.append(node(a,bound))
                except CertifiedEmptyNode: pass
        except NumericalFailure as exc:
            # Parent remains: no domain is discarded on a floating infeasibility status.
            status='requires_revision';failure=str(exc);break
        nodes.pop(k)
        nodes.extend(n for n in children if n[2]<witness_lo or n[3]>witness_hi)
    return dict(status=status,lower=outward(global_lo,True),upper=outward(global_hi,False),
        lower_exact=str(global_lo),upper_exact=str(global_hi),gap=outward(gap,False),
        witness_min=float(witness_lo),witness_max=float(witness_hi),nodes=evaluated,seconds=time.perf_counter()-start,reason=failure,
        exact_symmetry_used=symmetric,symmetry_permutation=permutation if symmetric else None)

def evaluate(counts,max_nodes=512,max_seconds=60):
    start=time.perf_counter();audit=Audit()
    try:
        region=from_counts(counts)
        recall=recall_projection(region,audit)
        precision=precision_projection(region,audit,max_nodes=max_nodes,max_seconds=max_seconds)
        status='certified' if recall['status']==precision['status']=='certified' else 'non_estimable' if 'non_estimable' in (recall['status'],precision['status']) else 'requires_revision'
        success=status=='certified' and F(precision['lower_exact'])>0 and F(recall['lower_exact'])>F(-1,50)
        result=dict(status=status,counts=counts,precision=precision,recall=recall,joint_success=success,
                    lp_calls=audit.lp_calls,seconds=time.perf_counter()-start)
    except (InvalidDesign,NumericalFailure) as exc:
        result=dict(status='invalid_or_uncertified',counts=counts,joint_success=False,reason=str(exc),seconds=time.perf_counter()-start)
    return result,audit
