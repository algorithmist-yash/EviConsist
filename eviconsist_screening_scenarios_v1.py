"""Frozen-revision synthetic populations. No detector/data collection imports."""
import numpy as np
TASKS=('Summary','QA','Data2txt')
SETTINGS=[('recall_boundary',.55,.90,.58,.88),('precision_boundary',.55,.85,.55,.90),('joint_null_corner',.55,.90,.55,.88)]
PROFILES=[('H1',[.4,.4,.4],1.,'single'),('H2',[.4,.4,.4],.5,'single'),('H3',[.05,.2,.5],1.,'single'),('H4',[.05,.2,.95],1.,'single'),('H5',[.01,.01,.01],1.,'single'),('H6',[.4,.4,.4],1.,'three'),('H7',[.05,.2,.5],.5,'unequal'),('H8',[.4,.4,.4],.5,'children')]

def paired(qe,qc,lam):
    both=max(0.,qe+qc-1)+lam*(min(qe,qc)-max(0.,qe+qc-1))
    p=np.array([1-qe-qc+both,qc-both,qe-both,both])
    if (p<0).any() or not np.isclose(p.sum(),1,rtol=0,atol=1e-14):raise ValueError('Infeasible cell probabilities; no clipping')
    return p

def definitions():
    table=[]
    for setting_index,(name,pe,re,pc,rc) in enumerate(SETTINGS):
        for profile_index,(profile,prevalence,lam,kind) in enumerate(PROFILES):
            mean=sum(prevalence)/3
            fe=mean*re*(1-pe)/((1-mean)*pe);fc=mean*rc*(1-pc)/((1-mean)*pc)
            if not (0<=fe<=1 and 0<=fc<=1):raise ValueError('Infeasible FPR')
            cells=[paired(fe,fc,lam).tolist(),paired(re,rc,lam).tolist()]
            # Independent analytic reconstruction by enumerating joint cells.
            precision=[];recall=[]
            for m in (0,1):
                selected=[2,3] if m==0 else [1,3]
                tp=sum(p*sum(cells[1][j] for j in selected)/3 for p in prevalence)
                fp=sum((1-p)*sum(cells[0][j] for j in selected)/3 for p in prevalence)
                precision.append(tp/(tp+fp));recall.append(tp/mean)
            np.testing.assert_allclose(precision,[pe,pc],atol=1e-14,rtol=0);np.testing.assert_allclose(recall,[re,rc],atol=1e-14,rtol=0)
            truth=[[.03,-.02],[0.,.05],[0.,-.02]][setting_index]
            table.append(dict(id=f'{name}_{profile}',setting_index=setting_index,profile_index=profile_index,setting=name,profile=profile,prevalence=prevalence,pairing=lam,group_structure=kind,groups_per_task=100,population_precision=[pe,pc],population_recall=[re,rc],negative_flag_rates=[fe,fc],paired_cells=cells,truth=truth,truth_checks=dict(precision=precision,recall=recall)))
    return table

def generate(scenario,index,stage=1):
    keys=(stage,scenario['setting_index'],scenario['profile_index'],index)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261019,spawn_key=keys)))
    output=[]
    for task,p in zip(TASKS,scenario['prevalence']):
        for g in range(100):
            kind=scenario['group_structure'];parent=f'{task}:{g:03}'
            n=1 if kind=='single' else 3 if kind=='three' else 2 if kind=='children' else int(rng.choice([1,3,8],p=[.5,.3,.2]))
            pp=p if kind=='single' else rng.beta(10*p,10*(1-p))
            distributions=[]
            for original in scenario['paired_cells']:
                a=np.array(original)
                if kind!='single':
                    positive=a>0;draw=np.zeros(4);draw[positive]=rng.dirichlet(5*a[positive]);a=draw
                distributions.append(a)
            for row in range(n):
                y=int(rng.random()<pp);pair=int(rng.choice(4,p=distributions[y]));e=int(pair//2);c=int(pair%2)
                output.append(dict(group=parent,task=task,y=y,e=e,c=c,child_source=f'{parent}:{row}' if kind=='children' else parent))
    return output,keys
