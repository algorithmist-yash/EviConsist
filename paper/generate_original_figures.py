"""Original diagrams/plots from local measured results; no external artwork."""
import os,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
D=R/'results/submission_preparation_2026_11_01/VERIFIED_PAPER_RESULTS.json'
if not D.exists():D=P/'reproducibility/VERIFIED_PAPER_RESULTS.json'
O=P/'figures';O.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(R/'results/submission_preparation_2026_11_01/mpl-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
data=json.loads(D.read_text(encoding='utf8'))
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
outputs=[]
def save(fig,name):
    for suffix in ['pdf','png','svg']:
        path=O/(name+'.'+suffix);fig.savefig(path,dpi=240,bbox_inches='tight');outputs.append(path)
    plt.close(fig)

# Original vector architecture diagram: supplied source is the evidence corpus.
fig,ax=plt.subplots(figsize=(3.42,2.55));ax.set_xlim(0,10);ax.set_ylim(0,8.4);ax.axis('off')
def box(x,y,w,h,text,color='#eaf1f8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12,rounding_size=.12',
        linewidth=.8,edgecolor='#264969',facecolor=color));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=6.5)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.9,color='#34495e'))
box(.2,6.65,4.15,1.3,'Source context\n120-word windows\nStride: 80 words')
box(5.3,6.65,4.15,1.3,'Target response\nSentence proxies\nPreserved offsets')
box(.8,4.7,8.1,1.0,'TF–IDF retrieval: up to 3 windows per sentence')
arrow((2.3,6.5),(3.2,5.85));arrow((7.4,6.5),(6.4,5.85))
box(.8,2.65,8.1,1.1,'Pinned NLI verifier: small OR base\nEntailment / contradiction / neutral',color='#fcebdc')
arrow((4.85,4.55),(4.85,3.92))
box(.8,.55,8.1,1.1,'Three response features → logistic regression\nValidation-locked threshold → response flag')
arrow((4.85,2.48),(4.85,1.85))
ax.text(5,.05,'Gold labels enter training/evaluation, not retrieval or NLI.',ha='center',va='center',fontsize=6.1,color='#555555')
save(fig,'pipeline_original')

# Post-hoc task plot at unchanged globally validation-selected thresholds.
tasks=['Summary','QA','Data2txt'];colors=['#2166ac','#b35806']
fig,axs=plt.subplots(2,1,figsize=(3.42,3.05),layout='constrained')
for ax,metric in zip(axs,['precision','recall']):
    x=np.arange(3);w=.32
    for k,offset,color,label in zip(['baseline','alternative'],[-w/2,w/2],colors,['Small','Base']):
        vals=[data['task_results'][t][k][metric] for t in tasks]
        bars=ax.bar(x+offset,vals,w,color=color,label=label,zorder=3)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.014,f'{v:.2f}',ha='center',fontsize=6.5)
    ax.set(xticks=x,xticklabels=tasks,ylim=(0,1.06),ylabel=metric.capitalize());ax.grid(axis='y',alpha=.18,zorder=0)
axs[0].legend(loc='upper left',fontsize=6.5,ncol=2);save(fig,'task_tradeoffs_original')

# Two confusion matrices use exactly the same frozen test responses.
fig,axs=plt.subplots(1,2,figsize=(3.42,1.65),layout='constrained')
for ax,k,label in zip(axs,['baseline','alternative'],['Small verifier','Base verifier']):
    a=np.array(data['overall'][k]['confusion'])
    ax.imshow(a,cmap='Blues',vmin=0,vmax=1300)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=8,color='white' if a[i,j]>750 else '#111111')
    ax.set(xticks=[0,1],xticklabels=['Neg.','Pos.'],yticks=[0,1],yticklabels=['Neg.','Pos.'],xlabel='Predicted',title=label)
    ax.tick_params(length=0);ax.set_ylabel('Actual' if k=='baseline' else '')
    for spine in ax.spines.values():spine.set_visible(False)
save(fig,'confusion_original')

# Sensitivity is descriptive and must not be mistaken for independent test data.
fig,ax=plt.subplots(figsize=(3.42,2.05),layout='constrained')
labels=['Original\nN=2,675','QA exclusions\nN=2,663','Flagged components\nN=2,603'];x=np.arange(3)
for k,color,label in zip(['baseline','alternative'],colors,['Small','Base']):
    vals=[data['overall'][k]['precision'],data['sensitivity']['remove_two_QA_sources'][k]['precision'],data['sensitivity']['remove_flagged_crossfold_components'][k]['precision']]
    ax.plot(x,vals,'o-',color=color,label=label,lw=1.2,ms=4)
    for xi,v in zip(x,vals):ax.annotate(f'{v:.4f}',(xi,v),xytext=(0,5 if k=='alternative' else -11),textcoords='offset points',ha='center',fontsize=6.4)
ax.set(xticks=x,xticklabels=labels,ylabel='Precision',ylim=(.36,.44));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=6.5,ncol=2,loc='upper left')
save(fig,'overlap_sensitivity_original')
manifest=dict(generated='2026-11-01',creator='Local Matplotlib scripts; AI-assisted code, no external figure/image reuse',
    input=str(D.relative_to(R)).replace('\\','/'),input_sha256=hashlib.sha256(D.read_bytes()).hexdigest(),
    figure_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
    note='pipeline_original is schematic; remaining figures use measured saved-prediction metrics. No error bars or new inferential claims are invented.')
(O/'ORIGINAL_FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
(O/'README.md').write_text('''# Original paper figures

All figures are generated locally, not copied from external papers or websites.

- `pipeline_original`: schematic of the implemented evidence-only pipeline; not a measured result.
- `verifier_diagnostics`: precision–recall, ten-bin reliability and low-risk acceptance curves from frozen test predictions. Generated by `deadline_paper_analysis_2026_11_01.py` at project root.
- `task_tradeoffs_original`: precision and recall by task at the locked global thresholds.
- `confusion_original`: measured small/base test confusion matrices; rows actual, columns predicted.
- `overlap_sensitivity_original`: optional standalone view of the post-hoc exclusion sensitivity. The paper uses the corresponding table to avoid redundant figures.

The four additional figures have PDF (LaTeX), SVG (editable vector), and PNG (preview) versions. Run `python paper/generate_original_figures.py` from the project root after `python deadline_paper_analysis_2026_11_01.py`. The latter checks archived metrics against saved test predictions. Input/output hashes are in ORIGINAL_FIGURE_MANIFEST.json. No stochastic model generation or detector fitting occurs.

Dataset provenance is still cited in the paper: originality of the artwork does not replace attribution of RAGTruth or the pretrained models. The sensitivity subsets are post-hoc, not fresh confirmation.
''',encoding='utf8')
print('Generated',len(outputs),'original figure assets under',O)
if __name__=='__main__':pass
