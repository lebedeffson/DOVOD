from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / 'research' / 'reference_impl' / 'results' / 'hierarchical_counterexample_carriers'
OUT = PACKAGE / 'results' / 'constraints'
FIG = PACKAGE / 'figures' / 'constraints'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

specs = [
    ('MECCANO', ROOT/'meccano_absent_relation_carriers.csv', 'counterexample_recordings', 11),
    ('IMPACT', ROOT/'impact_absent_relation_carriers.csv', 'counterexample_participants', 13),
]
rows=[]
dist_rows=[]
summary={}
for dataset, path, carrier_col, m in specs:
    df=pd.read_csv(path)
    counts=df[carrier_col].astype(int)
    n=len(df)
    vc=counts.value_counts().sort_index()
    for c,num in vc.items():
        dist_rows.append({'dataset':dataset,'carrier_count':int(c),'relations':int(num),'fraction':float(num/n)})
    ds={'observed_refuted_relations':int(n),'independent_units':int(m),'carrier_distribution':{str(int(k)):int(v) for k,v in vc.items()}}
    for t in range(0, min(5,m)+1):
        robust=int((counts>t).sum())
        rows.append({'dataset':dataset,'adversarial_carrier_deletions_t':t,'guaranteed_refuted_relations':robust,'fraction_of_observed_refutations':robust/n})
    ds['robust_to_one_carrier_loss']=int((counts>1).sum())
    ds['robust_to_two_carrier_loss']=int((counts>2).sum())
    ds['single_carrier_relations']=int((counts==1).sum())
    summary[dataset]=ds

rob=pd.DataFrame(rows)
dist=pd.DataFrame(dist_rows)
rob.to_csv(OUT/'paper_a_carrier_fault_tolerance.csv',index=False)
dist.to_csv(OUT/'paper_a_carrier_distribution.csv',index=False)
(OUT/'paper_a_carrier_fault_tolerance.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

fig,ax=plt.subplots(figsize=(6.4,4.0))
for dataset,g in rob.groupby('dataset'):
    ax.plot(g['adversarial_carrier_deletions_t'],100*g['fraction_of_observed_refutations'],marker='o',label=dataset)
ax.set_xlabel('Arbitrary independent carriers removed, t')
ax.set_ylabel('Observed refutations guaranteed to survive (%)')
ax.set_ylim(0,105)
ax.grid(True,alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG/'paper_a_carrier_fault_tolerance.png',dpi=220)
print(json.dumps(summary,indent=2))
