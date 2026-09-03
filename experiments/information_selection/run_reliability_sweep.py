from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json, sys
import pandas as pd
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[2]
PROJECT = PACKAGE / 'research' / 'reference_impl'
sys.path.insert(0, str(PROJECT))
from q1_uncertainty_source_aware import build_episodes
from source_aware_resolution_planner import SourceAwareResolutionPlanner, EPS

OUT=PACKAGE/'results'; OUT.mkdir(parents=True,exist_ok=True)
FIG=PACKAGE/'figures'; FIG.mkdir(parents=True,exist_ok=True)

class MyopicPlanner(SourceAwareResolutionPlanner):
    def solve_myopic(self, q):
        qkey=self._qkey(q); active=tuple(range(len(self.graphs)))
        cost, first=self._myopic_value(qkey,active)
        return cost, first

    @lru_cache(maxsize=None)
    def _myopic_value(self, qkey, active):
        before=self._uncertainty(qkey,active)
        if before <= EPS: return 0.0, None
        unresolved=[j for j in self.queryable if EPS < qkey[j] < 1.0-EPS]
        groups=self._semantic_groups(active); candidates=[]
        for j in unresolved:
            pj=float(qkey[j]); q1=list(qkey); q1[j]=1.0; q0=list(qkey); q0[j]=0.0; q1=tuple(q1); q0=tuple(q0)
            after=pj*self._uncertainty(q1,active)+(1-pj)*self._uncertainty(q0,active); reduction=before-after
            if reduction>EPS: candidates.append((reduction/self.physical_cost,'PHYSICAL_QUERY',j,(pj,q1,q0)))
        if len(groups)>1:
            denom=float(len(active)); after=sum((len(g)/denom)*self._uncertainty(qkey,g) for g in groups); reduction=before-after
            if reduction>EPS: candidates.append((reduction/self.semantic_cost,'SEMANTIC_REVIEW',self.action,groups))
        if not candidates: return float('inf'), None
        candidates.sort(key=lambda x:(x[0],x[1]=='SEMANTIC_REVIEW',-x[2]), reverse=True)
        _, kind, target, aux=candidates[0]
        if kind=='PHYSICAL_QUERY':
            pj,q1,q0=aux; c1,_=self._myopic_value(q1,active); c0,_=self._myopic_value(q0,active)
            return self.physical_cost + pj*c1 + (1-pj)*c0, (kind,int(target))
        denom=float(len(active)); fut=sum((len(g)/denom)*self._myopic_value(qkey,g)[0] for g in aux)
        return self.semantic_cost + fut, (kind,int(target))


def run():
    episodes, graphs=build_episodes(); mixed=[e for e in episodes if e['initial_semantic_variance']>1e-12 and e['initial_physical_variance']>1e-12]
    assert len(mixed)==187, len(mixed); scenarios=[]
    for base_cs in [1.0,2.0,5.0]:
        for rp in [1.0,0.9,0.75,0.5]:
            for rs in [1.0,0.9,0.75,0.5]:
                cp_eff=1.0/rp; cs_eff=base_cs/rs
                for policy in ['bellman','myopic']:
                    costs=[]; semfirst=0
                    for e in mixed:
                        if policy=='bellman':
                            p=SourceAwareResolutionPlanner(graphs,e['action'],e['masked_list'],physical_cost=cp_eff,semantic_cost=cs_eff); d=p.solve(e['q'],'optimal'); cost=d.expected_remaining_cost; first=d.kind
                        else:
                            p=MyopicPlanner(graphs,e['action'],e['masked_list'],physical_cost=cp_eff,semantic_cost=cs_eff); cost,dec=p.solve_myopic(e['q']); first='RESOLVED' if dec is None else dec[0]
                        costs.append(float(cost)); semfirst += int(first=='SEMANTIC_REVIEW')
                    scenarios.append({'base_semantic_cost':base_cs,'physical_reveal_success':rp,'semantic_reveal_success':rs,'effective_physical_cost':cp_eff,'effective_semantic_cost':cs_eff,'policy':policy,'episodes':len(mixed),'mean_expected_attempt_cost':sum(costs)/len(costs),'semantic_first_rate':semfirst/len(mixed)})
    df=pd.DataFrame(scenarios); df.to_csv(OUT/'paper_b_reveal_reliability_sweep.csv',index=False)
    focus_pairs=[(1.0,1.0),(0.75,1.0),(1.0,0.75),(0.75,0.75),(0.5,0.9),(0.9,0.5)]
    focus=df[(df.base_semantic_cost==1.0) & df.apply(lambda r:(r.physical_reveal_success,r.semantic_reveal_success) in focus_pairs,axis=1)].copy()
    focus.pivot_table(index=['physical_reveal_success','semantic_reveal_success'],columns='policy',values=['mean_expected_attempt_cost','semantic_first_rate']).reset_index().to_csv(OUT/'paper_b_reveal_reliability_focus.csv',index=False)
    wide=df.pivot_table(index=['base_semantic_cost','physical_reveal_success','semantic_reveal_success'],columns='policy',values='mean_expected_attempt_cost').reset_index(); wide['bellman_minus_myopic']=wide['bellman']-wide['myopic']; wide['bellman_relative_gain_pct']=(wide['myopic']-wide['bellman'])/wide['myopic']*100; wide.to_csv(OUT/'paper_b_reveal_reliability_policy_gap.csv',index=False)
    b=df[(df.base_semantic_cost==1.0)&(df.policy=='bellman')]; mat=b.pivot(index='physical_reveal_success',columns='semantic_reveal_success',values='semantic_first_rate').sort_index(ascending=False)
    fig,ax=plt.subplots(figsize=(5.5,4.4)); im=ax.imshow(mat.values,aspect='auto',vmin=0,vmax=max(0.01,mat.values.max())); ax.set_xticks(range(len(mat.columns)),[f'{x:.2g}' for x in mat.columns]); ax.set_yticks(range(len(mat.index)),[f'{x:.2g}' for x in mat.index]); ax.set_xlabel('Semantic-review reveal success probability'); ax.set_ylabel('Physical-query reveal success probability'); ax.set_title('Bellman semantic-first rate under no-information failures')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f'{100*mat.iloc[i,j]:.1f}%',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Semantic-first fraction'); fig.tight_layout(); fig.savefig(FIG/'paper_b_reveal_reliability_heatmap.png',dpi=220)
    summary={'mixed_episodes':len(mixed),'model':'Independent intervention attempts; with probability r an attempt perfectly reveals the requested variable/semantic alternative, otherwise returns no information. Exact Bellman recursion is equivalent to replacing c by c/r. Wrong/misleading evidence is not modeled.'}
    for rp,rs in focus_pairs:
        x=wide[(wide.base_semantic_cost==1.0)&(wide.physical_reveal_success==rp)&(wide.semantic_reveal_success==rs)].iloc[0]; br=df[(df.base_semantic_cost==1.0)&(df.physical_reveal_success==rp)&(df.semantic_reveal_success==rs)&(df.policy=='bellman')].iloc[0]
        summary[f'rp={rp},rs={rs}']={'bellman':float(x.bellman),'myopic':float(x.myopic),'bellman_gain_pct':float(x.bellman_relative_gain_pct),'semantic_first_rate':float(br.semantic_first_rate)}
    (OUT/'paper_b_reveal_reliability_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
