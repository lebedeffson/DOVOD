from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / 'research' / 'reference_impl'
sys.path.insert(0, str(ROOT))
from q1_uncertainty_source_aware import build_episodes

OUT = PACKAGE / 'results' / 'information_selection' / 'paper_b_noisy_observation'
OUT.mkdir(parents=True, exist_ok=True)
EPS=1e-12

RELIABILITY_PAIRS = [(0.90,0.90),(0.90,0.70),(0.70,0.90),(0.85,0.55),(0.55,0.85)]


def y_variance(q, graphs, action, weights=None):
    if weights is None:
        weights=np.full(len(graphs),1.0/len(graphs),dtype=float)
    else:
        weights=np.asarray(weights,dtype=float); weights=weights/weights.sum()
    probs=[]
    for g in graphs:
        p=1.0-float(q[action])
        for pred in g.get(action,[]): p*=float(q[int(pred)])
        probs.append(p)
    py=float(np.dot(weights,np.asarray(probs,dtype=float)))
    return py*(1.0-py)


def physical_expected_after(q, graphs, action, j, r, weights=None):
    p=float(q[j]); pz1=r*p+(1-r)*(1-p); pz0=1-pz1; out=0.0
    if pz1>EPS:
        post1=(r*p)/pz1; q1=np.asarray(q,dtype=float).copy(); q1[j]=post1
        out+=pz1*y_variance(q1,graphs,action,weights)
    if pz0>EPS:
        post0=((1-r)*p)/pz0; q0=np.asarray(q,dtype=float).copy(); q0[j]=post0
        out+=pz0*y_variance(q0,graphs,action,weights)
    return float(out)


def semantic_groups(graphs, action):
    groups=defaultdict(list)
    for i,g in enumerate(graphs): groups[tuple(sorted(g.get(action,[])))].append(i)
    return list(groups.values())


def semantic_expected_after(q, graphs, action, r):
    groups=semantic_groups(graphs,action); m=len(groups)
    if m<=1: return y_variance(q,graphs,action)
    prior_g=np.array([len(g)/len(graphs) for g in groups],dtype=float); out=0.0; wrong=(1-r)/(m-1)
    for z in range(m):
        likelihood=np.full(m,wrong,dtype=float); likelihood[z]=r
        pz=float(np.dot(prior_g,likelihood))
        if pz<=EPS: continue
        post_group=prior_g*likelihood/pz; w=np.zeros(len(graphs),dtype=float)
        for h,idxs in enumerate(groups):
            for i in idxs: w[i]=post_group[h]/len(idxs)
        out += pz*y_variance(q,graphs,action,w)
    return float(out)


def choose(q, graphs, action, masked, rp, rs, assumed_perfect=False):
    cur=y_variance(q,graphs,action); rp_use=1.0 if assumed_perfect else rp; rs_use=1.0 if assumed_perfect else rs; opts=[]
    for j in masked:
        after=physical_expected_after(q,graphs,action,int(j),rp_use); red=cur-after
        if red>EPS: opts.append((red,'PHYSICAL_QUERY',int(j)))
    if len(semantic_groups(graphs,action))>1:
        after=semantic_expected_after(q,graphs,action,rs_use); red=cur-after
        if red>EPS: opts.append((red,'SEMANTIC_REVIEW',int(action)))
    if not opts: return ('NONE',-1,cur)
    _,kind,target=max(opts,key=lambda x:(x[0],x[1]=='SEMANTIC_REVIEW',-x[2]))
    actual_after=physical_expected_after(q,graphs,action,target,rp) if kind=='PHYSICAL_QUERY' else semantic_expected_after(q,graphs,action,rs)
    return kind,target,float(actual_after)


def cluster_bootstrap(df, value_col, n_boot=10000, seed=20260903):
    recs=sorted(df.recording.unique()); per=df.groupby('recording')[value_col].mean(); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        s=rng.choice(recs,size=len(recs),replace=True); vals.append(float(np.mean([per.loc[x] for x in s])))
    return {'mean':float(df[value_col].mean()),'recording_equal_mean':float(per.mean()),'ci95_recording_bootstrap':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'recordings':len(recs)}


def main():
    episodes,graphs=build_episodes(); episodes=[e for e in episodes if e['initial_semantic_variance']>EPS]; rows=[]
    for rp,rs in RELIABILITY_PAIRS:
        for eid,ep in enumerate(episodes):
            a_kind,a_target,a_after=choose(ep['q'],graphs,ep['action'],ep['masked_list'],rp,rs,False)
            n_kind,n_target,n_after=choose(ep['q'],graphs,ep['action'],ep['masked_list'],rp,rs,True)
            cur=y_variance(ep['q'],graphs,ep['action'])
            rows.append({'episode':eid,'recording':ep['recording'],'frame':ep['frame'],'action':ep['action'],'mask_k':ep['mask_k'],'r_physical':rp,'r_semantic':rs,'initial_variance':cur,'aware_source':a_kind,'aware_target':a_target,'aware_expected_post_variance':a_after,'perfect_assumption_source':n_kind,'perfect_assumption_target':n_target,'perfect_assumption_actual_post_variance':n_after,'aware_minus_naive_post_variance':a_after-n_after,'aware_better':int(a_after<n_after-EPS),'same_source':int(a_kind==n_kind and a_target==n_target)})
    df=pd.DataFrame(rows); df.to_csv(OUT/'noisy_one_step_source_selection.csv',index=False); summaries=[]; boots={}
    for (rp,rs),g in df.groupby(['r_physical','r_semantic']):
        key=f'rp{rp:.2f}_rs{rs:.2f}'
        summaries.append({'r_physical':rp,'r_semantic':rs,'episodes':len(g),'recordings':g.recording.nunique(),'source_decision_change_rate':float(1-g.same_source.mean()),'aware_better_rate':float(g.aware_better.mean()),'mean_post_variance_aware':float(g.aware_expected_post_variance.mean()),'mean_post_variance_naive':float(g.perfect_assumption_actual_post_variance.mean()),'mean_delta_aware_minus_naive':float(g.aware_minus_naive_post_variance.mean()),'semantic_first_aware':float((g.aware_source=='SEMANTIC_REVIEW').mean()),'semantic_first_naive':float((g.perfect_assumption_source=='SEMANTIC_REVIEW').mean())})
        boots[key]=cluster_bootstrap(g,'aware_minus_naive_post_variance',seed=20260903+len(boots))
    pd.DataFrame(summaries).to_csv(OUT/'noisy_one_step_summary.csv',index=False)
    report={'schema':'tinyapv-paper-b-noisy-one-step-v1','model':'Each physical query passes through a binary symmetric channel with P(Z=X)=r_p. Semantic review returns the correct action-specific prerequisite alternative with probability r_s and otherwise a uniformly chosen wrong alternative. Priors are independent TRAIN-derived Bernoulli component completion probabilities and a uniform prior over the eight TRAIN-derived semantic graphs. One query is selected and Bayesian conditioning is exact under this controlled observation model.','boundary':'This is a one-step misinformation stress test, not a replacement for the sequential Bellman experiment. It tests source choice under explicitly incorrect observations; it does not model adversarial or temporally correlated errors.','summary':summaries,'recording_cluster_bootstrap':boots}
    (OUT/'noisy_one_step_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
