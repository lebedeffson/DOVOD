from __future__ import annotations
import csv, io, json, statistics, zipfile, os
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ZIP=Path(os.environ.get('PROCEDURAL_AI_MECCANO_PSR_ZIP', REPO/'data'/'external'/'MECCANO_PSR_Annotations.zip'))
OUT=ROOT/'results'/'hardening'
OUT.mkdir(parents=True,exist_ok=True)
N=17

def read_records():
    recs=[]
    with zipfile.ZipFile(ZIP) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('/PSR_labels_raw.csv'): continue
            parts=name.strip('/').split('/')
            si=next(i for i,p in enumerate(parts) if p in ('train','val','test'))
            rows=[]
            txt=io.StringIO(z.read(name).decode('utf-8-sig'))
            for r in csv.reader(txt):
                if r: rows.append((r[0],[int(float(x)) for x in r[1:]]))
            recs.append((parts[si],parts[si+1],rows))
    return recs

def infer(recs, excluded=None, support=1.0):
    pre=defaultdict(list)
    for split,rec,rows in recs:
        if split!='train' or rec==excluded: continue
        for k in range(len(rows)-1):
            s,s2=rows[k][1],rows[k+1][1]
            for j,(a,b) in enumerate(zip(s,s2)):
                if a!=1 and b==1: pre[j].append(s)
    deps={j:[] for j in range(N)}
    for j in range(N):
        states=pre[j]
        for i in range(N):
            if i==j or not states: continue
            if sum(s[i]==1 for s in states)/len(states) >= support:
                deps[j].append(i)
    return deps

def edges(d): return {(p,a) for a,ps in d.items() for p in ps}

def action_set(s,d): return [j for j in range(N) if s[j]!=1 and all(s[p]==1 for p in d[j])]

def next_events(recs, split=None, recording=None):
    out=[]
    for sp,rec,rows in recs:
        if split is not None and sp!=split: continue
        if recording is not None and rec!=recording: continue
        for k in range(len(rows)-1):
            s,s2=rows[k][1],rows[k+1][1]
            actual=[j for j,(a,b) in enumerate(zip(s,s2)) if a!=1 and b==1]
            if actual: out.append((rec,rows[k][0],s,actual))
    return out

def evaluate(ev,d):
    if not ev: return {'n':0,'recall':None,'mean_candidates':None}
    oks=[]; sizes=[]
    for rec,frame,s,actual in ev:
        c=action_set(s,d); oks.append(all(j in c for j in actual)); sizes.append(len(c))
    return {'n':len(ev),'recall':statistics.mean(oks),'mean_candidates':statistics.mean(sizes)}

def main():
    recs=read_records(); train_recs=sorted(rec for sp,rec,_ in recs if sp=='train')
    full=infer(recs); ef=edges(full)
    test_ev=next_events(recs,split='test')
    rows=[]
    for hold in train_recs:
        d=infer(recs,excluded=hold); e=edges(d)
        inter=len(ef&e); union=len(ef|e)
        held=evaluate(next_events(recs,split='train',recording=hold),d)
        test=evaluate(test_ev,d)
        rows.append({
            'held_out_recording':hold,'edges':len(e),'edge_jaccard_vs_full':inter/union if union else 1.0,
            'heldout_events':held['n'],'heldout_next_recall':held['recall'],'heldout_mean_candidates':held['mean_candidates'],
            'test_next_recall':test['recall'],'test_mean_candidates':test['mean_candidates'],
        })
    recalls=[r['heldout_next_recall'] for r in rows if r['heldout_next_recall'] is not None]
    testrec=[r['test_next_recall'] for r in rows]
    jac=[r['edge_jaccard_vs_full'] for r in rows]
    summary={
        'train_recordings':len(train_recs),'full_edges':len(ef),
        'edge_jaccard_mean':statistics.mean(jac),'edge_jaccard_min':min(jac),'edge_jaccard_max':max(jac),
        'heldout_next_recall_mean':statistics.mean(recalls),'heldout_next_recall_min':min(recalls),'heldout_next_recall_max':max(recalls),
        'test_recall_across_loro_graphs_mean':statistics.mean(testrec),'test_recall_min':min(testrec),'test_recall_max':max(testrec),
        'interpretation':'Each graph is inferred without one train recording; fixed test annotations are never used to infer edges.'
    }
    with (OUT/'loro_graph.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (OUT/'loro_graph_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));
    for r in rows: print(r)
if __name__=='__main__': main()
