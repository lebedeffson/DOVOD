from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from paper_b.count_dp import EvidenceCountDP
from paper_b.orientation import bayes_error_after_calibration_and_direct,calibration_risk_gain
from paper_b.pomcp import StaticWorldPOMCP
from paper_b.static_world import Query,cartesian_worlds
OUT=ROOT/'results'/'paper_b_orientation_planning.json'
def oriented_case(r,c=0.01):
    models=((0,),); worlds=cartesian_worlds(state_bits=1,models=models,physical_reliabilities=(r,),semantic_reliabilities=(0.8,),physical_orientations=(-1,1)); belief=(1/len(worlds),)*len(worlds); queries=(Query('calibrate-physical','calibrate_physical',0,c),Query('direct-state','state',0,c)); return worlds,belief,models,queries
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    closed=[]
    for r in (0.6,0.7,0.8,0.9,0.95):
        w,b,m,q=oriented_case(r); solver=EvidenceCountDP(b,w,m,q,horizon=2,false_allow=1.0,false_block=1.0); exact=solver.solve(); expected=min(0.5,0.02+bayes_error_after_calibration_and_direct(r)); closed.append({'r':r,'exact_value':exact.value,'exact_action':list(exact.action),'closed_form_optimal_value':expected,'abs_error':abs(exact.value-expected),'risk_gain_before_cost':calibration_risk_gain(r)})
    rows=[]
    for seed in range(5):
        w,b,m,q=oriented_case(0.86,0.015); solver=EvidenceCountDP(b,w,m,q,horizon=2,false_allow=1.0,false_block=1.0); exact=solver.solve(); vals=solver.root_action_values(); opt=min(vals.values()); optimal={a for a,v in vals.items() if v<=opt+1e-12}; approx=StaticWorldPOMCP(b,w,m,q,horizon=2,false_allow=1.0,false_block=1.0,seed=seed).solve(simulations=20000); rows.append({'seed':seed,'exact_value':exact.value,'pomcp_value':approx.value,'absolute_value_error':abs(approx.value-exact.value),'pomcp_action':list(approx.action),'pomcp_action_is_exact_optimal':approx.action in optimal})
    report={'schema':'dovod-paper-b-orientation-planning-v2','closed_form_regression':closed,'closed_form_all_match':all(x['abs_error']<1e-12 for x in closed),'pomcp':{'rows':rows,'mean_absolute_value_error':sum(x['absolute_value_error'] for x in rows)/len(rows),'max_absolute_value_error':max(x['absolute_value_error'] for x in rows),'exact_optimal_action_rate':sum(x['pomcp_action_is_exact_optimal'] for x in rows)/len(rows)},'claim_boundary':'Orientation is part of the same hidden-world model; POMCP is an independent in-package approximate baseline.'}
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
