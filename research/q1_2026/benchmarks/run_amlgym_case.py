from __future__ import annotations
import argparse,json,platform,tempfile,time,sys
from importlib.metadata import version as package_version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from paper_a.statistics import certify_binary_errors,paired_error_comparison
from paper_a.amlgym_bridge import DecisionObservation,decision_metrics,fit_operator_repair,parse_action_label,predict_operator_repair,stable_bucket
EXPECTED_AMLGYM_VERSION="1.0.11"

def action_label(op,args):
    args=tuple(map(str,args)); return f"({op}{(' '+' '.join(args)) if args else ''})"

def normalize_syntactic_metrics(value):
    return {str(k):float(v) for k,v in value.items()} if isinstance(value,dict) else {"mean":float(value)}

def collect(domain,reference,learned,max_problems,max_states,max_actions):
    from amlgym.benchmarks import get_problems_path,get_test_states
    from amlgym.modeling.UPEnv import UPEnv
    all_states=get_test_states(domain,kind="predictive_power"); rows=[]
    for problem_path in get_problems_path(domain,kind="predictive_power")[:max_problems]:
        problem_name=Path(problem_path).name
        if problem_name not in all_states: raise KeyError(f"predictive state file has no entry for {problem_name}")
        reference_env=UPEnv(reference,problem_path); learned_env=UPEnv(learned,problem_path)
        ranked=sorted(enumerate(all_states[problem_name]),key=lambda x:stable_bucket(f"state|{domain}|{problem_name}|{x[0]}",2**31-1))[:max_states]
        for state_index,state in ranked:
            state=tuple(map(str,state)); truth=reference_env.applicable_actions(set(state)); predicted=learned_env.applicable_actions(set(state)); labels=[]
            for operator,combos in sorted(reference_env.ground_actions.items()):
                actions=[action_label(operator,args) for args in combos]; actions.sort(key=lambda a:stable_bucket(f"action|{domain}|{problem_name}|{state_index}|{a}",2**31-1)); labels.extend(actions[:max_actions])
            bucket=stable_bucket(f"state-split|{domain}|{problem_name}|{state_index}",1000); split="repair" if bucket<500 else ("calibration" if bucket<750 else "test")
            for action in labels:
                operator,_=parse_action_label(action); rows.append((split,DecisionObservation(state,action,int(action in predicted.get(operator,set())),int(action in truth.get(operator,set())))))
    return rows

def run(args):
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path,get_trajectories_path
    from amlgym.metrics import syntactic_precision,syntactic_recall
    av=package_version("amlgym")
    if av!=EXPECTED_AMLGYM_VERSION: raise RuntimeError(f"AMLGym protocol requires {EXPECTED_AMLGYM_VERSION}, found {av}")
    reference=get_domain_path(args.domain); traces=get_trajectories_path(args.domain,kind="learning")[:args.trace_budget]
    if len(traces)<args.trace_budget: raise RuntimeError("not enough trajectories")
    learner=get_algorithm(args.algorithm,**({"noise":0.0} if args.algorithm.lower()=="nolam" else {})); started=time.perf_counter(); model_text=learner.learn(reference,traces); learn_seconds=time.perf_counter()-started
    with tempfile.TemporaryDirectory() as td:
        learned_path=Path(td)/"learned.pddl"; learned_path.write_text(str(model_text),encoding="utf-8")
        syntactic={"precision":normalize_syntactic_metrics(syntactic_precision(str(learned_path),reference)),"recall":normalize_syntactic_metrics(syntactic_recall(str(learned_path),reference))}
        rows=collect(args.domain,reference,str(learned_path),args.max_problems,args.max_states,args.max_actions); repair_rows=[o for split,o in rows if split=="repair"]; by_operator={}
        for o in repair_rows: by_operator.setdefault(parse_action_label(o.action_label)[0],[]).append(o)
        models={operator:fit_operator_repair(operator,obs,max_features=args.max_features,max_context_width=1,edit_penalty=args.edit_penalty) for operator,obs in by_operator.items() if len(obs)>=4}
        decision={}
        for split in ("repair","calibration","test"):
            obs=[o for s,o in rows if s==split]; base=[o.base_allow for o in obs]; repaired=[]
            for o in obs:
                operator,_=parse_action_label(o.action_label); repaired.append(o.base_allow if operator not in models else predict_operator_repair(models[operator],o))
            entry={"base":decision_metrics(obs,base),"dovod":decision_metrics(obs,repaired)}
            if obs:
                labels=[o.truth_allow for o in obs]; entry["paired"]=paired_error_comparison(labels,base,repaired).__dict__; entry["dovod_holdout_upper95"]=certify_binary_errors(labels,repaired).__dict__
            decision[split]=entry
    return {"schema":"dovod-q1-amlgym-case-v3","status":"ok","domain":args.domain,"algorithm":args.algorithm,"trace_budget":args.trace_budget,"amlgym_version":av,"python":platform.python_version(),"learn_seconds":learn_seconds,"syntactic":syntactic,"decision":decision,"repair_model_count":len(models),"pair_count":len(rows),"split_pair_counts":{split:sum(s==split for s,_ in rows) for split in ("repair","calibration","test")}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domain",required=True); p.add_argument("--algorithm",required=True); p.add_argument("--trace-budget",required=True,type=int); p.add_argument("--max-problems",type=int,default=2); p.add_argument("--max-states",type=int,default=12); p.add_argument("--max-actions",type=int,default=4); p.add_argument("--max-features",type=int,default=8); p.add_argument("--edit-penalty",type=float,default=0.25); p.add_argument("--output",required=True); args=p.parse_args()
    try: result=run(args)
    except Exception as exc: result={"schema":"dovod-q1-amlgym-case-v3","status":"failed","domain":args.domain,"algorithm":args.algorithm,"trace_budget":args.trace_budget,"error":f"{type(exc).__name__}: {exc}"}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["status"]=="ok" else 2)
if __name__=="__main__": main()
