from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def failed_case(domain,algorithm,trace_budget,error,*,returncode=None,stdout=None):
    case={"schema":"dovod-q1-amlgym-case-v3","status":"failed","domain":domain,"algorithm":algorithm,"trace_budget":int(trace_budget),"error":str(error)}
    if returncode is not None: case["returncode"]=int(returncode)
    if stdout: case["subprocess_output_tail"]=str(stdout)[-8000:]
    return case

def run_case(domain,algorithm,trace_budget,output_path,*,timeout_seconds=900):
    command=[sys.executable,str(ROOT/"benchmarks"/"run_amlgym_case.py"),"--domain",domain,"--algorithm",algorithm,"--trace-budget",str(trace_budget),"--output",str(output_path)]
    try:
        proc=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout_seconds,check=False)
    except subprocess.TimeoutExpired as exc:
        case=failed_case(domain,algorithm,trace_budget,f"timeout after {timeout_seconds}s",stdout=exc.stdout); output_path.write_text(json.dumps(case,indent=2)); return case
    except Exception as exc:
        case=failed_case(domain,algorithm,trace_budget,f"subprocess launch failed: {type(exc).__name__}: {exc}"); output_path.write_text(json.dumps(case,indent=2)); return case
    if not output_path.exists():
        case=failed_case(domain,algorithm,trace_budget,"no result file",returncode=proc.returncode,stdout=proc.stdout); output_path.write_text(json.dumps(case,indent=2)); return case
    try: case=json.loads(output_path.read_text())
    except Exception as exc:
        case=failed_case(domain,algorithm,trace_budget,f"invalid result JSON: {type(exc).__name__}: {exc}",returncode=proc.returncode,stdout=proc.stdout); output_path.write_text(json.dumps(case,indent=2)); return case
    case["returncode"]=proc.returncode
    if proc.returncode!=0 and case.get("status")=="ok":
        case=failed_case(domain,algorithm,trace_budget,"runner returned nonzero despite status=ok",returncode=proc.returncode,stdout=proc.stdout); output_path.write_text(json.dumps(case,indent=2))
    return case

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domains",nargs="*"); p.add_argument("--algorithms",nargs="*"); p.add_argument("--trace-budgets",nargs="*",type=int); p.add_argument("--output-dir",default=str(ROOT/"artifacts"/"amlgym")); p.add_argument("--summary",default=str(ROOT/"results"/"paper_a_amlgym_matrix.json")); a=p.parse_args()
    contract=json.loads((ROOT/"configs"/"amlgym_q1_contract.json").read_text()); domains=a.domains or contract["domains"]; algorithms=a.algorithms or contract["learner_families"]; budgets=a.trace_budgets or contract["trace_budgets"]; timeout_seconds=int(contract.get("runner",{}).get("cell_timeout_seconds",900)); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for domain in domains:
        for algorithm in algorithms:
            for budget in budgets:
                path=out/f"{domain}__{algorithm.lower()}__n{budget}.json"; case=run_case(domain,algorithm,budget,path,timeout_seconds=timeout_seconds); cases.append(case); print(case["status"],domain,algorithm,budget,flush=True)
    report={"schema":"dovod-q1-amlgym-matrix-v3","cases":cases,"case_count":len(cases),"ok_count":sum(x.get("status")=="ok" for x in cases),"failed_count":sum(x.get("status")!="ok" for x in cases),"claim_boundary":"Failures and timeouts are retained as explicit frozen cells; broad claims require the complete reviewed matrix."}
    summary=Path(a.summary); summary.parent.mkdir(parents=True,exist_ok=True); summary.write_text(json.dumps(report,indent=2))
if __name__=="__main__": main()
