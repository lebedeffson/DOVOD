from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from paper_a.statistics import exact_sign_test

def key(c): return (c.get("domain"),c.get("algorithm"),int(c.get("trace_budget",-1)))
def summarize(cases,all_ok):
    ok=[c for c in cases if c.get("status")=="ok"]; cell=[]; by={}
    for c in ok:
        t=c.get("decision",{}).get("test",{}); b=t.get("base",{}).get("risk"); d=t.get("dovod",{}).get("risk")
        if b is not None and d is not None:
            imp=float(b)-float(d); cell.append(imp); by.setdefault(c["domain"],[]).append(imp)
    means={d:sum(v)/len(v) for d,v in by.items()}
    return {"ok_count":len(ok),"failed_count":len(cases)-len(ok),"mean_test_risk_improvement_over_cells":sum(cell)/len(cell) if cell else None,"domain_mean_test_risk_improvement":means,"domain_sign_test":exact_sign_test(list(means.values())) if all_ok and len(means)==20 else None,"statistical_note":"The exact sign test treats domains, not learner/budget cells, as independent units and is emitted only when all frozen cells succeeded."}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",required=True); p.add_argument("--output",default=str(ROOT/"results"/"paper_a_amlgym_matrix.json")); a=p.parse_args(); contract=json.loads((ROOT/"configs"/"amlgym_q1_contract.json").read_text())
    expected={(d,l,int(b)) for d in contract["domains"] for l in contract["learner_families"] for b in contract["trace_budgets"]}; cases=[]
    for path in sorted(Path(a.input_dir).rglob("*.json")):
        try: obj=json.loads(path.read_text())
        except Exception as exc: cases.append({"status":"failed","path":str(path),"error":f"invalid JSON: {exc}"}); continue
        if "domain" in obj and "algorithm" in obj and "trace_budget" in obj: cases.append(obj)
    seen={key(c) for c in cases if c.get("domain") is not None}; missing=sorted(expected-seen); unexpected=sorted(seen-expected); complete=not missing and not unexpected and len(cases)==len(expected); all_ok=complete and all(c.get("status")=="ok" for c in cases)
    report={"schema":"dovod-q1-amlgym-merged-v3","expected_case_count":len(expected),"observed_case_count":len(cases),"complete":complete,"missing":[list(x) for x in missing],"unexpected":[list(x) for x in unexpected],"summary":summarize(cases,all_ok),"cases":cases,"claim_boundary":"Primary external benchmark artifact. Failed cells remain in the denominator and are never silently dropped."}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps({k:report[k] for k in ("complete","expected_case_count","observed_case_count","summary")},indent=2)); raise SystemExit(0 if complete else 3)
if __name__=="__main__": main()
