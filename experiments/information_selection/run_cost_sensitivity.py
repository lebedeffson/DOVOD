from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / 'research' / 'reference_impl'
OUT = PACKAGE / 'results' / 'information_selection' / 'cost_misspecification'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from q1_uncertainty_source_aware import build_episodes
from source_aware_resolution_planner import SourceAwareResolutionPlanner, EPS

COSTS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0)


def expected_true_cost_of_assumed_policy(planner_assumed: SourceAwareResolutionPlanner, q, true_semantic_cost: float):
    q0 = planner_assumed._qkey(q); a0 = tuple(range(len(planner_assumed.graphs)))
    @lru_cache(maxsize=None)
    def ev(qkey: tuple[float, ...], active: tuple[int, ...]) -> float:
        if planner_assumed._uncertainty(qkey, active) <= EPS: return 0.0
        _, decision = planner_assumed._value(qkey, active, 'optimal')
        if decision is None: return float('inf')
        kind, target = decision
        if kind == 'PHYSICAL_QUERY':
            p = float(qkey[target]); q1 = list(qkey); q1[target] = 1.0; q0_ = list(qkey); q0_[target] = 0.0
            return 1.0 + p * ev(tuple(q1), active) + (1.0 - p) * ev(tuple(q0_), active)
        if kind == 'SEMANTIC_REVIEW':
            groups = planner_assumed._semantic_groups(active); denom = float(len(active))
            return float(true_semantic_cost) + sum((len(g) / denom) * ev(qkey, g) for g in groups)
        raise AssertionError(kind)
    return float(ev(q0, a0))


def main():
    episodes, graphs = build_episodes(); episodes = [ep for ep in episodes if ep['initial_semantic_variance'] > EPS]; rows = []
    for eid, ep in enumerate(episodes):
        true_planners = {c: SourceAwareResolutionPlanner(graphs, ep['action'], ep['masked_list'], physical_cost=1.0, semantic_cost=c) for c in COSTS}
        true_opt = {c: true_planners[c].solve(ep['q'], mode='optimal').expected_remaining_cost for c in COSTS}
        true_first = {c: true_planners[c].solve(ep['q'], mode='optimal') for c in COSTS}
        for assumed in COSTS:
            pa = SourceAwareResolutionPlanner(graphs, ep['action'], ep['masked_list'], physical_cost=1.0, semantic_cost=assumed); a_first = pa.solve(ep['q'], mode='optimal')
            for tc in COSTS:
                eval_cost = expected_true_cost_of_assumed_policy(pa, ep['q'], tc); regret = eval_cost - true_opt[tc]
                rows.append({'episode': eid, 'recording': ep['recording'], 'frame': ep['frame'], 'action': ep['action'], 'mask_k': ep['mask_k'], 'true_semantic_cost': tc, 'assumed_semantic_cost': assumed, 'true_optimal_expected_cost': true_opt[tc], 'misspecified_policy_expected_true_cost': eval_cost, 'absolute_expected_regret': regret, 'relative_expected_regret': regret / true_opt[tc] if true_opt[tc] > 0 else 0.0, 'same_first_decision': int(a_first.kind == true_first[tc].kind and a_first.target == true_first[tc].target), 'assumed_first_intervention': a_first.kind, 'true_first_intervention': true_first[tc].kind})
    df = pd.DataFrame(rows); df.to_csv(OUT/'semantic_cost_misspecification_expected_episodes.csv', index=False)
    summ = df.groupby(['true_semantic_cost','assumed_semantic_cost'], as_index=False).agg(episodes=('episode','size'), recordings=('recording','nunique'), mean_true_optimal_expected_cost=('true_optimal_expected_cost','mean'), mean_misspecified_expected_cost=('misspecified_policy_expected_true_cost','mean'), mean_absolute_expected_regret=('absolute_expected_regret','mean'), mean_relative_expected_regret=('relative_expected_regret','mean'), max_relative_expected_regret=('relative_expected_regret','max'), first_decision_agreement=('same_first_decision','mean'))
    summ.to_csv(OUT/'semantic_cost_misspecification_expected_summary.csv', index=False)
    robust = []
    for tc, d in summ.groupby('true_semantic_cost'):
        for tol in (0.001,0.01,0.05):
            good=d[d.mean_relative_expected_regret <= tol + 1e-12]
            robust.append({'true_semantic_cost': float(tc), 'relative_regret_tolerance': tol, 'assumed_costs_within_tolerance': ';'.join(str(x) for x in good.assumed_semantic_cost.tolist())})
    robust = pd.DataFrame(robust); robust.to_csv(OUT/'semantic_cost_misspecification_expected_robust_ranges.csv', index=False)
    report = {'schema':'tinyapv-source-aware-cost-misspecification-expected-v1','method':'For each mixed-uncertainty episode, an exact Bellman policy is computed under an assumed semantic-review cost c_hat. The same contingent policy is evaluated recursively under a different true cost c. Regret is measured against the exact Bellman optimum computed with the true c.','episodes':len(episodes),'recordings':len({ep['recording'] for ep in episodes}),'cost_grid':list(COSTS),'robust_ranges':robust.to_dict(orient='records'),'claim_boundary':'Costs are normalized decision-layer costs; physical evidence remains a controlled perfect reveal and semantic review perfectly resolves the action-specific semantic alternative. Robustness on this grid does not identify real sensor latency, expert-review burden, or human utility.'}
    (OUT/'semantic_cost_misspecification_expected_report.json').write_text(json.dumps(report,indent=2), encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
