from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / 'research' / 'reference_impl'
sys.path.insert(0, str(ROOT))

from q1_uncertainty_source_aware import build_episodes
from recording_bootstrap import paired_recording_bootstrap
from source_aware_resolution_planner import SourceAwareResolutionPlanner, EPS

OUT = PACKAGE / 'results' / 'information_selection' / 'primary_bellman_myopic'
OUT.mkdir(parents=True, exist_ok=True)
SEMANTIC_COSTS = (1.0, 2.0, 5.0, 10.0)


def myopic_expected_cost(planner: SourceAwareResolutionPlanner, q, active=None):
    qkey = planner._qkey(q)
    if active is None:
        active = tuple(range(len(planner.graphs)))

    @lru_cache(maxsize=None)
    def value(qk: tuple[float, ...], act: tuple[int, ...]):
        current = planner._uncertainty(qk, act)
        if current <= EPS:
            return 0.0, None
        unresolved = [j for j in planner.queryable if EPS < qk[j] < 1.0 - EPS]
        groups = planner._semantic_groups(act)
        candidates = []
        for j in unresolved:
            pj = float(qk[j])
            q1 = list(qk); q1[j] = 1.0
            q0 = list(qk); q0[j] = 0.0
            q1 = tuple(q1); q0 = tuple(q0)
            u_after = pj * planner._uncertainty(q1, act) + (1.0 - pj) * planner._uncertainty(q0, act)
            reduction = current - u_after
            if reduction > EPS:
                future = pj * value(q1, act)[0] + (1.0 - pj) * value(q0, act)[0]
                total = planner.physical_cost + future
                candidates.append((reduction / planner.physical_cost, total, 'PHYSICAL_QUERY', int(j)))
        if len(groups) > 1:
            denom = float(len(act)); u_after = 0.0; future = 0.0
            for group in groups:
                mass = len(group) / denom
                u_after += mass * planner._uncertainty(qk, group)
                future += mass * value(qk, group)[0]
            reduction = current - u_after
            if reduction > EPS:
                candidates.append((reduction / planner.semantic_cost, planner.semantic_cost + future, 'SEMANTIC_REVIEW', planner.action))
        if not candidates:
            return float('inf'), None
        best = max(candidates, key=lambda x: (x[0], x[2] == 'SEMANTIC_REVIEW', -x[3]))
        return float(best[1]), (best[2], int(best[3]))
    return value(qkey, active)


def myopic_realized_cost(planner: SourceAwareResolutionPlanner, q, true_state, latent_graph_index: int):
    qkey = planner._qkey(q); active = tuple(range(len(planner.graphs))); latent = int(latent_graph_index)
    total = 0.0; physical = 0; semantic = 0; trace = []
    for _ in range(len(planner.queryable) + 3):
        current = planner._uncertainty(qkey, active)
        if current <= EPS: break
        unresolved = [j for j in planner.queryable if EPS < qkey[j] < 1.0 - EPS]
        groups = planner._semantic_groups(active); candidates = []
        for j in unresolved:
            pj = float(qkey[j]); q1 = list(qkey); q1[j] = 1.0; q0 = list(qkey); q0[j] = 0.0
            q1 = tuple(q1); q0 = tuple(q0)
            u_after = pj * planner._uncertainty(q1, active) + (1.0 - pj) * planner._uncertainty(q0, active)
            reduction = current - u_after
            if reduction > EPS: candidates.append((reduction / planner.physical_cost, 'PHYSICAL_QUERY', int(j)))
        if len(groups) > 1:
            denom = float(len(active)); u_after = sum((len(g) / denom) * planner._uncertainty(qkey, g) for g in groups)
            reduction = current - u_after
            if reduction > EPS: candidates.append((reduction / planner.semantic_cost, 'SEMANTIC_REVIEW', planner.action))
        if not candidates: break
        _, kind, target = max(candidates, key=lambda x: (x[0], x[1] == 'SEMANTIC_REVIEW', -x[2]))
        if kind == 'PHYSICAL_QUERY':
            ql = list(qkey); ql[target] = 1.0 if int(true_state[target]) == 1 else 0.0; qkey = tuple(ql)
            total += planner.physical_cost; physical += 1; trace.append(f'PHYSICAL:{target}')
        else:
            latent_alt = tuple(sorted(planner.graphs[latent].get(planner.action, [])))
            active = tuple(i for i in active if tuple(sorted(planner.graphs[i].get(planner.action, []))) == latent_alt)
            total += planner.semantic_cost; semantic += 1; trace.append(f'SEMANTIC:{planner.action}')
    final = planner._uncertainty(qkey, active)
    return {'realized_cost': float(total), 'physical_queries': physical, 'semantic_queries': semantic, 'resolved': int(final <= EPS), 'final_variance': float(final), 'trace': '>'.join(trace)}


def main():
    episodes, graphs = build_episodes(); exp_rows = []; real_rows = []
    for eid, ep in enumerate(episodes):
        if ep['initial_semantic_variance'] <= EPS: continue
        for sem_cost in SEMANTIC_COSTS:
            planner = SourceAwareResolutionPlanner(graphs, ep['action'], ep['masked_list'], physical_cost=1.0, semantic_cost=sem_cost)
            opt = planner.solve(ep['q'], mode='optimal'); my_cost, my_dec = myopic_expected_cost(planner, ep['q'])
            exp_rows += [
                {'episode': eid, 'recording': ep['recording'], 'frame': ep['frame'], 'action': ep['action'], 'mask_k': ep['mask_k'], 'semantic_cost': sem_cost, 'policy': 'bellman_optimal', 'expected_cost': opt.expected_remaining_cost, 'first_intervention': opt.kind, 'first_target': opt.target},
                {'episode': eid, 'recording': ep['recording'], 'frame': ep['frame'], 'action': ep['action'], 'mask_k': ep['mask_k'], 'semantic_cost': sem_cost, 'policy': 'myopic_gain_per_cost', 'expected_cost': my_cost, 'first_intervention': my_dec[0] if my_dec else 'RESOLVED', 'first_target': my_dec[1] if my_dec else -1},
            ]
            for latent in range(len(graphs)):
                opt_r = planner.realized_cost(ep['q'], ep['state'], latent, mode='optimal'); my_r = myopic_realized_cost(planner, ep['q'], ep['state'], latent)
                for policy, rr in [('bellman_optimal', opt_r), ('myopic_gain_per_cost', my_r)]:
                    real_rows.append({'episode': eid, 'latent_graph': latent, 'recording': ep['recording'], 'frame': ep['frame'], 'action': ep['action'], 'mask_k': ep['mask_k'], 'semantic_cost': sem_cost, 'policy': policy, **rr})
    exp = pd.DataFrame(exp_rows); rea = pd.DataFrame(real_rows)
    exp.to_csv(OUT / 'bellman_vs_myopic_expected.csv', index=False); rea.to_csv(OUT / 'bellman_vs_myopic_realized.csv', index=False)
    exp_summary = exp.groupby(['semantic_cost','policy'], as_index=False).agg(episodes=('episode','size'), recordings=('recording','nunique'), mean_expected_cost=('expected_cost','mean'), semantic_first_rate=('first_intervention', lambda s: float((s == 'SEMANTIC_REVIEW').mean())))
    real_summary = rea.groupby(['semantic_cost','policy'], as_index=False).agg(scenarios=('episode','size'), recordings=('recording','nunique'), mean_realized_cost=('realized_cost','mean'), mean_physical_queries=('physical_queries','mean'), mean_semantic_queries=('semantic_queries','mean'), resolution_rate=('resolved','mean'))
    exp_summary.to_csv(OUT / 'bellman_vs_myopic_expected_summary.csv', index=False); real_summary.to_csv(OUT / 'bellman_vs_myopic_realized_summary.csv', index=False)
    boot_exp = {}; boot_real = {}
    for c in SEMANTIC_COSTS:
        d = exp[exp.semantic_cost == c]
        boot_exp[f'c{int(c)}'] = paired_recording_bootstrap(d, 'bellman_optimal', 'myopic_gain_per_cost', 'expected_cost', index_cols=('recording','frame','action','mask_k'), n_boot=5000, seed=20260920+int(c))
        r = rea[rea.semantic_cost == c]
        boot_real[f'c{int(c)}'] = paired_recording_bootstrap(r, 'bellman_optimal', 'myopic_gain_per_cost', 'realized_cost', index_cols=('latent_graph','recording','frame','action','mask_k'), n_boot=5000, seed=20261020+int(c))
    epiv = exp.pivot_table(index=['episode','recording','frame','action','mask_k','semantic_cost'], columns='policy', values='expected_cost').reset_index(); epiv['myopic_minus_bellman'] = epiv['myopic_gain_per_cost'] - epiv['bellman_optimal']
    dec = exp.pivot_table(index=['episode','recording','frame','action','mask_k','semantic_cost'], columns='policy', values='first_intervention', aggfunc='first').reset_index(); dec['different_first_intervention'] = (dec['bellman_optimal'] != dec['myopic_gain_per_cost']).astype(int)
    details = []
    for c in SEMANTIC_COSTS:
        x = epiv[epiv.semantic_cost == c]; q = dec[dec.semantic_cost == c]
        details.append({'semantic_cost': c, 'episodes': int(len(x)), 'mean_expected_cost_bellman': float(x['bellman_optimal'].mean()), 'mean_expected_cost_myopic': float(x['myopic_gain_per_cost'].mean()), 'mean_myopic_minus_bellman': float(x['myopic_minus_bellman'].mean()), 'median_myopic_minus_bellman': float(x['myopic_minus_bellman'].median()), 'myopic_worse_fraction': float((x['myopic_minus_bellman'] > 1e-12).mean()), 'exact_tie_fraction': float((x['myopic_minus_bellman'].abs() <= 1e-12).mean()), 'first_intervention_disagreement_rate': float(q['different_first_intervention'].mean())})
    report = {'schema': 'tinyapv-bellman-vs-myopic-v1', 'boundary': 'Publication-layer analysis over the frozen research snapshot. The comparison uses the train-derived semantic version space and controlled masked-evidence MECCANO test episodes. Physical queries are perfect reveals and semantic review has normalized cost.', 'mixed_uncertainty_episodes': int(exp['episode'].nunique()), 'semantic_costs': list(SEMANTIC_COSTS), 'expected_summary': exp_summary.to_dict(orient='records'), 'realized_summary': real_summary.to_dict(orient='records'), 'pairwise_details': details, 'recording_cluster_bootstrap_expected_bellman_minus_myopic': boot_exp, 'recording_cluster_bootstrap_realized_bellman_minus_myopic': boot_real}
    (OUT / 'bellman_vs_myopic_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
