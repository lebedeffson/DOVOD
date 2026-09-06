from __future__ import annotations

import json
from importlib.metadata import version as package_version
from pathlib import Path

from pomdp_py.algorithms.value_function import qvalue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'paper_b_pomdp_py_crosscheck.json'

R = 0.86
QUERY_COST = 0.015
QUERY_BUDGET = 2
BAD_QUERY_REWARD = -10.0

ACTIONS = ['decide-0', 'decide-1', 'calibrate', 'direct']
OBS = [0, 1, 'none']
TERMINAL = ('terminal',)
WORLDS = [(x, orientation) for x in (0, 1) for orientation in (-1, 1)]
STATES = [(x, o, budget) for x, o in WORLDS for budget in range(QUERY_BUDGET + 1)] + [TERMINAL]


def is_terminal(s):
    return s == TERMINAL


class Transition:
    def probability(self, sp, s, action, normalized=False, **kwargs):
        if is_terminal(s):
            return 1.0 if sp == TERMINAL else 0.0
        x, o, budget = s
        if action.startswith('decide-'):
            return 1.0 if sp == TERMINAL else 0.0
        if action in {'calibrate', 'direct'}:
            if budget <= 0:
                return 1.0 if sp == TERMINAL else 0.0
            target = (x, o, budget - 1)
            return 1.0 if sp == target else 0.0
        raise ValueError(action)


class Observation:
    def probability(self, obs, sp, action, normalized=False, **kwargs):
        if is_terminal(sp):
            return 1.0 if obs == 'none' else 0.0
        x, orientation, _ = sp
        if action.startswith('decide-'):
            return 0.0
        if action not in {'calibrate', 'direct'}:
            raise ValueError(action)
        truth = 1 if action == 'calibrate' else x
        correct = R if orientation == 1 else 1.0 - R
        p1 = correct if truth == 1 else 1.0 - correct
        if obs == 1:
            return p1
        if obs == 0:
            return 1.0 - p1
        return 0.0


class Reward:
    def sample(self, s, action, sp, normalized=False):
        if is_terminal(s):
            return 0.0
        x, _, budget = s
        if action == 'decide-0':
            return 0.0 if x == 0 else -1.0
        if action == 'decide-1':
            return 0.0 if x == 1 else -1.0
        if action in {'calibrate', 'direct'}:
            return -QUERY_COST if budget > 0 else BAD_QUERY_REWARD
        raise ValueError(action)


TRANSITION = Transition()
OBSERVATION = Observation()
REWARD = Reward()


def initial_belief():
    b = {s: 0.0 for s in STATES}
    for x, o in WORLDS:
        b[(x, o, QUERY_BUDGET)] = 1.0 / len(WORLDS)
    return b


def main():
    b0 = initial_belief()
    qvals = {
        action: qvalue(
            b0,
            action,
            STATES,
            ACTIONS,
            OBS,
            TRANSITION,
            OBSERVATION,
            REWARD,
            1.0,
            horizon=QUERY_BUDGET + 1,
        )
        for action in ACTIONS
    }
    external_costs = {a: -float(v) for a, v in qvals.items()}
    expected_costs = {
        'decide-0': 0.5,
        'decide-1': 0.5,
        'calibrate': 2 * QUERY_COST + 2 * R * (1 - R),
        'direct': 2 * QUERY_COST + 2 * R * (1 - R),
    }
    abs_errors = {a: abs(external_costs[a] - expected_costs[a]) for a in ACTIONS}
    report = {
        'schema': 'dovod-paper-b-pomdp-py-crosscheck-v1',
        'library': 'pomdp-py',
        'library_version': package_version('pomdp-py'),
        'library_reference': 'Zheng and Tellex, pomdp_py, PlanRob/ICAPS 2020',
        'case': {'reliability': R, 'query_cost': QUERY_COST, 'query_budget': QUERY_BUDGET},
        'external_root_costs': external_costs,
        'closed_form_expected_costs': expected_costs,
        'absolute_errors': abs_errors,
        'max_absolute_error': max(abs_errors.values()),
        'best_external_actions': sorted(a for a, v in external_costs.items() if v <= min(external_costs.values()) + 1e-12),
        'expected_optimal_actions': ['calibrate', 'direct'],
        'match': max(abs_errors.values()) < 1e-10,
        'claim_boundary': (
            'This is an independent external-library Bellman cross-check on the finite two-query orientation case. '
            'pomdp_py exact value-function recursion is used directly; DOVOD count/history DP code is not imported.'
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report['match']:
        raise SystemExit('pomdp_py exact root-action values do not match closed form')


if __name__ == '__main__':
    main()
