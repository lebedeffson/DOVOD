from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

Action = tuple[str, int]


@dataclass(frozen=True)
class MinimaxRegretResult:
    action: Action
    worst_case_regret: float
    action_worst_case_regrets: tuple[tuple[Action, float], ...]
    scenario_optimal_values: tuple[tuple[Hashable, float], ...]


def minimax_regret_action(
    action_values_by_scenario: Mapping[Hashable, Mapping[Action, float]],
    *,
    atol: float = 1e-12,
) -> MinimaxRegretResult:
    """Choose the root action with the smallest worst-case regret.

    Each scenario supplies expected total loss for the same root actions, for
    example after rebuilding an exact planner under a different admissible cost
    vector. Regret is measured relative to that scenario's own optimal root
    action. Only actions available in every scenario are eligible.
    """
    if not action_values_by_scenario:
        raise ValueError("at least one scenario is required")
    scenario_items = list(action_values_by_scenario.items())
    common = set(scenario_items[0][1])
    for _, values in scenario_items[1:]:
        common.intersection_update(values)
    if not common:
        raise ValueError("scenarios have no common root action")

    optimal = {scenario: min(float(v) for v in values.values()) for scenario, values in scenario_items}
    worst = {
        action: max(float(values[action]) - optimal[scenario] for scenario, values in scenario_items)
        for action in common
    }
    best_value = min(worst.values())
    best_actions = [action for action, value in worst.items() if value <= best_value + float(atol)]
    action = min(best_actions)
    return MinimaxRegretResult(
        action=action,
        worst_case_regret=float(worst[action]),
        action_worst_case_regrets=tuple(sorted((a, float(v)) for a, v in worst.items())),
        scenario_optimal_values=tuple((scenario, float(optimal[scenario])) for scenario, _ in scenario_items),
    )
