from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from typing import Sequence

import numpy as np

from .static_world import Query, World, observation_probability, terminal_loss

Action = tuple[str, int]
History = tuple[tuple[int, int], ...]


@dataclass
class _ActionStat:
    visits: int = 0
    value_sum: float = 0.0
    @property
    def mean_reward(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class _Node:
    visits: int = 0
    actions: dict[Action, _ActionStat] = field(default_factory=dict)


@dataclass(frozen=True)
class POMCPResult:
    value: float
    action: Action
    simulations: int
    root_visits: int
    root_action_visits: tuple[tuple[Action, int], ...]


class StaticWorldPOMCP:
    def __init__(self, initial: Sequence[float], worlds: Sequence[World], models: Sequence[Sequence[int]], queries: Sequence[Query], *, horizon: int, false_allow: float = 2.0, false_block: float = 1.0, exploration: float = 1.25, seed: int = 0) -> None:
        self.worlds = tuple(worlds)
        self.models = tuple(tuple(m) for m in models)
        self.queries = tuple(queries)
        self.horizon = int(horizon)
        self.false_allow = float(false_allow)
        self.false_block = float(false_block)
        self.exploration = float(exploration)
        self.rng = Random(int(seed))
        b = np.asarray(initial, dtype=float)
        if len(b) != len(self.worlds) or len(b) == 0 or np.any(b < 0) or b.sum() <= 0:
            raise ValueError("invalid initial belief")
        self.initial = b / b.sum()
        self._lik = np.empty((len(self.queries),2,len(self.worlds)), dtype=float)
        for qi,q in enumerate(self.queries):
            for obs in (0,1):
                self._lik[qi,obs,:] = [observation_probability(q,w,self.models,obs) for w in self.worlds]
        self._loss = np.asarray([[terminal_loss(d,w,self.models,false_allow=self.false_allow,false_block=self.false_block) for w in self.worlds] for d in (0,1)], dtype=float)
        self.nodes: dict[History,_Node] = {}

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(("QUERY",i) for i in range(len(self.queries))) + (("DECIDE",0),("DECIDE",1))

    def _sample_world(self) -> int:
        u, c = self.rng.random(), 0.0
        for i,p in enumerate(self.initial):
            c += float(p)
            if u <= c:
                return i
        return len(self.worlds)-1

    def _sample_observation(self, qi: int, wi: int) -> int:
        return int(self.rng.random() < float(self._lik[int(qi),1,int(wi)]))

    def _posterior(self, history: History) -> np.ndarray:
        b = self.initial.copy()
        for qi,obs in history:
            b *= self._lik[int(qi),int(obs)]
            z = float(b.sum())
            if z <= 0:
                return self.initial.copy()
            b /= z
        return b

    def _rollout(self, wi: int, history: History) -> float:
        b = self._posterior(history)
        risks = self._loss @ b
        decision = int(np.argmin(risks))
        return -float(self._loss[decision,wi])

    def _valid_actions(self, history: History) -> tuple[Action, ...]:
        if len(history) >= self.horizon:
            return (("DECIDE",0),("DECIDE",1))
        return self.actions

    def _select_uct(self, node: _Node, valid: tuple[Action, ...]) -> Action:
        unvisited = [a for a in valid if node.actions.get(a,_ActionStat()).visits == 0]
        if unvisited:
            return unvisited[self.rng.randrange(len(unvisited))]
        logn = log(max(1,node.visits))
        scored = []
        for action in valid:
            stat = node.actions[action]
            scored.append((stat.mean_reward + self.exploration*sqrt(logn/stat.visits), action))
        return max(scored, key=lambda x:(x[0],x[1]))[1]

    def _simulate(self, wi: int, history: History) -> float:
        if len(history) >= self.horizon:
            return self._rollout(wi, history)
        node = self.nodes.get(history)
        if node is None:
            self.nodes[history] = _Node()
            return self._rollout(wi, history)
        action = self._select_uct(node, self._valid_actions(history))
        if action[0] == "DECIDE":
            reward = -float(self._loss[int(action[1]),wi])
        else:
            qi = int(action[1])
            obs = self._sample_observation(qi,wi)
            reward = -float(self.queries[qi].cost) + self._simulate(wi, history+((qi,obs),))
        node.visits += 1
        stat = node.actions.setdefault(action,_ActionStat())
        stat.visits += 1
        stat.value_sum += reward
        return reward

    def solve(self, simulations: int = 20000) -> POMCPResult:
        sims = int(simulations)
        if sims <= 0:
            raise ValueError("simulations must be positive")
        self.nodes.setdefault(tuple(),_Node())
        for _ in range(sims):
            self._simulate(self._sample_world(), tuple())
        root = self.nodes[tuple()]
        visited = [(a,s) for a,s in root.actions.items() if s.visits > 0]
        if not visited:
            raise RuntimeError("POMCP root has no visited action")
        action, stat = max(visited, key=lambda item:(item[1].visits,item[1].mean_reward))
        return POMCPResult(value=-float(stat.mean_reward), action=action, simulations=sims, root_visits=int(root.visits), root_action_visits=tuple(sorted((a,s.visits) for a,s in visited)))
