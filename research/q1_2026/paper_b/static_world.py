from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence


@dataclass(frozen=True)
class World:
    state: tuple[int, ...]
    model: int
    physical_reliability: float
    semantic_reliability: float
    physical_orientation: int = 1
    semantic_orientation: int = 1

    def __post_init__(self) -> None:
        if self.physical_orientation not in (-1, 1):
            raise ValueError("physical_orientation must be -1 or 1")
        if self.semantic_orientation not in (-1, 1):
            raise ValueError("semantic_orientation must be -1 or 1")
        if not 0.5 <= self.physical_reliability <= 1.0:
            raise ValueError("physical_reliability must lie in [0.5,1]")
        if not 0.5 <= self.semantic_reliability <= 1.0:
            raise ValueError("semantic_reliability must lie in [0.5,1]")


@dataclass(frozen=True)
class Query:
    name: str
    kind: str
    index: int
    cost: float

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("query cost must be non-negative")


def cartesian_worlds(
    *,
    state_bits: int,
    models: Sequence[Sequence[int]],
    physical_reliabilities: Sequence[float],
    semantic_reliabilities: Sequence[float],
    physical_orientations: Sequence[int] = (1,),
    semantic_orientations: Sequence[int] = (1,),
) -> tuple[World, ...]:
    return tuple(
        World(tuple(state), mi, float(rp), float(rs), int(op), int(os))
        for state in product((0, 1), repeat=state_bits)
        for mi in range(len(models))
        for rp in physical_reliabilities
        for rs in semantic_reliabilities
        for op in physical_orientations
        for os in semantic_orientations
    )


def query_truth(query: Query, world: World, models: Sequence[Sequence[int]]) -> int:
    if query.kind == "state":
        return int(world.state[query.index])
    if query.kind == "model_feature":
        return int(query.index in models[world.model])
    if query.kind in ("calibrate_physical", "calibrate_semantic"):
        return 1
    raise ValueError(f"unknown query kind {query.kind}")


def query_reliability(query: Query, world: World) -> float:
    if query.kind in ("state", "calibrate_physical"):
        return world.physical_reliability
    if query.kind in ("model_feature", "calibrate_semantic"):
        return world.semantic_reliability
    raise ValueError(f"unknown query kind {query.kind}")


def query_orientation(query: Query, world: World) -> int:
    if query.kind in ("state", "calibrate_physical"):
        return world.physical_orientation
    if query.kind in ("model_feature", "calibrate_semantic"):
        return world.semantic_orientation
    raise ValueError(f"unknown query kind {query.kind}")


def observation_probability(query: Query, world: World, models: Sequence[Sequence[int]], observation: int) -> float:
    truth = query_truth(query, world, models)
    r = query_reliability(query, world)
    orientation = query_orientation(query, world)
    correct = r if orientation == 1 else 1.0 - r
    return correct if int(observation) == truth else 1.0 - correct


def admissible(world: World, models: Sequence[Sequence[int]]) -> bool:
    return all(world.state[i] == 1 for i in models[world.model])


def terminal_loss(
    decision: int,
    world: World,
    models: Sequence[Sequence[int]],
    *,
    false_allow: float = 2.0,
    false_block: float = 1.0,
) -> float:
    truth = int(admissible(world, models))
    if int(decision) == truth:
        return 0.0
    return float(false_allow if int(decision) == 1 else false_block)
