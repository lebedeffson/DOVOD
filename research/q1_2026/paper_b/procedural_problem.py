from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Sequence

from .count_dp import CountDPResult, EvidenceCountDP
from .pomcp import POMCPResult, StaticWorldPOMCP
from .static_world import Query, World


@dataclass(frozen=True)
class SourceMode:
    """One persistent source mode used in the finite hidden-world model.

    ``orientation=1`` means an honest source and ``orientation=-1`` means a
    persistently inverted source.  ``weight`` is a prior mass before
    normalization.  Reliability is the probability of returning the oriented
    truth, exactly as in :mod:`paper_b.static_world`.
    """

    reliability: float
    orientation: int = 1
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 <= float(self.reliability) <= 1.0:
            raise ValueError("reliability must lie in [0.5,1]")
        if int(self.orientation) not in (-1, 1):
            raise ValueError("orientation must be -1 or 1")
        if float(self.weight) <= 0.0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True)
class ProceduralAcquisitionProblem:
    """Action-local procedural information-acquisition problem.

    The builder maps a next-action question into a small exact hidden-world
    model.  Physical uncertainty is represented by completion probabilities for
    the action itself and for candidate prerequisites.  Semantic uncertainty is
    represented by a finite set of alternative prerequisite sets.  Queries ask
    either for a physical completion fact or whether a candidate prerequisite
    belongs to the currently applicable procedure rule.

    By default state probabilities use an action-local independent-Bernoulli
    model.  A real upstream tracker/filter may instead supply an arbitrary joint
    posterior over full procedural states; correlations are then preserved after
    projection onto the action-local bits.
    """

    worlds: tuple[World, ...]
    initial: tuple[float, ...]
    models: tuple[tuple[int, ...], ...]
    queries: tuple[Query, ...]
    local_feature_names: tuple[str, ...]
    semantic_alternatives_original: tuple[tuple[int, ...], ...]

    def solve_exact(
        self,
        *,
        horizon: int,
        false_allow: float = 2.0,
        false_block: float = 1.0,
    ) -> CountDPResult:
        solver = EvidenceCountDP(
            self.initial,
            self.worlds,
            self.models,
            self.queries,
            horizon=horizon,
            false_allow=false_allow,
            false_block=false_block,
        )
        return solver.solve()

    def root_action_values(
        self,
        *,
        horizon: int,
        false_allow: float = 2.0,
        false_block: float = 1.0,
    ) -> dict[tuple[str, int], float]:
        solver = EvidenceCountDP(
            self.initial,
            self.worlds,
            self.models,
            self.queries,
            horizon=horizon,
            false_allow=false_allow,
            false_block=false_block,
        )
        return solver.root_action_values()

    def solve_pomcp(
        self,
        *,
        horizon: int,
        simulations: int = 20_000,
        false_allow: float = 2.0,
        false_block: float = 1.0,
        seed: int = 0,
    ) -> POMCPResult:
        return StaticWorldPOMCP(
            self.initial,
            self.worlds,
            self.models,
            self.queries,
            horizon=horizon,
            false_allow=false_allow,
            false_block=false_block,
            seed=seed,
        ).solve(simulations=simulations)


def _normalize_positive_weights(weights: Sequence[float], n: int, name: str) -> tuple[float, ...]:
    if len(weights) != n:
        raise ValueError(f"{name} length mismatch")
    vals = tuple(float(x) for x in weights)
    if any(x < 0.0 for x in vals) or sum(vals) <= 0.0:
        raise ValueError(f"{name} must be non-negative with positive total mass")
    z = sum(vals)
    return tuple(x / z for x in vals)


def _collapse_semantic_alternatives(
    alternatives: Sequence[Sequence[int]],
    weights: Sequence[float],
) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    if not alternatives:
        raise ValueError("semantic_prerequisites must be non-empty")
    if len(alternatives) != len(weights):
        raise ValueError("semantic_weights length mismatch")
    mass: dict[tuple[int, ...], float] = {}
    for alt, weight in zip(alternatives, weights):
        key = tuple(sorted(set(map(int, alt))))
        mass[key] = mass.get(key, 0.0) + float(weight)
    keys = tuple(sorted(mass))
    vals = _normalize_positive_weights(tuple(mass[k] for k in keys), len(keys), "semantic_weights")
    return keys, vals


def _project_joint_state_hypotheses(
    hypotheses: Sequence[tuple[Sequence[int], float]],
    *,
    action_index: int,
    prerequisite_indices: Sequence[int],
    state_width: int,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Project an arbitrary joint posterior onto the action-local bits.

    Multiple full states that induce the same local state are collapsed.  This
    preserves correlations among the target action and its prerequisites and
    lets an upstream tracker/filter provide a non-factorized posterior.
    """
    if not hypotheses:
        raise ValueError("physical_state_hypotheses must be non-empty")
    mass: dict[tuple[int, ...], float] = {}
    for full_state, weight in hypotheses:
        st = tuple(map(int, full_state))
        if len(st) != state_width or any(v not in (0, 1) for v in st):
            raise ValueError("joint physical states must be full-width binary vectors")
        w = float(weight)
        if w < 0.0:
            raise ValueError("joint physical-state weights must be non-negative")
        local = (1 - st[action_index],) + tuple(st[p] for p in prerequisite_indices)
        mass[local] = mass.get(local, 0.0) + w
    z = sum(mass.values())
    if z <= 0.0:
        raise ValueError("joint physical-state weights need positive total mass")
    return tuple((state, weight / z) for state, weight in sorted(mass.items()))


def build_prerequisite_acquisition_problem(
    completion_probabilities: Sequence[float],
    *,
    action_index: int,
    semantic_prerequisites: Sequence[Sequence[int]],
    semantic_weights: Sequence[float] | None = None,
    physical_modes: Sequence[SourceMode] = (SourceMode(0.9),),
    semantic_modes: Sequence[SourceMode] = (SourceMode(0.9),),
    physical_query_cost: float = 0.05,
    semantic_query_cost: float = 0.10,
    physical_calibration_cost: float | None = None,
    semantic_calibration_cost: float | None = None,
    physical_state_hypotheses: Sequence[tuple[Sequence[int], float]] | None = None,
    max_worlds: int = 65_536,
) -> ProceduralAcquisitionProblem:
    """Build an exact action-local source-selection problem.

    Parameters
    ----------
    completion_probabilities:
        Marginal probabilities that each procedural component/action is already
        complete.  Only the target action and predicates occurring in at least
        one semantic alternative are expanded into the local world model.
    action_index:
        Index of the candidate next action.  Applicability includes the condition
        that the action is *not already complete*.
    semantic_prerequisites:
        Alternative prerequisite sets for the action, e.g. a version space from
        the Paper-A repair stage or an authoritative rule library.
    semantic_weights:
        Prior mass over alternatives.  Duplicate alternatives are collapsed and
        their masses added.
    physical_modes / semantic_modes:
        Persistent source reliability/orientation priors.  Multiple modes model
        session-level uncertainty rather than independent per-query noise.
    *_calibration_cost:
        If provided, add a known-truth calibration query for that source.  This is
        necessary when persistent orientation makes direct answers non-identifying.
    physical_state_hypotheses:
        Optional arbitrary joint posterior over full binary procedural states.
        When supplied, it replaces the independent-Bernoulli physical prior while
        keeping ``completion_probabilities`` as the declared state schema/width.
    """

    q = tuple(float(x) for x in completion_probabilities)
    if not q or any(not 0.0 <= x <= 1.0 for x in q):
        raise ValueError("completion_probabilities must lie in [0,1]")
    a = int(action_index)
    if not 0 <= a < len(q):
        raise ValueError("action_index out of range")
    if physical_query_cost < 0.0 or semantic_query_cost < 0.0:
        raise ValueError("query costs must be non-negative")
    if physical_calibration_cost is not None and physical_calibration_cost < 0.0:
        raise ValueError("physical_calibration_cost must be non-negative")
    if semantic_calibration_cost is not None and semantic_calibration_cost < 0.0:
        raise ValueError("semantic_calibration_cost must be non-negative")
    if not physical_modes or not semantic_modes:
        raise ValueError("source mode sets must be non-empty")

    raw_alts = tuple(tuple(map(int, alt)) for alt in semantic_prerequisites)
    for alt in raw_alts:
        if any(p == a for p in alt):
            raise ValueError("target action cannot be its own prerequisite")
        if any(p < 0 or p >= len(q) for p in alt):
            raise ValueError("prerequisite index out of range")
    raw_weights = (
        tuple(1.0 for _ in raw_alts)
        if semantic_weights is None
        else tuple(float(x) for x in semantic_weights)
    )
    alts, alt_weights = _collapse_semantic_alternatives(raw_alts, raw_weights)

    original_prereqs = tuple(sorted({p for alt in alts for p in alt}))
    # Local bit 0 means "target action is incomplete".  Remaining bits represent
    # the physical completion state of each candidate prerequisite.
    local_feature_names = (f"action_incomplete:{a}",) + tuple(
        f"component_complete:{p}" for p in original_prereqs
    )
    local_probs = (1.0 - q[a],) + tuple(q[p] for p in original_prereqs)
    original_to_local = {p: i + 1 for i, p in enumerate(original_prereqs)}
    local_models = tuple(
        tuple(sorted((0,) + tuple(original_to_local[p] for p in alt))) for alt in alts
    )

    if physical_state_hypotheses is None:
        local_state_prior = tuple(
            (tuple(map(int, state)), prod(prob if bit else 1.0 - prob for bit, prob in zip(state, local_probs)))
            for state in product((0, 1), repeat=len(local_probs))
        )
    else:
        local_state_prior = _project_joint_state_hypotheses(
            physical_state_hypotheses,
            action_index=a,
            prerequisite_indices=original_prereqs,
            state_width=len(q),
        )
    local_state_prior = tuple((state, mass) for state, mass in local_state_prior if mass > 0.0)

    projected = len(local_state_prior) * len(local_models) * len(physical_modes) * len(semantic_modes)
    if projected > int(max_worlds):
        raise ValueError(
            f"action-local world expansion would create {projected} worlds; "
            f"max_worlds={int(max_worlds)}"
        )

    worlds: list[World] = []
    masses: list[float] = []
    for state, state_mass in local_state_prior:
        for mi, model_mass in enumerate(alt_weights):
            for pm in physical_modes:
                for sm in semantic_modes:
                    mass = state_mass * model_mass * float(pm.weight) * float(sm.weight)
                    if mass <= 0.0:
                        continue
                    worlds.append(
                        World(
                            tuple(map(int, state)),
                            mi,
                            float(pm.reliability),
                            float(sm.reliability),
                            int(pm.orientation),
                            int(sm.orientation),
                        )
                    )
                    masses.append(mass)
    z = sum(masses)
    if z <= 0.0:
        raise RuntimeError("constructed problem has zero prior mass")
    initial = tuple(m / z for m in masses)

    # Derive query marginals from the actual local prior.  This matters when a
    # correlated joint posterior is supplied: the declared marginal schema is
    # then not allowed to silently decide whether a bit is queryable.
    local_marginals = tuple(
        sum(mass for state, mass in local_state_prior if state[local_idx] == 1)
        for local_idx in range(len(local_feature_names))
    )

    queries: list[Query] = []
    for local_idx, name in enumerate(local_feature_names):
        prob = local_marginals[local_idx]
        # A known physical bit has zero VoI and does not need a query action.
        if 0.0 < prob < 1.0:
            queries.append(Query(f"physical:{name}", "state", local_idx, float(physical_query_cost)))

    # Ask only semantic questions whose answers vary across the current version space.
    for original_p in original_prereqs:
        local_idx = original_to_local[original_p]
        membership = {int(local_idx in model) for model in local_models}
        if len(membership) > 1:
            queries.append(
                Query(
                    f"semantic:requires_component:{original_p}",
                    "model_feature",
                    local_idx,
                    float(semantic_query_cost),
                )
            )

    if physical_calibration_cost is not None:
        queries.append(Query("calibrate:physical", "calibrate_physical", 0, float(physical_calibration_cost)))
    if semantic_calibration_cost is not None:
        queries.append(Query("calibrate:semantic", "calibrate_semantic", 0, float(semantic_calibration_cost)))

    return ProceduralAcquisitionProblem(
        worlds=tuple(worlds),
        initial=initial,
        models=local_models,
        queries=tuple(queries),
        local_feature_names=local_feature_names,
        semantic_alternatives_original=alts,
    )
