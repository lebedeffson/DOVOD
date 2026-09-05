from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .contextual_repair import RepairEdit, SoftRepairResult, RepairSample, build_frozen_vocabulary, repaired_allows, solve_contextual_repair_soft_milp

_LITERAL = re.compile(r"^\(([^()\s]+)(?:\s+([^()]+?))?\)$")


@dataclass(frozen=True)
class DecisionObservation:
    state_literals: tuple[str, ...]
    action_label: str
    base_allow: int
    truth_allow: int
    def __post_init__(self) -> None:
        if self.base_allow not in (0,1) or self.truth_allow not in (0,1):
            raise ValueError("base_allow and truth_allow must be binary")


@dataclass(frozen=True)
class OperatorRepair:
    operator: str
    feature_names: tuple[str, ...]
    vocabulary: tuple[RepairEdit, ...]
    fit: SoftRepairResult


def parse_action_label(label: str) -> tuple[str, tuple[str, ...]]:
    text=label.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"invalid action label: {label!r}")
    parts=text[1:-1].split()
    if not parts: raise ValueError(f"empty action label: {label!r}")
    return parts[0],tuple(parts[1:])


def _normalize_literal(literal: str, action_args: Sequence[str]) -> str | None:
    text=literal.strip()
    if text.startswith("(not "): return None
    match=_LITERAL.match(text)
    if not match: return None
    pred=match.group(1); args=tuple((match.group(2) or "").split()); amap={obj:f"?{i}" for i,obj in enumerate(action_args)}
    if args and not any(obj in amap for obj in args): return None
    mapped=tuple(amap.get(obj,"*") for obj in args)
    return f"{pred}({','.join(mapped)})" if mapped else f"{pred}()"


def action_local_features(state_literals: Iterable[str], action_label: str) -> frozenset[str]:
    _,args=parse_action_label(action_label)
    out=set()
    for literal in state_literals:
        feat=_normalize_literal(str(literal),args)
        if feat is not None: out.add(feat)
    return frozenset(out)


def select_unlabeled_features(observations: Sequence[DecisionObservation], *, max_features: int = 10, min_frequency: float = 0.05, max_frequency: float = 0.95) -> tuple[str, ...]:
    if max_features < 0: raise ValueError("max_features must be non-negative")
    n=len(observations)
    if n==0 or max_features==0: return tuple()
    counts: dict[str,int]={}
    for obs in observations:
        for feat in action_local_features(obs.state_literals,obs.action_label): counts[feat]=counts.get(feat,0)+1
    candidates=[]
    for feat,count in counts.items():
        freq=count/n
        if min_frequency <= freq <= max_frequency: candidates.append((abs(freq-0.5),feat))
    candidates.sort()
    return tuple(feat for _,feat in candidates[:max_features])


def encode_observation(obs: DecisionObservation, feature_names: Sequence[str]) -> tuple[int, ...]:
    present=action_local_features(obs.state_literals,obs.action_label)
    return (int(obs.base_allow),)+tuple(int(name in present) for name in feature_names)


def fit_operator_repair(operator: str, observations: Sequence[DecisionObservation], *, max_features: int = 10, max_context_width: int = 1, edit_penalty: float = 0.25, false_allow_weight: float = 1.0, false_block_weight: float = 1.0) -> OperatorRepair:
    observations=tuple(observations)
    if not observations: raise ValueError("observations must be non-empty")
    if any(parse_action_label(o.action_label)[0] != operator for o in observations): raise ValueError("all observations must belong to operator")
    features=select_unlabeled_features(observations,max_features=max_features)
    states=tuple(encode_observation(o,features) for o in observations)
    raw=build_frozen_vocabulary(states,(0,),context_indices=tuple(range(1,len(features)+1)),max_context_width=max_context_width)
    vocab=tuple(RepairEdit(e.kind,e.context,prerequisite=e.prerequisite,weight=float(edit_penalty)) for e in raw)
    samples=tuple(RepairSample(s,o.truth_allow) for s,o in zip(states,observations))
    fit=solve_contextual_repair_soft_milp((0,),vocab,samples,false_allow_weight=false_allow_weight,false_block_weight=false_block_weight)
    return OperatorRepair(operator,features,vocab,fit)


def predict_operator_repair(model: OperatorRepair, observation: DecisionObservation) -> int:
    return int(repaired_allows(encode_observation(observation,model.feature_names),(0,),model.fit.selected_edits))


def decision_metrics(observations: Sequence[DecisionObservation], predictions: Sequence[int]) -> dict[str,float|int]:
    observations=tuple(observations); predictions=tuple(map(int,predictions))
    if len(observations)!=len(predictions): raise ValueError("length mismatch")
    if not observations: return {"n":0,"risk":0.0,"false_allow_rate":0.0,"false_block_rate":0.0,"false_allows":0,"false_blocks":0}
    fa=sum(p==1 and o.truth_allow==0 for p,o in zip(predictions,observations)); fb=sum(p==0 and o.truth_allow==1 for p,o in zip(predictions,observations))
    neg=sum(o.truth_allow==0 for o in observations); pos=sum(o.truth_allow==1 for o in observations)
    return {"n":len(observations),"risk":(fa+fb)/len(observations),"false_allow_rate":fa/neg if neg else 0.0,"false_block_rate":fb/pos if pos else 0.0,"false_allows":fa,"false_blocks":fb,"positives":pos,"negatives":neg}


def stable_bucket(key: str, modulus: int = 1000) -> int:
    if modulus<=0: raise ValueError("modulus must be positive")
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8],"big") % modulus
