from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

State = tuple[int, ...]
Context = tuple[tuple[int, int], ...]


@dataclass(frozen=True, order=True)
class RepairEdit:
    kind: str
    context: Context
    prerequisite: int = -1
    weight: float = 1.0
    def __post_init__(self) -> None:
        if self.kind not in {"exception","guard"}:
            raise ValueError("kind must be 'exception' or 'guard'")
        if self.kind == "exception" and self.prerequisite < 0:
            raise ValueError("exception requires a prerequisite index")
        if self.kind == "guard" and self.prerequisite != -1:
            raise ValueError("guard must not name a prerequisite")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if tuple(sorted(self.context)) != self.context:
            raise ValueError("context literals must be sorted")
        if len({p for p,_ in self.context}) != len(self.context):
            raise ValueError("context cannot repeat a predicate")
        if any(v not in (0,1) for _,v in self.context):
            raise ValueError("context values must be binary")


@dataclass(frozen=True)
class RepairSample:
    state: State
    allow: int
    def __post_init__(self) -> None:
        if self.allow not in (0,1):
            raise ValueError("allow label must be binary")


@dataclass(frozen=True)
class RepairResult:
    selected_indices: tuple[int, ...]
    selected_edits: tuple[RepairEdit, ...]
    objective: float
    predictions: tuple[int, ...]


@dataclass(frozen=True)
class SoftRepairResult:
    selected_indices: tuple[int, ...]
    selected_edits: tuple[RepairEdit, ...]
    objective: float
    predictions: tuple[int, ...]
    error_indices: tuple[int, ...]
    false_allow_errors: int
    false_block_errors: int


def context_matches(state: State, context: Context) -> bool:
    return all(0 <= p < len(state) and int(state[p]) == int(v) for p,v in context)


def build_frozen_vocabulary(states: Sequence[State], base_prerequisites: Sequence[int], *, context_indices: Sequence[int] | None = None, max_context_width: int = 1, include_exceptions: bool = True, include_guards: bool = True) -> tuple[RepairEdit, ...]:
    states = tuple(tuple(map(int,s)) for s in states)
    if not states:
        raise ValueError("states must be non-empty")
    n = len(states[0])
    if any(len(s) != n or any(v not in (0,1) for v in s) for s in states):
        raise ValueError("states must have one common binary width")
    base = tuple(sorted(set(map(int,base_prerequisites))))
    ctx_idx = tuple(range(n)) if context_indices is None else tuple(sorted(set(map(int,context_indices))))
    width = int(max_context_width)
    contexts: set[Context] = set()
    if width == 0:
        contexts.add(tuple())
    for k in range(1,min(width,len(ctx_idx))+1):
        for inds in combinations(ctx_idx,k):
            contexts.update({tuple((p,int(s[p])) for p in inds) for s in states})
    edits: set[RepairEdit] = set()
    for context in sorted(contexts):
        if include_exceptions:
            for p in base:
                edits.add(RepairEdit("exception",context,prerequisite=p))
        if include_guards:
            edits.add(RepairEdit("guard",context))
    return tuple(sorted(edits))


def repaired_allows(state: State, base_prerequisites: Sequence[int], selected_edits: Iterable[RepairEdit]) -> bool:
    selected = tuple(selected_edits)
    for p in base_prerequisites:
        if int(state[int(p)]) != 1 and not any(e.kind=="exception" and e.prerequisite==int(p) and context_matches(state,e.context) for e in selected):
            return False
    if any(e.kind=="guard" and context_matches(state,e.context) for e in selected):
        return False
    return True


def _matching_exception_indices(state: State, prerequisite: int, vocabulary: Sequence[RepairEdit]) -> tuple[int, ...]:
    return tuple(i for i,e in enumerate(vocabulary) if e.kind=="exception" and e.prerequisite==prerequisite and context_matches(state,e.context))


def _matching_guard_indices(state: State, vocabulary: Sequence[RepairEdit]) -> tuple[int, ...]:
    return tuple(i for i,e in enumerate(vocabulary) if e.kind=="guard" and context_matches(state,e.context))


def _validate_inputs(base_prerequisites, vocabulary, samples):
    base = tuple(sorted(set(map(int,base_prerequisites))))
    vocab = tuple(vocabulary)
    samples = tuple(samples)
    if not samples:
        raise ValueError("samples must be non-empty")
    n = len(samples[0].state)
    if any(len(s.state) != n for s in samples):
        raise ValueError("sample state lengths differ")
    return base,vocab,samples


def solve_contextual_repair_milp(base_prerequisites, vocabulary, samples) -> RepairResult:
    base,vocab,samples = _validate_inputs(base_prerequisites,vocabulary,samples)
    m = len(vocab)
    aux=[]
    for si,sample in enumerate(samples):
        if sample.allow==0:
            for p in base:
                if int(sample.state[p]) != 1:
                    aux.append((si,p,_matching_exception_indices(sample.state,p,vocab)))
    z_offset=m
    z_index={(si,p):z_offset+k for k,(si,p,_) in enumerate(aux)}
    nvar=m+len(aux)
    rows=[]; lower=[]; upper=[]
    def add(coeffs,lo,hi):
        row=np.zeros(nvar,dtype=float)
        for idx,val in coeffs.items(): row[int(idx)]=float(val)
        rows.append(row); lower.append(float(lo)); upper.append(float(hi))
    for si,sample in enumerate(samples):
        state=sample.state
        if sample.allow==1:
            for p in base:
                if int(state[p])==1: continue
                covers=_matching_exception_indices(state,p,vocab)
                if not covers: raise RuntimeError(f"infeasible positive sample {si}")
                add({i:1.0 for i in covers},1.0,np.inf)
            for gi in _matching_guard_indices(state,vocab): add({gi:1.0},0.0,0.0)
        else:
            block={}
            for p in base:
                if int(state[p])==1: continue
                covers=_matching_exception_indices(state,p,vocab); zi=z_index[(si,p)]; block[zi]=1.0
                if not covers: add({zi:1.0},1.0,1.0)
                else:
                    for ei in covers: add({zi:1.0,ei:1.0},-np.inf,1.0)
                    coeff={zi:1.0}; coeff.update({ei:1.0 for ei in covers}); add(coeff,1.0,np.inf)
            for gi in _matching_guard_indices(state,vocab): block[gi]=block.get(gi,0.0)+1.0
            if not block: raise RuntimeError(f"infeasible negative sample {si}")
            add(block,1.0,np.inf)
    A=np.vstack(rows) if rows else np.zeros((0,nvar),dtype=float)
    c=np.zeros(nvar); c[:m]=[e.weight for e in vocab]
    res=milp(c=c,integrality=np.ones(nvar,dtype=int),bounds=Bounds(np.zeros(nvar),np.ones(nvar)),constraints=LinearConstraint(A,np.asarray(lower),np.asarray(upper)))
    if not res.success or res.x is None: raise RuntimeError(f"MILP failed: {res.message}")
    idx=tuple(i for i,x in enumerate(res.x[:m]) if x>=0.5); edits=tuple(vocab[i] for i in idx)
    preds=tuple(int(repaired_allows(s.state,base,edits)) for s in samples)
    return RepairResult(idx,edits,float(sum(vocab[i].weight for i in idx)),preds)


def solve_contextual_repair_bruteforce(base_prerequisites, vocabulary, samples) -> RepairResult:
    base,vocab,samples=_validate_inputs(base_prerequisites,vocabulary,samples)
    best=None
    for mask in product((0,1),repeat=len(vocab)):
        idx=tuple(i for i,bit in enumerate(mask) if bit); edits=tuple(vocab[i] for i in idx)
        preds=tuple(int(repaired_allows(s.state,base,edits)) for s in samples)
        if preds != tuple(s.allow for s in samples): continue
        candidate=(float(sum(vocab[i].weight for i in idx)),idx)
        if best is None or candidate < best: best=candidate
    if best is None: raise RuntimeError("infeasible")
    obj,idx=best; edits=tuple(vocab[i] for i in idx)
    return RepairResult(idx,edits,obj,tuple(int(repaired_allows(s.state,base,edits)) for s in samples))


def solve_contextual_repair_soft_milp(base_prerequisites, vocabulary, samples, *, false_allow_weight: float = 1.0, false_block_weight: float = 1.0) -> SoftRepairResult:
    base,vocab,samples=_validate_inputs(base_prerequisites,vocabulary,samples)
    m=len(vocab); ns=len(samples)
    aux=[]
    for si,sample in enumerate(samples):
        if sample.allow==0:
            for p in base:
                if int(sample.state[p]) != 1: aux.append((si,p,_matching_exception_indices(sample.state,p,vocab)))
    err_offset=m; z_offset=m+ns; z_index={(si,p):z_offset+k for k,(si,p,_) in enumerate(aux)}; nvar=m+ns+len(aux)
    rows=[]; lower=[]; upper=[]
    def add(coeffs,lo,hi):
        row=np.zeros(nvar)
        for idx,val in coeffs.items(): row[int(idx)]=float(val)
        rows.append(row); lower.append(float(lo)); upper.append(float(hi))
    for si,sample in enumerate(samples):
        state=sample.state; ei=err_offset+si
        if sample.allow==1:
            for p in base:
                if int(state[p])==1: continue
                covers=_matching_exception_indices(state,p,vocab)
                if covers:
                    coeff={i:1.0 for i in covers}; coeff[ei]=1.0; add(coeff,1.0,np.inf)
                else: add({ei:1.0},1.0,1.0)
            for gi in _matching_guard_indices(state,vocab): add({gi:1.0,ei:-1.0},-np.inf,0.0)
        else:
            block={ei:1.0}
            for p in base:
                if int(state[p])==1: continue
                covers=_matching_exception_indices(state,p,vocab); zi=z_index[(si,p)]; block[zi]=1.0
                if not covers: add({zi:1.0},1.0,1.0)
                else:
                    for edit_idx in covers: add({zi:1.0,edit_idx:1.0},-np.inf,1.0)
                    coeff={zi:1.0}; coeff.update({edit_idx:1.0 for edit_idx in covers}); add(coeff,1.0,np.inf)
            for gi in _matching_guard_indices(state,vocab): block[gi]=block.get(gi,0.0)+1.0
            add(block,1.0,np.inf)
    A=np.vstack(rows) if rows else np.zeros((0,nvar)); c=np.zeros(nvar); c[:m]=[e.weight for e in vocab]
    for si,sample in enumerate(samples): c[err_offset+si]=false_block_weight if sample.allow==1 else false_allow_weight
    res=milp(c=c,integrality=np.ones(nvar,dtype=int),bounds=Bounds(np.zeros(nvar),np.ones(nvar)),constraints=LinearConstraint(A,np.asarray(lower),np.asarray(upper)))
    if not res.success or res.x is None: raise RuntimeError(f"soft repair MILP failed: {res.message}")
    idx=tuple(i for i,x in enumerate(res.x[:m]) if x>=0.5); edits=tuple(vocab[i] for i in idx)
    preds=tuple(int(repaired_allows(s.state,base,edits)) for s in samples)
    errors=tuple(i for i,(pred,s) in enumerate(zip(preds,samples)) if pred!=s.allow)
    fa=sum(pred==1 and samples[i].allow==0 for i,pred in enumerate(preds)); fb=sum(pred==0 and samples[i].allow==1 for i,pred in enumerate(preds))
    obj=float(sum(e.weight for e in edits)+false_allow_weight*fa+false_block_weight*fb)
    return SoftRepairResult(idx,edits,obj,preds,errors,int(fa),int(fb))
