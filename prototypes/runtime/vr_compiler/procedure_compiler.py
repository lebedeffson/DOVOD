from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AltCondition:
    require_mask: int
    forbid_mask: int


@dataclass(frozen=True)
class CompiledAction:
    index: int
    action_id: str
    label: str
    kind: str
    canonical_order: int
    role: str
    irreversible: bool
    preconditions: tuple[AltCondition, ...]
    add_mask: int
    remove_mask: int


@dataclass(frozen=True)
class CompiledProcedure:
    procedure_id: str
    facts: tuple[str, ...]
    fact_index: dict[str, int]
    initial_mask: int
    goal_require_mask: int
    goal_forbid_mask: int
    actions: tuple[CompiledAction, ...]

    @property
    def action_by_id(self) -> dict[str, CompiledAction]:
        return {a.action_id:a for a in self.actions}

    def fact_set(self, mask: int) -> set[str]:
        return {name for i,name in enumerate(self.facts) if mask & (1<<i)}

    def feasible(self, mask: int, action: CompiledAction | str) -> bool:
        if isinstance(action,str): action=self.action_by_id[action]
        if action.add_mask and (mask & action.add_mask) == action.add_mask and action.remove_mask == 0:
            return False
        for alt in action.preconditions:
            if (mask & alt.require_mask) == alt.require_mask and (mask & alt.forbid_mask) == 0:
                nxt=self.apply_unchecked(mask,action)
                if nxt != mask:
                    return True
        return False

    def apply_unchecked(self, mask: int, action: CompiledAction) -> int:
        return (mask | action.add_mask) & ~action.remove_mask

    def apply(self, mask: int, action: CompiledAction | str) -> tuple[int,bool]:
        if isinstance(action,str): action=self.action_by_id[action]
        if not self.feasible(mask,action): return mask,False
        return self.apply_unchecked(mask,action),True

    def feasible_actions(self, mask: int, *, roles: set[str] | None=None) -> list[CompiledAction]:
        return [a for a in self.actions if (roles is None or a.role in roles) and self.feasible(mask,a)]

    def goal(self, mask: int) -> bool:
        return (mask & self.goal_require_mask)==self.goal_require_mask and (mask & self.goal_forbid_mask)==0

    def why_not(self, mask: int, action: CompiledAction | str) -> list[dict]:
        if isinstance(action,str): action=self.action_by_id[action]
        rows=[]
        for alt in action.preconditions:
            missing=[self.facts[i] for i in range(len(self.facts)) if alt.require_mask&(1<<i) and not mask&(1<<i)]
            forbidden=[self.facts[i] for i in range(len(self.facts)) if alt.forbid_mask&(1<<i) and mask&(1<<i)]
            rows.append({'missing':missing,'forbidden_present':forbidden})
        rows.sort(key=lambda x:(len(x['missing'])+len(x['forbidden_present']),x['missing'],x['forbidden_present']))
        return rows


def _mask(names: Iterable[str], index: dict[str,int]) -> int:
    m=0
    for n in names: m |= 1<<index[n]
    return m


def compile_procedure(path: str | Path) -> CompiledProcedure:
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    facts=tuple(map(str,raw['facts']))
    if len(facts)!=len(set(facts)): raise ValueError('duplicate fact')
    fi={f:i for i,f in enumerate(facts)}
    ids=[]; actions=[]
    for i,a in enumerate(raw['actions']):
        aid=str(a['id'])
        if aid in ids: raise ValueError(f'duplicate action id {aid}')
        ids.append(aid); alts=[]
        for alt in a.get('precondition_any_of',[{'all':[],'none':[]}]):
            req=set(map(str,alt.get('all',[]))); forb=set(map(str,alt.get('none',[]))); unknown=(req|forb)-set(fi)
            if unknown: raise ValueError(f'{aid}: unknown facts {sorted(unknown)}')
            if req&forb: raise ValueError(f'{aid}: fact both required and forbidden')
            alts.append(AltCondition(_mask(req,fi),_mask(forb,fi)))
        add=set(map(str,a.get('add',[]))); rem=set(map(str,a.get('remove',[]))); unknown=(add|rem)-set(fi)
        if unknown: raise ValueError(f'{aid}: unknown effect facts {sorted(unknown)}')
        if add&rem: raise ValueError(f'{aid}: fact both added and removed')
        if not add and not rem: raise ValueError(f'{aid}: action has no state effect')
        actions.append(CompiledAction(index=i,action_id=aid,label=str(a.get('label',aid)),kind=str(a.get('kind','generic')),canonical_order=int(a.get('canonical_order',i)),role=str(a.get('role','progress')),irreversible=bool(a.get('irreversible',False)),preconditions=tuple(alts),add_mask=_mask(add,fi),remove_mask=_mask(rem,fi)))
    initial=set(map(str,raw.get('initial',[]))); goal=raw['goal']; goal_all=set(map(str,goal.get('all',[])));goal_none=set(map(str,goal.get('none',[]))); unknown=(initial|goal_all|goal_none)-set(fi)
    if unknown: raise ValueError(f'unknown initial/goal facts {sorted(unknown)}')
    if goal_all&goal_none: raise ValueError('goal fact both required and forbidden')
    return CompiledProcedure(str(raw['procedure_id']),facts,fi,_mask(initial,fi),_mask(goal_all,fi),_mask(goal_none,fi),tuple(actions))


@dataclass
class ReachabilityGraph:
    proc: CompiledProcedure
    states: list[int]
    edges: dict[int,list[tuple[int,int]]]
    predecessor: dict[int,tuple[int,int] | None]

    def shortest_trace_to(self, target_mask: int) -> list[str]:
        cur=target_mask; out=[]
        while self.predecessor[cur] is not None:
            prev,ai=self.predecessor[cur]; out.append(self.proc.actions[ai].action_id);cur=prev
        return list(reversed(out))


def enumerate_reachable(proc: CompiledProcedure, max_states: int=1_000_000) -> ReachabilityGraph:
    q=deque([proc.initial_mask]); pred={proc.initial_mask:None}; edges={}
    while q:
        s=q.popleft(); es=[]
        for a in proc.actions:
            if not proc.feasible(s,a): continue
            t=proc.apply_unchecked(s,a); es.append((a.index,t))
            if t not in pred:
                pred[t]=(s,a.index);q.append(t)
                if len(pred)>max_states: raise RuntimeError('state limit exceeded')
        edges[s]=es
    return ReachabilityGraph(proc,list(pred),edges,pred)


def analyze_procedure(proc: CompiledProcedure):
    g=enumerate_reachable(proc); goal_states={s for s in g.states if proc.goal(s)}; rev={s:[] for s in g.states}
    for s,es in g.edges.items():
        for ai,t in es: rev[t].append((ai,s))
    dist={s:0 for s in goal_states}; q=deque(goal_states)
    while q:
        t=q.popleft()
        for ai,s in rev[t]:
            if s not in dist: dist[s]=dist[t]+1;q.append(s)
    trap_states=[s for s in g.states if s not in dist]; deadlocks=[s for s in g.states if not proc.goal(s) and not g.edges.get(s)]; reachable_actions={proc.actions[ai].action_id for es in g.edges.values() for ai,_ in es}; unreachable=[a.action_id for a in proc.actions if a.action_id not in reachable_actions]
    paths={s:1 for s in goal_states}
    if dist:
        for d in range(1,max(dist.values())+1):
            for s,sd in dist.items():
                if sd==d: paths[s]=sum(paths.get(t,0) for ai,t in g.edges.get(s,[]) if dist.get(t)==d-1)
    report={'schema':'dovod-procedure-compiler-analysis-v1','procedure_id':proc.procedure_id,'facts':len(proc.facts),'actions':len(proc.actions),'reachable_states':len(g.states),'goal_states':len(goal_states),'goal_reachable_from_initial':proc.initial_mask in dist,'shortest_goal_actions':dist.get(proc.initial_mask),'shortest_goal_plan_count':str(paths.get(proc.initial_mask,0)),'nonrecoverable_reachable_states':len(trap_states),'reachable_nongoal_deadlocks':len(deadlocks),'unreachable_actions':unreachable}
    return report,g,dist,paths


def action_enabling_distances(proc:CompiledProcedure,g:ReachabilityGraph,target_action:str)->dict[int,int]:
    a=proc.action_by_id[target_action]; seeds=[s for s in g.states if proc.feasible(s,a)]; rev={s:[] for s in g.states}
    for s,es in g.edges.items():
        for ai,t in es:rev[t].append(s)
    d={s:0 for s in seeds};q=deque(seeds)
    while q:
        t=q.popleft()
        for s in rev[t]:
            if s not in d:d[s]=d[t]+1;q.append(s)
    return d


def shortest_plan_to_action(proc:CompiledProcedure,g:ReachabilityGraph,start:int,target_action:str) -> list[str]:
    d=action_enabling_distances(proc,g,target_action); cur=d.get(start)
    if cur is None or cur==0:return []
    out=[];state=start
    while cur>0:
        candidates=[]
        for ai,t in g.edges.get(state,[]):
            if d.get(t)==cur-1:
                a=proc.actions[ai]; candidates.append((a.canonical_order,a.action_id,t))
        if not candidates:return []
        _,aid,state=min(candidates);out.append(aid);cur-=1
    return out
