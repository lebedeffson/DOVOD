from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vr_compiler'))
from procedure_compiler import compile_procedure,analyze_procedure,shortest_plan_to_action

@dataclass
class Decision:
    stage:str; mode:str; score_family:str; score:float; risk_action_label:int=-1; procedure_action_id:str=''; reason:str=''; recovery_action_id:str=''

class PlayableRuntimeV24:
    def __init__(self,root:Path=ROOT):
        self.root=root
        self.proc=compile_procedure(root/'replay_v24'/'IMPACT_PLAYABLE_PROCEDURE_V24.json')
        self.report,self.graph,self.goal_dist,_=analyze_procedure(self.proc)
        self.state=self.proc.initial_mask
        m=json.loads((root/'replay_v24'/'PLAYABLE_ACTION_MAP_V24.json').read_text())['actions']
        self.by_proc={x['procedureActionId']:x for x in m};self.by_action={int(x['riskActionLabel']):x for x in m};self.by_pair={(int(x['verb']),int(x['noun'])):x for x in m}
        cat=json.loads((root/'replay_v22'/'PROGRESSIVE_SEMANTIC_RISK_CATALOG_V22.json').read_text())
        self.global_risk=float(cat['globalRisk']);self.action_risk={int(x['id']):float(x['risk']) for x in cat['actions']};self.verb_risk={int(x['id']):float(x['risk']) for x in cat['verbs']};self.noun_risk={int(x['id']):float(x['risk']) for x in cat['nouns']};self.th=cat['thresholds']
    @property
    def facts(self):return sorted(self.proc.fact_set(self.state))
    @property
    def completed(self):return self.proc.goal(self.state)
    def reset(self):self.state=self.proc.initial_mask
    def feasible(self,procedure_action_id:str):return self.proc.feasible(self.state,procedure_action_id)
    def resolve(self,verb=-1,noun=-1,explicit_action=-1):
        if explicit_action>=0 and explicit_action in self.by_action:return self.by_action[explicit_action]
        return self.by_pair.get((verb,noun))
    def decide(self,verb=-1,noun=-1,explicit_action=-1,procedure_action_id=''):
        x=self.resolve(verb,noun,explicit_action)
        if x is not None:
            aid=int(x['riskActionLabel']);score=self.action_risk[aid];mode='ASSIST' if score>=self.th['action']['assist10'] else ('WATCH' if score>=self.th['action']['watch20'] else 'SILENCE')
            pa=procedure_action_id or x['procedureActionId']
            if pa and pa in self.proc.action_by_id and not self.feasible(pa):
                plan=shortest_plan_to_action(self.proc,self.graph,self.state,pa);rec=plan[0] if plan else ''
                return Decision('FULL_ACTION','HARD_BLOCK','hard_procedure_guard',score,aid,pa,'full_action_infeasible_in_playable_procedure',rec)
            return Decision('FULL_ACTION',mode,'action',score,aid,pa,'full_semantic_action_risk','')
        if verb>=0 and verb in self.verb_risk:
            score=self.verb_risk[verb];mode='PREARM' if score>=self.th['verb']['prearm10'] else 'SILENCE'
            return Decision('VERB_ONLY',mode,'verb',score,-1,procedure_action_id,'partial_verb_prearm_only','')
        if noun>=0 and noun in self.noun_risk:
            return Decision('NOUN_ONLY','OBSERVE','noun',self.noun_risk[noun],-1,procedure_action_id,'noun_only_not_user_facing','')
        return Decision('UNRESOLVED','SILENCE','global',self.global_risk,-1,procedure_action_id,'insufficient_semantic_evidence','')
    def commit(self,procedure_action_id:str):
        before=self.state;after,ok=self.proc.apply(before,procedure_action_id)
        if ok:self.state=after
        return ok,before,self.state
    def snapshot(self):return {'stateMask':self.state,'facts':self.facts,'completed':self.completed,'goalDistance':self.goal_dist.get(self.state,-1)}
