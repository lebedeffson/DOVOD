from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
import json, hashlib

@dataclass(frozen=True)
class EvidenceRef:
    predicate:int
    status:str  # VERIFIED | CONTRADICTED | UNKNOWN
    belief:float
    source_id:str
    timestamp:float
    calibration_version:str

@dataclass(frozen=True)
class GuidanceCertificate:
    certificate_id:str
    decision:str
    action:int|None
    query_predicate:int|None
    query_mode:str|None
    required_predicates:tuple[int,...]
    evidence:tuple[EvidenceRef,...]
    procedure_version:str
    action_risk:float
    created_at:float
    validity_horizon:float

    def to_dict(self): return asdict(self)

    def stable_hash(self):
        b=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(b).hexdigest()


def verify_certificate(cert:GuidanceCertificate, *, action_preconditions:dict[int,Sequence[int]], now:float,
                       min_belief:float=0.8, expected_procedure_version:str|None=None)->tuple[bool,list[str]]:
    reasons=[]
    if expected_procedure_version and cert.procedure_version!=expected_procedure_version:
        reasons.append('wrong_procedure_version')
    if now > cert.created_at + cert.validity_horizon:
        reasons.append('stale_certificate')
    ev={e.predicate:e for e in cert.evidence}
    if len(ev)!=len(cert.evidence): reasons.append('duplicate_evidence_predicate')
    if cert.decision=='GUIDE':
        if cert.action is None: reasons.append('guide_missing_action')
        else:
            expected=tuple(sorted(map(int,action_preconditions.get(cert.action,()))))
            if tuple(sorted(cert.required_predicates))!=expected:
                reasons.append('precondition_mapping_mismatch')
            for p in expected:
                e=ev.get(p)
                if e is None: reasons.append(f'missing_evidence:{p}'); continue
                if e.status!='VERIFIED': reasons.append(f'premise_not_verified:{p}')
                if e.belief < min_belief: reasons.append(f'premise_below_threshold:{p}')
                if e.timestamp > cert.created_at: reasons.append(f'future_evidence:{p}')
                if cert.created_at-e.timestamp > cert.validity_horizon: reasons.append(f'stale_evidence:{p}')
    elif cert.decision=='QUERY':
        if cert.query_predicate is None or not cert.query_mode: reasons.append('query_missing_target_or_mode')
    elif cert.decision=='WARN':
        if not any(e.status=='CONTRADICTED' for e in cert.evidence): reasons.append('warn_without_contradiction')
    elif cert.decision=='WAIT':
        pass
    else:
        reasons.append('unknown_decision')
    return (not reasons), reasons
