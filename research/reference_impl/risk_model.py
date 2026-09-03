from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass(frozen=True)
class RiskModel:
    weights: np.ndarray
    source: str
    version: str
    claimable: bool

    def weight(self, action:int)->float:
        return float(self.weights[int(action)])

    def save(self,path):
        Path(path).write_text(json.dumps({'weights':self.weights.tolist(),'source':self.source,'version':self.version,'claimable':self.claimable},indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        o=json.loads(Path(path).read_text())
        return cls(np.asarray(o['weights'],dtype=float),str(o['source']),str(o['version']),bool(o['claimable']))


def neutral_risk(n_actions:int)->RiskModel:
    return RiskModel(np.ones(n_actions,dtype=float),'NEUTRAL_EQUAL_RISK','neutral-v1',False)


def from_config(path:str|Path,n_actions:int,require_claimable:bool=False)->RiskModel:
    o=json.loads(Path(path).read_text()); w=np.asarray(o['weights'],dtype=float)
    if w.shape!=(n_actions,): raise ValueError(f'expected {n_actions} risk weights, got {w.shape}')
    if np.any(~np.isfinite(w)) or np.any(w<0): raise ValueError('risk weights must be finite and nonnegative')
    source=str(o.get('source','UNSPECIFIED'))
    claimable=bool(o.get('claimable',False)) and source not in {'UNSPECIFIED','SYNTHETIC','NEUTRAL_EQUAL_RISK'}
    model=RiskModel(w,source,str(o.get('version','unknown')),claimable)
    if require_claimable and not model.claimable:
        raise RuntimeError('Real risk-aware claim requested but risk source is not claimable/externally grounded')
    return model
