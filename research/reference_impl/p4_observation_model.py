from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence
import json
from pathlib import Path

import numpy as np

STATE_VALUES = (-1, 0, 1)
STATE_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_STATE = {0: -1, 1: 0, 2: 1}


@dataclass(frozen=True)
class ObservationModeSpec:
    name: str
    cost: float


@dataclass
class DirichletObservationModel:
    """Posterior over categorical observation channels."""
    alpha: np.ndarray
    mode_names: tuple[str, ...]
    mode_costs: np.ndarray
    component_names: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        a = np.asarray(self.alpha, dtype=float)
        if a.ndim != 4 or a.shape[2:] != (3, 3):
            raise ValueError(f"alpha must have shape [component, mode, 3, 3], got {a.shape}")
        if len(self.mode_names) != a.shape[1]:
            raise ValueError("mode_names mismatch")
        if len(self.component_names) != a.shape[0]:
            raise ValueError("component_names mismatch")
        if np.any(a <= 0):
            raise ValueError("Dirichlet alpha must be >0")
        self.alpha = a
        self.mode_costs = np.asarray(self.mode_costs, dtype=float)

    @property
    def n_components(self) -> int:
        return int(self.alpha.shape[0])

    def mode_index(self, mode: str) -> int:
        return self.mode_names.index(mode)

    def mean_confusion(self, component: int, mode: str | int) -> np.ndarray:
        m = self.mode_index(mode) if isinstance(mode, str) else int(mode)
        a = self.alpha[int(component), m]
        return a / a.sum(axis=-1, keepdims=True)

    def sample_confusion(self, component: int, mode: str | int, rng: np.random.Generator) -> np.ndarray:
        m = self.mode_index(mode) if isinstance(mode, str) else int(mode)
        a = self.alpha[int(component), m]
        return np.vstack([rng.dirichlet(a[t]) for t in range(3)])

    def sample_confusions(self, component: int, mode: str | int, rng: np.random.Generator, n: int) -> np.ndarray:
        m = self.mode_index(mode) if isinstance(mode, str) else int(mode)
        a = self.alpha[int(component), m]
        g = rng.gamma(shape=a[None, :, :], scale=1.0, size=(int(n), 3, 3))
        return g / g.sum(axis=-1, keepdims=True)

    def posterior_predictive_observation(self, state_probs: np.ndarray, component: int, mode: str | int) -> np.ndarray:
        state_probs = normalize_probs(state_probs)
        c = self.mean_confusion(component, mode)
        return state_probs @ c

    def bayes_update(self, prior: np.ndarray, observed_state: int, component: int, mode: str | int,
                     confusion: np.ndarray | None = None) -> np.ndarray:
        prior = normalize_probs(prior)
        o = STATE_TO_INDEX[int(observed_state)]
        c = self.mean_confusion(component, mode) if confusion is None else np.asarray(confusion, dtype=float)
        likelihood = c[:, o]
        post = prior * likelihood
        s = float(post.sum())
        if s <= 1e-15:
            return prior.copy()
        return post / s

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha.tolist(),
            "mode_names": list(self.mode_names),
            "mode_costs": self.mode_costs.tolist(),
            "component_names": list(self.component_names),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, obj: Mapping) -> "DirichletObservationModel":
        return cls(
            alpha=np.asarray(obj["alpha"], dtype=float),
            mode_names=tuple(obj["mode_names"]),
            mode_costs=np.asarray(obj["mode_costs"], dtype=float),
            component_names=tuple(obj["component_names"]),
            metadata=dict(obj.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DirichletObservationModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def normalize_probs(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 1e-15:
        return np.ones_like(p) / len(p)
    return p / s


def fit_hierarchical_dirichlet(
    observations: Iterable[Mapping],
    *,
    n_components: int,
    component_names: Sequence[str] | None,
    modes: Sequence[ObservationModeSpec],
    global_laplace: float = 1.0,
    pooling_strength: float = 6.0,
    metadata: dict | None = None,
) -> DirichletObservationModel:
    mode_names = tuple(m.name for m in modes)
    mode_to_idx = {m: i for i, m in enumerate(mode_names)}
    global_counts = np.full((len(modes), 3, 3), float(global_laplace), dtype=float)
    comp_counts = np.zeros((n_components, len(modes), 3, 3), dtype=float)
    n = 0
    for row in observations:
        j = int(row["component"])
        m = mode_to_idx[str(row["mode"])]
        t = STATE_TO_INDEX[int(row["true_state"])]
        o = STATE_TO_INDEX[int(row["observed_state"])]
        if not (0 <= j < n_components):
            raise ValueError(f"component out of range: {j}")
        global_counts[m, t, o] += 1.0
        comp_counts[j, m, t, o] += 1.0
        n += 1

    global_mean = global_counts / global_counts.sum(axis=-1, keepdims=True)
    alpha = np.zeros_like(comp_counts)
    for j in range(n_components):
        for m in range(len(modes)):
            for t in range(3):
                alpha[j, m, t] = pooling_strength * global_mean[m, t] + comp_counts[j, m, t]

    names = tuple(component_names or [f"component_{j}" for j in range(n_components)])
    md = dict(metadata or {})
    md.update({
        "n_observations": n,
        "global_laplace": global_laplace,
        "pooling_strength": pooling_strength,
        "global_confusion_mean": global_mean.tolist(),
    })
    return DirichletObservationModel(
        alpha=alpha,
        mode_names=mode_names,
        mode_costs=np.asarray([m.cost for m in modes], dtype=float),
        component_names=names,
        metadata=md,
    )


def sample_observation(true_state: int, confusion: np.ndarray, rng: np.random.Generator) -> int:
    t = STATE_TO_INDEX[int(true_state)]
    probs = normalize_probs(confusion[t])
    oi = int(rng.choice(3, p=probs))
    return INDEX_TO_STATE[oi]


def confusion_row_accuracy(confusion: np.ndarray) -> np.ndarray:
    c = np.asarray(confusion, dtype=float)
    return np.diag(c)
