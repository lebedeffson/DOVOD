from __future__ import annotations

"""Observation-model utilities retained for source-aware experiments.

The wider historical experiment also contains controlled perception-channel
simulation. The source-selection pipeline only requires TRAIN state priors;
this module computes those priors directly from the dataset-native PSR states
instead of storing a duplicate 300 kB intermediate JSON artifact.
"""

import numpy as np

from hardening_loro_graph import read_records
from p4_observation_model import STATE_TO_INDEX

N = 17


def train_state_priors() -> np.ndarray:
    """Laplace-smoothed TRAIN priors over {-1, 0, 1} for every component."""
    counts = np.ones((N, 3), dtype=float)
    for split, _, rows in read_records():
        if split != "train":
            continue
        for _, state in rows:
            for j, value in enumerate(state):
                counts[j, STATE_TO_INDEX[int(value)]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)
