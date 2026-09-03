from __future__ import annotations

"""Recording-level paired bootstrap utilities.

Real MECCANO recordings are the top-level independent observational units.
Repeated controlled seeds are technical replications within the same recording
and MUST NOT be promoted to independent clusters.
"""

from typing import Iterable, Sequence
import numpy as np
import pandas as pd


def paired_recording_bootstrap(
    df: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    value_col: str,
    *,
    index_cols: Sequence[str] = ("seed", "recording", "frame"),
    policy_col: str = "policy",
    recording_col: str = "recording",
    n_boot: int = 5000,
    seed: int = 20260829,
) -> dict:
    """Paired cluster bootstrap with recording as the only top-level cluster."""
    q = df[df[policy_col].isin([policy_a, policy_b])].copy()
    piv = q.pivot_table(index=list(index_cols), columns=policy_col, values=value_col, aggfunc="mean").dropna()
    if policy_a not in piv.columns or policy_b not in piv.columns:
        raise ValueError(f"Missing paired policies: {policy_a}, {policy_b}")
    piv["diff"] = piv[policy_a] - piv[policy_b]
    flat = piv.reset_index()
    if recording_col not in flat.columns:
        raise ValueError(f"{recording_col!r} not present in paired index")

    groups = {str(rec): g["diff"].to_numpy(dtype=float) for rec, g in flat.groupby(recording_col, sort=True)}
    recordings = list(groups)
    if len(recordings) < 2:
        raise ValueError("At least two independent recordings are required")

    rng = np.random.default_rng(seed)
    episode_boot = np.empty(int(n_boot), dtype=float)
    equal_recording_boot = np.empty(int(n_boot), dtype=float)
    rec_means = np.asarray([groups[r].mean() for r in recordings], dtype=float)

    for i in range(int(n_boot)):
        idx = rng.integers(0, len(recordings), size=len(recordings))
        sampled = [recordings[int(k)] for k in idx]
        episode_boot[i] = float(np.mean(np.concatenate([groups[r] for r in sampled])))
        equal_recording_boot[i] = float(np.mean(rec_means[idx]))

    return {
        "a": policy_a,
        "b": policy_b,
        "value": value_col,
        "mean_diff_episode_weighted": float(piv["diff"].mean()),
        "ci95_recording_cluster": [float(np.quantile(episode_boot, 0.025)), float(np.quantile(episode_boot, 0.975))],
        "mean_diff_equal_recording": float(rec_means.mean()),
        "ci95_equal_recording": [float(np.quantile(equal_recording_boot, 0.025)), float(np.quantile(equal_recording_boot, 0.975))],
        "n_independent_recordings": int(len(recordings)),
        "n_paired_episodes": int(len(piv)),
        "recording_means": {r: float(groups[r].mean()) for r in recordings},
    }
