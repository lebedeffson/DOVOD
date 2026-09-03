from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "research" / "reference_impl"))
from source_aware_resolution_planner import SourceAwareResolutionPlanner  # noqa: E402

OUT = ROOT / "results" / "information_selection" / "paper_b_exact_scaling"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures" / "information_selection" / "paper_b_exact_scaling.png"


def synthetic_instance(k: int, g: int):
    """Deterministic finite scaling instance.

    k binary physical predicates are unresolved at q=0.5. The target action is index k.
    The first g distinct prerequisite subsets in cardinality/lexicographic order define g
    semantic alternatives. This construction exists only to expose the exact planner's
    state-growth mechanism; it is not a procedural dataset.
    """
    if g > 2 ** k:
        raise ValueError((k, g))
    action = k
    subsets = list(
        itertools.chain.from_iterable(itertools.combinations(range(k), r) for r in range(k + 1))
    )[:g]
    graphs = [{action: list(s)} for s in subsets]
    q = [0.5] * k + [0.0]
    return graphs, action, tuple(range(k)), q


def count_memoized_value_states(k: int, g: int) -> int:
    SourceAwareResolutionPlanner._value.cache_clear()
    SourceAwareResolutionPlanner._uncertainty.cache_clear()
    SourceAwareResolutionPlanner._semantic_groups.cache_clear()
    graphs, action, queryable, q = synthetic_instance(k, g)
    planner = SourceAwareResolutionPlanner(
        graphs, action, queryable, physical_cost=1.0, semantic_cost=1.0
    )
    planner.solve(q, mode="optimal")
    return int(SourceAwareResolutionPlanner._value.cache_info().misses)


def main() -> None:
    rows = []
    for k in range(1, 11):
        for g in (2, 4, 8):
            if g <= 2 ** k:
                rows.append({
                    "k_unresolved": k,
                    "semantic_alternatives": g,
                    "memoized_value_states": count_memoized_value_states(k, g),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exact_bellman_state_scaling.csv", index=False)

    checkpoints = {}
    for k in (3, 5, 8, 10):
        checkpoints[str(k)] = {
            str(g): int(df[(df.k_unresolved == k) & (df.semantic_alternatives == g)].iloc[0].memoized_value_states)
            for g in (2, 4, 8)
            if not df[(df.k_unresolved == k) & (df.semantic_alternatives == g)].empty
        }
    report = {
        "schema": "tinyapv-paper-b-exact-scaling-v1",
        "construction": (
            "Synthetic finite-state mechanism stress with k unresolved binary physical predicates at q=0.5 and g distinct action-specific semantic prerequisite alternatives. "
            "The reported count is the number of unique Bellman value states memoized by the exact planner after a cold-cache solve."
        ),
        "checkpoints": checkpoints,
        "k10_state_range": [
            int(df[(df.k_unresolved == 10) & (df.semantic_alternatives == 2)].iloc[0].memoized_value_states),
            int(df[(df.k_unresolved == 10) & (df.semantic_alternatives == 8)].iloc[0].memoized_value_states),
        ],
        "interpretation": (
            "Exact source-aware planning grows exponentially with the number of unresolved physical predicates. "
            "It should be treated as a small-uncertainty exact solver with myopic gain-per-cost as a principled fallback when the exact state envelope becomes large or when prior experiments show no dynamic-planning advantage."
        ),
        "claim_boundary": (
            "This is an algorithmic scaling stress, not wall-clock benchmarking and not a claim about the distribution of real procedural episodes. "
            "The exact counts depend on the deterministic semantic-alternative construction stated above."
        ),
    }
    (OUT / "exact_bellman_state_scaling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for g, part in df.groupby("semantic_alternatives"):
        ax.plot(part["k_unresolved"], part["memoized_value_states"], marker="o", label=f"g={g}")
    ax.set_yscale("log")
    ax.set_xlabel("Unresolved binary physical variables k")
    ax.set_ylabel("Memoized Bellman states (log scale)")
    ax.set_title("Exact source-aware planning has exponential state growth")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
