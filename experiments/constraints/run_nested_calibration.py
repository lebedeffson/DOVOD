from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
PROJECT = PACKAGE / 'research' / 'reference_impl'
sys.path.insert(0, str(PROJECT))

from hardening_loro_graph import edges, evaluate, next_events, read_records
from q1_semantic_sample_complexity import infer_from_selected_train_recordings

OUT = PACKAGE / 'results' / 'constraints' / 'constraints' / 'nested_counterexample_calibration'
OUT.mkdir(parents=True, exist_ok=True)


def prune_with_recording(graph, rows):
    out = {a: list(ps) for a, ps in graph.items()}
    for k in range(len(rows) - 1):
        state, nxt = rows[k][1], rows[k + 1][1]
        for action, (before, after) in enumerate(zip(state, nxt)):
            if before != 1 and after == 1:
                out[action] = [p for p in out[action] if int(state[p]) == 1]
    return out


def bootstrap_outer(values, seed=20260901, b=20000):
    arr = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(b)])
    return {
        'mean': float(arr.mean()),
        'ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        'outer_folds': int(len(arr)),
    }


def main():
    recs = read_records()
    train_names = sorted(rec for sp, rec, _ in recs if sp == 'train')
    train_rows = {rec: rows for sp, rec, rows in recs if sp == 'train'}

    pair_rows = []
    outer_rows = []
    for held in train_names:
        held_events = next_events(recs, split='train', recording=held)
        pool = [r for r in train_names if r != held]
        local = []
        for calibrator in pool:
            fit9 = tuple(r for r in pool if r != calibrator)
            g9 = infer_from_selected_train_recordings(recs, fit9)
            pruned = prune_with_recording(g9, train_rows[calibrator])
            g10 = infer_from_selected_train_recordings(recs, tuple(pool))
            m9 = evaluate(held_events, g9)
            mp = evaluate(held_events, pruned)
            m10 = evaluate(held_events, g10)
            row = {
                'held_out_recording': held,
                'calibration_recording': calibrator,
                'fit_recordings': 9,
                'fit_edges': len(edges(g9)),
                'calibrated_edges': len(edges(pruned)),
                'refit10_edges': len(edges(g10)),
                'edges_pruned_by_one_calibration_recording': len(edges(g9) - edges(pruned)),
                'edges_added_vs_pruned_by_refit10': len(edges(g10) - edges(pruned)),
                'heldout_recall_fit9': m9['recall'],
                'heldout_recall_calibrated': mp['recall'],
                'heldout_recall_refit10': m10['recall'],
                'heldout_candidates_fit9': m9['mean_candidates'],
                'heldout_candidates_calibrated': mp['mean_candidates'],
                'heldout_candidates_refit10': m10['mean_candidates'],
                'recall_gain_calibrated_minus_fit9': mp['recall'] - m9['recall'],
                'candidate_change_calibrated_minus_fit9': mp['mean_candidates'] - m9['mean_candidates'],
                'pruning_equals_full_10_recording_refit': int(edges(pruned) == edges(g10)),
            }
            pair_rows.append(row)
            local.append(row)
        outer_rows.append({
            'held_out_recording': held,
            'mean_recall_fit9': float(np.mean([r['heldout_recall_fit9'] for r in local])),
            'mean_recall_calibrated': float(np.mean([r['heldout_recall_calibrated'] for r in local])),
            'mean_recall_gain': float(np.mean([r['recall_gain_calibrated_minus_fit9'] for r in local])),
            'mean_candidate_change': float(np.mean([r['candidate_change_calibrated_minus_fit9'] for r in local])),
            'mean_edges_pruned': float(np.mean([r['edges_pruned_by_one_calibration_recording'] for r in local])),
        })

    pair_df = pd.DataFrame(pair_rows)
    outer_df = pd.DataFrame(outer_rows)
    pair_df.to_csv(OUT / 'nested_calibration_pairs.csv', index=False)
    outer_df.to_csv(OUT / 'nested_calibration_outer_folds.csv', index=False)

    all_positive = bool((outer_df['mean_recall_gain'] > 0).all())

    report = {
        'schema': 'tinyapv-nested-counterexample-calibration-v1',
        'evidence_boundary': (
            'Strictly nested 11-fold recording analysis inside the MECCANO TRAIN split. For each outer held recording, nine recordings infer prerequisites, one calibration recording can only falsify/prune those prerequisites, and the outer recording is evaluated last. Validation/test data are never used.'
        ),
        'outer_recording_folds': len(outer_df),
        'fit_calibration_pairs': len(pair_df),
        'mean_fit9_edges': float(pair_df['fit_edges'].mean()),
        'mean_edges_pruned_by_one_independent_calibration_recording': float(pair_df['edges_pruned_by_one_calibration_recording'].mean()),
        'calibration_pruning_equals_full_10_recording_refit_pairs': int(pair_df['pruning_equals_full_10_recording_refit'].sum()),
        'all_pairs': int(len(pair_df)),
        'edges_ever_added_by_10_recording_refit_beyond_pruned_graph': int((pair_df['edges_added_vs_pruned_by_refit10'] > 0).sum()),
        'heldout_recall_fit9_mean': float(pair_df['heldout_recall_fit9'].mean()),
        'heldout_recall_calibrated_mean': float(pair_df['heldout_recall_calibrated'].mean()),
        'heldout_candidate_fit9_mean': float(pair_df['heldout_candidates_fit9'].mean()),
        'heldout_candidate_calibrated_mean': float(pair_df['heldout_candidates_calibrated'].mean()),
        'outer_recording_recall_gain_bootstrap': bootstrap_outer(outer_df['mean_recall_gain'].tolist(), seed=20260901),
        'outer_recording_candidate_change_bootstrap': bootstrap_outer(outer_df['mean_candidate_change'].tolist(), seed=20260902),
        'outer_recording_edges_pruned_bootstrap': bootstrap_outer(outer_df['mean_edges_pruned'].tolist(), seed=20260903),
        'all_outer_folds_have_positive_mean_recall_gain': all_positive,
        'bootstrap_interpretation': 'The outer-recording bootstrap interval is reported as a descriptive measure of variability across held recordings. Because training/calibration sets overlap across folds, it is not treated as an independence-based population hypothesis test.',
        'interpretation': (
            'A calibration recording improves held-recording authorization by providing negative evidence that removes accidental prerequisites. Equality between calibration-only pruning and the 10-recording support=1 refit is an expected monotonicity invariant of the learner and is used as an implementation/sanity check rather than as the primary empirical contribution.'
        ),
        'claim_boundary': (
            'The numerical gain is same-procedure held-recording generalization under the support=1 learner. It is not an independent-dataset result and does not prove that every removed edge is mechanically invalid.'
        ),
    }
    (OUT / 'nested_counterexample_calibration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
