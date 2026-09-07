from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'benchmarks' / 'run_paper_b_impact_psr_acquisition.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('impact_psr', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_concurrent_frames_are_excluded(tmp_path: Path):
    m = _load_module()
    folder = tmp_path / 'rec'
    folder.mkdir()
    (folder / 'PSR_labels.csv').write_text(
        '1.jpg,0,A\n2.jpg,1,B\n2.jpg,2,C\n3.jpg,3,D\n', encoding='utf-8'
    )
    assert m._load_sequence(folder) == [0, 3]


def test_exact_multistep_can_beat_myopic_on_constructed_case():
    m = _load_module()
    props = {
        2: {'component': 1, 'operation': 'install', 'quality': 'normal', 'description': 'Install B'},
        3: {'component': 1, 'operation': 'install', 'quality': 'incorrect', 'description': 'Incorrectly installed B'},
        4: {'component': 1, 'operation': 'remove', 'quality': 'normal', 'description': 'Remove B'},
    }
    prior = {2: 1/3, 3: 1/3, 4: 1/3}
    costs = {'component': 0.04, 'operation': 0.03, 'quality': 0.04}
    exact, exact_action, _ = m.solve_policy(prior, props, costs, horizon=3)
    myopic, myopic_action, _ = m.solve_policy(prior, props, costs, horizon=1)
    assert exact < myopic
    assert exact_action == ('QUERY', 'operation')
    assert myopic_action == ('QUERY', 'operation')


def test_mock_official_split_end_to_end(tmp_path: Path):
    m = _load_module()
    root = tmp_path / 'IMPACT'
    labels = root / 'dataset' / 'PSR' / 'labels'
    (labels / 'train' / 'r1').mkdir(parents=True)
    (labels / 'train' / 'r2').mkdir(parents=True)
    (labels / 'train' / 'r3').mkdir(parents=True)
    (labels / 'test' / 't1').mkdir(parents=True)
    props = [
        {'id': 0, 'description': 'Install A', 'install': True, 'state_idx': 0},
        {'id': 1, 'description': 'Remove A', 'install': False, 'state_idx': 0},
        {'id': 2, 'description': 'Install B', 'install': True, 'state_idx': 1},
        {'id': 3, 'description': 'Incorrectly installed B', 'install': True, 'state_idx': 1},
        {'id': 4, 'description': 'Remove B', 'install': False, 'state_idx': 1},
    ]
    (labels / 'procedure_info_IMPACT.json').write_text(json.dumps(props), encoding='utf-8')
    for name, mid in [('r1', 2), ('r2', 3), ('r3', 4)]:
        (labels / 'train' / name / 'PSR_labels.csv').write_text(
            f'1.jpg,0,A\n2.jpg,{mid},X\n3.jpg,1,R\n', encoding='utf-8'
        )
    (labels / 'test' / 't1' / 'PSR_labels.csv').write_text(
        '1.jpg,0,A\n2.jpg,3,X\n3.jpg,1,R\n', encoding='utf-8'
    )
    train = m._load_split(root, 'train')
    test = m._load_split(root, 'test')
    counts = m._transition_counts(train)
    rows = m._evaluate_costs(
        [(a, b) for seq in test for a, b in zip(seq, seq[1:])],
        counts,
        m._step_properties(root),
        {'component': 0.04, 'operation': 0.03, 'quality': 0.04},
        3,
    )
    assert len(rows) == 1
    assert rows[0]['current_step'] == 0
    assert rows[0]['true_next_step'] == 3
    assert rows[0]['exact_realized_cost'] < rows[0]['myopic_realized_cost']
