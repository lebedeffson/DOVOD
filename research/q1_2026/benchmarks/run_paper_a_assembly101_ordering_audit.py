from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

TOY_RE = re.compile(r"_(\d+)-([A-Za-z0-9]+)_\1_user_id_")


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    verb: str
    obj_a: str
    obj_b: str
    label: str
    remark: str

    @property
    def token(self) -> str:
        return "|".join((_norm(self.verb), _norm(self.obj_a), _norm(self.obj_b)))


def _norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _bucket(key: str, modulus: int = 1000) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def toy_id_from_name(name: str) -> str:
    match = TOY_RE.search(name)
    if not match:
        raise ValueError(f"cannot parse toy id from {name}")
    return match.group(2).lower()


def read_sequence(path: Path) -> tuple[Event, ...]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not str(x).strip() for x in row):
                continue
            if len(row) < 6:
                raise ValueError(f"{path.name}:{index}: expected at least 6 columns")
            try:
                start, end = int(row[0]), int(row[1])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{index}: invalid timestamps") from exc
            rows.append(Event(start, end, row[2], row[3], row[4], _norm(row[5]), _norm(row[6]) if len(row) > 6 else ""))
    rows.sort(key=lambda event: (event.start, event.end))
    return tuple(rows)


def split_paths(paths, *, unit: str, train_fraction: float, hash_key: str):
    if unit not in {"sequence", "toy_id"}:
        raise ValueError("split unit must be sequence or toy_id")
    cutoff = int(round(train_fraction * 1000))
    train, test = [], []
    groups = {}
    for path in sorted(map(Path, paths)):
        group = path.name if unit == "sequence" else toy_id_from_name(path.name)
        side = "train" if _bucket(f"{hash_key}|{group}") < cutoff else "test"
        groups[group] = side
        (train if side == "train" else test).append(path)
    if not train or not test:
        raise RuntimeError(f"degenerate {unit} split: train={len(train)} test={len(test)}")
    return tuple(train), tuple(test), groups


def learn_relations(
    sequences: dict[str, tuple[Event, ...]],
    *,
    target_labels=("correct",),
    history_labels=("correct",),
    min_target_support: int = 5,
    predecessor_frequency: float = 0.9,
):
    target_counts = Counter()
    pair_counts = Counter()
    history_labels = set(map(_norm, history_labels))
    target_labels = set(map(_norm, target_labels))
    for events in sequences.values():
        seen = set()
        for event in events:
            token = event.token
            if event.label in target_labels:
                target_counts[token] += 1
                for predecessor in seen:
                    if predecessor != token:
                        pair_counts[(predecessor, token)] += 1
            if event.label in history_labels:
                seen.add(token)
    relations = {}
    for (predecessor, target), count in pair_counts.items():
        support = target_counts[target]
        frequency = count / support if support else 0.0
        if support >= min_target_support and frequency >= predecessor_frequency:
            relations[(predecessor, target)] = {
                "train_target_support": int(support),
                "train_predecessor_count": int(count),
                "train_frequency": float(frequency),
            }
    return relations, target_counts


def evaluate_relations(
    sequences: dict[str, tuple[Event, ...]],
    relations: dict,
    *,
    history_labels=("correct", "correction"),
    ordering_mistake_remarks=("wrong order", "previous one is mistake"),
):
    by_target = defaultdict(list)
    for (predecessor, target), meta in relations.items():
        by_target[target].append((predecessor, meta))
    history_labels = set(map(_norm, history_labels))
    order_remarks = set(map(_norm, ordering_mistake_remarks))
    counts = Counter()
    refuted = set()
    for events in sequences.values():
        seen = set()
        for event in events:
            token = event.token
            rules = by_target.get(token, ())
            is_correct = event.label == "correct"
            is_order_mistake = event.label == "mistake" and event.remark in order_remarks
            if rules and (is_correct or is_order_mistake):
                kind = "correct" if is_correct else "order_mistake"
                counts[f"{kind}_eligible"] += 1
                missing = [pred for pred, _ in rules if pred not in seen]
                if missing:
                    counts[f"{kind}_violations"] += 1
                    if is_correct:
                        refuted.update((pred, token) for pred in missing)
            if event.label in history_labels:
                seen.add(token)
    correct_n = counts["correct_eligible"]
    order_n = counts["order_mistake_eligible"]
    correct_v = counts["correct_violations"]
    order_v = counts["order_mistake_violations"]
    denom = correct_v + order_v
    return {
        "correct_eligible": int(correct_n),
        "correct_violations": int(correct_v),
        "heldout_correct_violation_rate": None if correct_n == 0 else correct_v / correct_n,
        "order_mistake_eligible": int(order_n),
        "order_mistake_violations": int(order_v),
        "ordering_mistake_recall_within_covered_actions": None if order_n == 0 else order_v / order_n,
        "violation_precision_correct_vs_order_mistake_only": None if denom == 0 else order_v / denom,
        "relations_refuted_by_heldout_correct_behavior": len(refuted),
        "relations_unrefuted_on_heldout_correct_behavior": len(relations) - len(refuted),
    }


def _run_split(paths, sequences, split_cfg, induction_cfg, evaluation_cfg):
    train_paths, test_paths, groups = split_paths(
        paths,
        unit=split_cfg["unit"],
        train_fraction=float(split_cfg["train_fraction"]),
        hash_key=split_cfg["hash_key"],
    )
    train = {p.name: sequences[p.name] for p in train_paths}
    test = {p.name: sequences[p.name] for p in test_paths}
    thresholds = [
        float(induction_cfg["primary_predecessor_frequency"]),
        *map(float, induction_cfg.get("sensitivity_predecessor_frequencies", [])),
    ]
    threshold_rows = {}
    for threshold in sorted(set(thresholds)):
        relations, targets = learn_relations(
            train,
            target_labels=tuple(induction_cfg["target_labels"]),
            history_labels=tuple(induction_cfg["history_labels"]),
            min_target_support=int(induction_cfg["minimum_target_support"]),
            predecessor_frequency=threshold,
        )
        metrics = evaluate_relations(
            test,
            relations,
            history_labels=tuple(evaluation_cfg["history_labels"]),
            ordering_mistake_remarks=tuple(evaluation_cfg["ordering_mistake_remarks"]),
        )
        threshold_rows[f"{threshold:.2f}"] = {
            "candidate_relations": len(relations),
            "train_supported_target_actions": sum(count >= int(induction_cfg["minimum_target_support"]) for count in targets.values()),
            **metrics,
        }
    return {
        "unit": split_cfg["unit"],
        "train_sequences": len(train_paths),
        "test_sequences": len(test_paths),
        "train_groups": sum(side == "train" for side in groups.values()),
        "test_groups": sum(side == "test" for side in groups.values()),
        "thresholds": threshold_rows,
    }


def run(annotations_root: Path, contract: dict) -> dict:
    dataset = contract["dataset"]
    paths = tuple(sorted(annotations_root.glob("*.csv")))
    if len(paths) != int(dataset["expected_sequence_count"]):
        raise RuntimeError(f"expected {dataset['expected_sequence_count']} CSVs, found {len(paths)}")
    sequences = {path.name: read_sequence(path) for path in paths}
    label_counts = Counter(event.label for events in sequences.values() for event in events)
    remark_counts = Counter(event.remark for events in sequences.values() for event in events if event.remark)
    evaluation = contract["evaluation"]
    primary = _run_split(paths, sequences, evaluation["primary_split"], contract["induction"], evaluation)
    robustness = _run_split(paths, sequences, evaluation["robustness_split"], contract["induction"], evaluation)
    return {
        "schema": "dovod-paper-a-assembly101-ordering-audit-v1",
        "dataset": dataset,
        "sequence_count": len(paths),
        "event_count": sum(len(events) for events in sequences.values()),
        "label_counts": dict(sorted(label_counts.items())),
        "remark_counts": dict(sorted(remark_counts.items())),
        "primary_sequence_split": primary,
        "robustness_toy_group_split": robustness,
        "primary_threshold": float(contract["induction"]["primary_predecessor_frequency"]),
        "claim_boundary": contract["claim_boundary"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-root", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    report = run(Path(args.annotations_root), contract)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
