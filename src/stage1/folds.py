"""Leakage-safe fold manifests. Sequence is the unit. No modelling."""

from __future__ import annotations

from typing import Iterable

EUROC_ROOM = {
    "V1_01_easy": "EUROC_FIREFLY_VICON_ROOM1",
    "V1_02_medium": "EUROC_FIREFLY_VICON_ROOM1",
    "V1_03_difficult": "EUROC_FIREFLY_VICON_ROOM1",
    "V2_01_easy": "EUROC_FIREFLY_VICON_ROOM2",
    "V2_02_medium": "EUROC_FIREFLY_VICON_ROOM2",
    "V2_03_difficult": "EUROC_FIREFLY_VICON_ROOM2",
}
UZH_GROUP = {
    "indoor_forward_6_snapdragon": "indoor_forward",
    "indoor_forward_9_snapdragon": "indoor_forward",
    "indoor_forward_10_snapdragon": "indoor_forward",
    "indoor_45_2_snapdragon": "indoor_45",
    "indoor_45_4_snapdragon": "indoor_45",
    "indoor_45_13_snapdragon": "indoor_45",
    "indoor_45_14_snapdragon": "indoor_45",
    "outdoor_forward_1_snapdragon": "outdoor_forward",
}

FOLD_COLUMNS = [
    "fold_id",
    "evaluation_class",
    "source_or_domain",
    "split_role",
    "dataset",
    "sequence_id",
    "dependency_group",
    "notes",
]


def _row(**kwargs: str) -> dict[str, str]:
    return {k: kwargs.get(k, "") for k in FOLD_COLUMNS}


def leave_one_recording_out(dataset: str, sequences: Iterable[str], groups: dict[str, str]) -> list[dict[str, str]]:
    seqs = list(sequences)
    rows: list[dict[str, str]] = []
    eval_class = "WITHIN_EUROC" if dataset == "EUROC" else "WITHIN_UZH"
    for i, held in enumerate(seqs, start=1):
        fold = f"{dataset}_LORO_{i:02d}_{held}"
        for seq in seqs:
            role = "test" if seq == held else "train"
            rows.append(
                _row(
                    fold_id=fold,
                    evaluation_class=eval_class,
                    source_or_domain=dataset,
                    split_role=role,
                    dataset=dataset,
                    sequence_id=seq,
                    dependency_group=groups[seq],
                    notes="leave-one-recording-out; windows of held-out sequence never train",
                )
            )
    return rows


def leave_group_out(
    dataset: str,
    sequences: Iterable[str],
    groups: dict[str, str],
    *,
    evaluation_class: str,
) -> list[dict[str, str]]:
    seqs = list(sequences)
    uniq = []
    for seq in seqs:
        g = groups[seq]
        if g not in uniq:
            uniq.append(g)
    rows: list[dict[str, str]] = []
    for g_held in uniq:
        fold = f"{dataset}_LGO_{g_held}"
        n_held = sum(1 for s in seqs if groups[s] == g_held)
        note = "leave-trajectory-group-out source validation"
        if n_held == 1:
            note += "; SMALL_GROUP_SINGLE_RECORDING"
        for seq in seqs:
            role = "source_validation" if groups[seq] == g_held else "source_train"
            rows.append(
                _row(
                    fold_id=fold,
                    evaluation_class=evaluation_class,
                    source_or_domain=dataset,
                    split_role=role,
                    dataset=dataset,
                    sequence_id=seq,
                    dependency_group=groups[seq],
                    notes=note,
                )
            )
    return rows


def transfer_manifests(euroc: Iterable[str], uzh: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seq in euroc:
        rows.append(
            _row(
                fold_id="TRANSFER_EUROC_TO_UZH",
                evaluation_class="ZERO_SHOT_TRANSFER",
                source_or_domain="EUROC",
                split_role="source_train",
                dataset="EUROC",
                sequence_id=seq,
                dependency_group=EUROC_ROOM[seq],
                notes="source-only fitting; no UZH parameters",
            )
        )
    for seq in uzh:
        rows.append(
            _row(
                fold_id="TRANSFER_EUROC_TO_UZH",
                evaluation_class="ZERO_SHOT_TRANSFER",
                source_or_domain="EUROC",
                split_role="target_eval",
                dataset="UZH_FPV",
                sequence_id=seq,
                dependency_group=UZH_GROUP[seq],
                notes="frozen source system; no target-domain fitting",
            )
        )
    for seq in uzh:
        rows.append(
            _row(
                fold_id="TRANSFER_UZH_TO_EUROC",
                evaluation_class="ZERO_SHOT_TRANSFER",
                source_or_domain="UZH_FPV",
                split_role="source_train",
                dataset="UZH_FPV",
                sequence_id=seq,
                dependency_group=UZH_GROUP[seq],
                notes="source-only fitting; no EuRoC parameters",
            )
        )
    for seq in euroc:
        rows.append(
            _row(
                fold_id="TRANSFER_UZH_TO_EUROC",
                evaluation_class="ZERO_SHOT_TRANSFER",
                source_or_domain="UZH_FPV",
                split_role="target_eval",
                dataset="EUROC",
                sequence_id=seq,
                dependency_group=EUROC_ROOM[seq],
                notes="frozen source system; no target-domain fitting",
            )
        )
    return rows


def fold_is_sequence_disjoint(rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    """Train and held-out roles must not share a sequence inside one fold."""
    held_roles = {"test", "source_validation", "target_eval"}
    train_roles = {"train", "source_train"}
    issues: list[str] = []
    by_fold: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_fold.setdefault(row["fold_id"], []).append(row)
    for fold_id, items in by_fold.items():
        train = {(r["dataset"], r["sequence_id"]) for r in items if r["split_role"] in train_roles}
        held = {(r["dataset"], r["sequence_id"]) for r in items if r["split_role"] in held_roles}
        overlap = train & held
        if overlap:
            issues.append(f"{fold_id}: overlap {sorted(overlap)}")
    return (not issues), issues
