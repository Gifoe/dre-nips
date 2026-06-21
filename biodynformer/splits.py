from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import random


@dataclass(frozen=True)
class Split:
    name: str
    kind: str
    train_subjects: list[str]
    test_subjects: list[str]
    held_out_center: str | None = None


def _unique_patients(rows: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        sid = str(row["subject_id"])
        if sid in seen:
            continue
        seen.add(sid)
        out.append({"subject_id": sid, "center": str(row["center"]), "outcome_success": row.get("outcome_success")})
    return out


def build_five_fold_splits(patients: Sequence[dict], *, n_splits: int = 5, seed: int = 42) -> list[Split]:
    unique = _unique_patients(patients)
    if len(unique) < 2:
        raise ValueError("At least two patients are required.")
    k = max(2, min(int(n_splits), len(unique)))
    buckets: dict[str, list[str]] = {}
    for item in unique:
        buckets.setdefault(str(item.get("center", "")), []).append(str(item["subject_id"]))
    rng = random.Random(int(seed))
    folds = [[] for _ in range(k)]
    for _, subjects in sorted(buckets.items()):
        rng.shuffle(subjects)
        for idx, subject in enumerate(subjects):
            folds[idx % k].append(subject)
    all_subjects = {str(item["subject_id"]) for item in unique}
    splits = []
    for idx, test_subjects in enumerate(folds, start=1):
        test = sorted(test_subjects)
        train = sorted(all_subjects - set(test))
        splits.append(Split(name=f"fold_{idx}", kind="5fold", train_subjects=train, test_subjects=test))
    return splits


def build_leave_one_center_out_splits(patients: Sequence[dict]) -> list[Split]:
    unique = _unique_patients(patients)
    centers = sorted({str(item["center"]) for item in unique})
    splits = []
    for center in centers:
        test = sorted(str(item["subject_id"]) for item in unique if str(item["center"]) == center)
        train = sorted(str(item["subject_id"]) for item in unique if str(item["center"]) != center)
        if train and test:
            splits.append(Split(name=f"holdout_{center}", kind="loco", train_subjects=train, test_subjects=test, held_out_center=center))
    return splits
