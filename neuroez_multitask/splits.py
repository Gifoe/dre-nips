from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PatientSplit:
    name: str
    fold: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def _subject_ids(patient_index: Mapping[str, object] | Sequence[str]) -> list[str]:
    if isinstance(patient_index, Mapping):
        return sorted(str(key) for key in patient_index)
    return sorted(str(item) for item in patient_index)


def make_patient_splits(
    patient_index: Mapping[str, object] | Sequence[str],
    *,
    strategy: str = "5fold",
    n_splits: int = 5,
    seed: int = 42,
) -> list[PatientSplit]:
    subjects = _subject_ids(patient_index)
    if len(subjects) < 2:
        return [PatientSplit("single", 0, subjects, [], subjects)]
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    strategy_key = strategy.lower()
    if strategy_key in {"5fold", "kfold"}:
        k = max(2, min(int(n_splits), len(shuffled)))
        folds = [list(part) for part in np.array_split(np.asarray(shuffled, dtype=object), k)]
        splits: list[PatientSplit] = []
        for idx, test in enumerate(folds):
            val = folds[(idx + 1) % k]
            train = [sid for j, fold in enumerate(folds) if j not in {idx, (idx + 1) % k} for sid in fold]
            if not train:
                train = [sid for sid in shuffled if sid not in set(test)]
            splits.append(PatientSplit("5fold", idx, list(train), list(val), list(test)))
        return splits
    if strategy_key in {"lopo", "leave_one_patient_out"}:
        return [
            PatientSplit("lopo", idx, [sid for sid in shuffled if sid != test], [], [test])
            for idx, test in enumerate(shuffled)
        ]
    if strategy_key in {"leave_one_center_out", "loco"}:
        if not isinstance(patient_index, Mapping):
            raise ValueError("leave_one_center_out requires patient_index metadata with center fields.")
        centers = sorted({str(meta.get("center", "unknown")) if isinstance(meta, Mapping) else "unknown" for meta in patient_index.values()})
        out = []
        for idx, center in enumerate(centers):
            test = [sid for sid, meta in patient_index.items() if (str(meta.get("center", "unknown")) if isinstance(meta, Mapping) else "unknown") == center]
            train = [sid for sid in subjects if sid not in set(test)]
            out.append(PatientSplit("leave_one_center_out", idx, train, [], test))
        return out
    raise ValueError(f"Unknown split strategy: {strategy}")
