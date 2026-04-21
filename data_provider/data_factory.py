from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from data_provider.ez_dataset import build_or_load_patient_bags


def build_outer_splits(
    patient_bags: List[Dict[str, Any]],
    *,
    split_strategy: str = "5fold",
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, List[str]]]:
    subject_ids = [bag["subject_id"] for bag in patient_bags]
    if len(subject_ids) < 2:
        raise ValueError("At least two patients are required for TeChEZ cross-validation.")

    if split_strategy.lower() == "lopo":
        return [
            {
                "fold_idx": fold_idx + 1,
                "train_subjects": [sid for sid in subject_ids if sid != test_subject],
                "test_subjects": [test_subject],
            }
            for fold_idx, test_subject in enumerate(subject_ids)
        ]

    actual_splits = max(2, min(int(n_splits), len(subject_ids)))
    kfold = KFold(n_splits=actual_splits, shuffle=True, random_state=random_seed)
    subject_array = np.asarray(subject_ids)
    splits: List[Dict[str, List[str]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(subject_array), start=1):
        splits.append(
            {
                "fold_idx": fold_idx,
                "train_subjects": subject_array[train_idx].tolist(),
                "test_subjects": subject_array[test_idx].tolist(),
            }
        )
    return splits


def split_train_val_subjects(
    train_subjects: List[str],
    *,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    fold_idx: int = 0,
) -> Tuple[List[str], List[str]]:
    train_subjects = list(train_subjects)
    if len(train_subjects) < 2:
        return train_subjects, train_subjects

    rng = np.random.default_rng(seed=int(random_seed) + int(fold_idx))
    rng.shuffle(train_subjects)
    n_val = max(1, int(round(len(train_subjects) * float(val_ratio))))
    n_val = min(n_val, len(train_subjects) - 1)
    val_subjects = train_subjects[:n_val]
    fit_subjects = train_subjects[n_val:]
    return fit_subjects, val_subjects


def data_provider(args: Any):
    patient_bags = build_or_load_patient_bags(args)
    outer_splits = build_outer_splits(
        patient_bags,
        split_strategy=str(getattr(args, "split_strategy", "5fold")),
        n_splits=int(getattr(args, "n_splits", 5)),
        random_seed=int(getattr(args, "random_seed", 42)),
    )
    return patient_bags, outer_splits


__all__ = ["build_outer_splits", "data_provider", "split_train_val_subjects"]
