from __future__ import annotations

from neuroez_multitask.splits import make_patient_splits


def test_patient_splits_never_overlap_subjects():
    patient_index = {
        "p1": {"center": "lzu"},
        "p2": {"center": "hup"},
        "p3": {"center": "lzu"},
        "p4": {"center": "multicenter"},
        "p5": {"center": "pediatric"},
    }

    splits = make_patient_splits(patient_index, strategy="5fold", n_splits=3, seed=7)

    assert splits
    for split in splits:
        assert set(split.train_subjects).isdisjoint(split.val_subjects)
        assert set(split.train_subjects).isdisjoint(split.test_subjects)
        assert set(split.val_subjects).isdisjoint(split.test_subjects)
