from biodynformer.splits import build_five_fold_splits, build_leave_one_center_out_splits


def _patients():
    return [
        {"subject_id": "lzu:S1", "center": "lzu", "outcome_success": True},
        {"subject_id": "lzu:S2", "center": "lzu", "outcome_success": False},
        {"subject_id": "hup:S3", "center": "hup", "outcome_success": True},
        {"subject_id": "hup:S4", "center": "hup", "outcome_success": False},
        {"subject_id": "multicenter:S5", "center": "multicenter", "outcome_success": True},
        {"subject_id": "multicenter:S6", "center": "multicenter", "outcome_success": False},
        {"subject_id": "pediatric:S7", "center": "pediatric", "outcome_success": True},
        {"subject_id": "pediatric:S8", "center": "pediatric", "outcome_success": False},
        {"subject_id": "pediatric:S9", "center": "pediatric", "outcome_success": True},
        {"subject_id": "lzu:S10", "center": "lzu", "outcome_success": False},
    ]


def test_five_fold_splits_have_no_patient_leakage():
    splits = build_five_fold_splits(_patients(), n_splits=5, seed=7)

    assert len(splits) == 5
    for split in splits:
        assert set(split.train_subjects).isdisjoint(split.test_subjects)
        assert split.kind == "5fold"


def test_leave_one_center_out_holds_out_exactly_one_center():
    splits = build_leave_one_center_out_splits(_patients())

    assert {split.held_out_center for split in splits} == {"lzu", "hup", "multicenter", "pediatric"}
    for split in splits:
        held = split.held_out_center
        assert all(subject.startswith(f"{held}:") for subject in split.test_subjects)
        assert all(not subject.startswith(f"{held}:") for subject in split.train_subjects)
        assert split.kind == "loco"
