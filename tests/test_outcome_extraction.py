import argparse

from biodynformer.feature_bank import extract_patient_outcome
from scripts.build_feature_bank import build_parser


def test_extract_patient_outcome_from_success_used_channel_meta():
    patient = {
        "subject_id": "S1",
        "channel_meta": [{"success_used": False}],
    }

    assert extract_patient_outcome(patient) is False


def test_extract_patient_outcome_from_surgery_success_channel_meta():
    patient = {
        "subject_id": "S2",
        "channel_meta": [{"surgery_success": True}],
    }

    assert extract_patient_outcome(patient) is True


def test_extract_patient_outcome_from_raw_s_or_f_text():
    assert extract_patient_outcome({"outcome": "S"}) is True
    assert extract_patient_outcome({"outcome": "F"}) is False


def test_build_feature_bank_defaults_to_all_patients_not_success_only():
    args = build_parser().parse_args(["--output-dir", "D:\\tmp\\bank"])

    assert args.success_only is False
