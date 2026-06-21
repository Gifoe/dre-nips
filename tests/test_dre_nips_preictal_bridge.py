from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.build_patient_records_from_dre_nips_preictal_only as bridge
from scripts.build_patient_records_from_dre_nips_preictal_only import (
    build_parser,
    convert_dre_patients,
    load_valid_shard,
    summarize_patient_records,
)


def _fake_dre_patient():
    signal = np.ones((3, 1300), dtype=np.float32)
    seizure = SimpleNamespace(
        subject_id="P001",
        seizure_id="sz1",
        signal=signal,
        sfreq=10.0,
        channel_names=["A1", "A2", "B1"],
        seizure_onset_sec=125.0,
        seizure_offset_sec=140.0,
        labels=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        channel_meta=[
            {
                "source_path": "P001_sz1.edf",
                "success_used": False,
                "n_interictal_runs_subject": 3,
                "interictal_source_paths": "bad.edf",
            },
            {"source_path": "P001_sz1.edf", "success_used": False},
            {"source_path": "P001_sz1.edf", "success_used": False},
        ],
        interictal_sources=[{"edf_path": "bad.edf"}],
    )
    return SimpleNamespace(
        subject_id="P001",
        seizures=[seizure],
        canonical_channels=["A1", "A2", "B1"],
        labels=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        channel_meta=list(seizure.channel_meta),
    )


def test_parser_defaults_to_not_success_only():
    args = build_parser().parse_args(
        [
            "--output-pkl",
            "patients.pkl",
            "--output-summary-json",
            "summary.json",
            "--quality-audit-csv",
            "quality.csv",
            "--read-audit-dir",
            "read_audit",
        ]
    )

    assert args.success_only is False


def test_parser_no_longer_requires_external_dre_nips_root():
    help_text = build_parser().format_help()

    assert "dre-nips-root" not in help_text


def test_main_passes_quality_index_by_keyword(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "read_quality_reports", lambda args: [])

    called = {}

    def fake_convert(patients_by_center, *, quality_index):
        called["quality_index"] = quality_index
        return [], [], 0

    monkeypatch.setattr(bridge, "load_dre_center_patient_records", lambda args, center: [])
    monkeypatch.setattr(bridge, "convert_dre_patients", fake_convert)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_patient_records_from_dre_nips_preictal_only.py",
            "--output-pkl",
            str(tmp_path / "patients.pkl"),
            "--output-summary-json",
            str(tmp_path / "summary.json"),
            "--quality-audit-csv",
            str(tmp_path / "quality.csv"),
            "--read-audit-dir",
            str(tmp_path / "read_audit"),
            "--centers",
            "lzu",
            "--lzu-root",
            str(tmp_path),
            "--lzu-ez-annotations-path",
            str(tmp_path / "labels.xlsx"),
            "--lzu-seizure-times-path",
            str(tmp_path / "times.xlsx"),
        ],
    )

    bridge.main()

    assert called["quality_index"] == {}


def test_main_writes_each_center_shard_before_later_center_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "read_quality_reports", lambda args: [])

    def fake_load(args, center):
        if center == "hup":
            return [_fake_dre_patient()]
        raise MemoryError("boom")

    monkeypatch.setattr(bridge, "load_dre_center_patient_records", fake_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_patient_records_from_dre_nips_preictal_only.py",
            "--output-pkl",
            str(tmp_path / "patients.pkl"),
            "--output-summary-json",
            str(tmp_path / "summary.json"),
            "--quality-audit-csv",
            str(tmp_path / "quality.csv"),
            "--read-audit-dir",
            str(tmp_path / "read_audit"),
            "--centers",
            "hup,multicenter",
            "--hup-root",
            str(tmp_path),
            "--multicenter-root",
            str(tmp_path),
        ],
    )

    try:
        bridge.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit")

    shard = tmp_path / "patient_record_shards" / "hup.pkl"
    records = load_valid_shard(shard)
    assert records is not None
    assert len(records) == 1
    assert records[0].center == "hup"


def test_main_reuses_existing_valid_center_shard(monkeypatch, tmp_path):
    existing_records, _, _ = convert_dre_patients({"hup": [_fake_dre_patient()]}, quality_index={})
    shard_dir = tmp_path / "patient_record_shards"
    bridge.write_pickle(shard_dir / "hup.pkl", existing_records)
    monkeypatch.setattr(bridge, "read_quality_reports", lambda args: [])

    def fail_if_called(args, center):
        raise AssertionError(f"loader should not run for {center}")

    monkeypatch.setattr(bridge, "load_dre_center_patient_records", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_patient_records_from_dre_nips_preictal_only.py",
            "--output-pkl",
            str(tmp_path / "patients.pkl"),
            "--output-summary-json",
            str(tmp_path / "summary.json"),
            "--quality-audit-csv",
            str(tmp_path / "quality.csv"),
            "--read-audit-dir",
            str(tmp_path / "read_audit"),
            "--centers",
            "hup",
            "--hup-root",
            str(tmp_path),
        ],
    )

    bridge.main()

    merged = load_valid_shard(tmp_path / "patients.pkl")
    assert merged is not None
    assert [record.center for record in merged] == ["hup"]


def test_convert_dre_patients_keeps_failure_and_removes_forbidden_fields():
    records, quality_rows, removed_count = convert_dre_patients(
        {"lzu": [_fake_dre_patient()]},
        quality_index={},
    )

    assert removed_count >= 2
    assert len(records) == 1
    assert records[0].center == "lzu"
    assert records[0].outcome_success is False
    assert records[0].labels_ez.tolist() == [0.0, 1.0, 1.0]
    assert records[0].seizures[0].labels_ez.tolist() == [0.0, 1.0, 1.0]
    assert not hasattr(records[0].seizures[0], "interictal_sources")
    assert "n_interictal_runs_subject" not in records[0].channel_meta[0]
    assert quality_rows[0]["match_status"] == "missing"


def test_summary_reports_required_preictal_bridge_counts():
    records, _, removed_count = convert_dre_patients(
        {"lzu": [_fake_dre_patient()]},
        quality_index={},
    )

    summary = summarize_patient_records(records, interictal_fields_removed_count=removed_count)

    assert summary["total_patients"] == 1
    assert summary["total_seizures"] == 1
    assert summary["failure_patients_by_center"] == {"lzu": 1}
    assert summary["label_semantics"] == "0=EZ, 1=NEZ, inherited from dre-nips reader"
    assert summary["ez_channel_count"] == 1
    assert summary["not_enough_preictal_window_count"] == 0
