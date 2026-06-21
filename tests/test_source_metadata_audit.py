from __future__ import annotations

from pathlib import Path

import pandas as pd

from biodynformer.source_metadata import audit_source_metadata, read_quality_summary


def test_read_quality_summary_finds_required_chinese_columns(tmp_path: Path):
    report = tmp_path / "quality.xlsx"
    pd.DataFrame(
        [
            {"患者ID": "P001", "发作名称": "SZ1_onset", "质量评级": "GOOD", "文件路径": "x.xlsx"},
            {"患者ID": "P001", "发作名称": "SZ2_onset", "质量评级": "POOR", "文件路径": "y.xlsx"},
        ]
    ).to_excel(report, sheet_name="文件汇总", index=False)

    rows = read_quality_summary(report, center="lzu")

    assert [row.quality_rating for row in rows] == ["GOOD", "POOR"]
    assert rows[0].subject_id == "P001"
    assert rows[0].run_id == "SZ1_onset"


def test_audit_source_metadata_reports_outcome_and_missing_signal_paths(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [{"患者ID": "alice", "发作名称": "SZ1_onset", "质量评级": "REVIEW", "文件路径": "slope.xlsx"}]
    ).to_excel(metadata_dir / "slope_quality_report_en1修改.xlsx", sheet_name="文件汇总", index=False)
    pd.DataFrame([{"姓名": "alice", "Engel分级(S/F)": "S"}]).to_excel(
        metadata_dir / "label-seeg_with_engel.xlsx", sheet_name="Sheet1", index=False
    )

    audit = audit_source_metadata(metadata_dir=metadata_dir, centers=["lzu"])

    assert audit["centers"]["lzu"]["quality_rows"] == 1
    assert audit["centers"]["lzu"]["outcome_subjects"] == 1
    assert audit["centers"]["lzu"]["signal_files_found"] == 0
    assert audit["can_build_feature_bank"] is False


def test_audit_source_metadata_combines_success_and_failure_quality_reports(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for name, patient, rating in [
        ("slope_quality_report_HUP_S.xlsx", "HUP001", "GOOD"),
        ("slope_quality_report_HUP_F.xlsx", "HUP002", "POOR"),
    ]:
        pd.DataFrame([{"患者ID": patient, "发作名称": "task-ictal_run-01", "质量评级": rating}]).to_excel(
            metadata_dir / name, sheet_name="文件汇总", index=False
        )

    audit = audit_source_metadata(metadata_dir=metadata_dir, centers=["hup"])

    assert audit["centers"]["hup"]["quality_rows"] == 2
    assert audit["centers"]["hup"]["quality_subjects"] == 2
    assert audit["centers"]["hup"]["quality_ratings"] == {"GOOD": 1, "POOR": 1}


def test_audit_source_metadata_uses_hup_participants_outcomes(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [{"患者ID": "HUP001", "发作名称": "task-ictal_run-01", "质量评级": "GOOD"}]
    ).to_excel(metadata_dir / "slope_quality_report_HUP_S.xlsx", sheet_name="文件汇总", index=False)
    participants = tmp_path / "participants.tsv"
    participants.write_text(
        "participant_id\toutcome\tengel\n"
        "sub-HUP001\tS\t1A\n"
        "sub-HUP002\tF\t3A\n",
        encoding="utf-8",
    )

    audit = audit_source_metadata(metadata_dir=metadata_dir, centers=["hup"], hup_participants_path=participants)

    assert audit["centers"]["hup"]["outcome_subjects"] == 2
    assert audit["centers"]["hup"]["outcome_counts"] == {"F": 1, "S": 1}
    assert audit["centers"]["hup"]["quality_subjects_without_outcome"] == []
    assert audit["centers"]["hup"]["outcome_subjects_without_quality"] == ["HUP002"]
