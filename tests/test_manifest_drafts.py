from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from biodynformer.manifest_drafts import generate_manifest_drafts

COL_SUBJECT = "\u60a3\u8005ID"
COL_RUN = "\u53d1\u4f5c\u540d\u79f0"
COL_RATING = "\u8d28\u91cf\u8bc4\u7ea7"
COL_FILE = "\u6587\u4ef6\u8def\u5f84"
COL_FILE_RATING = "\u6587\u4ef6\u8d28\u91cf\u8bc4\u7ea7"
COL_CHANNEL = "\u901a\u9053\u540d"
COL_IS_EZ = "\u662f\u5426EZ"
SHEET_SUMMARY = "\u6587\u4ef6\u6c47\u603b"
SHEET_CHANNELS = "\u95ee\u9898\u901a\u9053"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def test_generate_lzu_draft_manifest_derives_onset_and_channel_labels(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    output_dir = tmp_path / "out"

    with pd.ExcelWriter(metadata_dir / "slope_quality_report_en1\u4fee\u6539.xlsx") as writer:
        pd.DataFrame(
            [
                {
                    COL_SUBJECT: "alice",
                    COL_RUN: "SZ1_onset",
                    COL_RATING: "GOOD",
                    COL_FILE: "slope\\alice_SZ1_onset_00-01-40_slope_results.xlsx",
                }
            ]
        ).to_excel(writer, sheet_name=SHEET_SUMMARY, index=False)
        pd.DataFrame(
            [
                {
                    COL_SUBJECT: "alice",
                    COL_RUN: "SZ1_onset",
                    COL_FILE_RATING: "GOOD",
                    COL_CHANNEL: "A1",
                    COL_IS_EZ: "NEZ",
                    COL_FILE: "slope\\alice_SZ1_onset_00-01-40_slope_results.xlsx",
                },
                {
                    COL_SUBJECT: "alice",
                    COL_RUN: "SZ1_onset",
                    COL_FILE_RATING: "GOOD",
                    COL_CHANNEL: "A2",
                    COL_IS_EZ: "EZ",
                    COL_FILE: "slope\\alice_SZ1_onset_00-01-40_slope_results.xlsx",
                },
            ]
        ).to_excel(writer, sheet_name=SHEET_CHANNELS, index=False)
    pd.DataFrame(
        [
            {
                "\u59d3\u540d": "alice",
                "\u53d1\u4f5c\u7f16\u53f7": "SZ1(2026.1.1)",
                "\u8111\u7535\u56fe\u8bb0\u5f55\u5f00\u59cb\u65f6\u95f4": "00:00:00",
                "Unnamed: 13": "00:01:40",
            }
        ]
    ).to_excel(metadata_dir / "SEEG\u6570\u636e\u5206\u6790\u65f6\u95f4\u6807\u7b7e.xlsx", sheet_name="Sheet1", index=False)
    pd.DataFrame([{"\u59d3\u540d": "alice", "Engel\u5206\u7ea7(S/F)": "S"}]).to_excel(
        metadata_dir / "label-seeg_with_engel.xlsx", sheet_name="Sheet1", index=False
    )

    summary = generate_manifest_drafts(metadata_dir=metadata_dir, output_dir=output_dir, centers=["lzu"])

    rows = _read_csv(output_dir / "lzu_all_draft_manifest.csv")
    assert summary["centers"]["lzu"]["all_draft_rows"] == 1
    assert rows[0]["subject_id"] == "alice"
    assert rows[0]["run_id"] == "SZ1_onset"
    assert rows[0]["seizure_onset_sec"] == "100.0"
    assert rows[0]["channel_names"] == "A1,A2"
    assert rows[0]["labels_ez"] == "0,1"
    assert rows[0]["outcome"] == "S"
    assert "signal_path" in rows[0]["missing_fields"]

    strict_rows = _read_csv(output_dir / "lzu_strict_pipeline_manifest.csv")
    assert strict_rows == []


def test_generate_pediatric_draft_manifest_uses_subject_channel_labels(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    output_dir = tmp_path / "out"

    pd.DataFrame(
        [
            {
                COL_SUBJECT: "S001",
                COL_RUN: "SZ1_seizure",
                COL_RATING: "REVIEW",
                COL_FILE: "D:\\missing\\S001_SZ1_seizure_slope_results.xlsx",
            }
        ]
    ).to_excel(metadata_dir / "\u513f\u79d1\u6570\u636e\u5206\u7c7b.xlsx", sheet_name="seizure_onsets", index=False)
    with pd.ExcelWriter(metadata_dir / "pediatric_ez_channels_final.xlsx") as writer:
        pd.DataFrame([{"subject_id": "S001", "surgery_result": "\u6210\u529f"}]).to_excel(
            writer, sheet_name="EZ_\u786e\u5b9a\u6c47\u603b", index=False
        )
        pd.DataFrame(
            [
                {
                    "subject_id": "S001",
                    "channel_order": 1,
                    "channel_name_norm": "A1",
                    "usable_channel_mask": 1,
                    "model_label_ez_excluding_bad": 1,
                },
                {
                    "subject_id": "S001",
                    "channel_order": 2,
                    "channel_name_norm": "A2",
                    "usable_channel_mask": 1,
                    "model_label_ez_excluding_bad": 0,
                },
            ]
        ).to_excel(writer, sheet_name="channel_level_labels", index=False)

    generate_manifest_drafts(metadata_dir=metadata_dir, output_dir=output_dir, centers=["pediatric"])

    rows = _read_csv(output_dir / "pediatric_good_review_draft_manifest.csv")
    assert len(rows) == 1
    assert rows[0]["channel_names"] == "A1,A2"
    assert rows[0]["labels_ez"] == "1,0"
    assert rows[0]["outcome"] == "S"
