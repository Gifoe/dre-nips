import json
from pathlib import Path

import numpy as np

from biodynformer.feature_bank import (
    FEATURE_SCHEMA,
    FORBIDDEN_BANK_KEYS,
    build_feature_bank_from_records,
    load_feature_bank_index,
)


def _toy_record():
    sfreq = 10.0
    samples = 1300
    signal = np.vstack(
        [
            np.sin(np.linspace(0, 20, samples)),
            np.cos(np.linspace(0, 20, samples)),
            np.sin(np.linspace(0, 40, samples)),
        ]
    ).astype(np.float32)
    return {
        "center": "lzu",
        "subject_id": "S1",
        "outcome_success": True,
        "seizures": [
            {
                "run_id": "run1",
                "seizure_id": "sz1",
                "quality_rating": "GOOD",
                "signal": signal,
                "sfreq": sfreq,
                "seizure_onset_sec": 125.0,
                "channel_names": ["A1", "A2", "B1"],
                "labels_ez": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
        ],
    }


def test_build_feature_bank_writes_expected_schema_and_tensors(tmp_path: Path):
    output_dir = tmp_path / "bank"

    summary = build_feature_bank_from_records([_toy_record()], output_dir=output_dir)
    index = load_feature_bank_index(output_dir)

    assert summary["num_runs"] == 1
    assert (output_dir / "feature_schema.json").exists()
    assert len(index) == 1

    arr = np.load(index[0]["tensor_path"], allow_pickle=True)
    expected_keys = {
        "node_features",
        "hfo_features",
        "quality_features",
        "causal_edge",
        "sync_edge",
        "structural_edge",
        "coverage_features",
        "window_mask",
        "channel_mask",
        "labels_ez",
        "outcome_success",
        "center",
        "subject_id",
        "run_id",
        "channel_names",
    }
    assert expected_keys.issubset(set(arr.files))
    assert not (set(arr.files) & FORBIDDEN_BANK_KEYS)
    assert arr["node_features"].shape[:2] == (4, 3)
    assert arr["causal_edge"].shape == (4, 3, 3)
    assert arr["window_mask"].tolist() == [True, True, True, True]

    schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["node_features"] == FEATURE_SCHEMA["node_features"]


def test_build_feature_bank_uses_window_mask_for_missing_preictal_window(tmp_path: Path):
    record = _toy_record()
    record["seizures"][0]["seizure_onset_sec"] = 20.0

    build_feature_bank_from_records([record], output_dir=tmp_path)
    index = load_feature_bank_index(tmp_path)
    arr = np.load(index[0]["tensor_path"], allow_pickle=True)

    assert arr["window_mask"].tolist() == [False, False, False, False]
