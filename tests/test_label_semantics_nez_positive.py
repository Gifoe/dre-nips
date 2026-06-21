from __future__ import annotations

import numpy as np

from scripts.build_physics_window_cache import build_cache_payload


def test_cache_preserves_nez_positive_labels_and_derives_ez_labels():
    patient_records = [
        {
            "center": "lzu",
            "subject_id": "p1",
            "outcome_success": True,
            "Engel": "I",
            "canonical_channels": ["A1", "A2", "A3"],
            "labels": np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
            "seizures": [
                {
                    "run_id": "s1",
                    "seizure_id": "s1",
                    "signal": np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32),
                    "sfreq": 128.0,
                    "seizure_onset_sec": 2.0,
                    "channel_names": ["A1", "A2", "A3"],
                    "labels": np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
                }
            ],
        }
    ]

    payload = build_cache_payload(
        patient_records,
        source_patient_records_pkl="synthetic.pkl",
        window_length_sec=0.5,
        window_step_sec=0.5,
        pre_onset_sec=1.0,
        post_onset_sec=1.0,
    )

    entry = payload["patient_index"]["lzu:p1"]
    np.testing.assert_array_equal(entry["labels"], np.asarray([0.0, 1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(entry["labels_nez"], np.asarray([0.0, 1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(entry["labels_ez"], np.asarray([1.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(entry["label_mask"], np.asarray([True, True, False]))
    assert "internal_positive_class=NEZ" in payload["cache_meta"]["label_semantics"]
    assert payload["cache_meta"]["report_semantics"] == "Task1 reports P(EZ)=1-P(NEZ)"
