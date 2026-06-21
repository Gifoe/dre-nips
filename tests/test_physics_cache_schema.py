from __future__ import annotations

import numpy as np

from scripts.build_physics_window_cache import build_cache_payload
from scripts.inspect_physics_cache import inspect_cache_payload


def _patient():
    signal = np.vstack(
        [
            np.sin(np.linspace(0, 10, 1024)),
            np.cos(np.linspace(0, 8, 1024)),
        ]
    ).astype(np.float32)
    return {
        "center": "hup",
        "subject_id": "sub-HUP001",
        "outcome_success": False,
        "Engel": "II",
        "canonical_channels": ["A1", "A2"],
        "labels": np.asarray([0.0, 1.0], dtype=np.float32),
        "seizures": [
            {
                "run_id": "run1",
                "seizure_id": "sz1",
                "signal": signal,
                "sfreq": 128.0,
                "seizure_onset_sec": 2.0,
                "channel_names": ["A1", "A2"],
                "labels": np.asarray([0.0, 1.0], dtype=np.float32),
            }
        ],
    }


def test_physics_cache_payload_contains_required_shapes():
    payload = build_cache_payload(
        [_patient()],
        source_patient_records_pkl="synthetic.pkl",
        window_length_sec=0.5,
        window_step_sec=0.5,
        pre_onset_sec=1.0,
        post_onset_sec=1.0,
    )

    report = inspect_cache_payload(payload)
    assert report["usable_physics_cache"] is True
    sample = payload["run_records"][0]["sample"]
    assert sample["window_features"].ndim == 3
    assert sample["physics_node_features"].shape[:2] == sample["window_features"].shape[:2]
    assert sample["tfccm_adjacency"].shape[:3] == (
        sample["window_features"].shape[0],
        sample["window_features"].shape[1],
        sample["window_features"].shape[1],
    )
    assert sample["tfccm_delay"].shape == sample["tfccm_adjacency"].shape
    assert sample["causal_node_features"].shape[:2] == sample["window_features"].shape[:2]
    assert sample["topology_graph_features"].ndim == 2
    assert sample["window_relative_centers_sec"].shape[0] == sample["window_features"].shape[0]
    assert sample["window_mask"].shape[0] == sample["window_features"].shape[0]
