from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroez_multitask.causal_graph_encoder import DelayAwareDirectedGraphEncoder, permute_graph_edges
from neuroez_multitask.evidence_views import CAUSAL_GRAPH_ALGORITHM, PHYSICS_FEATURE_LEVEL
from neuroez_multitask.experiments import model_kwargs_for_experiment
from neuroez_multitask.splits import PatientSplit
from run_task1_pgc_ez import (
    _save_fold_task1_checkpoint,
    _summary_from_task1_prediction_rows,
    _write_split_metadata,
)
from run_task2_pgc_outcome import _load_fold_task1_payload
from scripts.build_physics_window_cache import build_cache_payload
from scripts.inspect_physics_cache import inspect_cache_payload


def test_task1_summary_groups_channel_rows_back_to_patients():
    rows = [
        {"subject_id": "p1", "fold": 0, "label_nez": 0.0, "nez_prob": 0.4},
        {"subject_id": "p1", "fold": 0, "label_nez": 1.0, "nez_prob": 0.4},
    ]

    summary = _summary_from_task1_prediction_rows(rows)

    assert summary["patient_macro_F1"] == pytest.approx(1.0 / 3.0)


def test_task1_fold_checkpoint_records_split_scope_and_safety(tmp_path: Path):
    split = PatientSplit("5fold", 2, ["train-a"], ["val-a"], ["test-a"])
    model = torch.nn.Linear(2, 1)

    _save_fold_task1_checkpoint(
        output_dir=tmp_path,
        split=split,
        model=model,
        normalizer={"kind": "train-only"},
        experiment_name="T1_FULL_PGC",
        model_kwargs={"model_dim": 8},
        cache_meta={"cache_name": "toy.pkl"},
    )

    payload = torch.load(tmp_path / "fold_2" / "best_task1_backbone.pt", map_location="cpu", weights_only=False)
    assert payload["checkpoint_scope"] == "fold_specific"
    assert payload["safe_for_task2_fold_loading"] is True
    assert payload["train_subjects"] == ["train-a"]
    assert payload["test_subjects"] == ["test-a"]
    assert payload["normalizer"] == {"kind": "train-only"}


def test_split_metadata_written_per_fold_and_root(tmp_path: Path):
    splits = [
        PatientSplit("5fold", 0, ["a", "b"], ["c"], ["d"]),
        PatientSplit("5fold", 1, ["c", "d"], ["a"], ["b"]),
    ]

    _write_split_metadata(tmp_path, splits)

    root = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    fold0 = json.loads((tmp_path / "fold_0" / "split_subjects.json").read_text(encoding="utf-8"))
    assert root[0]["test_subjects"] == ["d"]
    assert fold0["train_subjects"] == ["a", "b"]
    assert fold0["val_subjects"] == ["c"]


def test_task2_rejects_single_task1_checkpoint_without_explicit_allow(tmp_path: Path):
    checkpoint = tmp_path / "best_task1_backbone.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    split = PatientSplit("5fold", 0, ["train"], [], ["test"])

    with pytest.raises(RuntimeError, match="Single --task1_checkpoint is unsafe"):
        _load_fold_task1_payload(
            checkpoint_dir=None,
            single_checkpoint=checkpoint,
            allow_external=False,
            split=split,
            device=torch.device("cpu"),
        )


def test_task2_fold_checkpoint_loader_detects_train_test_leakage(tmp_path: Path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    torch.save(
        {
            "model_state_dict": {},
            "train_subjects": ["leaked", "train"],
            "test_subjects": ["heldout"],
            "safe_for_task2_fold_loading": True,
        },
        fold_dir / "best_task1_backbone.pt",
    )
    split = PatientSplit("5fold", 0, ["train"], [], ["leaked"])

    with pytest.raises(RuntimeError, match="Task1 checkpoint leakage detected"):
        _load_fold_task1_payload(
            checkpoint_dir=tmp_path,
            single_checkpoint=None,
            allow_external=False,
            split=split,
            device=torch.device("cpu"),
        )


def test_task2_fold_checkpoint_loader_requires_matching_test_fold(tmp_path: Path):
    fold_dir = tmp_path / "fold_1"
    fold_dir.mkdir()
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "train_subjects": ["train"],
            "test_subjects": ["test"],
            "normalizer": {"kind": "fold"},
            "safe_for_task2_fold_loading": True,
        },
        fold_dir / "best_task1_backbone.pt",
    )
    split = PatientSplit("5fold", 1, ["train"], [], ["test"])

    payload = _load_fold_task1_payload(
        checkpoint_dir=tmp_path,
        single_checkpoint=None,
        allow_external=False,
        split=split,
        device=torch.device("cpu"),
    )

    assert payload["normalizer"] == {"kind": "fold"}
    assert payload["model_state_dict"]["weight"].item() == 1.0


def test_experiment_config_supports_random_weight_and_permute_modes():
    weight = model_kwargs_for_experiment("C2_FULL_PGC_RANDOM_WEIGHT", 8)
    permute = model_kwargs_for_experiment("C2_FULL_PGC_RANDOM_PERMUTE", 8)

    assert weight["random_graph"] is True
    assert weight["random_graph_mode"] == "weight"
    assert permute["random_graph"] is True
    assert permute["random_graph_mode"] == "permute"


def test_permute_graph_edges_preserves_non_diagonal_values_and_zero_diagonal():
    adjacency = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]]]],
        dtype=torch.float32,
    )

    shuffled = permute_graph_edges(adjacency)

    assert shuffled.shape == adjacency.shape
    assert torch.all(torch.diagonal(shuffled, dim1=-2, dim2=-1) == 0)
    edge_mask = ~torch.eye(3, dtype=torch.bool).reshape(-1)
    assert torch.sort(shuffled.reshape(-1, 9)[:, edge_mask]).values.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_random_graph_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported random_graph_mode"):
        DelayAwareDirectedGraphEncoder(model_dim=4, random_graph=True, random_graph_mode="shuffle")


def _patient_record() -> dict:
    signal = np.vstack(
        [
            np.sin(np.linspace(0, 10, 512)),
            np.cos(np.linspace(0, 8, 512)),
        ]
    ).astype(np.float32)
    return {
        "center": "hup",
        "subject_id": "sub-HUP001",
        "outcome_success": True,
        "canonical_channels": ["A1", "A2"],
        "labels": np.asarray([0.0, 1.0], dtype=np.float32),
        "seizures": [
            {
                "run_id": "run1",
                "signal": signal,
                "sfreq": 128.0,
                "seizure_onset_sec": 2.0,
                "channel_names": ["A1", "A2"],
                "labels": np.asarray([0.0, 1.0], dtype=np.float32),
            }
        ],
    }


def test_cache_meta_and_inspector_mark_v1_proxy_lite_status():
    payload = build_cache_payload(
        [_patient_record()],
        source_patient_records_pkl="synthetic.pkl",
        window_length_sec=0.5,
        window_step_sec=0.5,
        pre_onset_sec=1.0,
        post_onset_sec=1.0,
        physics_mode="proxy",
        causal_graph_mode="tfccm_lite",
        topology_mode="simple",
    )

    report = inspect_cache_payload(payload)

    assert CAUSAL_GRAPH_ALGORITHM == "tfccm_lite_nearest_neighbor_cross_mapping"
    assert PHYSICS_FEATURE_LEVEL == "physics_proxy_v1"
    assert report["causal_graph_algorithm"] == CAUSAL_GRAPH_ALGORITHM
    assert report["physics_feature_level"] == PHYSICS_FEATURE_LEVEL
    assert report["feature_names"]["b0"]
    assert report["feature_names"]["physics"]
    assert report["sample_shapes"]["tfccm_adjacency"][0] > 0


def test_inspector_warns_when_v1_feature_status_is_missing():
    payload = {
        "run_records": [
            {
                "subject_id": "p1",
                "sample": {
                    "window_features": np.zeros((1, 1, 1), dtype=np.float32),
                    "physics_node_features": np.zeros((1, 1, 1), dtype=np.float32),
                    "tfccm_adjacency": np.zeros((1, 1, 1), dtype=np.float32),
                    "tfccm_delay": np.zeros((1, 1, 1), dtype=np.float32),
                    "causal_node_features": np.zeros((1, 1, 1), dtype=np.float32),
                    "topology_graph_features": np.zeros((1, 8), dtype=np.float32),
                    "window_relative_centers_sec": np.zeros((1,), dtype=np.float32),
                    "window_mask": np.ones((1,), dtype=bool),
                },
            }
        ],
        "patient_index": {
            "p1": {
                "labels": np.asarray([0.0], dtype=np.float32),
                "labels_nez": np.asarray([0.0], dtype=np.float32),
                "labels_ez": np.asarray([1.0], dtype=np.float32),
            }
        },
        "outcome_index": {},
        "cache_meta": {},
    }

    report = inspect_cache_payload(payload)

    assert "causal_graph_algorithm missing" in report["warnings"]
    assert "physics_feature_level missing" in report["warnings"]
