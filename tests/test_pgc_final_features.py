from __future__ import annotations

import numpy as np
import pytest

from neuroez_multitask.causal_features import (
    circular_shift_surrogate,
    compute_tfccm_full_graph,
    standardize_channels,
)
from neuroez_multitask.evidence_views import PHYSICS_PROXY_FEATURE_NAMES, TOPOLOGY_FEATURE_NAMES
from neuroez_multitask.topology_features import (
    TOPOLOGY_FULL_FEATURE_NAMES,
    TOPOLOGY_SIMPLE_FEATURE_NAMES,
    compute_topology_features_full,
    sinkhorn_distance,
)
from scripts.build_physics_window_cache import (
    _parse_float_tuple,
    _parse_int_tuple,
    build_cache_payload,
    compute_lagged_corr_graph,
)
from scripts.inspect_physics_cache import inspect_cache_payload
from run_task1_pgc_ez import _model_kwargs as task1_model_kwargs
from run_task2_pgc_outcome import _model_kwargs as task2_model_kwargs


def _patient() -> dict:
    t = np.linspace(0, 10, 512, dtype=np.float32)
    signal = np.vstack(
        [
            np.sin(t),
            np.sin(np.roll(t, 3)),
            np.cos(t * 0.5),
        ]
    ).astype(np.float32)
    return {
        "center": "hup",
        "subject_id": "sub-HUP001",
        "outcome_success": True,
        "Engel": "I",
        "canonical_channels": ["A1", "A2", "A3"],
        "labels": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        "seizures": [
            {
                "run_id": "run1",
                "seizure_id": "sz1",
                "signal": signal,
                "sfreq": 128.0,
                "seizure_onset_sec": 2.0,
                "channel_names": ["A1", "A2", "A3"],
                "labels": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            }
        ],
    }


def test_standardize_channels_and_shift_surrogate_are_finite():
    data = np.asarray([[1.0, 1.0, 1.0], [1.0, np.nan, np.inf]], dtype=np.float32)

    standardized = standardize_channels(data)

    assert standardized.shape == data.shape
    assert np.isfinite(standardized).all()
    assert np.allclose(standardized[0], 0.0)
    assert circular_shift_surrogate(np.asarray([1, 2, 3]), 1).tolist() == [3, 1, 2]


def test_tfccm_full_graph_returns_diagnostic_matrices():
    t = np.linspace(0, 12, 256, dtype=np.float32)
    segment = np.vstack([np.sin(t), np.roll(np.sin(t), 2), np.cos(t)]).astype(np.float32)

    graph = compute_tfccm_full_graph(
        segment,
        128.0,
        embedding_dims=(2,),
        taus=(1,),
        max_delay_ms=20.0,
        library_fractions=(0.5, 1.0),
        n_surrogates=0,
        max_points=64,
    )

    assert set(graph) == {"adjacency", "delay", "pvalue", "convergence"}
    for value in graph.values():
        assert value.shape == (3, 3)
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    assert np.allclose(np.diag(graph["adjacency"]), 0.0)


def test_lagged_corr_graph_is_available_as_debug_baseline():
    t = np.linspace(0, 6, 128, dtype=np.float32)
    segment = np.vstack([np.sin(t), np.roll(np.sin(t), 3)]).astype(np.float32)

    adjacency, delay = compute_lagged_corr_graph(segment, 64.0, max_delay_ms=80.0)

    assert adjacency.shape == (2, 2)
    assert delay.shape == (2, 2)
    assert np.all(adjacency >= 0.0)
    assert np.allclose(np.diag(adjacency), 0.0)


def test_full_topology_features_and_sinkhorn_requirements():
    adjacency = np.asarray(
        [
            [[0.0, 0.2, 0.1], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]],
            [[0.0, 0.1, 0.4], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.2], [0.3, 0.0, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    centers = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    cost = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)

    features = compute_topology_features_full(adjacency, centers, structural_cost_matrix=cost, enable_sinkhorn=True)

    assert TOPOLOGY_SIMPLE_FEATURE_NAMES == TOPOLOGY_FEATURE_NAMES
    assert features.shape == (3, len(TOPOLOGY_FULL_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert sinkhorn_distance(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]), np.ones((2, 2), dtype=np.float32)) >= 0.0
    with pytest.raises(RuntimeError, match="structural_cost_matrix"):
        compute_topology_features_full(adjacency, centers, enable_sinkhorn=True)


def test_final_cache_defaults_include_full_modes_and_diagnostics():
    payload = build_cache_payload(
        [_patient()],
        source_patient_records_pkl="synthetic.pkl",
        window_length_sec=0.5,
        window_step_sec=0.5,
        pre_onset_sec=1.0,
        post_onset_sec=1.0,
        tfccm_embedding_dims=(2,),
        tfccm_taus=(1,),
        tfccm_library_fractions=(0.5, 1.0),
        tfccm_n_surrogates=0,
        tfccm_max_points=64,
    )

    report = inspect_cache_payload(payload)
    sample = payload["run_records"][0]["sample"]
    meta = payload["cache_meta"]

    assert meta["cache_name"] == "all_window_cache_physics_final.pkl"
    assert meta["cache_version"] == "physics_final_v1"
    assert report["physics_mode"] == "strict"
    assert report["causal_graph_mode"] == "tfccm_full"
    assert report["topology_mode"] == "full"
    assert sample["tfccm_pvalue"].shape == sample["tfccm_adjacency"].shape
    assert sample["tfccm_convergence"].shape == sample["tfccm_adjacency"].shape
    assert sample["topology_graph_features"].shape[-1] == len(TOPOLOGY_FULL_FEATURE_NAMES)
    assert "causal_graph_mode=tfccm_full but tfccm_pvalue missing" not in report["warnings"]
    assert "causal_graph_mode=tfccm_full but tfccm_convergence missing" not in report["warnings"]


def test_debug_modes_preserve_proxy_lite_simple_defaults():
    payload = build_cache_payload(
        [_patient()],
        source_patient_records_pkl="synthetic.pkl",
        window_length_sec=0.5,
        window_step_sec=0.5,
        pre_onset_sec=1.0,
        post_onset_sec=1.0,
        physics_mode="proxy",
        causal_graph_mode="tfccm_lite",
        topology_mode="simple",
    )

    sample = payload["run_records"][0]["sample"]
    meta = payload["cache_meta"]

    assert meta["physics_mode"] == "proxy"
    assert meta["causal_graph_mode"] == "tfccm_lite"
    assert meta["topology_mode"] == "simple"
    assert meta["feature_names_physics"] == PHYSICS_PROXY_FEATURE_NAMES
    assert meta["feature_names_topology"] == TOPOLOGY_SIMPLE_FEATURE_NAMES
    assert "tfccm_pvalue" not in sample
    assert sample["topology_graph_features"].shape[-1] == len(TOPOLOGY_SIMPLE_FEATURE_NAMES)


def test_parser_helpers_reject_empty_values():
    assert _parse_int_tuple("2, 3,4") == (2, 3, 4)
    assert _parse_float_tuple("0.3, 1.0") == (0.3, 1.0)
    with pytest.raises(ValueError):
        _parse_int_tuple("")
    with pytest.raises(ValueError):
        _parse_float_tuple("")


def test_training_model_kwargs_follow_cache_topology_dim():
    cache_meta = {"feature_names_topology": list(TOPOLOGY_FULL_FEATURE_NAMES)}

    task1_kwargs = task1_model_kwargs("T1_FULL_PGC", 16, cache_meta=cache_meta)
    task2_kwargs = task2_model_kwargs("T2_FULL_ATTENTION_TOPOLOGY", 16, cache_meta=cache_meta)

    assert task1_kwargs["topology_dim"] == len(TOPOLOGY_FULL_FEATURE_NAMES)
    assert task2_kwargs["topology_dim"] == len(TOPOLOGY_FULL_FEATURE_NAMES)
