from __future__ import annotations

import numpy as np
import torch

from neuroez_multitask.dataset import PhysicsCacheDataset, collate_patient_batch
from neuroez_multitask.evidence_views import b0_self_reference_features, compute_tfccm_graph
from neuroez_multitask.model import PGCSEEGModel
from neuroez_multitask.normalization import fit_multiview_normalizer
from neuroez_multitask.train_task2 import estimate_task2_pos_weight
from run_task1_pgc_ez import _checkpoint_state_dict, _model_kwargs


def _minimal_sample(channel_names: list[str], value_by_channel: dict[str, float]) -> dict:
    values = np.asarray([[value_by_channel[name] for name in channel_names]], dtype=np.float32)
    return {
        "window_features": values[..., None],
        "physics_node_features": values[..., None],
        "tfccm_adjacency": np.zeros((1, len(channel_names), len(channel_names)), dtype=np.float32),
        "tfccm_delay": np.zeros((1, len(channel_names), len(channel_names)), dtype=np.float32),
        "causal_node_features": np.zeros((1, len(channel_names), 1), dtype=np.float32),
        "topology_graph_features": np.zeros((1, 8), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([-1.0], dtype=np.float32),
        "window_mask": np.asarray([True]),
        "channel_names": channel_names,
    }


def _cache_for_channel_alignment() -> dict:
    return {
        "patient_index": {
            "p1": {
                "canonical_channels": ["A", "B", "C"],
                "labels": np.asarray([0.0, -1.0, 1.0], dtype=np.float32),
                "labels_nez": np.asarray([0.0, -1.0, 1.0], dtype=np.float32),
                "labels_ez": np.asarray([1.0, -1.0, 0.0], dtype=np.float32),
                "label_mask": np.asarray([True, False, True]),
                "center": "lzu",
            }
        },
        "outcome_index": {"p1": {"success_failure": 1, "Engel": "I"}},
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names": ["C", "A"],
                "sample": _minimal_sample(["C", "A"], {"A": 10.0, "C": 30.0}),
            }
        ],
    }


def test_collate_maps_local_run_channels_to_patient_canonical_order():
    ds = PhysicsCacheDataset(_cache_for_channel_alignment())
    batch = collate_patient_batch([ds[0]])

    assert batch["b0_features"][0, 0, 0, :, 0].tolist() == [10.0, 0.0, 30.0]
    assert batch["seizure_channel_mask"][0, 0].tolist() == [True, False, True]


def test_self_reference_expands_abs_delta_zdelta_ratio_from_baseline_windows():
    raw = np.asarray(
        [
            [[2.0], [4.0]],
            [[4.0], [8.0]],
            [[8.0], [16.0]],
        ],
        dtype=np.float32,
    )
    centers = np.asarray([-2.0, -1.0, 1.0], dtype=np.float32)

    out = b0_self_reference_features(raw, centers)

    assert out.shape == (3, 2, 4)
    np.testing.assert_allclose(out[2, 0], [8.0, 5.0, 5.0, np.log(8.0 / 3.0)], rtol=1e-5)


def test_model_kwargs_keep_node_only_and_graph_ablation_distinct():
    node = _model_kwargs("T1_B0_TFCCM_NODE", 16)
    graph = _model_kwargs("T1_B0_TFCCM_GRAPH_NO_DELAY", 16)

    assert node["use_causal_graph"] is False
    assert node["use_causal_node_features"] is True
    assert graph["use_causal_graph"] is True
    assert graph["use_delay"] is False


def test_task2_experiment_kwargs_select_real_readout_variants():
    global_kwargs = _model_kwargs("T2_FULL_GLOBAL", 16)
    attn_kwargs = _model_kwargs("T2_FULL_ATTENTION", 16)
    topo_kwargs = _model_kwargs("T2_FULL_ATTENTION_TOPOLOGY", 16)

    assert global_kwargs["outcome_readout_type"] == "global"
    assert global_kwargs["use_topology_features"] is False
    assert attn_kwargs["outcome_readout_type"] == "attention"
    assert attn_kwargs["use_topology_features"] is False
    assert topo_kwargs["outcome_readout_type"] == "attention"
    assert topo_kwargs["use_topology_features"] is True


def test_train_only_multiview_normalizer_does_not_fit_test_patients():
    train_cache = _cache_for_channel_alignment()
    test_cache = _cache_for_channel_alignment()
    train_cache["patient_index"] = {"train": {**train_cache["patient_index"]["p1"], "canonical_channels": ["A"]}}
    train_cache["outcome_index"] = {"train": {"success_failure": 1}}
    train_cache["run_records"] = [
        {"subject_id": "train", "channel_names": ["A"], "sample": _minimal_sample(["A"], {"A": 1.0})}
    ]
    test_cache["patient_index"] = {"test": {**test_cache["patient_index"]["p1"], "canonical_channels": ["A"]}}
    test_cache["outcome_index"] = {"test": {"success_failure": 0}}
    test_cache["run_records"] = [
        {"subject_id": "test", "channel_names": ["A"], "sample": _minimal_sample(["A"], {"A": 100.0})}
    ]
    train_ds = PhysicsCacheDataset(train_cache)
    test_ds = PhysicsCacheDataset(test_cache)

    normalizer = fit_multiview_normalizer(train_ds)

    assert float(normalizer.b0.mean[0]) == 1.0
    assert float(normalizer.transform_sample(test_ds[0]["runs"][0]["sample"])["window_features"][0, 0, 0]) == 99.0


def test_task2_pos_weight_is_failure_over_success_for_train_subjects_only():
    cache = {
        "outcome_index": {
            "s1": {"success_failure": 1},
            "s2": {"success_failure": 0},
            "s3": {"success_failure": 0},
            "heldout": {"success_failure": 0},
        }
    }

    pos_weight = estimate_task2_pos_weight(cache, ["s1", "s2", "s3"])

    assert torch.isclose(pos_weight, torch.tensor(2.0))


def _old_lagged_corr(segment: np.ndarray, sfreq: float, max_delay_ms: float = 80.0) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    channels, samples = data.shape
    max_lag = max(1, int(round(max_delay_ms * sfreq / 1000.0)))
    max_lag = min(max_lag, samples // 4)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1) + 1e-6
    adj = np.zeros((channels, channels), dtype=np.float32)
    for src in range(channels):
        for dst in range(channels):
            if src == dst:
                continue
            best = 0.0
            for lag in range(1, max_lag + 1):
                score = float(np.mean(centered[src, :-lag] * centered[dst, lag:]) / (std[src] * std[dst]))
                if abs(score) > abs(best):
                    best = score
            adj[src, dst] = max(best, 0.0)
    return adj


def test_tfccm_graph_is_not_the_old_lagged_correlation_proxy():
    rng = np.random.default_rng(123)
    source = rng.normal(size=240).astype(np.float32)
    target = np.roll(np.tanh(source), 4) + 0.02 * rng.normal(size=240).astype(np.float32)
    segment = np.stack([source, target], axis=0)

    adj, delay = compute_tfccm_graph(segment, sfreq=100.0, max_delay_ms=80.0)
    old = _old_lagged_corr(segment, sfreq=100.0, max_delay_ms=80.0)

    assert not np.allclose(adj, old)
    assert adj[0, 1] > adj[1, 0]
    assert delay[0, 1] > 0.0


def test_global_outcome_readout_is_supported_without_attention_or_topology():
    model = PGCSEEGModel(
        model_dim=8,
        topology_dim=8,
        use_physics_branch=False,
        use_causal_graph=False,
        outcome_readout_type="global",
        use_topology_features=False,
    )
    batch = {
        "b0_features": torch.randn(2, 1, 2, 3, 9),
        "physics_features": torch.randn(2, 1, 2, 3, 6),
        "causal_adjacency": torch.zeros(2, 1, 2, 3, 3),
        "causal_delay": torch.zeros(2, 1, 2, 3, 3),
        "causal_node_features": torch.zeros(2, 1, 2, 3, 7),
        "topology_features": torch.randn(2, 8),
        "labels_nez": torch.zeros(2, 3),
        "labels_ez": torch.ones(2, 3),
        "channel_mask": torch.ones(2, 3, dtype=torch.bool),
        "outcome_label": torch.tensor([1.0, 0.0]),
        "outcome_mask": torch.ones(2, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 1, 3, dtype=torch.bool),
        "window_mask": torch.ones(2, 1, 2, dtype=torch.bool),
    }

    outputs = model(batch)

    assert outputs["outcome_logit"].shape == (2,)
    assert torch.all(outputs["outcome_attention"] == 0)


def test_checkpoint_state_dict_skips_lazy_branches_unused_by_b0_baseline():
    model = PGCSEEGModel(**_model_kwargs("T1_B0_BASELINE", 8))
    batch = {
        "b0_features": torch.randn(1, 1, 2, 3, 36),
        "physics_features": torch.randn(1, 1, 2, 3, 24),
        "causal_adjacency": torch.zeros(1, 1, 2, 3, 3),
        "causal_delay": torch.zeros(1, 1, 2, 3, 3),
        "causal_node_features": torch.zeros(1, 1, 2, 3, 7),
        "topology_features": torch.randn(1, 8),
        "labels_nez": torch.zeros(1, 3),
        "labels_ez": torch.ones(1, 3),
        "channel_mask": torch.ones(1, 3, dtype=torch.bool),
        "outcome_label": torch.tensor([1.0]),
        "outcome_mask": torch.ones(1, dtype=torch.bool),
        "seizure_mask": torch.ones(1, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "window_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }
    model(batch)

    state = _checkpoint_state_dict(model)

    assert state
    assert "physics_encoder.net.0.weight" not in state
    assert "causal_graph_encoder.delay_encoder.0.weight" not in state
    assert all(hasattr(value, "shape") for value in state.values())
    PGCSEEGModel(**_model_kwargs("T1_B0_BASELINE", 8)).load_state_dict(state, strict=False)
