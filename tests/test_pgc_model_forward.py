from __future__ import annotations

import torch

from neuroez_multitask.model import PGCSEEGModel


def _batch(labels_value: float = 0.0):
    return {
        "b0_features": torch.randn(2, 2, 3, 4, 9),
        "physics_features": torch.randn(2, 2, 3, 4, 6),
        "causal_adjacency": torch.rand(2, 2, 3, 4, 4),
        "causal_delay": torch.rand(2, 2, 3, 4, 4),
        "causal_node_features": torch.randn(2, 2, 3, 4, 7),
        "topology_features": torch.randn(2, 8),
        "labels_nez": torch.full((2, 4), labels_value),
        "labels_ez": torch.full((2, 4), 1.0 - labels_value),
        "channel_mask": torch.ones(2, 4, dtype=torch.bool),
        "outcome_label": torch.tensor([1.0, 0.0]),
        "outcome_mask": torch.ones(2, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 2, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 2, 4, dtype=torch.bool),
        "window_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "subject_id": ["p1", "p2"],
        "center": ["lzu", "hup"],
    }


def test_pgc_forward_returns_nez_internal_and_ez_report_probabilities():
    model = PGCSEEGModel(model_dim=8, topology_dim=8)
    outputs = model(_batch())

    assert outputs["nez_logits"].shape == (2, 4)
    assert outputs["nez_prob"].shape == (2, 4)
    assert outputs["ez_prob"].shape == (2, 4)
    assert torch.allclose(outputs["ez_prob"], 1.0 - outputs["nez_prob"], atol=1e-6)
    assert outputs["outcome_logit"].shape == (2,)
    assert outputs["outcome_prob"].shape == (2,)
    assert outputs["patient_channel_embedding"].shape[:2] == (2, 4)
    assert outputs["outcome_attention"].shape == (2, 4)


def test_task2_forward_does_not_depend_on_ground_truth_ez_labels():
    torch.manual_seed(123)
    model = PGCSEEGModel(model_dim=8, topology_dim=8)
    model.eval()
    batch_a = _batch(labels_value=0.0)
    batch_b = dict(batch_a)
    batch_b["labels_nez"] = torch.ones_like(batch_a["labels_nez"])
    batch_b["labels_ez"] = torch.zeros_like(batch_a["labels_ez"])

    with torch.no_grad():
        out_a = model(batch_a)["outcome_prob"]
        out_b = model(batch_b)["outcome_prob"]

    assert torch.allclose(out_a, out_b, atol=1e-6)
