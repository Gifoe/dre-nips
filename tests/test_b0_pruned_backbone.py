from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import exp_ez_hybrid
from exp_ez_hybrid import Exp_EZHybridLocalization
from neuroez_c.dataset import build_patient_examples, collate_patient_ez_batch, fit_window_tensor_normalizer
from neuroez_c.evidence_views import b0_self_reference_features
from neuroez_c.model import NeuroEZCModel


def _args(**overrides):
    values = {
        "b0_feature_parts": "abs,delta,zdelta,ratio",
        "b0_feature_groups": "spectral_classical",
        "self_compare_eps": 1e-5,
        "model_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "use_patient_relative_z": True,
        "positive_label": "nez",
        "class_weight_mode": "none",
        "device": "cpu",
        "output_dir": ".",
        "num_workers": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _features(offset: float = 0.0) -> np.ndarray:
    return np.arange(4 * 3 * 20, dtype=np.float32).reshape(4, 3, 20) + 1.0 + offset


def _sample(subject_id: str = "p1", offset: float = 0.0) -> dict:
    return {
        "subject_id": subject_id,
        "run_id": f"{subject_id}_r1",
        "sample_id": f"{subject_id}_s1",
        "channel_names_norm": ["a", "b", "c"],
        "labels": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "window_features": _features(offset),
        "window_adjacency": np.ones((4, 3, 3), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
    }


def _patient_index(*subject_ids: str) -> dict:
    return {
        subject_id: {
            "canonical_channels": ["a", "b", "c"],
            "labels": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "label_mask": np.asarray([True, True, True]),
        }
        for subject_id in subject_ids
    }


class B0PrunedBackboneTests(unittest.TestCase):
    def test_self_reference_defaults_to_pruned_spectral_classical_features(self):
        centers = np.asarray([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)

        out = b0_self_reference_features(_features(), centers, _args())

        self.assertEqual(out.shape, (4, 3, 36))
        graph_only_names = {"degree_norm", "strength_norm", "pagerank"}
        self.assertTrue(graph_only_names.isdisjoint(set(getattr(b0_self_reference_features, "selected_feature_names"))))


    def test_old_window_cache_schema_builds_pruned_batch_without_adjacency_view(self):
        sample = _sample("p1")
        args = _args()
        normalizer = fit_window_tensor_normalizer([sample], args=args)

        examples = build_patient_examples([sample], _patient_index("p1"), normalizer=normalizer, args=args)
        batch = collate_patient_ez_batch(examples)

        self.assertEqual(batch["b0_features"].shape[-1], 36)
        self.assertNotIn("adjacency", batch)
        self.assertNotIn("inter_features", batch)
        self.assertNotIn("phys_features", batch)
        self.assertNotIn("raw_waveform", batch)


    def test_model_forward_omits_rank_count_threshold_outputs_and_ignores_adjacency(self):
        batch = {
            "b0_features": torch.randn(2, 1, 4, 3, 36),
            "labels": torch.ones(2, 3),
            "labels_nez": torch.ones(2, 3),
            "labels_ez": torch.zeros(2, 3),
            "channel_mask": torch.ones(2, 3, dtype=torch.bool),
            "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
            "seizure_channel_mask": torch.ones(2, 1, 3, dtype=torch.bool),
            "window_mask": torch.ones(2, 1, 4, dtype=torch.bool),
            "adjacency": torch.randn(2, 1, 4, 3, 3),
        }

        out = NeuroEZCModel(_args())(batch)

        self.assertEqual(out["logits"].shape, (2, 3))
        self.assertEqual(out["scores"].shape, (2, 3))
        for removed in ("predicted_count", "score_mass", "graph_sparsity_loss"):
            self.assertNotIn(removed, out)


    def test_loss_is_bce_only(self):
        exp = object.__new__(Exp_EZHybridLocalization)
        exp.args = _args()
        outputs = {"logits": torch.tensor([[0.0, 1.0, -1.0]])}
        batch = {
            "labels": torch.tensor([[1.0, 0.0, 1.0]]),
            "labels_ez": torch.tensor([[0.0, 1.0, 0.0]]),
            "channel_mask": torch.tensor([[True, True, True]]),
        }

        loss, parts = exp._compute_loss(outputs, batch, torch.tensor(1.0))

        expected = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch["labels"])
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(parts, {"bce": float(expected.detach().cpu())})


    def test_synthetic_two_fold_smoke_run_writes_summary(self):
        samples = [_sample(f"p{i}", offset=float(i)) for i in range(4)]
        patient_index = _patient_index(*(f"p{i}" for i in range(4)))
        run_records = [
            {
                "subject_id": sample["subject_id"],
                "run_id": sample["run_id"],
                "channel_names_norm": sample["channel_names_norm"],
                "labels": sample["labels"],
                "sample": {
                    "sample_id": sample["sample_id"],
                    "window_features": sample["window_features"],
                    "window_adjacency": sample["window_adjacency"],
                    "window_relative_centers_sec": sample["window_relative_centers_sec"],
                },
            }
            for sample in samples
        ]
        splits = [
            {"fold_idx": 1, "train_subjects": ["p0", "p1"], "test_subjects": ["p2", "p3"]},
            {"fold_idx": 2, "train_subjects": ["p2", "p3"], "test_subjects": ["p0", "p1"]},
        ]
        original_data_provider = exp_ez_hybrid.data_provider
        try:
            exp_ez_hybrid.data_provider = lambda args: (run_records, patient_index, splits)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                args = _args(output_dir=str(output_dir), epochs=1, patience=0, batch_size=2, patient_batch_size=2)
                records = Exp_EZHybridLocalization(args).run()

                self.assertEqual(len(records), 4)
                self.assertTrue((output_dir / "heldout_summary_neuroez_v3.json").exists())
                self.assertTrue((output_dir / "test_channel_predictions_neuroez_v2_fold_1.csv").exists())
        finally:
            exp_ez_hybrid.data_provider = original_data_provider


if __name__ == "__main__":
    unittest.main()
