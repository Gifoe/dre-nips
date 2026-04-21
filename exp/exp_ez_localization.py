from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider, split_train_val_subjects
from exp.exp_basic import Exp_Basic
from report_threshold import build_patient_prediction, select_best_threshold


class Exp_EZLocalization(Exp_Basic):
    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.patient_bags, self.outer_splits = data_provider(args)
        self.patient_map = {bag["subject_id"]: bag for bag in self.patient_bags}

    def _build_model(self):
        model_name = str(getattr(self.args, "model", "TeChEZ"))
        return self.model_dict[model_name](self.args).to(self.device)

    def _fit_feature_scalers(self, patient_bags: Iterable[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        feat_sum = None
        feat_sq_sum = None
        conn_sum = None
        conn_sq_sum = None
        feat_count = 0
        conn_count = 0

        for bag in patient_bags:
            for run in bag["runs"]:
                valid_mask = run["channel_mask"].astype(bool)
                if not np.any(valid_mask):
                    continue
                feat = run["x_feat"][:, valid_mask, :].reshape(-1, run["x_feat"].shape[-1])
                conn = run["node_conn"][:, valid_mask, :].reshape(-1, run["node_conn"].shape[-1])

                current_feat_sum = feat.sum(axis=0)
                current_feat_sq_sum = np.square(feat).sum(axis=0)
                current_conn_sum = conn.sum(axis=0)
                current_conn_sq_sum = np.square(conn).sum(axis=0)

                feat_sum = current_feat_sum if feat_sum is None else feat_sum + current_feat_sum
                feat_sq_sum = current_feat_sq_sum if feat_sq_sum is None else feat_sq_sum + current_feat_sq_sum
                conn_sum = current_conn_sum if conn_sum is None else conn_sum + current_conn_sum
                conn_sq_sum = current_conn_sq_sum if conn_sq_sum is None else conn_sq_sum + current_conn_sq_sum
                feat_count += feat.shape[0]
                conn_count += conn.shape[0]

        if feat_count == 0 or conn_count == 0:
            raise ValueError("Unable to fit TeChEZ feature scalers because the training set is empty.")

        feat_mean = feat_sum / float(feat_count)
        feat_var = feat_sq_sum / float(feat_count) - np.square(feat_mean)
        conn_mean = conn_sum / float(conn_count)
        conn_var = conn_sq_sum / float(conn_count) - np.square(conn_mean)

        return {
            "feat_mean": feat_mean.astype(np.float32),
            "feat_std": np.sqrt(np.clip(feat_var, 1e-8, None)).astype(np.float32),
            "conn_mean": conn_mean.astype(np.float32),
            "conn_std": np.sqrt(np.clip(conn_var, 1e-8, None)).astype(np.float32),
        }

    def _normalize_bag(self, bag: Dict[str, Any], scalers: Dict[str, np.ndarray]) -> Dict[str, Any]:
        bag_copy = copy.deepcopy(bag)
        for run in bag_copy["runs"]:
            run["x_feat"] = ((run["x_feat"] - scalers["feat_mean"]) / scalers["feat_std"]).astype(np.float32)
            run["node_conn"] = ((run["node_conn"] - scalers["conn_mean"]) / scalers["conn_std"]).astype(np.float32)
        return bag_copy

    def _to_device_bag(self, bag: Dict[str, Any]) -> Dict[str, Any]:
        device_bag = {
            "subject_id": bag["subject_id"],
            "canonical_channels": list(bag["canonical_channels"]),
            "channel_meta": list(bag["channel_meta"]),
            "labels": torch.as_tensor(bag["labels"], dtype=torch.float32, device=self.device),
            "label_mask": torch.as_tensor(bag["label_mask"], dtype=torch.bool, device=self.device),
            "runs": [],
        }

        for run in bag["runs"]:
            device_bag["runs"].append(
                {
                    "run_id": run["run_id"],
                    "task": run["task"],
                    "phase_group": run["phase_group"],
                    "phase_ids": torch.as_tensor(run["phase_ids"], dtype=torch.long, device=self.device),
                    "quality_weight": float(run["quality_weight"]),
                    "n_windows": int(run["n_windows"]),
                    "x_feat": torch.as_tensor(run["x_feat"], dtype=torch.float32, device=self.device),
                    "node_conn": torch.as_tensor(run["node_conn"], dtype=torch.float32, device=self.device),
                    "edge_index": torch.as_tensor(run["edge_index"], dtype=torch.long, device=self.device),
                    "edge_attr": torch.as_tensor(run["edge_attr"], dtype=torch.float32, device=self.device),
                    "channel_mask": torch.as_tensor(run["channel_mask"], dtype=torch.bool, device=self.device),
                }
            )
        return device_bag

    def _prepare_split_bags(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_bags = [self.patient_map[sid] for sid in train_subjects]
        val_bags = [self.patient_map[sid] for sid in val_subjects]
        test_bags = [self.patient_map[sid] for sid in test_subjects]

        scalers = self._fit_feature_scalers(train_bags)
        train_device = [self._to_device_bag(self._normalize_bag(bag, scalers)) for bag in train_bags]
        val_device = [self._to_device_bag(self._normalize_bag(bag, scalers)) for bag in val_bags]
        test_device = [self._to_device_bag(self._normalize_bag(bag, scalers)) for bag in test_bags]
        return train_device, val_device, test_device

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alpha = float(getattr(self.args, "focal_alpha", 0.75))
        gamma = float(getattr(self.args, "focal_gamma", 2.0))
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(labels == 1.0, alpha, 1.0 - alpha)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (alpha_t * torch.pow(1.0 - pt, gamma) * bce_loss).mean()

    def _rank_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos = logits[labels == 1.0]
        neg = logits[labels == 0.0]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_tensor(0.0)
        margin = float(getattr(self.args, "rank_margin", 0.4))
        pairwise = margin - pos.unsqueeze(1) + neg.unsqueeze(0)
        return torch.relu(pairwise).mean()

    def _train_one_patient(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        patient_bag: Dict[str, Any],
    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(patient_bag)
        logits = output["patient_logits"][patient_bag["label_mask"]]
        labels = patient_bag["labels"][patient_bag["label_mask"]]

        focal_loss = self._focal_loss(logits, labels)
        rank_loss = self._rank_loss(logits, labels)
        total_loss = focal_loss + float(getattr(self.args, "rank_lambda", 0.2)) * rank_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "focal_loss": float(focal_loss.detach().cpu().item()),
            "rank_loss": float(rank_loss.detach().cpu().item()),
        }

    def _evaluate_patients(
        self,
        model: torch.nn.Module,
        patient_bags: Iterable[Dict[str, Any]],
        fold_idx: int,
    ) -> List[Dict[str, Any]]:
        model.eval()
        outputs: List[Dict[str, Any]] = []
        with torch.no_grad():
            for patient_bag in patient_bags:
                output = model(patient_bag)
                run_summaries = []
                for run_output, run_scores in zip(output["run_outputs"], output["run_channel_scores"]):
                    run_summaries.append(
                        {
                            "run_id": run_output["run_id"],
                            "quality_weight": float(run_output["quality_weight"]),
                            "n_selected_windows": int(
                                next(run["n_windows"] for run in patient_bag["runs"] if run["run_id"] == run_output["run_id"])
                            ),
                            "channel_scores": run_scores.detach().cpu().numpy().tolist(),
                        }
                    )

                outputs.append(
                    {
                        "subject_id": patient_bag["subject_id"],
                        "fold_idx": int(fold_idx),
                        "canonical_channels": list(patient_bag["canonical_channels"]),
                        "labels": patient_bag["labels"].detach().cpu().numpy().astype(np.float32),
                        "scores": output["patient_scores"].detach().cpu().numpy().astype(np.float32),
                        "run_summaries": run_summaries,
                    }
                )
        return outputs

    def run(self) -> List[Dict[str, Any]]:
        all_predictions: List[Dict[str, Any]] = []
        epochs = int(getattr(self.args, "epochs", 40))
        patience = int(getattr(self.args, "patience", 8))
        learning_rate = float(getattr(self.args, "learning_rate", 1e-3))
        weight_decay = float(getattr(self.args, "weight_decay", 1e-4))
        val_ratio = float(getattr(self.args, "val_ratio", 0.2))

        for split in self.outer_splits:
            fold_idx = int(split["fold_idx"])
            train_subjects, val_subjects = split_train_val_subjects(
                split["train_subjects"],
                val_ratio=val_ratio,
                random_seed=int(getattr(self.args, "random_seed", 42)),
                fold_idx=fold_idx,
            )
            test_subjects = list(split["test_subjects"])

            train_bags, val_bags, test_bags = self._prepare_split_bags(
                train_subjects,
                val_subjects,
                test_subjects,
            )

            model = self._build_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_state = copy.deepcopy(model.state_dict())
            best_tau = 0.5
            best_score = -1e9
            epochs_without_improvement = 0

            for _ in range(epochs):
                order = np.random.default_rng(seed=int(getattr(self.args, "random_seed", 42)) + fold_idx).permutation(len(train_bags))
                for patient_idx in order:
                    self._train_one_patient(model, optimizer, train_bags[int(patient_idx)])

                val_outputs = self._evaluate_patients(model, val_bags, fold_idx=fold_idx)
                tau, summary = select_best_threshold(val_outputs)
                val_score = summary["macro_f1"] + 0.2 * summary["macro_auc_pr"]

                if val_score > best_score + 1e-5:
                    best_score = val_score
                    best_tau = float(tau)
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            model.load_state_dict(best_state)
            test_outputs = self._evaluate_patients(model, test_bags, fold_idx=fold_idx)
            for patient_output in test_outputs:
                all_predictions.append(build_patient_prediction(patient_output, best_tau))

        return all_predictions


__all__ = ["Exp_EZLocalization"]
