from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data_factory import data_provider, split_train_val_subjects
from ez_dataset import flatten_window_samples


def _resolve_runtime(args: Any) -> Dict[str, Any]:
    from neuroez_c.dataset import (
        PatientNeuroEZCDataset,
        build_patient_examples as build_neuroez_c_examples,
        collate_patient_ez_batch as collate_neuroez_c_batch,
        fit_window_tensor_normalizer as fit_neuroez_c_normalizer,
    )
    from neuroez_c.model import NeuroEZCModel

    return {
        "name": "B0-Pruned-EZBackbone",
        "dataset_cls": PatientNeuroEZCDataset,
        "build_patient_examples": build_neuroez_c_examples,
        "collate_fn": collate_neuroez_c_batch,
        "fit_normalizer": fit_neuroez_c_normalizer,
        "model_cls": NeuroEZCModel,
    }


def _set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _acquire_device(args: Any) -> torch.device:
    preferred = str(getattr(args, "device", "auto")).lower()
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def _move_tensors_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_ez: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight_mode: str,
    ez_negative_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask & (labels >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
    if str(class_weight_mode).lower() == "ez_negative":
        weights = torch.ones_like(losses)
        weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()


def _estimate_ez_negative_weight(dataset: Any, cap: float = 20.0) -> float:
    ez = 0.0
    nez = 0.0
    for item in dataset.patient_examples:
        labels_ez = np.asarray(item["labels_ez"], dtype=np.float32)
        mask = np.asarray(item["channel_mask"], dtype=bool)
        ez += float(((labels_ez == 1.0) & mask).sum())
        nez += float(((labels_ez == 0.0) & mask).sum())
    if ez <= 0.0:
        return 1.0
    return float(np.clip(nez / max(ez, 1.0), 1.0, float(cap)))


def _select_topk(scores: np.ndarray, k: int, valid_mask: np.ndarray, *, descending: bool = True) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return pred
    k = max(1, min(int(k), int(valid_idx.size)))
    order = valid_idx[np.argsort(scores[valid_idx])]
    if descending:
        order = order[::-1]
    pred[order[:k]] = True
    return pred


def _reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive_ranks = np.where(y_true[order] == 1)[0]
    return float(1.0 / float(positive_ranks[0] + 1)) if positive_ranks.size else 0.0


def _recall_at_true_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    true_count = int((y_true == 1).sum())
    if y_true.size == 0 or true_count <= 0:
        return 0.0
    pred = _select_topk(scores, true_count, np.ones_like(y_true, dtype=bool), descending=True)
    return float(((y_true == 1) & pred).sum() / max(true_count, 1))


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_prediction_records(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    patient_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "nez_precision": [],
        "nez_recall": [],
        "nez_f1": [],
        "ez_precision": [],
        "ez_recall": [],
        "ez_f1": [],
        "auroc_nez": [],
        "auprc_nez": [],
        "auroc_ez": [],
        "auprc_ez": [],
        "ez_recall_at_true_count": [],
        "ez_mrr": [],
    }
    pooled_y_nez: List[np.ndarray] = []
    pooled_pred_nez: List[np.ndarray] = []
    pooled_score_nez: List[np.ndarray] = []
    pooled_score_ez: List[np.ndarray] = []
    enriched: List[Dict[str, Any]] = []

    for record in records:
        labels_nez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float32)
        labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
        score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
        score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
        valid_mask = np.asarray(record["channel_mask"], dtype=bool)
        if not np.any(valid_mask):
            continue

        y_nez = labels_nez[valid_mask].astype(int)
        y_ez = labels_ez[valid_mask].astype(int)
        score_nez_valid = score_nez[valid_mask]
        score_ez_valid = score_ez[valid_mask]
        pred_ez_mask = _select_topk(score_ez, int(y_ez.sum()), valid_mask, descending=True)
        pred_nez_mask = (~pred_ez_mask) & valid_mask
        pred_nez = pred_nez_mask[valid_mask].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        patient_accuracy = float(accuracy_score(y_nez, pred_nez))
        patient_balanced_accuracy = float(balanced_accuracy_score(y_nez, pred_nez))
        patient_macro_f1 = float(f1_score(y_nez, pred_nez, average="macro", zero_division=0))
        patient_weighted_f1 = float(f1_score(y_nez, pred_nez, average="weighted", zero_division=0))
        patient_metrics["accuracy"].append(patient_accuracy)
        patient_metrics["balanced_accuracy"].append(patient_balanced_accuracy)
        patient_metrics["macro_f1"].append(patient_macro_f1)
        patient_metrics["weighted_f1"].append(patient_weighted_f1)
        patient_metrics["nez_precision"].append(float(precision[0]))
        patient_metrics["nez_recall"].append(float(recall[0]))
        patient_metrics["nez_f1"].append(float(f1[0]))
        patient_metrics["ez_precision"].append(float(precision[1]))
        patient_metrics["ez_recall"].append(float(recall[1]))
        patient_metrics["ez_f1"].append(float(f1[1]))
        patient_metrics["ez_recall_at_true_count"].append(_recall_at_true_count(y_ez, score_ez_valid))
        patient_metrics["ez_mrr"].append(_reciprocal_rank(y_ez, score_ez_valid))

        if np.unique(y_nez).size > 1:
            patient_metrics["auroc_nez"].append(float(roc_auc_score(y_nez, score_nez_valid)))
            patient_metrics["auprc_nez"].append(float(average_precision_score(y_nez, score_nez_valid)))
            patient_metrics["auroc_ez"].append(float(roc_auc_score(y_ez, score_ez_valid)))
            patient_metrics["auprc_ez"].append(float(average_precision_score(y_ez, score_ez_valid)))

        pooled_y_nez.append(y_nez)
        pooled_pred_nez.append(pred_nez)
        pooled_score_nez.append(score_nez_valid)
        pooled_score_ez.append(score_ez_valid)

        channel_names = list(record["canonical_channels"])
        enriched_record = dict(record)
        enriched_record["predicted_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_ez_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_nez_mask"] = pred_nez_mask.astype(int).tolist()
        enriched_record["predicted_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_ez_mask) if flag]
        enriched_record["predicted_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_nez_mask) if flag]
        enriched_record["predicted_channels"] = enriched_record["predicted_ez_channels"]
        enriched_record["true_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_ez == 1.0) if flag]
        enriched_record["true_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_nez == 1.0) if flag]
        enriched_record["patient_accuracy"] = patient_accuracy
        enriched_record["patient_balanced_accuracy"] = patient_balanced_accuracy
        enriched_record["patient_macro_f1"] = patient_macro_f1
        enriched_record["patient_weighted_f1"] = patient_weighted_f1
        enriched_record["patient_nez_precision"] = float(precision[0])
        enriched_record["patient_nez_recall"] = float(recall[0])
        enriched_record["patient_nez_f1"] = float(f1[0])
        enriched_record["patient_ez_precision"] = float(precision[1])
        enriched_record["patient_ez_recall"] = float(recall[1])
        enriched_record["patient_ez_f1"] = float(f1[1])
        enriched_record["ez_recall_at_true_count"] = patient_metrics["ez_recall_at_true_count"][-1]
        enriched_record["ez_mrr"] = patient_metrics["ez_mrr"][-1]
        enriched_record["true_nez_count"] = float((labels_nez[valid_mask] == 1.0).sum())
        enriched_record["true_ez_count"] = float((labels_ez[valid_mask] == 1.0).sum())
        enriched_record["predicted_nez_count"] = float(pred_nez_mask.sum())
        enriched_record["predicted_ez_count"] = float(pred_ez_mask.sum())
        enriched.append(enriched_record)

    summary = {
        "patient_macro_accuracy": _safe_mean(patient_metrics["accuracy"]),
        "patient_macro_balanced_accuracy": _safe_mean(patient_metrics["balanced_accuracy"]),
        "patient_macro_f1": _safe_mean(patient_metrics["macro_f1"]),
        "patient_weighted_f1": _safe_mean(patient_metrics["weighted_f1"]),
        "patient_macro_nez_precision": _safe_mean(patient_metrics["nez_precision"]),
        "patient_macro_nez_recall": _safe_mean(patient_metrics["nez_recall"]),
        "patient_macro_nez_f1": _safe_mean(patient_metrics["nez_f1"]),
        "patient_macro_ez_precision": _safe_mean(patient_metrics["ez_precision"]),
        "patient_macro_ez_recall": _safe_mean(patient_metrics["ez_recall"]),
        "patient_macro_ez_f1": _safe_mean(patient_metrics["ez_f1"]),
        "patient_macro_auroc_nez": _safe_mean(patient_metrics["auroc_nez"]),
        "patient_macro_auprc_nez": _safe_mean(patient_metrics["auprc_nez"]),
        "patient_macro_auroc_ez": _safe_mean(patient_metrics["auroc_ez"]),
        "patient_macro_auprc_ez": _safe_mean(patient_metrics["auprc_ez"]),
        "patient_macro_ez_recall_at_true_count": _safe_mean(patient_metrics["ez_recall_at_true_count"]),
        "patient_macro_ez_mrr": _safe_mean(patient_metrics["ez_mrr"]),
    }
    summary["macro_topk_recall"] = summary["patient_macro_ez_recall_at_true_count"]
    summary["ez_recall_at_true_count"] = summary["patient_macro_ez_recall_at_true_count"]

    if pooled_y_nez:
        y_nez_all = np.concatenate(pooled_y_nez)
        pred_nez_all = np.concatenate(pooled_pred_nez)
        score_nez_all = np.concatenate(pooled_score_nez)
        score_ez_all = np.concatenate(pooled_score_ez)
        y_ez_all = 1 - y_nez_all
        summary.update(
            {
                "pooled_accuracy": float(accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_balanced_accuracy": float(balanced_accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_macro_f1": float(f1_score(y_nez_all, pred_nez_all, average="macro", zero_division=0)),
                "pooled_weighted_f1": float(f1_score(y_nez_all, pred_nez_all, average="weighted", zero_division=0)),
                "pooled_auroc_nez": float(roc_auc_score(y_nez_all, score_nez_all)) if np.unique(y_nez_all).size > 1 else 0.0,
                "pooled_auprc_nez": float(average_precision_score(y_nez_all, score_nez_all)) if np.unique(y_nez_all).size > 1 else 0.0,
                "pooled_auroc_ez": float(roc_auc_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
                "pooled_auprc_ez": float(average_precision_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
            }
        )
    else:
        summary.update(
            {
                "pooled_accuracy": 0.0,
                "pooled_balanced_accuracy": 0.0,
                "pooled_macro_f1": 0.0,
                "pooled_weighted_f1": 0.0,
                "pooled_auroc_nez": 0.0,
                "pooled_auprc_nez": 0.0,
                "pooled_auroc_ez": 0.0,
                "pooled_auprc_ez": 0.0,
            }
        )
    return summary, enriched


def _summary_score(summary: Dict[str, float], args: Any, *, val_loss: float = 0.0) -> float:
    metric = str(getattr(args, "early_stop_metric", "patient_macro_f1")).lower()
    if metric in {"loss", "val_loss"}:
        return -float(val_loss)
    aliases = {
        "macro_topk_recall": "ez_recall_at_true_count",
        "patient_topk_recall": "ez_recall_at_true_count",
        "patient_macro_f1": "patient_macro_f1",
        "patient_balanced_accuracy": "patient_macro_balanced_accuracy",
        "pooled_macro_f1": "pooled_macro_f1",
        "pooled_balanced_accuracy": "pooled_balanced_accuracy",
    }
    key = aliases.get(metric, metric)
    return float(summary.get(key, summary.get("patient_macro_f1", 0.0)))


def select_best_decision_rule(records: Sequence[Dict[str, Any]], args: Any | None = None):
    del args
    summary, enriched = _summarize_prediction_records(records)
    return {"strategy": "top_true_ez_count"}, summary, enriched


class Exp_EZHybridLocalization:
    """BCE-only patient-level B0-Pruned backbone for EZ localization."""

    @staticmethod
    def _log(message: str) -> None:
        print(f"[B0-Pruned][Train] {message}", flush=True)

    def __init__(self, args: Any) -> None:
        self.args = args
        self.runtime = _resolve_runtime(args)
        self.device = _acquire_device(args)
        self.run_records, self.patient_index, self.outer_splits = data_provider(args)
        self.current_epoch = 0
        self._log(
            f"Experiment ready with {len(self.run_records)} ictal records, "
            f"{len(self.patient_index)} patients, {len(self.outer_splits)} outer split(s), "
            f"runtime={self.runtime['name']}, device={self.device}."
        )

    def _make_loader(self, dataset: Any, *, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            collate_fn=self.runtime["collate_fn"],
            pin_memory=self.device.type == "cuda",
        )

    def _build_datasets(
        self,
        fit_subjects: Sequence[str],
        val_subjects: Sequence[str],
        test_subjects: Sequence[str],
    ) -> Tuple[Any, Any, Any, Any]:
        fit_samples = flatten_window_samples(self.run_records, subject_ids=fit_subjects)
        val_samples = flatten_window_samples(self.run_records, subject_ids=val_subjects)
        test_samples = flatten_window_samples(self.run_records, subject_ids=test_subjects)
        normalizer = self.runtime["fit_normalizer"](fit_samples, args=self.args)
        fit_examples = self.runtime["build_patient_examples"](fit_samples, self.patient_index, normalizer=normalizer, subject_ids=fit_subjects, args=self.args)
        val_examples = self.runtime["build_patient_examples"](val_samples, self.patient_index, normalizer=normalizer, subject_ids=val_subjects, args=self.args)
        test_examples = self.runtime["build_patient_examples"](test_samples, self.patient_index, normalizer=normalizer, subject_ids=test_subjects, args=self.args)
        if not val_examples:
            val_examples = fit_examples
        dataset_cls = self.runtime["dataset_cls"]
        return dataset_cls(fit_examples), dataset_cls(val_examples), dataset_cls(test_examples), normalizer

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], ez_negative_weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = _masked_bce_loss(
            outputs["logits"],
            batch["labels"],
            batch["labels_ez"],
            batch["channel_mask"],
            class_weight_mode=str(getattr(self.args, "class_weight_mode", "ez_negative")),
            ez_negative_weight=ez_negative_weight,
        )
        return bce, {"bce": float(bce.detach().cpu())}

    def _dry_initialize_lazy_layers(self, model: torch.nn.Module, loader: DataLoader) -> None:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                _ = model(_move_tensors_to_device(batch, self.device))
                return

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ez_negative_weight: torch.Tensor,
    ) -> Dict[str, float]:
        model.train()
        losses: List[float] = []
        bce_losses: List[float] = []
        for batch in loader:
            batch = _move_tensors_to_device(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, parts = self._compute_loss(outputs, batch, ez_negative_weight)
            loss.backward()
            grad_clip = float(getattr(self.args, "grad_clip", 1.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(parts["bce"])
        return {"loss": _safe_mean(losses), "bce": _safe_mean(bce_losses)}

    def _evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        ez_negative_weight: torch.Tensor,
    ) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        model.eval()
        losses: List[float] = []
        records: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                batch_device = _move_tensors_to_device(batch, self.device)
                outputs = model(batch_device)
                loss, _ = self._compute_loss(outputs, batch_device, ez_negative_weight)
                losses.append(float(loss.detach().cpu()))
                score_nez = outputs["score_nez"].detach().cpu().numpy()
                score_ez = outputs["score_ez"].detach().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                labels_nez = batch["labels_nez"].cpu().numpy()
                labels_ez = batch["labels_ez"].cpu().numpy()
                channel_mask = batch["channel_mask"].cpu().numpy().astype(bool)
                for idx, subject_id in enumerate(batch["subject_id"]):
                    c = len(batch["canonical_channels"][idx])
                    records.append(
                        {
                            "subject_id": str(subject_id),
                            "canonical_channels": list(batch["canonical_channels"][idx]),
                            "channel_meta": list(batch.get("channel_meta", [[]])[idx]),
                            "labels": labels[idx, :c].astype(np.float32),
                            "labels_nez": labels_nez[idx, :c].astype(np.float32),
                            "labels_ez": labels_ez[idx, :c].astype(np.float32),
                            "scores": score_nez[idx, :c].astype(np.float32),
                            "score_nez": score_nez[idx, :c].astype(np.float32),
                            "score_ez": score_ez[idx, :c].astype(np.float32),
                            "channel_mask": channel_mask[idx, :c],
                            "run_ids": list(batch.get("run_ids", [[]])[idx]),
                            "sample_ids": list(batch.get("sample_ids", [[]])[idx]),
                        }
                    )
        summary, enriched = _summarize_prediction_records(records)
        summary["selection_score"] = _summary_score(summary, self.args, val_loss=_safe_mean(losses))
        summary["strategy"] = "top_true_ez_count"
        return _safe_mean(losses), summary, enriched

    def _save_outputs(self, records: Sequence[Dict[str, Any]], *, fold_idx: int, split_name: str) -> None:
        output_dir = Path(getattr(self.args, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        patient_rows = []
        channel_rows = []
        for record in records:
            patient_rows.append(
                {
                    "fold_idx": int(fold_idx),
                    "split": split_name,
                    "subject_id": record["subject_id"],
                    "true_nez_count": float(record.get("true_nez_count", 0.0)),
                    "true_ez_count": float(record.get("true_ez_count", 0.0)),
                    "predicted_nez_count": float(record.get("predicted_nez_count", 0.0)),
                    "predicted_ez_count": float(record.get("predicted_ez_count", 0.0)),
                    "patient_macro_f1": float(record.get("patient_macro_f1", 0.0)),
                    "patient_balanced_accuracy": float(record.get("patient_balanced_accuracy", 0.0)),
                    "patient_weighted_f1": float(record.get("patient_weighted_f1", 0.0)),
                    "patient_nez_precision": float(record.get("patient_nez_precision", 0.0)),
                    "patient_nez_recall": float(record.get("patient_nez_recall", 0.0)),
                    "patient_nez_f1": float(record.get("patient_nez_f1", 0.0)),
                    "patient_ez_precision": float(record.get("patient_ez_precision", 0.0)),
                    "patient_ez_recall": float(record.get("patient_ez_recall", 0.0)),
                    "patient_ez_f1": float(record.get("patient_ez_f1", 0.0)),
                    "patient_accuracy": float(record.get("patient_accuracy", 0.0)),
                    "ez_mrr": float(record.get("ez_mrr", 0.0)),
                    "ez_recall_at_true_count": float(record.get("ez_recall_at_true_count", 0.0)),
                    "predicted_ez_channels": ";".join(record.get("predicted_ez_channels", record.get("predicted_channels", []))),
                    "predicted_nez_channels": ";".join(record.get("predicted_nez_channels", [])),
                    "true_ez_channels": ";".join(record.get("true_ez_channels", [])),
                    "true_nez_channels": ";".join(record.get("true_nez_channels", [])),
                    "n_seizures": len(record.get("run_ids", [])),
                }
            )
            labels_nez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float32)
            labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
            score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
            score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
            pred_ez = np.asarray(record.get("predicted_ez_mask", record.get("predicted_mask", np.zeros_like(labels_nez))), dtype=int)
            pred_nez = np.asarray(record.get("predicted_nez_mask", 1 - pred_ez), dtype=int)
            valid_mask = np.asarray(record["channel_mask"], dtype=bool)
            rank_ez_desc = np.full(score_ez.shape[0], -1, dtype=int)
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size > 0:
                for rank, idx in enumerate(valid_idx[np.argsort(score_ez[valid_idx])[::-1]], start=1):
                    rank_ez_desc[idx] = rank
            for channel_idx, channel_name in enumerate(record["canonical_channels"]):
                channel_rows.append(
                    {
                        "fold_idx": int(fold_idx),
                        "split": split_name,
                        "subject_id": record["subject_id"],
                        "channel_name": channel_name,
                        "true_nez": float(labels_nez[channel_idx]),
                        "true_ez": float(labels_ez[channel_idx]),
                        "score_nez_probability": float(score_nez[channel_idx]),
                        "score_ez_probability": float(score_ez[channel_idx]),
                        "rank_ez_desc": int(rank_ez_desc[channel_idx]),
                        "predicted_nez": int(pred_nez[channel_idx]),
                        "predicted_ez": int(pred_ez[channel_idx]),
                    }
                )
        pd.DataFrame(patient_rows).to_csv(output_dir / f"{split_name}_patient_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)
        pd.DataFrame(channel_rows).to_csv(output_dir / f"{split_name}_channel_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)

    def run(self) -> List[Dict[str, Any]]:
        all_test_records: List[Dict[str, Any]] = []
        base_seed = int(getattr(self.args, "random_seed", 42))
        batch_size = int(getattr(self.args, "patient_batch_size", getattr(self.args, "batch_size", 2)))
        epochs = int(getattr(self.args, "epochs", 50))
        patience = int(getattr(self.args, "patience", 15))
        min_epochs_before_early_stop = max(0, int(getattr(self.args, "min_epochs_before_early_stop", 0)))
        log_interval = int(getattr(self.args, "log_interval", 1))

        self._log(
            "Starting B0-Pruned cross-validation | "
            f"folds={len(self.outer_splits)} | epochs={epochs} | batch_size={batch_size} | "
            "loss=masked_channel_bce | selection=top_true_ez_count_for_reporting"
        )

        for split in self.outer_splits:
            fold_idx = int(split["fold_idx"])
            _set_random_seed(base_seed + fold_idx)
            fit_subjects, val_subjects = split_train_val_subjects(
                list(split["train_subjects"]),
                val_ratio=float(getattr(self.args, "val_ratio", 0.2)),
                random_seed=base_seed,
                fold_idx=fold_idx,
            )
            test_subjects = list(split["test_subjects"])
            train_dataset, val_dataset, test_dataset, normalizer = self._build_datasets(fit_subjects, val_subjects, test_subjects)
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                self._log(f"Fold {fold_idx}: skipped because train/test dataset is empty.")
                continue

            train_loader = self._make_loader(train_dataset, shuffle=True, batch_size=batch_size)
            val_loader = self._make_loader(val_dataset, shuffle=False, batch_size=batch_size)
            test_loader = self._make_loader(test_dataset, shuffle=False, batch_size=batch_size)

            model = self.runtime["model_cls"](self.args).to(self.device)
            self._dry_initialize_lazy_layers(model, train_loader)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(self.args, "learning_rate", 1e-4)),
                weight_decay=float(getattr(self.args, "weight_decay", 1e-3)),
            )
            ez_weight_arg = str(getattr(self.args, "ez_negative_weight", "2")).lower()
            ez_negative_weight_value = (
                _estimate_ez_negative_weight(train_dataset, cap=float(getattr(self.args, "ez_negative_weight_cap", 20.0)))
                if ez_weight_arg == "auto"
                else float(ez_weight_arg)
            )
            ez_negative_weight = torch.tensor(ez_negative_weight_value, dtype=torch.float32, device=self.device)

            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} ready | "
                f"fit_patients={len(train_dataset)} | val_patients={len(val_dataset)} | test_patients={len(test_dataset)} | "
                f"window_feature_dim={normalizer.feature_dim} | ez_negative_weight={ez_negative_weight_value:.3f}"
            )

            best_state = copy.deepcopy(model.state_dict())
            best_summary: Dict[str, float] = {"patient_macro_f1": 0.0, "selection_score": -1.0}
            best_score = -1.0
            epochs_without_improvement = 0
            fold_dir = Path(getattr(self.args, "output_dir", "outputs")) / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = fold_dir / "best_b0_pruned_model.pth"

            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                train_metrics = self._train_one_epoch(model, train_loader, optimizer, ez_negative_weight)
                val_loss, val_summary, val_records = self._evaluate(model, val_loader, ez_negative_weight)
                val_score = _summary_score(val_summary, self.args, val_loss=val_loss)
                val_summary["selection_score"] = float(val_score)
                improved = val_score > best_score + 1e-6
                if improved:
                    best_score = val_score
                    best_state = copy.deepcopy(model.state_dict())
                    best_summary = dict(val_summary)
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_summary": best_summary,
                            "normalizer_mean": normalizer.mean,
                            "normalizer_std": normalizer.std,
                            "label_semantics": "1=NEZ,0=EZ",
                            "score_semantics": "scores=p_NEZ, score_ez=1-p_NEZ",
                            "positive_label": str(getattr(self.args, "positive_label", "nez")),
                            "ez_negative_weight": float(ez_negative_weight_value),
                        },
                        best_model_path,
                    )
                    self._save_outputs(val_records, fold_idx=fold_idx, split_name="val")
                else:
                    epochs_without_improvement += 1

                if epoch == 1 or epoch % log_interval == 0 or improved:
                    self._log(
                        f"Fold {fold_idx} epoch {epoch}/{epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} | val_loss={val_loss:.4f} | "
                        f"val_patient_macro_f1={float(val_summary.get('patient_macro_f1', 0.0)):.4f} | "
                        f"val_ez_recall_at_true_count={float(val_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                        f"{'improved' if improved else 'no_improve'}"
                    )
                if patience > 0 and epoch > min_epochs_before_early_stop and epochs_without_improvement >= patience:
                    self._log(f"Fold {fold_idx}: early stop at epoch {epoch}.")
                    break

            model.load_state_dict(best_state)
            test_loss, test_summary, test_records = self._evaluate(model, test_loader, ez_negative_weight)
            self._save_outputs(test_records, fold_idx=fold_idx, split_name="test")
            all_test_records.extend(test_records)
            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} complete | "
                f"test_loss={test_loss:.4f} | test_patient_macro_f1={float(test_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"test_ez_recall_at_true_count={float(test_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                f"best_val_patient_macro_f1={float(best_summary.get('patient_macro_f1', 0.0)):.4f}"
            )

        if all_test_records:
            overall_summary, _ = _summarize_prediction_records(all_test_records)
            output_dir = Path(getattr(self.args, "output_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([overall_summary]).to_csv(output_dir / "heldout_summary_neuroez_v3.csv", index=False)
            pd.Series(overall_summary).to_json(output_dir / "heldout_summary_neuroez_v3.json", indent=2)
            self._log(
                "Cross-validation held-out mean | "
                f"patients={len(all_test_records)} | "
                f"patient_macro_f1={float(overall_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"ez_recall_at_true_count={float(overall_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                f"pooled_auroc_nez={float(overall_summary.get('pooled_auroc_nez', 0.0)):.4f} | "
                f"pooled_auprc_ez={float(overall_summary.get('pooled_auprc_ez', 0.0)):.4f}"
            )
        self._log(f"Cross-validation finished. Total held-out patient records: {len(all_test_records)}")
        return all_test_records


__all__ = ["Exp_EZHybridLocalization", "select_best_decision_rule"]
