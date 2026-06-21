from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else 0.0


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score)) if np.unique(y_true).size > 1 else 0.0


def _reciprocal_rank(y_true_ez: np.ndarray, ez_prob: np.ndarray) -> float:
    positives = np.flatnonzero(y_true_ez > 0.5)
    if positives.size == 0:
        return 0.0
    order = np.argsort(ez_prob)[::-1]
    for rank, idx in enumerate(order, start=1):
        if idx in positives:
            return 1.0 / float(rank)
    return 0.0


def _recall_at_true_count(y_true_ez: np.ndarray, ez_prob: np.ndarray) -> float:
    true_count = int(np.sum(y_true_ez > 0.5))
    if true_count == 0:
        return 0.0
    pred = np.zeros_like(y_true_ez, dtype=bool)
    pred[np.argsort(ez_prob)[::-1][:true_count]] = True
    return float(np.sum(pred & (y_true_ez > 0.5)) / true_count)


def summarize_task1_predictions(records: Sequence[dict[str, Any]]) -> dict[str, float | str]:
    pooled_labels = []
    pooled_scores = []
    patient_macro = []
    ez_f1s = []
    ez_precisions = []
    ez_recalls = []
    recalls_at_count = []
    mrrs = []
    balanced = []
    for record in records:
        labels_nez = np.asarray(record["labels_nez"], dtype=np.float32)
        nez_prob = np.asarray(record["nez_prob"], dtype=np.float32)
        mask = np.asarray(record.get("channel_mask", labels_nez >= 0.0), dtype=bool) & (labels_nez >= 0.0)
        if not np.any(mask):
            continue
        label_ez = 1.0 - labels_nez[mask]
        ez_prob = 1.0 - nez_prob[mask]
        pred_ez = ez_prob >= 0.5
        pooled_labels.append(label_ez)
        pooled_scores.append(ez_prob)
        patient_macro.append(float(f1_score(label_ez.astype(int), pred_ez.astype(int), average="macro", zero_division=0)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_ez.astype(int), pred_ez.astype(int), labels=[1], zero_division=0
        )
        ez_precisions.append(float(precision[0]))
        ez_recalls.append(float(recall[0]))
        ez_f1s.append(float(f1[0]))
        balanced.append(float(balanced_accuracy_score(label_ez.astype(int), pred_ez.astype(int))))
        recalls_at_count.append(_recall_at_true_count(label_ez, ez_prob))
        mrrs.append(_reciprocal_rank(label_ez, ez_prob))
    if not pooled_labels:
        y_true = np.asarray([], dtype=np.float32)
        y_score = np.asarray([], dtype=np.float32)
    else:
        y_true = np.concatenate(pooled_labels)
        y_score = np.concatenate(pooled_scores)
    return {
        "internal_positive_class": "NEZ",
        "reported_positive_class": "EZ",
        "task1_probability_reported": "P(EZ)=1-P(NEZ)",
        "AUROC": _safe_auc(y_true, y_score) if y_true.size else 0.0,
        "AUPRC": _safe_auprc(y_true, y_score) if y_true.size else 0.0,
        "patient_macro_F1": float(np.mean(patient_macro)) if patient_macro else 0.0,
        "EZ_F1": float(np.mean(ez_f1s)) if ez_f1s else 0.0,
        "EZ_precision": float(np.mean(ez_precisions)) if ez_precisions else 0.0,
        "EZ_recall": float(np.mean(ez_recalls)) if ez_recalls else 0.0,
        "EZ_recall_at_true_count": float(np.mean(recalls_at_count)) if recalls_at_count else 0.0,
        "EZ_MRR": float(np.mean(mrrs)) if mrrs else 0.0,
        "balanced_accuracy": float(np.mean(balanced)) if balanced else 0.0,
    }


def summarize_task2_predictions(labels: Sequence[float], probs: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.float32)
    y_score = np.asarray(probs, dtype=np.float32)
    pred = y_score >= 0.5
    if y_true.size == 0:
        return {key: 0.0 for key in ["AUROC", "AUPRC", "balanced_accuracy", "F1", "sensitivity", "specificity", "Brier_score"]}
    tn = float(np.sum((y_true == 0) & (~pred)))
    fp = float(np.sum((y_true == 0) & pred))
    fn = float(np.sum((y_true == 1) & (~pred)))
    tp = float(np.sum((y_true == 1) & pred))
    return {
        "AUROC": _safe_auc(y_true, y_score),
        "AUPRC": _safe_auprc(y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true.astype(int), pred.astype(int))) if np.unique(y_true).size > 1 else 0.0,
        "F1": float(f1_score(y_true.astype(int), pred.astype(int), zero_division=0)),
        "sensitivity": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
        "Brier_score": float(brier_score_loss(y_true, y_score)),
    }
