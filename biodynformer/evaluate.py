from __future__ import annotations

from typing import Any

import numpy as np


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision * y_sorted) / max(float(y_sorted.sum()), 1.0))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def binary_metrics(y_true: np.ndarray, prob_pos: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    p = np.asarray(prob_pos, dtype=np.float32).reshape(-1)
    if len(y) == 0:
        return {f"{prefix}{name}": float("nan") for name in ["balanced_accuracy", "macro_f1", "mcc", "auroc", "auprc", "brier", "ece", "sensitivity", "specificity"]}
    pred = (p >= 0.5).astype(np.float32)
    tp = float(np.sum((pred == 1.0) & (y == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y == 0.0)))
    fp = float(np.sum((pred == 1.0) & (y == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y == 1.0)))
    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    precision_pos = tp / max(tp + fp, 1.0)
    precision_neg = tn / max(tn + fn, 1.0)
    f1_pos = 2.0 * precision_pos * sensitivity / max(precision_pos + sensitivity, 1e-6)
    f1_neg = 2.0 * precision_neg * specificity / max(precision_neg + specificity, 1e-6)
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    out = {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "mcc": (tp * tn - fp * fn) / float(denom),
        "auroc": _binary_auc(y, p),
        "auprc": _average_precision(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _ece(y, p),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return {f"{prefix}{key}": float(value) for key, value in out.items()}


def summarize_task1(y_true: np.ndarray, prob_nez: np.ndarray, subjects: list[str]) -> dict[str, Any]:
    pooled = binary_metrics(y_true, prob_nez, prefix="pooled_")
    patient_rows = []
    for subject in sorted(set(subjects)):
        idx = np.asarray([item == subject for item in subjects], dtype=bool)
        patient_rows.append(binary_metrics(np.asarray(y_true)[idx], np.asarray(prob_nez)[idx]))
    patient_summary = {}
    for key in ["balanced_accuracy", "macro_f1", "mcc"]:
        values = [row[key] for row in patient_rows if not np.isnan(row[key])]
        patient_summary[f"patient_macro_{key}"] = float(np.mean(values)) if values else float("nan")
    auprc_values = [
        _average_precision(np.asarray(y_true), np.asarray(prob_nez)),
        _average_precision(1.0 - np.asarray(y_true), 1.0 - np.asarray(prob_nez)),
    ]
    finite_auprc = [value for value in auprc_values if not np.isnan(value)]
    macro_auprc = float(np.mean(finite_auprc)) if finite_auprc else float("nan")
    return {**pooled, **patient_summary, "macro_auprc": float(macro_auprc), "brier": pooled["pooled_brier"], "ece": pooled["pooled_ece"]}


def summarize_task2(y_true: np.ndarray, prob_success: np.ndarray) -> dict[str, Any]:
    return binary_metrics(y_true, prob_success)
