from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).dot(self.weights) + float(self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    epochs: int = 200,
    l2: float = 1e-3,
) -> LogisticModel:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        return LogisticModel(np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float32), 0.0, np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float32), np.ones((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float32))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    z_x = (x - mean) / std
    w = np.zeros((x.shape[1],), dtype=np.float32)
    b = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0.5, neg / pos, 1.0).astype(np.float32)
    for _ in range(int(epochs)):
        logits = z_x.dot(w) + b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        err = (p - y) * weights
        grad_w = z_x.T.dot(err) / float(x.shape[0]) + float(l2) * w
        grad_b = float(err.mean())
        w -= float(learning_rate) * grad_w.astype(np.float32)
        b -= float(learning_rate) * grad_b
    return LogisticModel(w.astype(np.float32), float(b), mean.astype(np.float32), std.astype(np.float32))
