from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)


def fit_feature_normalizer(arrays: Iterable[np.ndarray], fallback_dim: int) -> FeatureNormalizer:
    flattened = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float32)
        if values.size and values.shape[-1] > 0:
            flattened.append(values.reshape(-1, values.shape[-1]))
    if not flattened:
        return FeatureNormalizer(np.zeros((fallback_dim,), dtype=np.float32), np.ones((fallback_dim,), dtype=np.float32))
    merged = np.concatenate(flattened, axis=0)
    mean = np.mean(merged, axis=0).astype(np.float32)
    std = np.std(merged, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureNormalizer(mean, std)
