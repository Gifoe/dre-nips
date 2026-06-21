from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .evidence_views import b0_self_reference_features, physics_self_reference_features


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)


@dataclass
class MultiViewNormalizer:
    b0: FeatureNormalizer
    physics: FeatureNormalizer
    causal_node: FeatureNormalizer
    topology: FeatureNormalizer

    def transform_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if sample.get("_pgc_normalized"):
            return dict(sample)
        prepared = prepare_multiview_sample(sample)
        out = dict(prepared)
        out["window_features"] = self.b0.transform(prepared["window_features"])
        out["physics_node_features"] = self.physics.transform(prepared["physics_node_features"])
        out["causal_node_features"] = self.causal_node.transform(prepared["causal_node_features"])
        out["topology_graph_features"] = self.topology.transform(prepared["topology_graph_features"])
        out["_pgc_normalized"] = True
        return out


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


def _centers_for_sample(sample: dict[str, Any], num_windows: int) -> np.ndarray:
    centers = np.asarray(sample.get("window_relative_centers_sec", np.arange(num_windows, dtype=np.float32)), dtype=np.float32)
    if centers.shape[0] != num_windows:
        centers = np.arange(num_windows, dtype=np.float32)
    return centers


def prepare_multiview_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if sample.get("_pgc_prepared"):
        return dict(sample)
    out = dict(sample)
    b0 = np.asarray(sample["window_features"], dtype=np.float32)
    physics = np.asarray(sample["physics_node_features"], dtype=np.float32)
    centers = _centers_for_sample(sample, b0.shape[0])
    out["window_features"] = b0_self_reference_features(b0, centers)
    out["physics_node_features"] = physics_self_reference_features(physics, centers)
    out["causal_node_features"] = np.asarray(sample["causal_node_features"], dtype=np.float32)
    out["topology_graph_features"] = np.asarray(sample["topology_graph_features"], dtype=np.float32)
    out["window_relative_centers_sec"] = centers
    out["_pgc_prepared"] = True
    return out


def fit_multiview_normalizer(dataset: Iterable[dict[str, Any]]) -> MultiViewNormalizer:
    b0_rows = []
    physics_rows = []
    causal_rows = []
    topology_rows = []
    b0_dim = physics_dim = causal_dim = topology_dim = 1
    for item in dataset:
        for run in item.get("runs", []):
            sample = prepare_multiview_sample(run["sample"])
            b0_rows.append(sample["window_features"])
            physics_rows.append(sample["physics_node_features"])
            causal_rows.append(sample["causal_node_features"])
            topology_rows.append(sample["topology_graph_features"])
            b0_dim = int(sample["window_features"].shape[-1])
            physics_dim = int(sample["physics_node_features"].shape[-1])
            causal_dim = int(sample["causal_node_features"].shape[-1])
            topology_dim = int(sample["topology_graph_features"].shape[-1])
    return MultiViewNormalizer(
        b0=fit_feature_normalizer(b0_rows, b0_dim),
        physics=fit_feature_normalizer(physics_rows, physics_dim),
        causal_node=fit_feature_normalizer(causal_rows, causal_dim),
        topology=fit_feature_normalizer(topology_rows, topology_dim),
    )
