from __future__ import annotations

import numpy as np


TOPOLOGY_SIMPLE_FEATURE_NAMES = [
    "causal_density_mean",
    "causal_density_std",
    "source_concentration",
    "sink_concentration",
    "driver_entropy",
    "hwc_mean",
    "hwc_std",
    "hwc_slope",
]

TOPOLOGY_FULL_FEATURE_NAMES = [
    "causal_density_mean",
    "causal_density_std",
    "causal_density_slope",
    "source_concentration_mean",
    "source_concentration_std",
    "source_concentration_slope",
    "sink_concentration_mean",
    "sink_concentration_std",
    "sink_concentration_slope",
    "driver_entropy_mean",
    "driver_entropy_std",
    "driver_entropy_slope",
    "hwc_mean",
    "hwc_std",
    "hwc_slope",
    "hwc_preictal_mean",
    "hwc_ictal_mean",
    "hwc_pre_to_ictal_delta",
    "network_instability",
    "topology_trajectory_length",
    "sinkhorn_cost_mean",
    "sinkhorn_cost_std",
    "sinkhorn_cost_max",
    "sinkhorn_cost_slope",
]


def _safe_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 1e-8:
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=np.float32)
    return (arr / total).astype(np.float32)


def graph_scalar_series(adjacency_sequence: np.ndarray) -> dict[str, np.ndarray]:
    """
    Input:
      adjacency_sequence: [T, C, C]
    Return per-window scalar series:
      causal_density
      source_concentration
      sink_concentration
      driver_entropy
      hwc
    """
    graphs = np.nan_to_num(np.asarray(adjacency_sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if graphs.ndim != 3 or graphs.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "causal_density": empty,
            "source_concentration": empty,
            "sink_concentration": empty,
            "driver_entropy": empty,
            "hwc": empty,
        }
    density = []
    source_concentration = []
    sink_concentration = []
    driver_entropy = []
    hwc = []
    for graph in graphs:
        channels = graph.shape[0]
        non_diag = ~np.eye(channels, dtype=bool)
        values = graph[non_diag]
        density.append(float(np.mean(values > 0.05)) if values.size else 0.0)
        out_strength = np.sum(graph, axis=1)
        in_strength = np.sum(graph, axis=0)
        total_out = float(np.sum(out_strength))
        total_in = float(np.sum(in_strength))
        source_concentration.append(float(np.max(out_strength) / total_out) if total_out > 1e-8 and out_strength.size else 0.0)
        sink_concentration.append(float(np.max(in_strength) / total_in) if total_in > 1e-8 and in_strength.size else 0.0)
        driver = np.maximum(out_strength - in_strength, 0.0)
        p = _safe_distribution(driver)
        entropy = float(-np.sum(p * np.log(p + 1e-8)) / np.log(max(channels, 2))) if p.size else 0.0
        if float(np.sum(driver)) <= 1e-8:
            entropy = 0.0
        driver_entropy.append(entropy)
        hwc.append(float(np.std(out_strength - in_strength)))
    return {
        "causal_density": np.asarray(density, dtype=np.float32),
        "source_concentration": np.asarray(source_concentration, dtype=np.float32),
        "sink_concentration": np.asarray(sink_concentration, dtype=np.float32),
        "driver_entropy": np.asarray(driver_entropy, dtype=np.float32),
        "hwc": np.asarray(hwc, dtype=np.float32),
    }


def safe_slope(values: np.ndarray, centers: np.ndarray | None = None) -> float:
    y = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if y.size < 2:
        return 0.0
    if centers is None:
        x = np.arange(y.size, dtype=np.float32)
    else:
        x = np.nan_to_num(np.asarray(centers, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)[: y.size]
        if x.size < y.size:
            x = np.arange(y.size, dtype=np.float32)
    x_centered = x - float(np.mean(x))
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 1e-8:
        return 0.0
    slope = float(np.sum((y - float(np.mean(y))) * x_centered) / denom)
    return slope if np.isfinite(slope) else 0.0


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    left = _safe_distribution(a)
    right = _safe_distribution(b)
    cost_arr = np.nan_to_num(np.asarray(cost, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if cost_arr.shape != (left.size, right.size):
        raise ValueError(f"cost shape must be {(left.size, right.size)}, got {cost_arr.shape}")
    eps = max(float(epsilon), 1e-6)
    k_mat = np.exp(-cost_arr / eps).astype(np.float32)
    k_mat = np.maximum(k_mat, 1e-12)
    u = np.ones_like(left, dtype=np.float32)
    v = np.ones_like(right, dtype=np.float32)
    for _ in range(max(1, int(max_iter))):
        prev_u = u.copy()
        u = left / np.maximum(k_mat @ v, 1e-12)
        v = right / np.maximum(k_mat.T @ u, 1e-12)
        if float(np.max(np.abs(u - prev_u))) < float(tol):
            break
    plan = (u[:, None] * k_mat) * v[None, :]
    value = float(np.sum(plan * cost_arr))
    return value if np.isfinite(value) else 0.0


def _summary(values: np.ndarray, centers: np.ndarray) -> tuple[float, float, float]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr)), safe_slope(arr, centers)


def compute_topology_features_full(
    adjacency_sequence: np.ndarray,
    centers: np.ndarray,
    *,
    structural_cost_matrix: np.ndarray | None = None,
    enable_sinkhorn: bool = False,
) -> np.ndarray:
    graphs = np.nan_to_num(np.asarray(adjacency_sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if graphs.ndim != 3 or graphs.shape[0] == 0:
        return np.zeros((1, len(TOPOLOGY_FULL_FEATURE_NAMES)), dtype=np.float32)
    center_arr = np.asarray(centers, dtype=np.float32).reshape(-1)
    if center_arr.size < graphs.shape[0]:
        center_arr = np.arange(graphs.shape[0], dtype=np.float32)
    else:
        center_arr = center_arr[: graphs.shape[0]]
    scalars = graph_scalar_series(graphs)
    density_mean, density_std, density_slope = _summary(scalars["causal_density"], center_arr)
    source_mean, source_std, source_slope = _summary(scalars["source_concentration"], center_arr)
    sink_concentration_mean, sink_concentration_std, sink_concentration_slope = _summary(scalars["sink_concentration"], center_arr)
    entropy_mean, entropy_std, entropy_slope = _summary(scalars["driver_entropy"], center_arr)
    hwc = scalars["hwc"]
    hwc_mean, hwc_std, hwc_slope = _summary(hwc, center_arr)
    pre_mask = center_arr < 0
    ictal_mask = center_arr >= 0
    hwc_pre = float(np.mean(hwc[pre_mask])) if np.any(pre_mask) else 0.0
    hwc_ictal = float(np.mean(hwc[ictal_mask])) if np.any(ictal_mask) else 0.0
    diffs = np.diff(graphs, axis=0)
    diff_norms = np.sqrt(np.sum(diffs * diffs, axis=(1, 2))) if diffs.size else np.zeros((0,), dtype=np.float32)
    network_instability = float(np.mean(diff_norms)) if diff_norms.size else 0.0
    trajectory_length = float(np.sum(diff_norms)) if diff_norms.size else 0.0
    sinkhorn_values = np.zeros((max(graphs.shape[0] - 1, 0),), dtype=np.float32)
    if enable_sinkhorn:
        if structural_cost_matrix is None:
            raise RuntimeError("structural_cost_matrix is required when enable_sinkhorn=True")
        cost = np.asarray(structural_cost_matrix, dtype=np.float32)
        if cost.shape != graphs.shape[1:]:
            raise ValueError(f"structural_cost_matrix shape must be {graphs.shape[1:]}, got {cost.shape}")
        costs = []
        for idx in range(graphs.shape[0] - 1):
            left = _safe_distribution(np.sum(graphs[idx], axis=1))
            right = _safe_distribution(np.sum(graphs[idx + 1], axis=1))
            costs.append(sinkhorn_distance(left, right, cost))
        sinkhorn_values = np.asarray(costs, dtype=np.float32)
    sinkhorn_mean = float(np.mean(sinkhorn_values)) if sinkhorn_values.size else 0.0
    sink_std_value = float(np.std(sinkhorn_values)) if sinkhorn_values.size else 0.0
    sink_max = float(np.max(sinkhorn_values)) if sinkhorn_values.size else 0.0
    sink_slope_value = safe_slope(sinkhorn_values, center_arr[:-1]) if sinkhorn_values.size else 0.0
    row = np.asarray(
        [
            density_mean,
            density_std,
            density_slope,
            source_mean,
            source_std,
            source_slope,
            sink_concentration_mean,
            sink_concentration_std,
            sink_concentration_slope,
            entropy_mean,
            entropy_std,
            entropy_slope,
            hwc_mean,
            hwc_std,
            hwc_slope,
            hwc_pre,
            hwc_ictal,
            hwc_ictal - hwc_pre,
            network_instability,
            trajectory_length,
            sinkhorn_mean,
            sink_std_value,
            sink_max,
            sink_slope_value,
        ],
        dtype=np.float32,
    )
    out = np.tile(row[None, :], (graphs.shape[0], 1))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
