from __future__ import annotations

from typing import Sequence

import numpy as np


def standardize_channels(data: np.ndarray) -> np.ndarray:
    """
    Input:
      data: [C, N]
    Output:
      standardized data: [C, N]
    """
    arr = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        raise ValueError("data must be shaped [channels, samples].")
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def embedding_points(
    series: np.ndarray,
    target: np.ndarray,
    *,
    lag: int,
    embedding_dim: int,
    tau: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build delay embedding from `series` and aligned target values.
    Return:
      points: [M, embedding_dim]
      target_values: [M]
    """
    x = np.nan_to_num(np.asarray(series, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(target, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n == 0:
        return np.zeros((0, embedding_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    x = x[:n]
    y = y[:n]
    lag = max(0, int(lag))
    embedding_dim = max(1, int(embedding_dim))
    tau = max(1, int(tau))
    start = (embedding_dim - 1) * tau + lag
    if n <= start:
        return np.zeros((0, embedding_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    idx = np.arange(start, n, dtype=np.int64)
    if idx.shape[0] < embedding_dim + 2:
        return np.zeros((0, embedding_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    points = np.stack([x[idx - dim * tau] for dim in range(embedding_dim)], axis=1)
    target_values = y[idx - lag]
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, int(max_points)).round().astype(np.int64)
        points = points[keep]
        target_values = target_values[keep]
    return points.astype(np.float32, copy=False), target_values.astype(np.float32, copy=False)


def cross_map_skill(
    manifold_series: np.ndarray,
    target_series: np.ndarray,
    *,
    lag: int,
    embedding_dim: int,
    tau: int,
    library_fraction: float,
    max_points: int,
) -> float:
    """
    Use nearest-neighbor simplex projection to estimate target_series from manifold_series.
    Return Pearson correlation between predicted and true target.
    """
    points, target = embedding_points(
        manifold_series,
        target_series,
        lag=lag,
        embedding_dim=embedding_dim,
        tau=tau,
        max_points=max_points,
    )
    n = int(points.shape[0])
    k = int(embedding_dim) + 1
    if n < k + 2 or float(np.std(target)) < 1e-6:
        return 0.0
    fraction = float(np.clip(library_fraction, 0.0, 1.0))
    lib_size = max(k + 1, int(round(n * fraction)))
    lib_size = min(lib_size, n)
    if lib_size <= k:
        return 0.0
    library = points[:lib_size]
    library_target = target[:lib_size]
    preds = np.zeros((n,), dtype=np.float32)
    for idx in range(n):
        dist = np.sqrt(np.sum((library - points[idx]) ** 2, axis=1))
        if idx < lib_size and lib_size > 1:
            dist[idx] = np.inf
        finite_order = np.argsort(dist)
        nn = finite_order[np.isfinite(dist[finite_order])][:k]
        if nn.shape[0] == 0:
            preds[idx] = float(np.mean(library_target))
            continue
        d0 = max(float(np.min(dist[nn])), 1e-6)
        weights = np.exp(-dist[nn] / d0)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-8:
            preds[idx] = float(np.mean(library_target[nn]))
        else:
            preds[idx] = float(np.sum(weights * library_target[nn]) / weight_sum)
    if float(np.std(preds)) < 1e-6 or float(np.std(target)) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(preds, target)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def circular_shift_surrogate(series: np.ndarray, shift: int) -> np.ndarray:
    """
    Circularly shift one time series.
    """
    return np.roll(np.asarray(series), int(shift))


def _zero_graphs(channels: int) -> dict[str, np.ndarray]:
    zeros = np.zeros((channels, channels), dtype=np.float32)
    return {
        "adjacency": zeros.copy(),
        "delay": zeros.copy(),
        "pvalue": np.ones((channels, channels), dtype=np.float32),
        "convergence": zeros.copy(),
    }


def compute_tfccm_full_graph(
    segment: np.ndarray,
    sfreq: float,
    *,
    embedding_dims: Sequence[int] = (2, 3, 4),
    taus: Sequence[int] = (1, 2, 3),
    max_delay_ms: float = 80.0,
    library_fractions: Sequence[float] = (0.3, 0.5, 0.7, 1.0),
    n_surrogates: int = 20,
    alpha: float = 0.05,
    max_points: int = 128,
    surrogate_method: str = "circular_shift",
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    data = np.asarray(segment, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment must be shaped [channels, samples].")
    channels, samples = data.shape
    if channels < 2 or samples < 8:
        return _zero_graphs(channels)
    max_lag = int(float(max_delay_ms) * float(sfreq) / 1000.0)
    max_lag = min(max_lag, samples // 4)
    if max_lag <= 0:
        return _zero_graphs(channels)
    dims = tuple(max(1, int(value)) for value in embedding_dims)
    tau_values = tuple(max(1, int(value)) for value in taus)
    fractions = tuple(float(value) for value in library_fractions)
    if not dims or not tau_values or not fractions:
        return _zero_graphs(channels)
    normalized = standardize_channels(data)
    adjacency = np.zeros((channels, channels), dtype=np.float32)
    delay = np.zeros((channels, channels), dtype=np.float32)
    pvalue = np.zeros((channels, channels), dtype=np.float32) if n_surrogates <= 0 else np.ones((channels, channels), dtype=np.float32)
    convergence_arr = np.zeros((channels, channels), dtype=np.float32)
    rng = np.random.default_rng(int(random_seed))
    for src in range(channels):
        for dst in range(channels):
            if src == dst:
                continue
            best_score = 0.0
            best_convergence = 0.0
            best_lag = 0
            best_dim = dims[0]
            best_tau = tau_values[0]
            for embedding_dim in dims:
                for tau in tau_values:
                    for lag in range(1, max_lag + 1):
                        skills = [
                            cross_map_skill(
                                normalized[dst],
                                normalized[src],
                                lag=lag,
                                embedding_dim=embedding_dim,
                                tau=tau,
                                library_fraction=fraction,
                                max_points=max_points,
                            )
                            for fraction in fractions
                        ]
                        score_full = float(skills[-1])
                        convergence = score_full - float(skills[0])
                        objective = score_full * convergence
                        if score_full > 0.0 and convergence > 0.0 and objective > best_score * best_convergence:
                            best_score = score_full
                            best_convergence = convergence
                            best_lag = lag
                            best_dim = embedding_dim
                            best_tau = tau
            if best_score <= 0.0 or best_convergence <= 0.0:
                continue
            edge_p = 0.0
            if n_surrogates > 0:
                if surrogate_method != "circular_shift":
                    raise ValueError(f"Unsupported surrogate_method={surrogate_method!r}")
                surrogate_scores = []
                for _ in range(int(n_surrogates)):
                    shift = int(rng.integers(1, max(samples, 2)))
                    surrogate_src = circular_shift_surrogate(normalized[src], shift)
                    surrogate_scores.append(
                        cross_map_skill(
                            normalized[dst],
                            surrogate_src,
                            lag=best_lag,
                            embedding_dim=best_dim,
                            tau=best_tau,
                            library_fraction=fractions[-1],
                            max_points=max_points,
                        )
                    )
                count = int(np.sum(np.asarray(surrogate_scores, dtype=np.float32) >= best_score))
                edge_p = (1.0 + count) / (float(n_surrogates) + 1.0)
                if edge_p >= float(alpha):
                    pvalue[src, dst] = np.float32(edge_p)
                    continue
            adjacency[src, dst] = np.float32(best_score * best_convergence)
            delay[src, dst] = np.float32(best_lag / max(float(sfreq), 1e-6))
            convergence_arr[src, dst] = np.float32(best_convergence)
            pvalue[src, dst] = np.float32(edge_p)
    return {
        "adjacency": np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "delay": np.nan_to_num(delay, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "pvalue": np.nan_to_num(pvalue, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32),
        "convergence": np.nan_to_num(convergence_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }
