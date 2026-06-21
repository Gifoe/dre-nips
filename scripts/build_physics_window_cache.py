from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_multitask.causal_features import compute_tfccm_full_graph, standardize_channels
from neuroez_multitask.evidence_views import (
    B0_FEATURE_NAMES,
    CAUSAL_GRAPH_ALGORITHM,
    CAUSAL_GRAPH_WARNING,
    CAUSAL_NODE_FEATURE_NAMES,
    PHYSICS_FEATURE_LEVEL,
    PHYSICS_FEATURE_NAMES,
    PHYSICS_PROXY_FEATURE_NAMES,
    PHYSICS_STRICT_FEATURE_NAMES,
    PHYSICS_FEATURE_WARNING,
    WindowConfig,
    compute_b0_features,
    compute_causal_node_features,
    compute_physics_features,
    compute_tfccm_graph,
    compute_topology_features,
    derive_ez_labels,
    extract_onset_windows,
    label_mask,
    stack_or_empty,
)
from neuroez_multitask.physics_features import compute_physics_features_strict
from neuroez_multitask.topology_features import (
    TOPOLOGY_FULL_FEATURE_NAMES,
    TOPOLOGY_SIMPLE_FEATURE_NAMES,
    compute_topology_features_full,
)


STRICT_PHYSICS_FEATURE_LEVEL = "physics_strict_v2"
STRICT_PHYSICS_FEATURE_WARNING = (
    "Strict mode uses robust Welch/FFT aperiodic fitting, envelope-based HFO detection, "
    "PAC vector length, and local synchrony. It is an EDF-derived v2 implementation, "
    "not a specparam/FOOOF-validated clinical biomarker package."
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _center(patient: Any) -> str:
    return str(_get(patient, "center", _get(patient, "source_center", "unknown"))).strip().lower() or "unknown"


def _subject_id(center: str, subject_id: Any) -> str:
    text = str(subject_id).strip()
    return text if text.startswith(f"{center}:") else f"{center}:{text}"


def _labels_nez(obj: Any, *, input_label_semantics: str = "nez-positive") -> np.ndarray:
    raw = _get(obj, "labels", None)
    if raw is None:
        raw = _get(obj, "labels_nez", None)
    if raw is None:
        raw = _get(obj, "labels_ez", None)
        if raw is None:
            return np.zeros((0,), dtype=np.float32)
        values = np.asarray(raw, dtype=np.float32)
        if input_label_semantics == "ez-positive":
            return np.where(values >= 0.0, 1.0 - values, -1.0).astype(np.float32)
        return values.astype(np.float32)
    return np.asarray(raw, dtype=np.float32)


def _outcome(patient: Any) -> tuple[str | None, int | None]:
    engel = _get(patient, "Engel", _get(patient, "engel", None))
    success = _get(patient, "success_failure", _get(patient, "outcome_success", _get(patient, "surgery_success", None)))
    if success is None and engel is not None:
        text = str(engel).strip().lower().replace(" ", "")
        if text in {"i", "1", "engeli", "engel1"}:
            success = True
        elif text in {"ii", "iii", "iv", "2", "3", "4", "engelii", "engeliii", "engeliv"}:
            success = False
    if success is None:
        return None if engel is None else str(engel), None
    if isinstance(success, str):
        text = success.strip().lower()
        success = text in {"1", "true", "yes", "s", "success", "engeli", "engel i", "i"}
    return None if engel is None else str(engel), int(bool(success))


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated ints, got {text!r}")
    return values


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated floats, got {text!r}")
    return values


def compute_lagged_corr_graph(
    segment: np.ndarray,
    sfreq: float,
    *,
    max_delay_ms: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(segment, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment must be shaped [channels, samples].")
    channels, samples = data.shape
    adj = np.zeros((channels, channels), dtype=np.float32)
    delay = np.zeros((channels, channels), dtype=np.float32)
    if channels < 2 or samples < 3:
        return adj, delay
    max_lag = int(float(max_delay_ms) * float(sfreq) / 1000.0)
    max_lag = min(max_lag, samples // 4)
    if max_lag <= 0:
        return adj, delay
    normalized = standardize_channels(data)
    for src in range(channels):
        for dst in range(channels):
            if src == dst:
                continue
            best = 0.0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                left = normalized[src, :-lag]
                right = normalized[dst, lag:]
                if left.size < 2 or float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
                    continue
                corr = float(np.corrcoef(left, right)[0, 1])
                if np.isfinite(corr) and corr > best:
                    best = corr
                    best_lag = lag
            adj[src, dst] = np.float32(max(best, 0.0))
            delay[src, dst] = np.float32(best_lag / max(float(sfreq), 1e-6))
    return np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0), delay


def _load_structural_cost_matrix(
    structural_cost_matrix_dir: Path | None,
    subject_id: str,
    channel_names: list[str],
) -> np.ndarray | None:
    if structural_cost_matrix_dir is None:
        return None

    candidates = [
        structural_cost_matrix_dir / f"{subject_id}.npy",
        structural_cost_matrix_dir / f"{subject_id.replace(':', '_')}.npy",
    ]

    for path in candidates:
        if path.exists():
            cost = np.load(path).astype(np.float32)
            expected = (len(channel_names), len(channel_names))
            if cost.shape != expected:
                raise ValueError(
                    f"Structural cost matrix shape mismatch for {subject_id}: "
                    f"expected {expected}, got {cost.shape}"
                )
            return cost

    raise FileNotFoundError(f"Missing structural cost matrix for {subject_id}")


def build_run_sample(
    seizure: Any,
    *,
    config: WindowConfig,
    tfccm_max_delay_ms: float = 80.0,
    physics_mode: str = "strict",
    causal_graph_mode: str = "tfccm_full",
    topology_mode: str = "full",
    line_noise_hz: float | None = 50.0,
    tfccm_embedding_dims: tuple[int, ...] = (2, 3, 4),
    tfccm_taus: tuple[int, ...] = (1, 2, 3),
    tfccm_library_fractions: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0),
    tfccm_n_surrogates: int = 20,
    tfccm_alpha: float = 0.05,
    tfccm_max_points: int = 128,
    tfccm_random_seed: int = 42,
    structural_cost_matrix: np.ndarray | None = None,
    enable_sinkhorn: bool = False,
) -> dict[str, np.ndarray]:
    signal = np.asarray(_get(seizure, "signal"), dtype=np.float32)
    channel_names = [str(name) for name in _get(seizure, "channel_names", [])]
    sfreq = float(_get(seizure, "sfreq"))
    onset_sec = float(_get(seizure, "seizure_onset_sec", _get(seizure, "onset_sec", 0.0)))
    windows, centers, window_mask = extract_onset_windows(signal, sfreq=sfreq, onset_sec=onset_sec, config=config)
    b0_rows = []
    physics_rows = []
    adj_rows = []
    delay_rows = []
    pvalue_rows = []
    convergence_rows = []
    causal_rows = []
    mode = physics_mode.lower().strip()
    if mode not in {"proxy", "strict"}:
        raise ValueError(f"Unsupported physics_mode={physics_mode!r}.")
    graph_mode = causal_graph_mode.lower().strip()
    if graph_mode not in {"lagged_corr", "tfccm_lite", "tfccm_full"}:
        raise ValueError(f"Unsupported causal_graph_mode={causal_graph_mode!r}.")
    topo_mode = topology_mode.lower().strip()
    if topo_mode not in {"simple", "full"}:
        raise ValueError(f"Unsupported topology_mode={topology_mode!r}.")
    physics_feature_names = PHYSICS_STRICT_FEATURE_NAMES if mode == "strict" else PHYSICS_PROXY_FEATURE_NAMES
    for valid, segment in zip(window_mask, windows):
        pvalue = None
        convergence = None
        if not valid:
            channels = signal.shape[0]
            b0_rows.append(np.zeros((channels, len(B0_FEATURE_NAMES)), dtype=np.float32))
            physics_rows.append(np.zeros((channels, len(physics_feature_names)), dtype=np.float32))
            adj = np.zeros((channels, channels), dtype=np.float32)
            delay = np.zeros((channels, channels), dtype=np.float32)
            if graph_mode == "tfccm_full":
                pvalue = np.ones((channels, channels), dtype=np.float32)
                convergence = np.zeros((channels, channels), dtype=np.float32)
        else:
            b0_rows.append(compute_b0_features(segment, sfreq))
            if mode == "strict":
                physics_rows.append(compute_physics_features_strict(segment, sfreq, line_noise_hz=line_noise_hz))
            else:
                physics_rows.append(compute_physics_features(segment, sfreq))
            if graph_mode == "tfccm_full":
                graph = compute_tfccm_full_graph(
                    segment,
                    sfreq,
                    embedding_dims=tfccm_embedding_dims,
                    taus=tfccm_taus,
                    max_delay_ms=tfccm_max_delay_ms,
                    library_fractions=tfccm_library_fractions,
                    n_surrogates=tfccm_n_surrogates,
                    alpha=tfccm_alpha,
                    max_points=tfccm_max_points,
                    random_seed=tfccm_random_seed,
                )
                adj = graph["adjacency"]
                delay = graph["delay"]
                pvalue = graph["pvalue"]
                convergence = graph["convergence"]
            elif graph_mode == "tfccm_lite":
                adj, delay = compute_tfccm_graph(segment, sfreq, max_delay_ms=tfccm_max_delay_ms)
            elif graph_mode == "lagged_corr":
                adj, delay = compute_lagged_corr_graph(segment, sfreq, max_delay_ms=tfccm_max_delay_ms)
            else:
                raise ValueError(f"Unsupported causal_graph_mode={causal_graph_mode!r}")
        adj_rows.append(adj)
        delay_rows.append(delay)
        if pvalue is not None:
            pvalue_rows.append(pvalue)
        if convergence is not None:
            convergence_rows.append(convergence)
        causal_rows.append(compute_causal_node_features(adj, delay))
    adjacency = stack_or_empty(adj_rows, (signal.shape[0], signal.shape[0]))
    topology_features = (
        compute_topology_features_full(
            adjacency,
            centers,
            structural_cost_matrix=structural_cost_matrix,
            enable_sinkhorn=enable_sinkhorn,
        )
        if topo_mode == "full"
        else compute_topology_features(adjacency, centers)
    )
    sample = {
        "window_features": stack_or_empty(b0_rows, (signal.shape[0], len(B0_FEATURE_NAMES))),
        "physics_node_features": stack_or_empty(physics_rows, (signal.shape[0], len(physics_feature_names))),
        "tfccm_adjacency": adjacency,
        "tfccm_delay": stack_or_empty(delay_rows, (signal.shape[0], signal.shape[0])),
        "causal_node_features": stack_or_empty(causal_rows, (signal.shape[0], len(CAUSAL_NODE_FEATURE_NAMES))),
        "topology_graph_features": topology_features,
        "window_relative_centers_sec": centers.astype(np.float32),
        "window_mask": window_mask.astype(bool),
        "channel_names": channel_names,
    }
    if graph_mode == "tfccm_full":
        sample["tfccm_pvalue"] = stack_or_empty(pvalue_rows, (signal.shape[0], signal.shape[0]))
        sample["tfccm_convergence"] = stack_or_empty(convergence_rows, (signal.shape[0], signal.shape[0]))
    return sample


def build_cache_payload(
    patient_records: Sequence[Any],
    *,
    source_patient_records_pkl: str,
    window_length_sec: float = 2.0,
    window_step_sec: float = 1.0,
    pre_onset_sec: float = 60.0,
    post_onset_sec: float = 120.0,
    input_label_semantics: str = "nez-positive",
    tfccm_max_delay_ms: float = 80.0,
    physics_mode: str = "strict",
    causal_graph_mode: str = "tfccm_full",
    topology_mode: str = "full",
    line_noise_hz: float | None = 50.0,
    tfccm_embedding_dims: tuple[int, ...] = (2, 3, 4),
    tfccm_taus: tuple[int, ...] = (1, 2, 3),
    tfccm_library_fractions: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0),
    tfccm_n_surrogates: int = 20,
    tfccm_alpha: float = 0.05,
    tfccm_max_points: int = 128,
    tfccm_random_seed: int = 42,
    enable_sinkhorn: bool = False,
    structural_cost_matrix_dir: Path | None = None,
) -> dict[str, Any]:
    config = WindowConfig(
        window_length_sec=window_length_sec,
        window_step_sec=window_step_sec,
        pre_onset_sec=pre_onset_sec,
        post_onset_sec=post_onset_sec,
    )
    run_records: list[dict[str, Any]] = []
    patient_index: dict[str, dict[str, Any]] = {}
    outcome_index: dict[str, dict[str, Any]] = {}
    mode = physics_mode.lower().strip()
    if mode not in {"proxy", "strict"}:
        raise ValueError(f"Unsupported physics_mode={physics_mode!r}.")
    graph_mode = causal_graph_mode.lower().strip()
    if graph_mode not in {"lagged_corr", "tfccm_lite", "tfccm_full"}:
        raise ValueError(f"Unsupported causal_graph_mode={causal_graph_mode!r}.")
    topo_mode = topology_mode.lower().strip()
    if topo_mode not in {"simple", "full"}:
        raise ValueError(f"Unsupported topology_mode={topology_mode!r}.")
    if enable_sinkhorn and structural_cost_matrix_dir is None:
        raise RuntimeError("--enable-sinkhorn requires --structural-cost-matrix-dir")
    physics_feature_names = PHYSICS_STRICT_FEATURE_NAMES if mode == "strict" else PHYSICS_PROXY_FEATURE_NAMES
    physics_feature_level = STRICT_PHYSICS_FEATURE_LEVEL if mode == "strict" else PHYSICS_FEATURE_LEVEL
    physics_feature_warning = STRICT_PHYSICS_FEATURE_WARNING if mode == "strict" else PHYSICS_FEATURE_WARNING
    topology_feature_names = TOPOLOGY_FULL_FEATURE_NAMES if topo_mode == "full" else TOPOLOGY_SIMPLE_FEATURE_NAMES
    for patient in patient_records:
        center = _center(patient)
        sid = _subject_id(center, _get(patient, "subject_id", "unknown"))
        seizures = list(_get(patient, "seizures", []) or [])
        labels = _labels_nez(patient, input_label_semantics=input_label_semantics)
        if labels.size == 0 and seizures:
            labels = _labels_nez(seizures[0], input_label_semantics=input_label_semantics)
        channels = list(_get(patient, "canonical_channels", []) or [])
        if not channels and seizures:
            channels = [str(name) for name in _get(seizures[0], "channel_names", [])]
        structural_cost_matrix = (
            _load_structural_cost_matrix(structural_cost_matrix_dir, sid, channels) if enable_sinkhorn else None
        )
        labels = labels.astype(np.float32, copy=False)
        labels_ez = derive_ez_labels(labels)
        mask = label_mask(labels)
        patient_index[sid] = {
            "canonical_channels": channels,
            "labels": labels,
            "labels_nez": labels,
            "labels_ez": labels_ez,
            "label_mask": mask,
            "channel_meta": list(_get(patient, "channel_meta", []) or []),
            "center": center,
        }
        engel, success = _outcome(patient)
        outcome_index[sid] = {
            "Engel": engel,
            "success_failure": success,
            "followup_months": _get(patient, "followup_months", None),
            "center": center,
            "clinical_features_optional": _get(patient, "clinical_features_optional", {}),
        }
        for seizure in seizures:
            run_labels = _labels_nez(seizure, input_label_semantics=input_label_semantics)
            if run_labels.size == 0:
                run_labels = labels
            channel_names = [str(name) for name in _get(seizure, "channel_names", channels)]
            sample = build_run_sample(
                seizure,
                config=config,
                tfccm_max_delay_ms=tfccm_max_delay_ms,
                physics_mode=mode,
                causal_graph_mode=graph_mode,
                topology_mode=topo_mode,
                line_noise_hz=line_noise_hz,
                tfccm_embedding_dims=tfccm_embedding_dims,
                tfccm_taus=tfccm_taus,
                tfccm_library_fractions=tfccm_library_fractions,
                tfccm_n_surrogates=tfccm_n_surrogates,
                tfccm_alpha=tfccm_alpha,
                tfccm_max_points=tfccm_max_points,
                tfccm_random_seed=tfccm_random_seed,
                structural_cost_matrix=structural_cost_matrix,
                enable_sinkhorn=enable_sinkhorn,
            )
            run_records.append(
                {
                    "subject_id": sid,
                    "run_id": str(_get(seizure, "run_id", _get(seizure, "seizure_id", f"run{len(run_records)}"))),
                    "seizure_id": str(_get(seizure, "seizure_id", _get(seizure, "run_id", f"run{len(run_records)}"))),
                    "center": center,
                    "task": "ictal",
                    "phase_group": "ictal",
                    "channel_names_norm": channel_names,
                    "labels": run_labels.astype(np.float32, copy=False),
                    "labels_nez": run_labels.astype(np.float32, copy=False),
                    "labels_ez": derive_ez_labels(run_labels),
                    "sample": sample,
                }
            )
    return {
        "run_records": run_records,
        "patient_index": patient_index,
        "outcome_index": outcome_index,
        "cache_meta": {
            "cache_name": "all_window_cache_physics_final.pkl"
            if (mode == "strict" and graph_mode == "tfccm_full" and topo_mode == "full")
            else ("all_window_cache_physics_v2.pkl" if mode == "strict" else "all_window_cache_physics_v1.pkl"),
            "label_semantics": "internal_positive_class=NEZ; labels_nez: 1=NEZ, 0=EZ",
            "report_semantics": "Task1 reports P(EZ)=1-P(NEZ)",
            "cache_version": "physics_final_v1"
            if (mode == "strict" and graph_mode == "tfccm_full" and topo_mode == "full")
            else ("physics_strict_v2" if mode == "strict" else "physics_proxy_lite_v1"),
            "physics_mode": mode,
            "causal_graph_mode": graph_mode,
            "topology_mode": topo_mode,
            "causal_graph_algorithm": "tfccm_full_surrogate_tested"
            if graph_mode == "tfccm_full"
            else ("lagged_positive_correlation" if graph_mode == "lagged_corr" else CAUSAL_GRAPH_ALGORITHM),
            "causal_graph_warning": None if graph_mode == "tfccm_full" else CAUSAL_GRAPH_WARNING,
            "physics_feature_level": physics_feature_level,
            "physics_feature_warning": physics_feature_warning,
            "feature_names_b0": B0_FEATURE_NAMES,
            "feature_names_physics": physics_feature_names,
            "feature_names_causal_node": CAUSAL_NODE_FEATURE_NAMES,
            "feature_names_topology": topology_feature_names,
            "physics_params": {
                "line_noise_hz": line_noise_hz,
            },
            "window_length_sec": float(window_length_sec),
            "window_step_sec": float(window_step_sec),
            "pre_onset_sec": float(pre_onset_sec),
            "post_onset_sec": float(post_onset_sec),
            "tfccm_params": {
                "max_delay_ms": float(tfccm_max_delay_ms),
                "embedding_dims": list(tfccm_embedding_dims),
                "taus": list(tfccm_taus),
                "library_fractions": list(tfccm_library_fractions),
                "n_surrogates": int(tfccm_n_surrogates),
                "alpha": float(tfccm_alpha),
                "max_points": int(tfccm_max_points),
                "random_seed": int(tfccm_random_seed),
                "surrogate_test_enabled": bool(tfccm_n_surrogates > 0 and graph_mode == "tfccm_full"),
            },
            "sinkhorn_enabled": bool(enable_sinkhorn),
            "structural_cost_matrix_required": bool(enable_sinkhorn),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_patient_records_pkl": source_patient_records_pkl,
            "input_label_semantics": input_label_semantics,
        },
    }


def load_patient_records(path: Path) -> list[Any]:
    with open(path, "rb") as fin:
        payload = pickle.load(fin)
    if isinstance(payload, Mapping):
        payload = payload.get("records", payload.get("patient_records", payload))
    if not isinstance(payload, list):
        raise TypeError("patient_records.pkl must contain a list or a mapping with records.")
    return payload


def write_cache(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PGC-SEEG physics window cache from patient_records.pkl.")
    parser.add_argument("--patient-records-pkl", "--patient_records_pkl", dest="patient_records_pkl", type=Path, default=None)
    parser.add_argument("--output-cache", "--output_cache", dest="output_cache", type=Path, required=True)
    parser.add_argument("--window-length-sec", type=float, default=2.0)
    parser.add_argument("--window-step-sec", type=float, default=1.0)
    parser.add_argument("--pre-onset-sec", type=float, default=60.0)
    parser.add_argument("--post-onset-sec", type=float, default=120.0)
    parser.add_argument("--input-label-semantics", choices=["nez-positive", "ez-positive"], default="nez-positive")
    parser.add_argument("--tfccm-max-delay-ms", type=float, default=80.0)
    parser.add_argument("--physics-mode", choices=["proxy", "strict"], default="strict")
    parser.add_argument(
        "--causal-graph-mode",
        choices=["lagged_corr", "tfccm_lite", "tfccm_full"],
        default="tfccm_full",
    )
    parser.add_argument("--topology-mode", choices=["simple", "full"], default="full")
    parser.add_argument("--tfccm-embedding-dims", type=str, default="2,3,4")
    parser.add_argument("--tfccm-taus", type=str, default="1,2,3")
    parser.add_argument("--tfccm-library-fractions", type=str, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--tfccm-n-surrogates", type=int, default=20)
    parser.add_argument("--tfccm-alpha", type=float, default=0.05)
    parser.add_argument("--tfccm-max-points", type=int, default=128)
    parser.add_argument("--tfccm-random-seed", type=int, default=42)
    parser.add_argument("--enable-sinkhorn", action="store_true")
    parser.add_argument("--structural-cost-matrix-dir", type=Path, default=None)
    parser.add_argument("--line-noise-hz", type=float, default=50.0)
    parser.add_argument("--edf_root", type=Path, default=None)
    parser.add_argument("--annotation_dir", type=Path, default=None)
    args = parser.parse_args()
    if args.patient_records_pkl is None:
        raise SystemExit("--patient_records_pkl is required in this local build. Generate it first with scripts/build_patient_records_from_dre_nips.py.")
    tfccm_embedding_dims = _parse_int_tuple(args.tfccm_embedding_dims)
    tfccm_taus = _parse_int_tuple(args.tfccm_taus)
    tfccm_library_fractions = _parse_float_tuple(args.tfccm_library_fractions)
    payload = build_cache_payload(
        load_patient_records(args.patient_records_pkl),
        source_patient_records_pkl=str(args.patient_records_pkl),
        window_length_sec=args.window_length_sec,
        window_step_sec=args.window_step_sec,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        input_label_semantics=args.input_label_semantics,
        tfccm_max_delay_ms=args.tfccm_max_delay_ms,
        physics_mode=args.physics_mode,
        causal_graph_mode=args.causal_graph_mode,
        topology_mode=args.topology_mode,
        line_noise_hz=args.line_noise_hz,
        tfccm_embedding_dims=tfccm_embedding_dims,
        tfccm_taus=tfccm_taus,
        tfccm_library_fractions=tfccm_library_fractions,
        tfccm_n_surrogates=args.tfccm_n_surrogates,
        tfccm_alpha=args.tfccm_alpha,
        tfccm_max_points=args.tfccm_max_points,
        tfccm_random_seed=args.tfccm_random_seed,
        enable_sinkhorn=args.enable_sinkhorn,
        structural_cost_matrix_dir=args.structural_cost_matrix_dir,
    )
    write_cache(payload, args.output_cache)
    print(json.dumps({"output_cache": str(args.output_cache), "run_records": len(payload["run_records"]), "patients": len(payload["patient_index"])}, indent=2))


if __name__ == "__main__":
    main()
