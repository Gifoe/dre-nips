from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_multitask.evidence_views import PHYSICS_STRICT_FEATURE_NAMES
from neuroez_multitask.topology_features import TOPOLOGY_FULL_FEATURE_NAMES


REQUIRED_SAMPLE_KEYS = [
    "window_features",
    "physics_node_features",
    "tfccm_adjacency",
    "tfccm_delay",
    "causal_node_features",
    "topology_graph_features",
    "window_relative_centers_sec",
    "window_mask",
]


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def inspect_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    run_records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    outcome_index = payload.get("outcome_index")
    cache_meta = payload.get("cache_meta")
    if not isinstance(run_records, list) or not run_records:
        errors.append("run_records is missing or empty")
    if not isinstance(patient_index, dict) or not patient_index:
        errors.append("patient_index is missing or empty")
    if not isinstance(outcome_index, dict):
        errors.append("outcome_index is missing")
    if not isinstance(cache_meta, dict):
        errors.append("cache_meta is missing")
        cache_meta = {}
    if isinstance(cache_meta, dict):
        if not cache_meta.get("causal_graph_algorithm"):
            warnings.append("causal_graph_algorithm missing")
        if not cache_meta.get("physics_feature_level"):
            warnings.append("physics_feature_level missing")
        for key in ("feature_names_b0", "feature_names_physics", "feature_names_causal_node", "feature_names_topology"):
            if key not in cache_meta:
                warnings.append(f"{key} missing")
    shapes: dict[str, Any] = {}
    if isinstance(run_records, list) and run_records:
        sample = run_records[0].get("sample", {})
        for key in REQUIRED_SAMPLE_KEYS:
            if key not in sample:
                errors.append(f"first sample missing {key}")
            else:
                shapes[key] = _shape(sample[key])
        for key in ("tfccm_pvalue", "tfccm_convergence"):
            if key in sample:
                shapes[key] = _shape(sample[key])
            else:
                shapes[key] = "missing"
        if "window_features" in sample:
            t, c = np.asarray(sample["window_features"]).shape[:2]
            for key in ("physics_node_features", "causal_node_features"):
                if key in sample and tuple(np.asarray(sample[key]).shape[:2]) != (t, c):
                    errors.append(f"{key} first two dims do not match window_features")
            for key in ("tfccm_adjacency", "tfccm_delay"):
                if key in sample and tuple(np.asarray(sample[key]).shape[:3]) != (t, c, c):
                    errors.append(f"{key} shape must be [T,C,C]")
    if isinstance(patient_index, dict):
        for sid, entry in patient_index.items():
            labels = np.asarray(entry.get("labels", []), dtype=np.float32)
            labels_nez = np.asarray(entry.get("labels_nez", []), dtype=np.float32)
            labels_ez = np.asarray(entry.get("labels_ez", []), dtype=np.float32)
            if labels.shape != labels_nez.shape or labels.shape != labels_ez.shape:
                errors.append(f"{sid}: label shapes mismatch")
            valid = labels >= 0.0
            if np.any(valid) and not np.allclose(labels_ez[valid], 1.0 - labels_nez[valid]):
                errors.append(f"{sid}: labels_ez is not derived from labels_nez")
    meta = cache_meta if isinstance(cache_meta, dict) else {}
    feature_names = {
        "b0": list(meta.get("feature_names_b0", [])),
        "physics": list(meta.get("feature_names_physics", [])),
        "causal_node": list(meta.get("feature_names_causal_node", [])),
        "topology": list(meta.get("feature_names_topology", [])),
    }
    feature_dims: dict[str, int | None] = {
        "b0": shapes.get("window_features", [None, None, None])[-1],
        "physics": shapes.get("physics_node_features", [None, None, None])[-1],
        "causal_node": shapes.get("causal_node_features", [None, None, None])[-1],
        "topology": shapes.get("topology_graph_features", [None, None])[-1],
    }
    if feature_names["physics"] and feature_dims["physics"] is not None and len(feature_names["physics"]) != feature_dims["physics"]:
        warnings.append("physics feature names length does not match physics feature dim")
    if feature_names["topology"] and feature_dims["topology"] is not None and len(feature_names["topology"]) != feature_dims["topology"]:
        warnings.append("topology feature names length does not match topology feature dim")
    physics_mode = meta.get("physics_mode")
    causal_graph_mode = meta.get("causal_graph_mode")
    topology_mode = meta.get("topology_mode")
    if causal_graph_mode == "tfccm_full" and shapes.get("tfccm_pvalue") == "missing":
        warnings.append("causal_graph_mode=tfccm_full but tfccm_pvalue missing")
    if causal_graph_mode == "tfccm_full" and shapes.get("tfccm_convergence") == "missing":
        warnings.append("causal_graph_mode=tfccm_full but tfccm_convergence missing")
    if topology_mode == "full" and feature_dims["topology"] != len(TOPOLOGY_FULL_FEATURE_NAMES):
        warnings.append(
            f"topology_mode=full but topology dim mismatch: expected {len(TOPOLOGY_FULL_FEATURE_NAMES)}, got {feature_dims['topology']}"
        )
    if physics_mode == "strict" and feature_dims["physics"] != len(PHYSICS_STRICT_FEATURE_NAMES):
        warnings.append(
            f"physics_mode=strict but physics dim mismatch: expected {len(PHYSICS_STRICT_FEATURE_NAMES)}, got {feature_dims['physics']}"
        )
    center_distribution: dict[str, int] = {}
    if isinstance(patient_index, dict):
        for entry in patient_index.values():
            center = str(entry.get("center", "unknown"))
            center_distribution[center] = center_distribution.get(center, 0) + 1
    outcome_distribution: dict[str, int] = {}
    num_patients_with_outcome = 0
    if isinstance(outcome_index, dict):
        for entry in outcome_index.values():
            outcome = entry.get("success_failure")
            if outcome is not None:
                num_patients_with_outcome += 1
            key = "missing" if outcome is None else str(int(bool(outcome)))
            outcome_distribution[key] = outcome_distribution.get(key, 0) + 1

    return {
        "usable_physics_cache": not errors,
        "cache_version": meta.get("cache_version"),
        "cache_name": meta.get("cache_name"),
        "label_semantics": meta.get("label_semantics"),
        "physics_mode": physics_mode,
        "causal_graph_mode": causal_graph_mode,
        "topology_mode": topology_mode,
        "causal_graph_algorithm": meta.get("causal_graph_algorithm"),
        "physics_feature_level": meta.get("physics_feature_level"),
        "num_patients": len(patient_index) if isinstance(patient_index, dict) else 0,
        "num_runs": len(run_records) if isinstance(run_records, list) else 0,
        "num_patients_with_outcome": num_patients_with_outcome,
        "center_distribution": center_distribution,
        "outcome_distribution": outcome_distribution,
        "counts": {
            "run_records": len(run_records) if isinstance(run_records, list) else 0,
            "patients": len(patient_index) if isinstance(patient_index, dict) else 0,
            "outcomes": len(outcome_index) if isinstance(outcome_index, dict) else 0,
        },
        "feature_dims": feature_dims,
        "feature_names": feature_names,
        "sample_shapes": shapes,
        "shapes": shapes,
        "cache_meta": meta,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all_window_cache_physics_v1.pkl schema.")
    parser.add_argument("--cache-path", type=Path, required=True)
    args = parser.parse_args()
    with open(args.cache_path, "rb") as fin:
        payload = pickle.load(fin)
    print(json.dumps(inspect_cache_payload(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
