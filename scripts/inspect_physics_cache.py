from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


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
    shapes: dict[str, list[int]] = {}
    if isinstance(run_records, list) and run_records:
        sample = run_records[0].get("sample", {})
        for key in REQUIRED_SAMPLE_KEYS:
            if key not in sample:
                errors.append(f"first sample missing {key}")
            else:
                shapes[key] = _shape(sample[key])
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
    return {
        "usable_physics_cache": not errors,
        "counts": {
            "run_records": len(run_records) if isinstance(run_records, list) else 0,
            "patients": len(patient_index) if isinstance(patient_index, dict) else 0,
            "outcomes": len(outcome_index) if isinstance(outcome_index, dict) else 0,
        },
        "shapes": shapes,
        "cache_meta": cache_meta if isinstance(cache_meta, dict) else {},
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
