from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


BASE_SPECTRAL_FEATURE_NAMES = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_alpha",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "log_total_power",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
    "hjorth_mobility",
    "hjorth_complexity",
)
PRUNED_SPECTRAL_FEATURE_NAMES = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
)
WINDOW_NODE_FEATURE_NAMES = tuple(BASE_SPECTRAL_FEATURE_NAMES) + (
    "degree_norm",
    "strength_norm",
    "clustering_coeff",
    "eigenvector_centrality",
    "pagerank",
    "kcore_norm",
    "local_efficiency",
)


SAMPLE_REQUIRED_FOR_FLATTEN = (
    "window_features",
    "window_relative_centers_sec",
)

RUN_REQUIRED_FOR_FLATTEN = (
    "subject_id",
    "run_id",
    "channel_names_norm",
    "labels",
    "sample",
)


def _shape(value: Any) -> list[int] | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    return [int(dim) for dim in arr.shape]


def _first_existing_sample(run_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in run_records:
        sample = record.get("sample") if isinstance(record, dict) else None
        if isinstance(sample, dict):
            return sample
    return None


def inspect_cache(path: Path, max_records: int = 200) -> dict[str, Any]:
    with path.open("rb") as fin:
        payload = pickle.load(fin)

    report: dict[str, Any] = {
        "path": str(path),
        "top_level_type": type(payload).__name__,
        "usable_window_cache": False,
        "errors": [],
        "warnings": [],
        "counts": {},
        "shapes": {},
        "pruned_backbone_support": {},
    }

    if not isinstance(payload, dict):
        report["errors"].append("Top-level object is not a dict. This is probably patient_records.pkl, not a window cache.")
        return report

    run_records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    if not isinstance(run_records, list) or not isinstance(patient_index, dict):
        report["errors"].append("Missing top-level run_records list and patient_index dict.")
        return report
    if not run_records:
        report["errors"].append("run_records is empty.")
    if not patient_index:
        report["errors"].append("patient_index is empty.")

    report["counts"]["run_records"] = len(run_records)
    report["counts"]["patients"] = len(patient_index)

    inspected = [record for record in run_records[: max(1, int(max_records))] if isinstance(record, dict)]
    subjects = {str(record.get("subject_id")) for record in inspected if "subject_id" in record}
    report["counts"]["inspected_records"] = len(inspected)
    report["counts"]["inspected_subjects"] = len(subjects)

    missing_run_keys: dict[str, int] = {key: 0 for key in RUN_REQUIRED_FOR_FLATTEN}
    missing_sample_keys: dict[str, int] = {key: 0 for key in SAMPLE_REQUIRED_FOR_FLATTEN}
    feature_shapes: list[tuple[int, int, int]] = []
    center_lengths: list[int] = []
    has_negative_centers = 0
    has_post_centers = 0
    channel_mismatch = 0
    finite_feature_failures = 0

    for record in inspected:
        for key in RUN_REQUIRED_FOR_FLATTEN:
            if key not in record:
                missing_run_keys[key] += 1
        sample = record.get("sample")
        if not isinstance(sample, dict):
            continue
        for key in SAMPLE_REQUIRED_FOR_FLATTEN:
            if key not in sample:
                missing_sample_keys[key] += 1

        labels = np.asarray(record.get("labels", []))
        channels = list(record.get("channel_names_norm", []))
        features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)

        if features.ndim == 3:
            feature_shapes.append(tuple(int(dim) for dim in features.shape))
            if not np.isfinite(features).all():
                finite_feature_failures += 1
        if centers.ndim == 1:
            center_lengths.append(int(centers.shape[0]))
            if np.any(centers < 0.0):
                has_negative_centers += 1
            if np.any(centers >= 0.0):
                has_post_centers += 1
        if features.ndim != 3 or features.shape[0] == 0 or features.shape[2] == 0:
            finite_feature_failures += 1
        elif features.shape[1] != len(channels) or features.shape[1] != labels.shape[0]:
            channel_mismatch += 1

    first_sample = _first_existing_sample(run_records)
    first_feature = None
    if first_sample is not None:
        first_feature = np.asarray(first_sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        report["shapes"]["first_window_features"] = _shape(first_sample.get("window_features"))
        report["shapes"]["first_window_relative_centers_sec"] = _shape(first_sample.get("window_relative_centers_sec"))

    base_dim = len(WINDOW_NODE_FEATURE_NAMES)
    spectral_dim = len(BASE_SPECTRAL_FEATURE_NAMES)
    pruned_dim = len(PRUNED_SPECTRAL_FEATURE_NAMES)
    feature_dim = int(first_feature.shape[-1]) if isinstance(first_feature, np.ndarray) and first_feature.ndim == 3 else 0
    graph_dim = max(0, feature_dim - spectral_dim)

    report["counts"]["records_with_negative_centers"] = has_negative_centers
    report["counts"]["records_with_post_centers"] = has_post_centers
    report["counts"]["channel_shape_mismatches"] = channel_mismatch
    report["counts"]["records_with_nonfinite_window_features"] = finite_feature_failures
    report["feature_names"] = {
        "expected_base_dim": base_dim,
        "expected_spectral_classical_dim": spectral_dim,
        "expected_pruned_spectral_dim": pruned_dim,
        "observed_first_feature_dim": feature_dim,
        "observed_graph_dim_if_standard_order": graph_dim,
        "pruned_spectral_names": list(PRUNED_SPECTRAL_FEATURE_NAMES),
        "standard_order": list(WINDOW_NODE_FEATURE_NAMES),
    }

    missing_run_keys = {key: value for key, value in missing_run_keys.items() if value}
    missing_sample_keys = {key: value for key, value in missing_sample_keys.items() if value}
    if missing_run_keys:
        report["errors"].append(f"Missing run-level keys required by flatten_window_samples: {missing_run_keys}")
    if missing_sample_keys:
        report["errors"].append(f"Missing sample-level keys required by flatten_window_samples: {missing_sample_keys}")
    if channel_mismatch:
        report["errors"].append(f"{channel_mismatch} inspected records have channel/label/window feature shape mismatches.")
    if finite_feature_failures:
        report["errors"].append(f"{finite_feature_failures} inspected records contain non-finite window_features.")

    if feature_dim < base_dim:
        report["warnings"].append(
            f"window_features dim={feature_dim}, smaller than expected spectral/classical dim={spectral_dim}; B0-Pruned cannot select its spectral features."
        )
    if has_negative_centers == 0:
        report["warnings"].append("No inspected records have negative pre-onset centers; self-reference zdelta/ratio will use all windows as baseline.")
    enough_flatten = not missing_run_keys and not missing_sample_keys and channel_mismatch == 0 and finite_feature_failures == 0
    b0_feature_parts = enough_flatten and feature_dim >= spectral_dim
    temporal = enough_flatten and bool(feature_shapes) and max(shape[0] for shape in feature_shapes) > 0
    report["usable_window_cache"] = enough_flatten and len(run_records) > 0 and len(patient_index) > 0
    report["pruned_backbone_support"] = {
        "self_reference_abs_delta_zdelta_ratio": {
            "supported_by_cache": b0_feature_parts,
            "code_note": "Uses window_features plus window_relative_centers_sec.",
        },
        "spectral_classical_pruned_features": {
            "supported_by_cache": b0_feature_parts,
            "code_note": "Uses the first 13 spectral/classical cache features and selects the pruned subset.",
        },
        "temporal_mean_and_cross_seizure_stats": {
            "supported_by_cache": temporal,
            "code_note": "Uses available windows and multiple run_records per patient when present.",
        },
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NeuroEZ window cache support for B0-Pruned-EZBackbone.")
    parser.add_argument("caches", nargs="*", type=Path, help="One or more *_window_cache.pkl files.")
    parser.add_argument("--cache-path", dest="cache_paths", action="append", type=Path, help="Cache path to inspect. Can be repeated.")
    parser.add_argument("--max-records", type=int, default=200, help="Maximum records per cache to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    cache_paths = list(args.caches)
    if args.cache_paths:
        cache_paths.extend(args.cache_paths)
    if not cache_paths:
        parser.error("provide at least one cache path as a positional argument or with --cache-path")

    reports = [inspect_cache(path, max_records=args.max_records) for path in cache_paths]
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    for report in reports:
        print("=" * 88)
        print(report["path"])
        print(f"usable_window_cache: {report['usable_window_cache']}")
        print(f"counts: {json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}")
        print(f"shapes: {json.dumps(report['shapes'], ensure_ascii=False, sort_keys=True)}")
        print(f"feature_names: observed_dim={report['feature_names']['observed_first_feature_dim']} expected_base_dim={report['feature_names']['expected_base_dim']}")
        if report["errors"]:
            print("errors:")
            for item in report["errors"]:
                print(f"  - {item}")
        if report["warnings"]:
            print("warnings:")
            for item in report["warnings"]:
                print(f"  - {item}")
        print("pruned_backbone_support:")
        for name, item in report["pruned_backbone_support"].items():
            print(f"  - {name}: cache={item['supported_by_cache']} | {item['code_note']}")


if __name__ == "__main__":
    main()
