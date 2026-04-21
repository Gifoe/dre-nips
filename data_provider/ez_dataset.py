from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from module1_file_discovery import discover_all_bids_files
from ez_features import extract_run_feature_record


def _cache_path(args: Any) -> Path:
    output_dir = Path(getattr(args, "output_dir", "outputs"))
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (
        "techez_cache_"
        f"sr{int(float(getattr(args, 'target_sfreq', 512.0)))}_"
        f"win{int(float(getattr(args, 'win_len_sec', 15.0)))}_"
        f"step{int(float(getattr(args, 'step_sec', 5.0)))}.pkl"
    )


def _channel_sort_key(channel_meta: Dict[str, Any]) -> tuple:
    number = channel_meta.get("contact_number")
    number_key = int(number) if number is not None else 10**9
    return (
        str(channel_meta.get("contact_group", "")),
        number_key,
        str(channel_meta.get("channel_name_norm", "")),
    )


def _build_patient_bag(subject_id: str, run_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    run_records = list(run_records)
    channel_meta_map: Dict[str, Dict[str, Any]] = {}
    label_map: Dict[str, float] = {}

    for run in run_records:
        for idx, channel_name in enumerate(run["channel_names_norm"]):
            if channel_name not in channel_meta_map:
                channel_meta_map[channel_name] = {
                    "channel_name_norm": channel_name,
                    "contact_group": run["contact_groups"][idx],
                    "contact_number": run["contact_numbers"][idx],
                }
            label_map[channel_name] = max(
                float(label_map.get(channel_name, 0.0)),
                float(run["labels"][idx]),
            )

    canonical_meta = sorted(channel_meta_map.values(), key=_channel_sort_key)
    canonical_channels = [item["channel_name_norm"] for item in canonical_meta]
    canonical_index = {name: idx for idx, name in enumerate(canonical_channels)}
    patient_labels = np.asarray(
        [label_map.get(name, 0.0) for name in canonical_channels],
        dtype=np.float32,
    )

    aligned_runs: List[Dict[str, Any]] = []
    for run in run_records:
        num_windows = int(run["x_feat"].shape[0])
        patient_channels = len(canonical_channels)
        feat_dim = int(run["x_feat"].shape[-1])
        conn_dim = int(run["node_conn"].shape[-1])

        aligned_feat = np.zeros((num_windows, patient_channels, feat_dim), dtype=np.float32)
        aligned_conn = np.zeros((num_windows, patient_channels, conn_dim), dtype=np.float32)
        channel_mask = np.zeros(patient_channels, dtype=bool)
        local_to_canonical: Dict[int, int] = {}

        for local_idx, channel_name in enumerate(run["channel_names_norm"]):
            patient_idx = canonical_index[channel_name]
            local_to_canonical[local_idx] = patient_idx
            aligned_feat[:, patient_idx, :] = run["x_feat"][:, local_idx, :]
            aligned_conn[:, patient_idx, :] = run["node_conn"][:, local_idx, :]
            channel_mask[patient_idx] = True

        if run["edge_index"].size > 0:
            src = np.asarray([local_to_canonical[int(idx)] for idx in run["edge_index"][0]], dtype=np.int64)
            dst = np.asarray([local_to_canonical[int(idx)] for idx in run["edge_index"][1]], dtype=np.int64)
            edge_index = np.stack([src, dst], axis=0)
            edge_attr = np.asarray(run["edge_attr"], dtype=np.float32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.zeros((num_windows, 0, 4), dtype=np.float32)

        aligned_runs.append(
            {
                "run_id": run["run_id"],
                "task": run["task"],
                "phase_group": run["phase_group"],
                "phase_ids": np.asarray(run["phase_ids"], dtype=np.int64),
                "quality_weight": float(run["quality_weight"]),
                "n_windows": int(run["n_windows"]),
                "x_feat": aligned_feat,
                "node_conn": aligned_conn,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "channel_mask": channel_mask,
            }
        )

    return {
        "subject_id": subject_id,
        "canonical_channels": canonical_channels,
        "channel_meta": canonical_meta,
        "labels": patient_labels,
        "label_mask": np.ones(len(canonical_channels), dtype=bool),
        "runs": aligned_runs,
    }


def build_or_load_patient_bags(args: Any) -> List[Dict[str, Any]]:
    cache_path = _cache_path(args)
    force_rebuild = bool(getattr(args, "force_rebuild_cache", False))
    if cache_path.exists() and not force_rebuild:
        with open(cache_path, "rb") as fin:
            return pickle.load(fin)

    runs_df = discover_all_bids_files(
        getattr(args, "dataset_dir", "."),
        participants_path=getattr(args, "participants_path", None),
        subject_filter=getattr(args, "subject_filter", None),
        success_only=bool(getattr(args, "success_only", True)),
    )
    if runs_df.empty:
        raise ValueError("No BIDS runs were discovered for TeChEZ.")

    run_records: List[Dict[str, Any]] = []
    for row in runs_df.to_dict(orient="records"):
        record = extract_run_feature_record(
            row,
            device=getattr(args, "device", None),
            target_sfreq=float(getattr(args, "target_sfreq", 512.0)),
            win_len_sec=float(getattr(args, "win_len_sec", 15.0)),
            step_sec=float(getattr(args, "step_sec", 5.0)),
            ez_definition=str(getattr(args, "ez_definition", "soz_or_resected")),
        )
        if record is not None:
            run_records.append(record.to_dict())

    if not run_records:
        raise ValueError("TeChEZ feature extraction produced zero valid runs.")

    grouped_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in run_records:
        grouped_runs[run["subject_id"]].append(run)

    patient_bags = [
        _build_patient_bag(subject_id, grouped_runs[subject_id])
        for subject_id in sorted(grouped_runs.keys())
    ]

    with open(cache_path, "wb") as fout:
        pickle.dump(patient_bags, fout)
    return patient_bags


__all__ = ["build_or_load_patient_bags"]
