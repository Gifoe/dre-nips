from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def _log(message: str) -> None:
    print(f"[NeuroEZ-C0][Data] {message}", flush=True)


def _extract_run_records_from_cache_payload(cached_payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(cached_payload, dict):
        raise ValueError("Unsupported external cache format: top-level object is not a dict.")
    run_records = cached_payload.get("run_records")
    patient_index = cached_payload.get("patient_index")
    if not isinstance(run_records, list) or not isinstance(patient_index, dict):
        raise ValueError("Unsupported external cache format: expected run_records list and patient_index dict.")
    return run_records, patient_index


def _load_external_cache(cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)
    return _extract_run_records_from_cache_payload(cached_payload)


def _patient_source_center(subject_id: str, patient_meta: dict[str, Any]) -> str:
    for key in ("source_center", "center", "source_dataset"):
        value = str(patient_meta.get(key, "")).strip().lower()
        if value:
            return value
    if ":" in subject_id:
        return subject_id.split(":", 1)[0].strip().lower()
    return ""


def _filter_high_ez_fraction_lzu(
    run_records: list[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not bool(getattr(args, "drop_high_ez_fraction_lzu", False)):
        return run_records, patient_index

    max_fraction = float(getattr(args, "lzu_max_ez_fraction", 0.40))
    dropped: set[str] = set()
    for subject_id, patient_meta in patient_index.items():
        if _patient_source_center(str(subject_id), patient_meta) != "lzu":
            continue
        labels = np.asarray(patient_meta.get("labels", []), dtype=np.float32)
        mask = np.asarray(patient_meta.get("label_mask", np.ones(labels.shape, dtype=bool)), dtype=bool)
        valid = mask & (labels >= 0.0)
        if not np.any(valid):
            continue
        ez_fraction = float(np.sum(labels[valid] > 0.5) / max(int(np.sum(valid)), 1))
        if ez_fraction > max_fraction:
            dropped.add(str(subject_id))

    if not dropped:
        _log(f"LZU EZ-fraction filter enabled at {max_fraction:.2f}; no patients dropped.")
        return run_records, patient_index

    filtered_records = [record for record in run_records if str(record.get("subject_id")) not in dropped]
    filtered_index = {sid: meta for sid, meta in patient_index.items() if str(sid) not in dropped}
    _log(
        f"LZU EZ-fraction filter enabled at {max_fraction:.2f}; "
        f"dropped {len(dropped)} patient(s), kept {len(filtered_index)} patient(s) and {len(filtered_records)} record(s)."
    )
    return filtered_records, filtered_index


def build_or_load_run_records(args: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cache_path_raw = getattr(args, "sample_cache_path", None) or getattr(args, "window_cache_path", None)
    if not cache_path_raw:
        raise ValueError("C0 ablation requires --window_cache_path pointing to a validated window cache pkl.")
    cache_path = Path(str(cache_path_raw)).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"External sample cache does not exist: {cache_path}")
    run_records, patient_index = _load_external_cache(cache_path)
    run_records, patient_index = _filter_high_ez_fraction_lzu(run_records, patient_index, args)
    _log(f"Loaded {len(run_records)} run records and {len(patient_index)} patients from {cache_path}.")
    return run_records, patient_index


def flatten_window_samples(
    run_records: Iterable[dict[str, Any]],
    *,
    subject_ids: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    samples: list[dict[str, Any]] = []
    for run_record in run_records:
        subject_id = str(run_record["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        sample = dict(run_record["sample"])
        samples.append(
            {
                "subject_id": subject_id,
                "run_id": str(run_record["run_id"]),
                "sample_id": str(sample.get("sample_id", run_record["run_id"])),
                "channel_names_norm": list(run_record["channel_names_norm"]),
                "labels": np.asarray(run_record["labels"], dtype=np.float32),
                "window_features": np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_adjacency": np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_relative_centers_sec": np.asarray(
                    sample.get("window_relative_centers_sec", np.zeros((0,))),
                    dtype=np.float32,
                ),
            }
        )
    return samples


__all__ = ["build_or_load_run_records", "flatten_window_samples"]
