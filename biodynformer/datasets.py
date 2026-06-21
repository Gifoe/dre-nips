from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _load_npz(row: dict[str, Any]) -> dict[str, Any]:
    arr = np.load(row["tensor_path"], allow_pickle=True)
    return {key: arr[key] for key in arr.files}


def _mean_valid(values: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(window_mask, dtype=bool)
    if values.shape[0] != mask.shape[0] or not mask.any():
        return np.zeros(values.shape[1:], dtype=np.float32)
    return values[mask].mean(axis=0).astype(np.float32)


def version_channel_matrix(payload: dict[str, Any], version: str) -> np.ndarray:
    version = str(version).lower()
    mask = payload["window_mask"]
    node = _mean_valid(payload["node_features"], mask)
    pieces = [node]
    if version in {"v2", "final"}:
        pieces.extend(
            [
                _mean_valid(payload["hfo_features"], mask),
                _mean_valid(payload["quality_features"], mask),
                payload["coverage_features"].astype(np.float32),
                _mean_valid(payload["causal_edge"], mask).mean(axis=1, keepdims=True),
                _mean_valid(payload["sync_edge"], mask).mean(axis=1, keepdims=True),
            ]
        )
    if version == "final":
        trend_source = payload["node_features"]
        valid = np.asarray(mask, dtype=bool)
        if valid.sum() >= 2:
            trend = trend_source[valid][-1] - trend_source[valid][0]
        else:
            trend = np.zeros_like(node)
        pieces.append(trend.astype(np.float32))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def build_task_examples(index: Sequence[dict[str, Any]], *, task: str, version: str, subjects: set[str] | None = None) -> list[dict[str, Any]]:
    task = str(task).lower()
    examples: list[dict[str, Any]] = []
    for row in index:
        subject_id = str(row["subject_id"])
        if subjects is not None and subject_id not in subjects:
            continue
        payload = _load_npz(row)
        outcome_raw = int(payload["outcome_success"])
        outcome = None if outcome_raw < 0 else bool(outcome_raw)
        if task == "task1" and outcome is not True:
            continue
        if task == "task2" and outcome is None:
            continue
        features = version_channel_matrix(payload, version)
        labels_ez = np.asarray(payload["labels_ez"], dtype=np.float32)
        labels_nez = np.where(labels_ez >= 0.0, 1.0 - labels_ez, -1.0).astype(np.float32)
        examples.append(
            {
                "center": str(payload["center"]),
                "subject_id": subject_id,
                "run_id": str(payload["run_id"]),
                "x_channel": features,
                "labels_ez": labels_ez,
                "labels_nez": labels_nez,
                "outcome_success": outcome,
                "window_mask": np.asarray(payload["window_mask"], dtype=bool),
            }
        )
    return examples


def task1_xy(examples: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, subjects = [], [], []
    for example in examples:
        labels = np.asarray(example["labels_nez"], dtype=np.float32)
        valid = labels >= 0.0
        xs.append(np.asarray(example["x_channel"], dtype=np.float32)[valid])
        ys.append(labels[valid])
        subjects.extend([str(example["subject_id"])] * int(valid.sum()))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), subjects


def task2_xy(examples: Sequence[dict[str, Any]], *, mode: str = "full") -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, subjects = [], [], []
    for example in examples:
        channel_x = np.asarray(example["x_channel"], dtype=np.float32)
        labels_ez = np.asarray(example["labels_ez"], dtype=np.float32)
        label_features = np.asarray([labels_ez.mean(), labels_ez.sum(), channel_x.shape[0]], dtype=np.float32)
        biomarker = channel_x.mean(axis=0)
        if mode == "label_only":
            feat = label_features
        elif mode == "biomarker_only":
            feat = biomarker
        elif mode == "metadata_only":
            feat = np.asarray([channel_x.shape[0]], dtype=np.float32)
        else:
            feat = np.concatenate([biomarker, label_features], axis=0)
        xs.append(feat)
        ys.append(float(bool(example["outcome_success"])))
        subjects.append(str(example["subject_id"]))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), subjects
