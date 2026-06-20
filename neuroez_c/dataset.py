from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .evidence_views import (
    EvidenceNormalizer,
    b0_self_reference_features,
    fit_normalizer,
)


def _as_window_tensors(sample: dict[str, Any], feature_dim_fallback: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
    window_adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
    window_centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)

    if window_features.ndim == 3 and window_features.shape[0] > 0 and window_features.shape[-1] > 0:
        if window_adjacency.ndim != 3 or window_adjacency.shape[0] != window_features.shape[0]:
            num_windows, num_channels = int(window_features.shape[0]), int(window_features.shape[1])
            window_adjacency = np.zeros((num_windows, num_channels, num_channels), dtype=np.float32)
        if window_centers.shape[0] != window_features.shape[0]:
            window_centers = np.arange(window_features.shape[0], dtype=np.float32)
        return window_features, window_adjacency.astype(np.float32, copy=False), window_centers.astype(np.float32, copy=False)

    num_channels = int(np.asarray(sample.get("labels", []), dtype=np.float32).shape[0])
    features = np.zeros((1, num_channels, max(1, int(feature_dim_fallback))), dtype=np.float32)
    adjacency = np.zeros((features.shape[0], num_channels, num_channels), dtype=np.float32)
    centers = np.zeros((features.shape[0],), dtype=np.float32)
    return features, adjacency, centers


def _prepared_views(sample: dict[str, Any], args: Any | None = None) -> dict[str, np.ndarray]:
    raw_features, _, centers = _as_window_tensors(sample, feature_dim_fallback=1)
    return {
        "b0": b0_self_reference_features(raw_features, centers, args),
        "centers": np.asarray(centers, dtype=np.float32),
    }


def fit_window_tensor_normalizer(window_samples: Iterable[dict[str, Any]], args: Any | None = None) -> EvidenceNormalizer:
    samples = list(window_samples)
    views = [_prepared_views(sample, args=args) for sample in samples]
    return fit_normalizer((view["b0"] for view in views))


def build_patient_examples(
    window_samples: Sequence[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    *,
    normalizer: EvidenceNormalizer | None = None,
    subject_ids: Sequence[str] | None = None,
    args: Any | None = None,
) -> list[dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in window_samples:
        subject_id = str(sample["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        grouped[subject_id].append(sample)

    positive_label = str(getattr(args, "positive_label", "nez") if args is not None else "nez").lower()
    examples: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        if subject_id not in patient_index:
            continue
        patient_meta = patient_index[subject_id]
        canonical_channels = list(patient_meta["canonical_channels"])
        channel_to_idx = {name: idx for idx, name in enumerate(canonical_channels)}
        labels_ez = np.asarray(patient_meta["labels"], dtype=np.float32)
        labels_nez = np.where(labels_ez >= 0.0, 1.0 - labels_ez, -1.0).astype(np.float32, copy=False)
        labels = labels_nez if positive_label == "nez" else labels_ez
        channel_mask = np.asarray(patient_meta.get("label_mask", np.ones(len(canonical_channels), dtype=bool)), dtype=bool)
        num_patient_channels = len(canonical_channels)

        b0_items: list[np.ndarray] = []
        window_masks: list[np.ndarray] = []
        seizure_channel_masks: list[np.ndarray] = []
        run_ids: list[str] = []
        sample_ids: list[str] = []

        for sample in grouped[subject_id]:
            local_channels = list(sample["channel_names_norm"])
            local_to_patient = [channel_to_idx.get(channel_name) for channel_name in local_channels]
            views = _prepared_views(sample, args=args)
            b0 = normalizer.transform(views["b0"]) if normalizer is not None else views["b0"]

            t = int(b0.shape[0])
            aligned_b0 = np.zeros((t, num_patient_channels, b0.shape[-1]), dtype=np.float32)
            aligned_channel_mask = np.zeros((num_patient_channels,), dtype=bool)

            for local_idx, patient_idx in enumerate(local_to_patient):
                if patient_idx is None:
                    continue
                aligned_b0[:, patient_idx, :] = b0[:, local_idx, :]
                aligned_channel_mask[patient_idx] = True

            b0_items.append(aligned_b0)
            window_masks.append(np.ones((t,), dtype=bool))
            seizure_channel_masks.append(aligned_channel_mask)
            run_ids.append(str(sample["run_id"]))
            sample_ids.append(str(sample["sample_id"]))

        if not b0_items:
            continue
        examples.append(
            {
                "subject_id": subject_id,
                "canonical_channels": canonical_channels,
                "channel_meta": list(patient_meta.get("channel_meta", [])),
                "labels": labels,
                "labels_nez": labels_nez,
                "labels_ez": labels_ez,
                "label_semantics": "1=NEZ,0=EZ" if positive_label == "nez" else "1=EZ,0=NEZ",
                "channel_mask": channel_mask,
                "b0_features": b0_items,
                "features": b0_items,
                "window_mask": window_masks,
                "seizure_channel_mask": seizure_channel_masks,
                "run_ids": run_ids,
                "sample_ids": sample_ids,
            }
        )
    return examples


class PatientNeuroEZCDataset(Dataset):
    def __init__(self, patient_examples: Sequence[dict[str, Any]]) -> None:
        self.patient_examples = list(patient_examples)

    def __len__(self) -> int:
        return len(self.patient_examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.patient_examples[index]


def _allocate_view(batch: Sequence[dict[str, Any]], key: str, max_seizures: int, max_windows: int, max_channels: int) -> torch.Tensor:
    dim = max(int(item[key][s].shape[-1]) for item in batch for s in range(len(item[key])))
    return torch.zeros((len(batch), max_seizures, max_windows, max_channels, dim), dtype=torch.float32)


def collate_patient_ez_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("collate_patient_ez_batch received an empty batch.")

    batch_size = len(batch)
    max_seizures = max(len(item["b0_features"]) for item in batch)
    max_windows = max(int(arr.shape[0]) for item in batch for arr in item["b0_features"])
    max_channels = max(int(item["labels"].shape[0]) for item in batch)

    b0_features = _allocate_view(batch, "b0_features", max_seizures, max_windows, max_channels)
    labels = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_ez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    seizure_mask = torch.zeros((batch_size, max_seizures), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool)
    window_mask = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.bool)

    subject_ids: list[str] = []
    canonical_channels: list[list[str]] = []
    channel_meta: list[list[dict[str, Any]]] = []
    run_ids: list[list[str]] = []
    sample_ids: list[list[str]] = []
    label_semantics: list[str] = []

    for batch_idx, item in enumerate(batch):
        c = int(item["labels"].shape[0])
        labels[batch_idx, :c] = torch.as_tensor(item["labels"], dtype=torch.float32)
        labels_nez[batch_idx, :c] = torch.as_tensor(item["labels_nez"], dtype=torch.float32)
        labels_ez[batch_idx, :c] = torch.as_tensor(item["labels_ez"], dtype=torch.float32)
        channel_mask[batch_idx, :c] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)
        subject_ids.append(str(item["subject_id"]))
        canonical_channels.append(list(item["canonical_channels"]))
        channel_meta.append(list(item.get("channel_meta", [])))
        run_ids.append(list(item.get("run_ids", [])))
        sample_ids.append(list(item.get("sample_ids", [])))
        label_semantics.append(str(item.get("label_semantics", "1=NEZ,0=EZ")))

        for seizure_idx, b0 in enumerate(item["b0_features"]):
            mask = np.asarray(item["seizure_channel_mask"][seizure_idx], dtype=bool)
            win_mask = np.asarray(item["window_mask"][seizure_idx], dtype=bool)
            b0 = np.asarray(b0, dtype=np.float32)
            t, current_c = int(b0.shape[0]), int(b0.shape[1])
            b0_features[batch_idx, seizure_idx, :t, :current_c, : b0.shape[-1]] = torch.as_tensor(b0)
            seizure_channel_mask[batch_idx, seizure_idx, :current_c] = torch.as_tensor(mask, dtype=torch.bool)
            window_mask[batch_idx, seizure_idx, :t] = torch.as_tensor(win_mask, dtype=torch.bool)
            seizure_mask[batch_idx, seizure_idx] = True

    return {
        "features": b0_features,
        "b0_features": b0_features,
        "labels": labels,
        "labels_nez": labels_nez,
        "labels_ez": labels_ez,
        "channel_mask": channel_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_mask": window_mask,
        "subject_id": subject_ids,
        "canonical_channels": canonical_channels,
        "channel_meta": channel_meta,
        "run_ids": run_ids,
        "sample_ids": sample_ids,
        "label_semantics": label_semantics,
    }


__all__ = [
    "EvidenceNormalizer",
    "PatientNeuroEZCDataset",
    "build_patient_examples",
    "collate_patient_ez_batch",
    "fit_window_tensor_normalizer",
]
