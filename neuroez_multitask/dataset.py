from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PhysicsCacheDataset(Dataset):
    def __init__(self, cache_payload: dict[str, Any], subject_ids: set[str] | None = None) -> None:
        self.patient_index = cache_payload["patient_index"]
        self.outcome_index = cache_payload.get("outcome_index", {})
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for record in cache_payload["run_records"]:
            sid = str(record["subject_id"])
            if subject_ids is not None and sid not in subject_ids:
                continue
            by_subject.setdefault(sid, []).append(record)
        self.subject_ids = sorted(by_subject)
        self.by_subject = by_subject

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sid = self.subject_ids[index]
        patient = self.patient_index[sid]
        outcome = self.outcome_index.get(sid, {})
        runs = self.by_subject[sid]
        return {
            "subject_id": sid,
            "center": patient.get("center", ""),
            "runs": runs,
            "channel_names": list(patient["canonical_channels"]),
            "labels": np.asarray(patient["labels"], dtype=np.float32),
            "labels_nez": np.asarray(patient["labels_nez"], dtype=np.float32),
            "labels_ez": np.asarray(patient["labels_ez"], dtype=np.float32),
            "channel_mask": np.asarray(patient["label_mask"], dtype=bool),
            "outcome_label": -1.0 if outcome.get("success_failure") is None else float(outcome.get("success_failure")),
            "outcome_mask": outcome.get("success_failure") is not None,
            "topology_features": _patient_topology(runs),
        }


def _patient_topology(runs: Sequence[dict[str, Any]]) -> np.ndarray:
    values = []
    for run in runs:
        sample = run["sample"]
        topo = np.asarray(sample["topology_graph_features"], dtype=np.float32)
        values.append(topo.mean(axis=0) if topo.ndim == 2 else topo)
    if not values:
        return np.zeros((8,), dtype=np.float32)
    return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)


def collate_patient_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_s = max(len(item["runs"]) for item in batch)
    max_t = max(np.asarray(run["sample"]["window_features"]).shape[0] for item in batch for run in item["runs"])
    max_c = max(len(item["channel_names"]) for item in batch)
    f_b0 = np.asarray(batch[0]["runs"][0]["sample"]["window_features"]).shape[-1]
    f_phys = np.asarray(batch[0]["runs"][0]["sample"]["physics_node_features"]).shape[-1]
    f_causal = np.asarray(batch[0]["runs"][0]["sample"]["causal_node_features"]).shape[-1]
    f_topology = np.asarray(batch[0]["topology_features"]).shape[-1]

    b0 = torch.zeros((batch_size, max_s, max_t, max_c, f_b0), dtype=torch.float32)
    phys = torch.zeros((batch_size, max_s, max_t, max_c, f_phys), dtype=torch.float32)
    adj = torch.zeros((batch_size, max_s, max_t, max_c, max_c), dtype=torch.float32)
    delay = torch.zeros_like(adj)
    causal = torch.zeros((batch_size, max_s, max_t, max_c, f_causal), dtype=torch.float32)
    topology = torch.zeros((batch_size, f_topology), dtype=torch.float32)
    labels = torch.full((batch_size, max_c), -1.0, dtype=torch.float32)
    labels_nez = torch.full((batch_size, max_c), -1.0, dtype=torch.float32)
    labels_ez = torch.full((batch_size, max_c), -1.0, dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_c), dtype=torch.bool)
    outcome_label = torch.zeros((batch_size,), dtype=torch.float32)
    outcome_mask = torch.zeros((batch_size,), dtype=torch.bool)
    seizure_mask = torch.zeros((batch_size, max_s), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_s, max_c), dtype=torch.bool)
    window_mask = torch.zeros((batch_size, max_s, max_t), dtype=torch.bool)

    for bidx, item in enumerate(batch):
        c = len(item["channel_names"])
        labels[bidx, :c] = torch.as_tensor(item["labels"], dtype=torch.float32)
        labels_nez[bidx, :c] = torch.as_tensor(item["labels_nez"], dtype=torch.float32)
        labels_ez[bidx, :c] = torch.as_tensor(item["labels_ez"], dtype=torch.float32)
        channel_mask[bidx, :c] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)
        outcome_label[bidx] = float(item["outcome_label"])
        outcome_mask[bidx] = bool(item["outcome_mask"])
        topology[bidx] = torch.as_tensor(item["topology_features"], dtype=torch.float32)
        for sidx, run in enumerate(item["runs"]):
            sample = run["sample"]
            wf = np.asarray(sample["window_features"], dtype=np.float32)
            pf = np.asarray(sample["physics_node_features"], dtype=np.float32)
            ca = np.asarray(sample["tfccm_adjacency"], dtype=np.float32)
            cd = np.asarray(sample["tfccm_delay"], dtype=np.float32)
            cn = np.asarray(sample["causal_node_features"], dtype=np.float32)
            wm = np.asarray(sample["window_mask"], dtype=bool)
            t = wf.shape[0]
            b0[bidx, sidx, :t, :c] = torch.as_tensor(wf)
            phys[bidx, sidx, :t, :c] = torch.as_tensor(pf)
            adj[bidx, sidx, :t, :c, :c] = torch.as_tensor(ca)
            delay[bidx, sidx, :t, :c, :c] = torch.as_tensor(cd)
            causal[bidx, sidx, :t, :c] = torch.as_tensor(cn)
            seizure_mask[bidx, sidx] = True
            seizure_channel_mask[bidx, sidx, :c] = True
            window_mask[bidx, sidx, :t] = torch.as_tensor(wm, dtype=torch.bool)

    return {
        "b0_features": b0,
        "physics_features": phys,
        "causal_adjacency": adj,
        "causal_delay": delay,
        "causal_node_features": causal,
        "topology_features": topology,
        "labels": labels,
        "labels_nez": labels_nez,
        "labels_ez": labels_ez,
        "channel_mask": channel_mask,
        "outcome_label": outcome_label,
        "outcome_mask": outcome_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_mask": window_mask,
        "subject_id": [item["subject_id"] for item in batch],
        "center": [item["center"] for item in batch],
        "channel_names": [item["channel_names"] for item in batch],
    }
