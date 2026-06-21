from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from torch.utils.data import DataLoader

from neuroez_multitask.dataset import PhysicsCacheDataset, collate_patient_batch
from neuroez_multitask.metrics import summarize_task1_predictions
from neuroez_multitask.model import PGCSEEGModel
from neuroez_multitask.normalization import fit_multiview_normalizer
from neuroez_multitask.splits import make_patient_splits
from neuroez_multitask.train_task1 import task1_loss, task1_prediction_rows


def _model_kwargs(experiment_name: str, model_dim: int) -> dict[str, Any]:
    name = experiment_name.upper()
    kwargs: dict[str, Any] = {"model_dim": model_dim}
    if name in {"T1_B0_BASELINE", "T2_B0_GLOBAL"}:
        kwargs.update(
            use_physics_branch=False,
            use_causal_graph=False,
            use_causal_node_features=False,
            use_delay=False,
            outcome_readout_type="global",
            use_topology_features=False,
        )
    elif name == "T1_B0_PHYS_GATED":
        kwargs.update(use_physics_branch=True, use_causal_graph=False, use_causal_node_features=False, use_delay=False)
    elif name == "T1_B0_TFCCM_NODE":
        kwargs.update(use_physics_branch=False, use_causal_graph=False, use_causal_node_features=True, use_delay=False)
    elif name == "T1_B0_TFCCM_GRAPH_NO_DELAY":
        kwargs.update(use_physics_branch=False, use_causal_graph=True, use_causal_node_features=False, use_delay=False)
    elif name == "T1_B0_TFCCM_GRAPH_DELAY":
        kwargs.update(use_physics_branch=False, use_causal_graph=True, use_causal_node_features=False, use_delay=True)
    elif name == "T2_FULL_GLOBAL":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="global",
            use_topology_features=False,
        )
    elif name == "T2_FULL_ATTENTION":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="attention",
            use_topology_features=False,
        )
    else:
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="attention",
            use_topology_features=True,
        )
    return kwargs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "empty"}]
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, (UninitializedParameter, UninitializedBuffer)):
            continue
        try:
            state[key] = value.detach().cpu().clone()
        except ValueError as exc:
            if "uninitialized" in str(exc).lower():
                continue
            raise
    return state


def _evaluate(model: PGCSEEGModel, loader: DataLoader, device: torch.device, fold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(tensor_batch)
            nez_prob = outputs["nez_prob"].cpu().numpy()
            labels = batch["labels_nez"].cpu().numpy()
            masks = batch["channel_mask"].cpu().numpy()
            for idx, sid in enumerate(batch["subject_id"]):
                c = len(batch["channel_names"][idx])
                records.append(
                    {
                        "subject_id": sid,
                        "labels_nez": labels[idx, :c],
                        "nez_prob": nez_prob[idx, :c],
                        "channel_mask": masks[idx, :c],
                    }
                )
                rows.extend(
                    task1_prediction_rows(
                        subject_id=sid,
                        center=batch["center"][idx],
                        fold=fold,
                        channel_names=batch["channel_names"][idx],
                        labels_nez=labels[idx, :c],
                        nez_prob=nez_prob[idx, :c],
                        channel_mask=masks[idx, :c],
                    )
                )
    return summarize_task1_predictions(records), rows


def _metric_score(metrics: dict[str, Any]) -> float:
    return float(metrics.get("AUPRC", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task1 PGC-SEEG EZ/NEZ model with NEZ-positive internal labels.")
    parser.add_argument("--window_cache_path", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="T1_FULL_PGC")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_strategy", type=str, default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    with open(args.window_cache_path, "rb") as fin:
        cache = pickle.load(fin)
    splits = make_patient_splits(cache["patient_index"], strategy=args.split_strategy, n_splits=args.n_splits, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    pred_rows: list[dict[str, Any]] = []
    best_state = None
    best_score = -1.0
    for split in splits:
        raw_train_ds = PhysicsCacheDataset(cache, set(split.train_subjects))
        normalizer = fit_multiview_normalizer(raw_train_ds)
        train_ds = PhysicsCacheDataset(cache, set(split.train_subjects), normalizer=normalizer)
        val_ds = PhysicsCacheDataset(cache, set(split.val_subjects), normalizer=normalizer)
        test_ds = PhysicsCacheDataset(cache, set(split.test_subjects), normalizer=normalizer)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patient_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        model = PGCSEEGModel(**_model_kwargs(args.experiment_name, args.model_dim)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best_fold_state = None
        best_fold_score = -1.0
        bad_epochs = 0
        selection_loader = val_loader if len(val_ds) > 0 else train_loader
        for _ in range(max(args.epochs, 0)):
            model.train()
            for batch in train_loader:
                tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = task1_loss(model(tensor_batch), tensor_batch)
                loss.backward()
                optimizer.step()
            val_metrics, _ = _evaluate(model, selection_loader, device, split.fold)
            val_score = _metric_score(val_metrics)
            if val_score > best_fold_score:
                best_fold_score = val_score
                best_fold_state = _checkpoint_state_dict(model)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= max(int(args.patience), 1):
                    break
        if best_fold_state is not None:
            model.load_state_dict(best_fold_state, strict=False)
        metrics, rows = _evaluate(model, test_loader, device, split.fold)
        fold_rows.append({"fold": split.fold, **metrics})
        pred_rows.extend(rows)
        score = best_fold_score if best_fold_score >= 0.0 else _metric_score(metrics)
        if score > best_score:
            best_score = score
            best_state = _checkpoint_state_dict(model)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "patient_predictions.csv", pred_rows)
    summary = {k: v for k, v in summarize_task1_predictions([
        {
            "subject_id": row["subject_id"],
            "labels_nez": np.asarray([row["label_nez"]], dtype=np.float32),
            "nez_prob": np.asarray([row["nez_prob"]], dtype=np.float32),
            "channel_mask": np.asarray([True]),
        }
        for row in pred_rows
    ]).items()}
    summary.update(
        {
            "internal_positive_class": "NEZ",
            "reported_positive_class": "EZ",
            "task1_probability_reported": "P(EZ)=1-P(NEZ)",
        }
    )
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_checkpoint.pt")
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_task1_backbone.pt")
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_task1_full.pt")


if __name__ == "__main__":
    main()
