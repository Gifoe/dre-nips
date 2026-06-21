from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuroez_multitask.dataset import PhysicsCacheDataset, collate_patient_batch
from neuroez_multitask.metrics import summarize_task1_predictions, summarize_task2_predictions
from neuroez_multitask.model import PGCSEEGModel
from neuroez_multitask.splits import make_patient_splits
from neuroez_multitask.train_task1 import task1_loss, task1_prediction_rows
from neuroez_multitask.train_task2 import task2_loss
from run_task1_pgc_ez import _model_kwargs


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


def _load_checkpoint(model: PGCSEEGModel, path: Path | None, device: torch.device) -> None:
    if path is None or not path.exists():
        return
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)


def _evaluate(model: PGCSEEGModel, loader: DataLoader, device: torch.device, fold: int, cache: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    task1_records = []
    task1_rows = []
    outcome_labels = []
    outcome_probs = []
    task2_rows = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(tensor_batch)
            nez_prob = outputs["nez_prob"].cpu().numpy()
            labels = batch["labels_nez"].cpu().numpy()
            masks = batch["channel_mask"].cpu().numpy()
            outcome_prob = outputs["outcome_prob"].cpu().numpy()
            outcome_label = batch["outcome_label"].cpu().numpy()
            outcome_mask = batch["outcome_mask"].cpu().numpy()
            for idx, sid in enumerate(batch["subject_id"]):
                c = len(batch["channel_names"][idx])
                task1_records.append(
                    {
                        "subject_id": sid,
                        "labels_nez": labels[idx, :c],
                        "nez_prob": nez_prob[idx, :c],
                        "channel_mask": masks[idx, :c],
                    }
                )
                task1_rows.extend(
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
                if outcome_mask[idx]:
                    outcome_labels.append(float(outcome_label[idx]))
                    outcome_probs.append(float(outcome_prob[idx]))
                    outcome = cache.get("outcome_index", {}).get(sid, {})
                    task2_rows.append(
                        {
                            "subject_id": sid,
                            "Engel": outcome.get("Engel"),
                            "success_failure": float(outcome_label[idx]),
                            "outcome_prob": float(outcome_prob[idx]),
                            "fold": fold,
                            "center": batch["center"][idx],
                        }
                    )
    metrics = {
        **{f"task1_{key}": value for key, value in summarize_task1_predictions(task1_records).items()},
        **{f"task2_{key}": value for key, value in summarize_task2_predictions(outcome_labels, outcome_probs).items()},
    }
    return metrics, task1_rows, task2_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint fine-tune PGC-SEEG with L_task1 + lambda_outcome * L_task2.")
    parser.add_argument("--window_cache_path", type=Path, required=True)
    parser.add_argument("--task1_checkpoint", type=Path, default=None)
    parser.add_argument("--task2_checkpoint", type=Path, default=None)
    parser.add_argument("--experiment_name", type=str, default="T1_FULL_PGC")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_strategy", type=str, default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_outcome", type=float, default=0.2)
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
    task1_rows: list[dict[str, Any]] = []
    task2_rows: list[dict[str, Any]] = []
    best_state = None
    best_score = -1.0
    for split in splits:
        train_ds = PhysicsCacheDataset(cache, set(split.train_subjects))
        test_ds = PhysicsCacheDataset(cache, set(split.test_subjects))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patient_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        model = PGCSEEGModel(**_model_kwargs(args.experiment_name, args.model_dim)).to(device)
        _load_checkpoint(model, args.task1_checkpoint, device)
        _load_checkpoint(model, args.task2_checkpoint, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for _ in range(max(args.epochs, 0)):
            model.train()
            for batch in train_loader:
                tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(tensor_batch)
                loss = task1_loss(outputs, tensor_batch) + float(args.lambda_outcome) * task2_loss(outputs, tensor_batch)
                loss.backward()
                optimizer.step()
        metrics, rows1, rows2 = _evaluate(model, test_loader, device, split.fold, cache)
        fold_rows.append({"fold": split.fold, **metrics})
        task1_rows.extend(rows1)
        task2_rows.extend(rows2)
        score = float(metrics.get("task1_AUPRC", 0.0)) + float(args.lambda_outcome) * float(metrics.get("task2_AUPRC", 0.0))
        if score > best_score:
            best_score = score
            best_state = model.state_dict()
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "task1_patient_predictions.csv", task1_rows)
    _write_csv(args.output_dir / "task2_patient_predictions.csv", task2_rows)
    summary = {
        "num_task1_rows": len(task1_rows),
        "num_task2_rows": len(task2_rows),
        "lambda_outcome": float(args.lambda_outcome),
    }
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_checkpoint.pt")


if __name__ == "__main__":
    main()
