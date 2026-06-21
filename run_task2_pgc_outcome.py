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
from neuroez_multitask.metrics import summarize_task2_predictions
from neuroez_multitask.model import PGCSEEGModel
from neuroez_multitask.splits import make_patient_splits
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


def _freeze_backbone(model: PGCSEEGModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("outcome_head")


def _evaluate(model: PGCSEEGModel, loader: DataLoader, device: torch.device, fold: int, cache: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels = []
    probs = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(tensor_batch)
            outcome_prob = outputs["outcome_prob"].cpu().numpy()
            outcome_label = batch["outcome_label"].cpu().numpy()
            outcome_mask = batch["outcome_mask"].cpu().numpy()
            for idx, sid in enumerate(batch["subject_id"]):
                if not outcome_mask[idx]:
                    continue
                labels.append(float(outcome_label[idx]))
                probs.append(float(outcome_prob[idx]))
                outcome = cache.get("outcome_index", {}).get(sid, {})
                rows.append(
                    {
                        "subject_id": sid,
                        "Engel": outcome.get("Engel"),
                        "success_failure": float(outcome_label[idx]),
                        "outcome_prob": float(outcome_prob[idx]),
                        "fold": fold,
                        "center": batch["center"][idx],
                    }
                )
    return summarize_task2_predictions(labels, probs), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task2 PGC-SEEG outcome model.")
    parser.add_argument("--window_cache_path", type=Path, required=True)
    parser.add_argument("--task1_checkpoint", type=Path, default=None)
    parser.add_argument("--experiment_name", type=str, default="T2_FULL_ATTENTION_TOPOLOGY")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_strategy", type=str, default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
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
        train_ds = PhysicsCacheDataset(cache, set(split.train_subjects))
        test_ds = PhysicsCacheDataset(cache, set(split.test_subjects))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patient_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        model = PGCSEEGModel(**_model_kwargs(args.experiment_name, args.model_dim)).to(device)
        if args.task1_checkpoint and args.task1_checkpoint.exists():
            payload = torch.load(args.task1_checkpoint, map_location=device)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=False)
        if args.freeze_backbone:
            _freeze_backbone(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        for _ in range(max(args.epochs, 0)):
            model.train()
            for batch in train_loader:
                tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = task2_loss(model(tensor_batch), tensor_batch)
                loss.backward()
                optimizer.step()
        metrics, rows = _evaluate(model, test_loader, device, split.fold, cache)
        fold_rows.append({"fold": split.fold, **metrics})
        pred_rows.extend(rows)
        score = float(metrics.get("AUPRC", 0.0))
        if score > best_score:
            best_score = score
            best_state = model.state_dict()
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "patient_predictions.csv", pred_rows)
    summary = summarize_task2_predictions([row["success_failure"] for row in pred_rows], [row["outcome_prob"] for row in pred_rows])
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_checkpoint.pt")
        torch.save({"model_state_dict": best_state, "experiment_name": args.experiment_name}, args.output_dir / "best_task2_outcome.pt")


if __name__ == "__main__":
    main()
