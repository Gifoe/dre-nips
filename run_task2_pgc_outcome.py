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
from neuroez_multitask.experiments import model_kwargs_for_experiment
from neuroez_multitask.metrics import summarize_task2_predictions
from neuroez_multitask.model import PGCSEEGModel
from neuroez_multitask.normalization import fit_multiview_normalizer
from neuroez_multitask.splits import PatientSplit, make_patient_splits
from neuroez_multitask.train_task2 import estimate_task2_pos_weight, task2_loss
from run_task1_pgc_ez import _checkpoint_state_dict, _write_split_metadata


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


def _torch_load_payload(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_fold_task1_payload(
    *,
    checkpoint_dir: Path | None,
    single_checkpoint: Path | None,
    allow_external: bool,
    split: PatientSplit,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_dir is not None:
        fold_ckpt = checkpoint_dir / f"fold_{split.fold}" / "best_task1_backbone.pt"
        if not fold_ckpt.exists():
            raise FileNotFoundError(f"Missing fold-specific Task1 checkpoint: {fold_ckpt}")
        payload = _torch_load_payload(fold_ckpt, device)
        if not isinstance(payload, dict):
            payload = {"model_state_dict": payload, "checkpoint_scope": "legacy_fold_specific"}

        task1_train = set(map(str, payload.get("train_subjects", [])))
        task1_test = set(map(str, payload.get("test_subjects", [])))
        task2_test = set(map(str, split.test_subjects))
        leakage = task1_train & task2_test
        if leakage:
            raise RuntimeError(
                "Task1 checkpoint leakage detected. "
                f"Task2 test subjects appear in Task1 train subjects: {sorted(leakage)}"
            )
        if task1_test and task1_test != task2_test:
            raise RuntimeError(
                "Task1 and Task2 folds are not aligned. "
                f"Task1 test={sorted(task1_test)}, Task2 test={sorted(task2_test)}"
            )
        if payload.get("safe_for_task2_fold_loading") is False:
            raise RuntimeError("This Task1 checkpoint is marked unsafe for Task2 fold loading.")
        return payload

    if single_checkpoint is not None:
        if not allow_external:
            raise RuntimeError(
                "Single --task1_checkpoint is unsafe for fold evaluation. "
                "Use --task1_checkpoint_dir with fold-specific checkpoints. "
                "Use --allow_external_task1_checkpoint only for external-cohort pretraining/debug."
            )
        if not single_checkpoint.exists():
            raise FileNotFoundError(f"Missing Task1 checkpoint: {single_checkpoint}")
        payload = _torch_load_payload(single_checkpoint, device)
        if isinstance(payload, dict):
            return payload
        return {"model_state_dict": payload, "checkpoint_scope": "external_or_debug"}

    return None


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


def _metric_score(metrics: dict[str, float]) -> float:
    return float(metrics.get("AUPRC", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task2 PGC-SEEG outcome model.")
    parser.add_argument("--window_cache_path", type=Path, required=True)
    parser.add_argument("--task1_checkpoint", type=Path, default=None)
    parser.add_argument("--task1_checkpoint_dir", type=Path, default=None)
    parser.add_argument("--allow_external_task1_checkpoint", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="T2_FULL_ATTENTION_TOPOLOGY")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_strategy", type=str, default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    with open(args.window_cache_path, "rb") as fin:
        cache = pickle.load(fin)
    splits = make_patient_splits(cache["patient_index"], strategy=args.split_strategy, n_splits=args.n_splits, seed=args.seed)
    _write_split_metadata(args.output_dir, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    pred_rows: list[dict[str, Any]] = []
    best_state = None
    best_score = -1.0
    for split in splits:
        task1_payload = _load_fold_task1_payload(
            checkpoint_dir=args.task1_checkpoint_dir,
            single_checkpoint=args.task1_checkpoint,
            allow_external=args.allow_external_task1_checkpoint,
            split=split,
            device=device,
        )
        raw_train_ds = PhysicsCacheDataset(cache, set(split.train_subjects))
        if task1_payload is not None and task1_payload.get("normalizer") is not None:
            normalizer = task1_payload["normalizer"]
        else:
            normalizer = fit_multiview_normalizer(raw_train_ds)
        train_ds = PhysicsCacheDataset(cache, set(split.train_subjects), normalizer=normalizer)
        val_ds = PhysicsCacheDataset(cache, set(split.val_subjects), normalizer=normalizer)
        test_ds = PhysicsCacheDataset(cache, set(split.test_subjects), normalizer=normalizer)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patient_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patient_batch)
        model_kwargs = model_kwargs_for_experiment(args.experiment_name, args.model_dim)
        model = PGCSEEGModel(**model_kwargs).to(device)
        if task1_payload is not None:
            state = task1_payload.get("model_state_dict", task1_payload)
            model.load_state_dict(state, strict=False)
        if args.freeze_backbone:
            _freeze_backbone(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        pos_weight = estimate_task2_pos_weight(cache, split.train_subjects).to(device)
        best_fold_state = None
        best_fold_score = -1.0
        bad_epochs = 0
        selection_loader = val_loader if len(val_ds) > 0 else train_loader
        for _ in range(max(args.epochs, 0)):
            model.train()
            for batch in train_loader:
                tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = task2_loss(model(tensor_batch), tensor_batch, pos_weight=pos_weight)
                loss.backward()
                optimizer.step()
            val_metrics, _ = _evaluate(model, selection_loader, device, split.fold, cache)
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
        metrics, rows = _evaluate(model, test_loader, device, split.fold, cache)
        fold_rows.append({"fold": split.fold, **metrics})
        pred_rows.extend(rows)
        score = best_fold_score if best_fold_score >= 0.0 else _metric_score(metrics)
        if score > best_score:
            best_score = score
            best_state = _checkpoint_state_dict(model)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "patient_predictions.csv", pred_rows)
    summary = summarize_task2_predictions([row["success_failure"] for row in pred_rows], [row["outcome_prob"] for row in pred_rows])
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_state is not None:
        payload = {
            "model_state_dict": best_state,
            "experiment_name": args.experiment_name,
            "model_kwargs": model_kwargs_for_experiment(args.experiment_name, args.model_dim),
            "checkpoint_scope": "task2_global_best",
        }
        torch.save(payload, args.output_dir / "best_checkpoint.pt")
        torch.save(payload, args.output_dir / "best_task2_outcome.pt")


if __name__ == "__main__":
    main()
