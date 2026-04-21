from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_predictions(scores: np.ndarray, tau: float) -> np.ndarray:
    pred_mask = np.asarray(scores >= float(tau), dtype=bool)
    if pred_mask.size > 0 and not pred_mask.any():
        pred_mask[int(np.argmax(scores))] = True
    return pred_mask


def compute_patient_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    pred_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    labels_bool = np.asarray(labels, dtype=bool)
    pred_bool = np.asarray(pred_mask, dtype=bool)

    if len(np.unique(labels_bool.astype(int))) > 1:
        auc_val: Optional[float] = float(roc_auc_score(labels_bool, scores))
        auc_pr_val: Optional[float] = float(average_precision_score(labels_bool, scores))
    else:
        auc_val = None
        auc_pr_val = None

    acc = float(accuracy_score(labels_bool, pred_bool))
    prec = float(precision_score(labels_bool, pred_bool, zero_division=0))
    rec = float(recall_score(labels_bool, pred_bool, zero_division=0))
    f1 = float(f1_score(labels_bool, pred_bool, zero_division=0))
    mcc = float(matthews_corrcoef(labels_bool, pred_bool))

    cm = confusion_matrix(labels_bool, pred_bool, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp + 1e-8))
    npv = float(tn / (tn + fn + 1e-8))

    return {
        "n_true_ez": int(labels_bool.sum()),
        "n_pred_ez": int(pred_bool.sum()),
        "ACC": acc,
        "PREC": prec,
        "REC": rec,
        "SPEC": spec,
        "NPV": npv,
        "F1": f1,
        "AUC": auc_val,
        "AUC_PR": auc_pr_val,
        "MCC": mcc,
    }


def select_best_threshold(
    patient_outputs: Iterable[Dict[str, np.ndarray]],
    candidate_taus: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    patient_outputs = list(patient_outputs)
    if not patient_outputs:
        raise ValueError("select_best_threshold received no patient outputs.")

    if candidate_taus is None:
        candidate_taus = np.linspace(0.20, 0.80, 25)

    best_tau = float(candidate_taus[0])
    best_score = -1e9
    best_summary: Dict[str, float] = {}

    for tau in candidate_taus:
        metric_list = []
        auc_pr_values = []
        for item in patient_outputs:
            pred_mask = threshold_predictions(item["scores"], float(tau))
            metrics = compute_patient_metrics(item["labels"], item["scores"], pred_mask)
            metric_list.append(metrics)
            if metrics["AUC_PR"] is not None:
                auc_pr_values.append(float(metrics["AUC_PR"]))

        macro_f1 = float(np.mean([m["F1"] for m in metric_list]))
        macro_recall = float(np.mean([m["REC"] for m in metric_list]))
        macro_precision = float(np.mean([m["PREC"] for m in metric_list]))
        macro_auc_pr = float(np.mean(auc_pr_values)) if auc_pr_values else 0.0
        score = macro_f1 + 0.2 * macro_recall + 0.1 * macro_precision + 0.2 * macro_auc_pr

        if score > best_score:
            best_score = score
            best_tau = float(tau)
            best_summary = {
                "tau": best_tau,
                "macro_f1": macro_f1,
                "macro_recall": macro_recall,
                "macro_precision": macro_precision,
                "macro_auc_pr": macro_auc_pr,
            }

    return best_tau, best_summary


def build_patient_prediction(
    patient_output: Dict[str, np.ndarray],
    tau: float,
) -> Dict[str, object]:
    scores = np.asarray(patient_output["scores"], dtype=np.float32)
    labels = np.asarray(patient_output["labels"], dtype=np.float32)
    pred_mask = threshold_predictions(scores, tau)
    metrics = compute_patient_metrics(labels, scores, pred_mask)

    channels = list(patient_output["canonical_channels"])
    true_ez = [channels[idx] for idx, value in enumerate(labels) if value == 1.0]
    pred_ez = [channels[idx] for idx, value in enumerate(pred_mask) if value]
    tp = [channels[idx] for idx, value in enumerate(pred_mask & (labels == 1.0)) if value]
    fp = [channels[idx] for idx, value in enumerate(pred_mask & (labels == 0.0)) if value]
    fn = [channels[idx] for idx, value in enumerate((~pred_mask) & (labels == 1.0)) if value]

    channel_rows = pd.DataFrame(
        {
            "channel_name_norm": channels,
            "patient_score": scores,
            "is_true_ez": labels.astype(int),
            "predicted_ez": pred_mask.astype(int),
        }
    ).sort_values("patient_score", ascending=False)
    channel_rows["rank"] = np.arange(1, len(channel_rows) + 1)

    return {
        "subject_id": patient_output["subject_id"],
        "fold_idx": int(patient_output["fold_idx"]),
        "tau": float(tau),
        "metrics": metrics,
        "true_ez": true_ez,
        "predicted_ez": pred_ez,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "channel_table": channel_rows,
        "run_summaries": patient_output.get("run_summaries", []),
    }


def save_patient_report(prediction: Dict[str, object], output_dir: str) -> None:
    output_root = Path(output_dir)
    patient_dir = output_root / "per_patient" / str(prediction["subject_id"])
    patient_dir.mkdir(parents=True, exist_ok=True)

    report_json = {
        "subject_id": prediction["subject_id"],
        "fold_idx": int(prediction["fold_idx"]),
        "tau": float(prediction["tau"]),
        "metrics": prediction["metrics"],
        "results": {
            "true_ez": prediction["true_ez"],
            "predicted_ez": prediction["predicted_ez"],
            "true_positive": prediction["true_positive"],
            "false_positive": prediction["false_positive"],
            "false_negative": prediction["false_negative"],
        },
        "run_summaries": prediction["run_summaries"],
    }
    with open(patient_dir / "patient_report_threshold.json", "w", encoding="utf-8") as fout:
        json.dump(report_json, fout, indent=2)

    prediction["channel_table"].to_csv(patient_dir / "channel_scores_threshold.csv", index=False)


def summarize_cv_results(predictions: List[Dict[str, object]], output_dir: str) -> Dict[str, object]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for prediction in predictions:
        save_patient_report(prediction, output_dir)
        row = {"subject_id": prediction["subject_id"], "fold_idx": prediction["fold_idx"], "tau": prediction["tau"]}
        row.update(prediction["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "channel_results_threshold_all_folds.csv", index=False)

    summary = {}
    if not df.empty:
        means = df.mean(numeric_only=True)
        stds = df.std(numeric_only=True)
        summary = {
            f"{column}_mean": float(value)
            for column, value in means.to_dict().items()
        }
        summary.update(
            {
                f"{column}_std": float(value)
                for column, value in stds.to_dict().items()
            }
        )

    combined = {
        "n_patients": int(len(predictions)),
        "selection_strategy": "patient_level_threshold_with_top1_fallback",
        "summary_metrics": summary,
    }
    with open(output_root / "summary_metrics_threshold.json", "w", encoding="utf-8") as fout:
        json.dump(combined, fout, indent=2)
    return combined


__all__ = [
    "build_patient_prediction",
    "compute_patient_metrics",
    "select_best_threshold",
    "summarize_cv_results",
    "threshold_predictions",
]
