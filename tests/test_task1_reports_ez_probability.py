from __future__ import annotations

import numpy as np

from neuroez_multitask.metrics import summarize_task1_predictions


def test_task1_summary_reports_ez_positive_metrics_only():
    records = [
        {
            "subject_id": "p1",
            "labels_nez": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "nez_prob": np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
            "channel_mask": np.asarray([True, True, True, True]),
        }
    ]

    summary = summarize_task1_predictions(records)

    assert summary["internal_positive_class"] == "NEZ"
    assert summary["reported_positive_class"] == "EZ"
    assert summary["task1_probability_reported"] == "P(EZ)=1-P(NEZ)"
    assert "AUROC" in summary
    assert "AUPRC" in summary
    assert "auroc_nez" not in summary
    assert "auroc_ez" not in summary
    assert "ezauc" not in summary
    assert "nezauc" not in summary
