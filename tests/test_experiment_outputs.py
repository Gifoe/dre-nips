from __future__ import annotations

from neuroez_multitask.train_task1 import task1_prediction_rows


def test_task1_prediction_rows_include_nez_and_ez_probabilities():
    rows = task1_prediction_rows(
        subject_id="p1",
        center="lzu",
        fold=0,
        channel_names=["A1"],
        labels_nez=[0.0],
        nez_prob=[0.25],
        channel_mask=[True],
    )

    assert rows == [
        {
            "subject_id": "p1",
            "channel_name": "A1",
            "label_nez": 0.0,
            "label_ez": 1.0,
            "nez_prob": 0.25,
            "ez_prob": 0.75,
            "fold": 0,
            "center": "lzu",
        }
    ]
