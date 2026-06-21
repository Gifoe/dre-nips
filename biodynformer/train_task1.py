from __future__ import annotations

from typing import Any, Sequence

from .datasets import build_task_examples, task1_xy
from .evaluate import summarize_task1
from .models import LogisticModel, fit_logistic_regression
import numpy as np


def train_and_evaluate_task1(
    index: Sequence[dict[str, Any]],
    *,
    version: str,
    train_subjects: set[str],
    test_subjects: set[str],
    learning_rate: float = 0.05,
    epochs: int = 200,
) -> dict[str, Any]:
    train_examples = build_task_examples(index, task="task1", version=version, subjects=train_subjects)
    test_examples = build_task_examples(index, task="task1", version=version, subjects=test_subjects)
    x_train, y_train, _ = task1_xy(train_examples)
    x_test, y_test, test_subject_rows = task1_xy(test_examples)
    if x_train.shape[0] == 0 and x_test.ndim == 2:
        dim = x_test.shape[1]
        model = LogisticModel(np.zeros((dim,), dtype=np.float32), 0.0, np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32))
    else:
        model = fit_logistic_regression(x_train, y_train, learning_rate=learning_rate, epochs=epochs)
    prob = model.predict_proba(x_test) if x_test.size else []
    metrics = summarize_task1(y_test, prob, test_subject_rows)
    return {"metrics": metrics, "model": model.to_dict(), "num_train_examples": len(train_examples), "num_test_examples": len(test_examples)}
