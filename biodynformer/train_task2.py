from __future__ import annotations

from typing import Any, Sequence

from .datasets import build_task_examples, task2_xy
from .evaluate import summarize_task2
from .models import LogisticModel, fit_logistic_regression
import numpy as np


def train_and_evaluate_task2(
    index: Sequence[dict[str, Any]],
    *,
    version: str,
    train_subjects: set[str],
    test_subjects: set[str],
    mode: str = "full",
    learning_rate: float = 0.05,
    epochs: int = 200,
) -> dict[str, Any]:
    train_examples = build_task_examples(index, task="task2", version=version, subjects=train_subjects)
    test_examples = build_task_examples(index, task="task2", version=version, subjects=test_subjects)
    x_train, y_train, _ = task2_xy(train_examples, mode=mode)
    x_test, y_test, _ = task2_xy(test_examples, mode=mode)
    if x_train.shape[0] == 0 and x_test.ndim == 2:
        dim = x_test.shape[1]
        model = LogisticModel(np.zeros((dim,), dtype=np.float32), 0.0, np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32))
    else:
        model = fit_logistic_regression(x_train, y_train, learning_rate=learning_rate, epochs=epochs)
    prob = model.predict_proba(x_test) if x_test.size else []
    return {"metrics": summarize_task2(y_test, prob), "model": model.to_dict(), "num_train_examples": len(train_examples), "num_test_examples": len(test_examples), "mode": mode}
