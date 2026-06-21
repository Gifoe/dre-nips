from biodynformer.datasets import build_task_examples


def test_task1_uses_only_successful_patients(small_feature_bank):
    examples = build_task_examples(small_feature_bank, task="task1", version="v1")

    assert examples
    assert all(example["outcome_success"] is True for example in examples)


def test_task2_requires_outcome_labels(small_feature_bank):
    examples = build_task_examples(small_feature_bank, task="task2", version="v1")

    assert examples
    assert all("outcome_success" in example for example in examples)
