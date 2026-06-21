from biodynformer.feature_bank import load_feature_bank_index
from biodynformer.train_task1 import train_and_evaluate_task1


def test_task1_empty_train_fold_does_not_crash(small_feature_bank):
    index = small_feature_bank
    first_success = next(row["subject_id"] for row in index if row["outcome_success"] is True)

    result = train_and_evaluate_task1(
        index,
        version="v1",
        train_subjects=set(),
        test_subjects={first_success},
        epochs=1,
    )

    assert result["num_train_examples"] == 0
    assert result["num_test_examples"] == 1
    assert "metrics" in result
