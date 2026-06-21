import numpy as np

from biodynformer.models import fit_logistic_regression


def test_empty_training_matrix_uses_predictable_constant_model_shape():
    model = fit_logistic_regression(np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    prob = model.predict_proba(np.zeros((3, 7), dtype=np.float32))

    assert prob.shape == (3,)
    assert np.allclose(prob, 0.5)
