import numpy as np

from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquetNeuralClassifier


def test_choquet_neural_classifier_returns_calibrated_shape_and_labels(
    binary_classification_sample,
):
    X, y = binary_classification_sample
    model = ChoquetNeuralClassifier(
        universe=VariableUniverse(("x0", "x1")),
        n_hidden=2,
        max_iter=200,
        random_state=2,
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(X)) == {"negative", "positive"}
    for capacity in model.capacities_:
        capacity.validate()
