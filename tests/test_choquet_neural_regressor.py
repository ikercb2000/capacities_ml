import numpy as np
from sklearn.base import clone

from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquetNeuralRegressor


def test_choquet_neural_regressor_learns_valid_hidden_capacities(
    binary_classification_sample,
):
    X, _ = binary_classification_sample
    y = X[:, 0] + X[:, 1]
    model = ChoquetNeuralRegressor(
        universe=VariableUniverse(("x0", "x1")),
        n_hidden=2,
        max_iter=200,
        random_state=1,
    ).fit(X, y)

    assert np.mean((model.predict(X) - y) ** 2) < 0.01
    assert len(model.capacities_) == 2
    for capacity in model.capacities_:
        capacity.validate()
    assert clone(model).n_hidden == 2
