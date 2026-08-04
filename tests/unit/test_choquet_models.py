import numpy as np

from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquetClassifier
from capacities_ml.optimization import Solver
from capacities_ml.models import ChoquetRegressor


def test_choquet_regressor_fits_capacity_and_intercept_with_cvxpy():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.5, 1.0, 1.0, 1.5])

    model = ChoquetRegressor(
        universe=VariableUniverse(("x0", "x1")),
        solver=Solver.CVXPY,
    ).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)
    assert model.problem_.parameter_layout.slice("intercept") == slice(4, 5)


def test_choquet_classifier_optimizes_capacity_and_threshold():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = ChoquetClassifier(
        universe=VariableUniverse(("x0", "x1")),
        solver=Solver.PYMOO,
        solver_options={
            "population_size": 30,
            "n_generations": 20,
            "seed": 4,
        },
    ).fit(X, y)

    assert model.result_.success
    assert 0.0 <= model.threshold_ <= 1.0
    assert model.problem_.parameter_layout.slice("threshold") == slice(4, 5)
