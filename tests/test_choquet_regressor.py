import numpy as np

from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquetRegressor
from capacities_ml.optimization import Solver


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
