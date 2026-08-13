import numpy as np

from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import L2Penalty, Solver


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

    penalty = L2Penalty(weight=0.0, selection=[0])
    model = ChoquetRegressor(
        universe=VariableUniverse(("x0", "x1")),
        solver=Solver.CVXPY,
        penalty=penalty,
    ).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)
    assert model.problem_.parameter_layout.slice("intercept") == slice(4, 5)
    assert model.problem_.objective.penalty is penalty
