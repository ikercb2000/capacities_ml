import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity, L2Penalty, Solver
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer


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


@pytest.mark.parametrize("solver", (Solver.SCIPY, Solver.CVXPY))
def test_choquet_regressor_is_serializable(
    estimator_roundtrip,
    solver,
):
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
        solver=solver,
    ).fit(X, y)

    restored = estimator_roundtrip(model)

    np.testing.assert_allclose(restored.predict(X), model.predict(X))
    assert restored.problem_.name == model.problem_.name
    assert restored.result_.solver_name == model.result_.solver_name
    assert restored.capacity_.to_named_dict() == model.capacity_.to_named_dict()


def test_fitted_sklearn_search_with_choquet_regressor_is_serializable(
    estimator_roundtrip,
):
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )
    y = 0.5 + 0.3 * X[:, 0] + 0.7 * X[:, 1]
    estimator = Pipeline(
        [
            ("normalize", CapacityNormalizer()),
            (
                "model",
                ChoquetRegressor(
                    universe=VariableUniverse(("x0", "x1")),
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator,
        {"model__sparsity": [KAdditivity(1), KAdditivity(2)]},
        cv=2,
        error_score="raise",
    ).fit(X, y)

    restored = estimator_roundtrip(search)

    np.testing.assert_allclose(restored.predict(X), search.predict(X))
    assert restored.best_params_ == search.best_params_
