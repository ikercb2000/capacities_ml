import numpy as np
import pytest

from capacities_ml_fin.base.capacities import validate_capacity
from capacities_ml_fin.ml.models import ScaledChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity, Solver


def _scaled_data():
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
    aggregate = 0.25 * X[:, 0] + 0.75 * X[:, 1]
    return X, 1.5 + 3.0 * aggregate


def test_scaled_choquet_regressor_recovers_q_intercept_and_constraints():
    X, y = _scaled_data()

    model = ScaledChoquetRegressor(sparsity=KAdditivity(1)).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)
    assert np.isclose(model.q_, 3.0, atol=1e-5)
    assert np.isclose(model.intercept_, 1.5, atol=1e-5)
    assert model.problem_.parameter_layout.slice("q") == slice(2, 3)
    assert model.problem_.parameter_layout.slice("intercept") == slice(3, 4)
    assert model.result_.diagnostics["maximum_constraint_violation"] < 1e-7
    validate_capacity(model.capacity_)


def test_scaled_choquet_regressor_is_separate_and_serializable(
    estimator_roundtrip,
):
    X, y = _scaled_data()
    model = ScaledChoquetRegressor().fit(X, y)

    restored = estimator_roundtrip(model)

    np.testing.assert_allclose(restored.predict(X), model.predict(X))
    assert restored.q_ == model.q_
    assert restored.problem_.name == "scaled_choquet_regression"


def test_scaled_choquet_regressor_defaults_to_nonnegative_q():
    X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    y = 2.0 - 4.0 * X[:, 0]

    increasing = ScaledChoquetRegressor().fit(X, y)
    decreasing = ScaledChoquetRegressor(q_bounds=(-np.inf, np.inf)).fit(X, y)

    assert increasing.q_ >= -1e-9
    assert np.isclose(increasing.q_, 0.0, atol=1e-7)
    assert np.isclose(decreasing.q_, -4.0, atol=1e-6)
    np.testing.assert_allclose(decreasing.predict(X), y, atol=1e-6)


def test_scaled_choquet_regressor_rejects_cvxpy_bilinear_problem():
    X, y = _scaled_data()

    with pytest.raises(ValueError, match="non-convex"):
        ScaledChoquetRegressor(solver=Solver.CVXPY).fit(X, y)


@pytest.mark.parametrize(
    "bounds, error, match",
    [
        ((0.0,), ValueError, "exactly"),
        ((2.0, 1.0), ValueError, "lower bound"),
        ((0.0, np.nan), ValueError, "NaN"),
        ("invalid", TypeError, "pair"),
    ],
)
def test_scaled_choquet_regressor_validates_q_bounds(bounds, error, match):
    X, y = _scaled_data()

    with pytest.raises(error, match=match):
        ScaledChoquetRegressor(q_bounds=bounds).fit(X, y)


def test_scaled_choquet_regressor_uses_sklearn_input_validation():
    X, y = _scaled_data()
    model = ScaledChoquetRegressor().fit(X, y)

    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))
