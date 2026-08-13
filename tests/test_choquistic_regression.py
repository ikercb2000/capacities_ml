import numpy as np
import pytest
from sklearn.base import clone

from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral_mobius
from capacities_ml_fin.ml.models import ChoquisticRegression
from capacities_ml_fin.ml.optimization import CapacityRepresentation, L2Penalty


def test_choquistic_regression_is_probabilistic_and_cloneable(
    binary_classification_sample,
):
    X, y = binary_classification_sample
    penalty = L2Penalty(weight=0.001, selection=[0])
    model = ChoquisticRegression(
        universe=VariableUniverse(("x0", "x1")),
        class_weight="balanced",
        penalty=penalty,
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        np.exp(model.predict_log_proba(X)), probabilities, atol=1e-15
    )
    assert np.mean(model.predict(X) == y) >= 5 / 6
    assert model.gamma_ > 0.0
    assert 0.0 <= model.beta_ <= 1.0
    gamma_slice = model.problem_.parameter_layout.slice("gamma")
    beta_slice = model.problem_.parameter_layout.slice("beta")
    assert gamma_slice.stop - gamma_slice.start == 1
    assert beta_slice.stop - beta_slice.start == 1
    assert model.problem_.representation is CapacityRepresentation.MOBIUS
    assert model.problem_.objective.penalty is penalty
    utilities = batch_choquet_integral_mobius(X, model.capacity_)
    np.testing.assert_allclose(model.utility_function(X), utilities)
    np.testing.assert_allclose(
        model.decision_function(X),
        model.gamma_ * (utilities - model.beta_),
    )
    assert clone(model).get_params()["class_weight"] == "balanced"


def test_choquistic_regression_requires_normalized_predictors(
    binary_classification_sample,
):
    X, y = binary_classification_sample
    model = ChoquisticRegression(universe=VariableUniverse(("x0", "x1")))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.fit(2.0 * X, y)

    model.fit(X, y)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.predict(2.0 * X)


def test_choquistic_regression_is_serializable(
    binary_classification_sample,
    estimator_roundtrip,
):
    X, y = binary_classification_sample
    model = ChoquisticRegression(
        universe=VariableUniverse(("x0", "x1")),
        class_weight="balanced",
    ).fit(X, y)

    restored = estimator_roundtrip(model)

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_allclose(
        restored.decision_function(X),
        model.decision_function(X),
    )
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_allclose(restored.utility_function(X), model.utility_function(X))
    assert restored.problem_.name == model.problem_.name
    assert restored.result_.solver_name == model.result_.solver_name
    assert restored.capacity_.to_named_dict() == model.capacity_.to_named_dict()
