import pytest
from sklearn.utils.estimator_checks import (
    check_estimators_unfitted,
    check_parameters_default_constructible,
)

from capacities_ml_fin.ml.models import (
    ChoquetClassifier,
    ChoquetNeuralClassifier,
    ChoquetNeuralRegressor,
    ChoquetRegressor,
    ChoquisticRegression,
    ScaledChoquetRegressor,
)

ESTIMATORS = (
    ChoquetRegressor(),
    ChoquetClassifier(),
    ChoquisticRegression(),
    ChoquetNeuralRegressor(),
    ChoquetNeuralClassifier(),
    ScaledChoquetRegressor(),
)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimator_defaults_follow_sklearn_constructor_contract(estimator):
    name = type(estimator).__name__
    check_parameters_default_constructible(name, estimator)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_unfitted_estimators_raise_not_fitted_error(estimator):
    check_estimators_unfitted(type(estimator).__name__, estimator)


def test_classifier_tags_describe_binary_normalized_inputs():
    for estimator in (ChoquetClassifier(), ChoquisticRegression()):
        tags = estimator.__sklearn_tags__()
        assert tags.classifier_tags.multi_class is False
        assert tags.input_tags.positive_only is True


def test_stochastic_estimators_report_non_determinism_without_a_seed():
    assert ChoquetClassifier().__sklearn_tags__().non_deterministic is True
    assert ChoquetNeuralRegressor().__sklearn_tags__().non_deterministic is True
    assert ChoquetNeuralClassifier().__sklearn_tags__().non_deterministic is True
    assert (
        ChoquetNeuralClassifier(random_state=0).__sklearn_tags__().non_deterministic
        is False
    )
