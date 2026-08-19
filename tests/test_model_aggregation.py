import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax

from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.base.interpretation import (
    pairwise_interactions,
    shapley_indices,
)
from capacities_ml_fin.ml.aggregation import (
    aggregate_binary_probabilities,
    aggregate_regression_predictions,
)
from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity, L2Penalty


@pytest.mark.parametrize("order", (1, 2))
def test_aggregate_regression_predictions_supports_numpy_k_additivity(order):
    fit_predictions = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )
    y_fit = 0.25 * fit_predictions[:, 0] + 0.75 * fit_predictions[:, 1]
    predict_predictions = np.array([[0.4, 0.6], [0.7, 0.3]])
    penalty = L2Penalty(weight=0.0, selection=np.arange(3 if order == 2 else 2))

    result = aggregate_regression_predictions(
        fit_predictions,
        y_fit,
        predict_predictions,
        sparsity=KAdditivity(order=order),
        penalty=penalty,
    )

    assert result.predictions.shape == (2,)
    assert result.model_names == ("x0", "x1")
    assert result.fitted_model.fit_intercept is False
    assert result.fitted_model.intercept_ == 0.0
    assert result.fitted_model.problem_.objective.penalty is penalty
    np.testing.assert_allclose(
        result.predictions,
        0.25 * predict_predictions[:, 0] + 0.75 * predict_predictions[:, 1],
        atol=1e-5,
    )
    result.capacity.validate()


def test_dataframe_aggregation_preserves_model_names_and_interpretation():
    fit_predictions = pd.DataFrame(
        {
            "ridge": [0.0, 0.0, 1.0, 1.0, 0.2, 0.8],
            "forest": [0.0, 1.0, 0.0, 1.0, 0.8, 0.2],
        }
    )
    y_fit = 0.4 * fit_predictions["ridge"] + 0.6 * fit_predictions["forest"]
    predict_predictions = pd.DataFrame({"ridge": [0.3, 0.9], "forest": [0.7, 0.1]})

    result = aggregate_regression_predictions(
        fit_predictions,
        y_fit,
        predict_predictions,
        sparsity=KAdditivity(order=2),
    )

    assert result.model_names == ("ridge", "forest")
    assert result.capacity.var_names == result.model_names
    assert set(shapley_indices(result.capacity)) == {"ridge", "forest"}
    assert set(pairwise_interactions(result.capacity)) == {("ridge", "forest")}


def test_binary_probability_aggregation_supports_named_models_and_weights():
    fit_probabilities = pd.DataFrame(
        {
            "logistic": [0.05, 0.15, 0.25, 0.75, 0.85, 0.95],
            "boosting": [0.10, 0.30, 0.20, 0.70, 0.90, 0.80],
        }
    )
    y_fit = np.array(["no", "no", "no", "yes", "yes", "yes"])
    predict_probabilities = pd.DataFrame(
        {"logistic": [0.2, 0.8], "boosting": [0.3, 0.7]}
    )

    result = aggregate_binary_probabilities(
        fit_probabilities,
        y_fit,
        predict_probabilities,
        sparsity=KAdditivity(order=2),
        class_weight="balanced",
        sample_weight=np.ones(y_fit.size),
    )

    assert result.probabilities.shape == (2,)
    assert np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))
    assert result.model_names == ("logistic", "boosting")
    assert result.capacity.var_names == result.model_names
    np.testing.assert_array_equal(result.classes, ["no", "yes"])
    assert result.positive_class == "yes"
    result.capacity.validate()
    np.testing.assert_allclose(
        result.capacity.event_value(np.zeros(2, dtype=int)), 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        result.capacity.event_value(np.ones(2, dtype=int)), 1.0, atol=1e-9
    )


def test_binary_aggregation_uses_one_capacity_for_both_classes_and_only_fits_it():
    fit = np.array(
        [
            [0.05, 0.10],
            [0.15, 0.30],
            [0.25, 0.20],
            [0.75, 0.70],
            [0.85, 0.90],
            [0.95, 0.80],
        ]
    )
    target = np.array([0, 0, 0, 1, 1, 1])
    predict = np.array([[0.2, 0.4], [0.8, 0.6]])

    result = aggregate_binary_probabilities(fit, target, predict)
    model = result.fitted_model
    scores = model.class_scores(predict)

    np.testing.assert_allclose(
        scores[:, 0], batch_choquet_integral(1.0 - predict, result.capacity)
    )
    np.testing.assert_allclose(
        scores[:, 1], batch_choquet_integral(predict, result.capacity)
    )
    assert model.problem_.n_parameters == len(model.result_.parameters)
    assert model.problem_.parameter_layout.slice("capacity") == slice(
        0, model.problem_.n_parameters
    )
    assert not hasattr(model, "beta_")
    assert not hasattr(model, "gamma_")


def test_binary_softmax_matches_scaled_score_difference():
    fit = np.array([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])
    target = np.array([0, 0, 1, 1])
    predict = np.array([[0.25, 0.4], [0.7, 0.85]])
    scale = 4.5

    result = aggregate_binary_probabilities(fit, target, predict, softmax_scale=scale)
    scores = result.fitted_model.class_scores(predict)

    np.testing.assert_allclose(
        result.probabilities,
        expit(scale * (scores[:, 1] - scores[:, 0])),
    )
    np.testing.assert_allclose(
        result.fitted_model.predict_proba(predict),
        softmax(scale * scores, axis=1),
    )


@pytest.mark.parametrize("scale", (0.0, -1.0, np.inf, np.nan))
def test_binary_aggregation_requires_positive_finite_softmax_scale(scale):
    fit = np.array([[0.1, 0.2], [0.8, 0.9]])
    with pytest.raises(ValueError, match="positive finite"):
        aggregate_binary_probabilities(
            fit, np.array([0, 1]), fit[:1], softmax_scale=scale
        )


def test_choquet_regressor_fit_intercept_false_and_default_compatibility():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.mean(X, axis=1)

    without_intercept = ChoquetRegressor(
        sparsity=KAdditivity(order=1), fit_intercept=False
    ).fit(X, y)
    default = ChoquetRegressor(sparsity=KAdditivity(order=1)).fit(X, y)

    assert without_intercept.intercept_ == 0.0
    with pytest.raises(KeyError, match="intercept"):
        without_intercept.problem_.parameter_layout.slice("intercept")
    assert default.fit_intercept is True
    assert default.problem_.parameter_layout.slice("intercept") == slice(2, 3)
    np.testing.assert_allclose(without_intercept.predict(X), y, atol=1e-6)


def test_aggregation_rejects_mismatched_dataframe_models_and_order():
    fit = pd.DataFrame({"first": [0.1, 0.9], "second": [0.2, 0.8]})
    target = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="same models.*same order"):
        aggregate_regression_predictions(fit, target, fit[["second", "first"]])
    with pytest.raises(ValueError, match="same models.*same order"):
        aggregate_regression_predictions(
            fit,
            target,
            pd.DataFrame({"first": [0.1], "third": [0.2]}),
        )


@pytest.mark.parametrize(
    ("fit_values", "predict_values", "message"),
    [
        (np.ones((3, 2)), np.ones((2, 3)), "same number of models"),
        (np.array([[0.1, np.nan], [0.2, 0.3]]), np.ones((1, 2)), "finite"),
        (np.ones(3), np.ones((1, 1)), "two-dimensional"),
    ],
)
def test_regression_aggregation_validates_dimensions_and_finite_values(
    fit_values, predict_values, message
):
    with pytest.raises(ValueError, match=message):
        aggregate_regression_predictions(
            fit_values, np.arange(len(fit_values), dtype=float), predict_values
        )


@pytest.mark.parametrize(
    ("fit_probabilities", "predict_probabilities"),
    [
        (np.array([[0.1, 1.1], [0.2, 0.8]]), np.array([[0.2, 0.8]])),
        (np.array([[0.1, 0.9], [0.2, 0.8]]), np.array([[-0.1, 0.8]])),
    ],
)
def test_binary_aggregation_requires_unit_interval_probabilities(
    fit_probabilities, predict_probabilities
):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_binary_probabilities(
            fit_probabilities,
            np.array([0, 1]),
            predict_probabilities,
        )


@pytest.mark.parametrize("target", (np.array([0, 0]), np.array([0, 1, 2])))
def test_binary_aggregation_requires_exactly_two_classes(target):
    probabilities = np.array([[0.1, 0.2], [0.8, 0.9], [0.4, 0.5]])
    fit = probabilities[: target.size]
    with pytest.raises(ValueError, match="exactly two"):
        aggregate_binary_probabilities(fit, target, probabilities[:1])
