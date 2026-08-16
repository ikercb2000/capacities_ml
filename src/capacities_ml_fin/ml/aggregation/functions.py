# imports
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.ml.aggregation.results import (
    BinaryProbabilityAggregationResult,
    RegressionAggregationResult,
)
from capacities_ml_fin.ml.models.classification import ChoquisticRegression
from capacities_ml_fin.ml.models.regression import ChoquetRegressor
from capacities_ml_fin.ml.optimization import Solver
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# regression prediction aggregation
def aggregate_regression_predictions(
    fit_predictions: ArrayLike,
    y_fit: ArrayLike,
    predict_predictions: ArrayLike,
    *,
    sparsity: CapacitySparsity | None = None,
    penalty: Any = None,
    solver: Solver | str = "scipy",
) -> RegressionAggregationResult:
    """Learn a capacity over regressors and aggregate new predictions.

    Rows are observations and columns are source models. The capacity is fitted
    without an intercept, so the returned prediction is exactly
    ``C_mu(model_1_prediction, ..., model_K_prediction)``.
    """
    fit_input, predict_input, model_names = _validated_prediction_pair(
        fit_predictions,
        predict_predictions,
        probabilities=False,
    )
    target = _validated_regression_target(y_fit, len(fit_input))
    model = ChoquetRegressor(
        sparsity=sparsity,
        penalty=penalty,
        solver=solver,
        fit_intercept=False,
    ).fit(fit_input, target)
    predictions = np.asarray(model.predict(predict_input), dtype=float)
    return RegressionAggregationResult(
        predictions=predictions,
        capacity=model.capacity_,
        model_names=model_names,
        optimization_result=model.result_,
        fitted_model=model,
    )


# binary probability aggregation
def aggregate_binary_probabilities(
    fit_probabilities: ArrayLike,
    y_fit: ArrayLike,
    predict_probabilities: ArrayLike,
    *,
    sparsity: CapacitySparsity | None = None,
    penalty: Any = None,
    class_weight: dict[Any, float] | str | None = None,
    sample_weight: ArrayLike | None = None,
    solver: Solver | str = "scipy",
) -> BinaryProbabilityAggregationResult:
    """Learn a capacity over binary classifiers' positive probabilities.

    Every matrix column must contain the probability of the same positive
    class from one source model. The positive class in the result is the second
    class according to scikit-learn's deterministic label ordering.
    """
    fit_input, predict_input, model_names = _validated_prediction_pair(
        fit_probabilities,
        predict_probabilities,
        probabilities=True,
    )
    target = _validated_binary_target(y_fit, len(fit_input))
    model = ChoquisticRegression(
        sparsity=sparsity,
        penalty=penalty,
        class_weight=class_weight,
        solver=solver,
    ).fit(fit_input, target, sample_weight=sample_weight)
    probabilities = np.asarray(model.predict_proba(predict_input)[:, 1], dtype=float)
    return BinaryProbabilityAggregationResult(
        probabilities=probabilities,
        capacity=model.capacity_,
        model_names=model_names,
        optimization_result=model.result_,
        fitted_model=model,
        classes=model.classes_.copy(),
        positive_class=model.classes_[1],
    )


# paired model-output validation
def _validated_prediction_pair(
    fit_values: ArrayLike,
    predict_values: ArrayLike,
    *,
    probabilities: bool,
) -> tuple[ArrayLike, ArrayLike, tuple[str, ...]]:
    fit_is_frame = isinstance(fit_values, pd.DataFrame)
    predict_is_frame = isinstance(predict_values, pd.DataFrame)
    if fit_is_frame != predict_is_frame:
        raise TypeError(
            "fit and predict model outputs must both be pandas DataFrames or "
            "both be array-like."
        )

    if fit_is_frame:
        fit_frame = fit_values
        predict_frame = predict_values
        if not all(isinstance(name, str) for name in fit_frame.columns):
            raise TypeError("DataFrame model names must all be strings.")
        if not fit_frame.columns.is_unique:
            raise ValueError("DataFrame model names must be unique.")
        if tuple(fit_frame.columns) != tuple(predict_frame.columns):
            raise ValueError(
                "fit and predict DataFrames must contain the same models in "
                "the same order."
            )
        model_names = tuple(fit_frame.columns)
        fit_input: ArrayLike = fit_frame
        predict_input: ArrayLike = predict_frame
    else:
        fit_input = np.asarray(fit_values, dtype=float)
        predict_input = np.asarray(predict_values, dtype=float)
        fit_columns = fit_input.shape[1] if fit_input.ndim == 2 else 0
        model_names = tuple(f"x{index}" for index in range(fit_columns))

    fit_matrix = _validated_matrix(fit_input, name="fit model outputs")
    predict_matrix = _validated_matrix(predict_input, name="predict model outputs")
    if fit_matrix.shape[1] != predict_matrix.shape[1]:
        raise ValueError(
            "fit and predict model outputs must contain the same number of models."
        )
    if probabilities:
        for name, matrix in (
            ("fit probabilities", fit_matrix),
            ("predict probabilities", predict_matrix),
        ):
            if np.any((matrix < 0.0) | (matrix > 1.0)):
                raise ValueError(f"{name} must lie in [0, 1].")
    return fit_input, predict_input, model_names


# numeric matrix validation
def _validated_matrix(values: ArrayLike, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values.") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain observations and model columns.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


# regression target validation
def _validated_regression_target(values: ArrayLike, n_samples: int) -> np.ndarray:
    try:
        target = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("y_fit must contain numeric regression targets.") from error
    if target.ndim != 1 or target.shape[0] != n_samples:
        raise ValueError("y_fit must be one-dimensional with one value per fit row.")
    if not np.all(np.isfinite(target)):
        raise ValueError("y_fit must contain only finite values.")
    return target


# binary target validation
def _validated_binary_target(values: ArrayLike, n_samples: int) -> np.ndarray:
    target = np.asarray(values)
    if target.ndim != 1 or target.shape[0] != n_samples:
        raise ValueError("y_fit must be one-dimensional with one label per fit row.")
    if np.any(pd.isna(target)):
        raise ValueError("y_fit must not contain missing labels.")
    try:
        classes = np.unique(target)
    except TypeError as error:
        raise TypeError("y_fit labels must be mutually comparable.") from error
    if classes.size != 2:
        raise ValueError("y_fit must contain exactly two distinct classes.")
    return target
