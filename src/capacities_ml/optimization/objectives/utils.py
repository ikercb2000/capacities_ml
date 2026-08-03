# imports
from __future__ import annotations
from collections.abc import Callable
from typing import Any
from numpy.typing import ArrayLike

# modules
from capacities_ml.optimization.objectives.objectives import (
    AbsoluteErrorObjective,
    LogisticNegativeLogLikelihood,
    QuantileLossObjective,
    SquaredErrorObjective,
    ZeroOneLossObjective,
)

# objective factories
def squared_error_objective(
    target: ArrayLike,
    predictor: Callable,
    *,
    penalty: Callable | None = None,
    mean: bool = True,
    symbolic_predictor: Any = None,
) -> SquaredErrorObjective:
    return SquaredErrorObjective(target, predictor, penalty, mean, symbolic_predictor)


# absolute error factory
def absolute_error_objective(
    target: ArrayLike,
    predictor: Callable,
    *,
    penalty: Callable | None = None,
    mean: bool = True,
    symbolic_predictor: Any = None,
) -> AbsoluteErrorObjective:
    return AbsoluteErrorObjective(target, predictor, penalty, mean, symbolic_predictor)


# quantile loss factory
def quantile_loss_objective(
    target: ArrayLike,
    predictor: Callable,
    quantile: float,
    *,
    penalty: Callable | None = None,
    mean: bool = True,
    symbolic_predictor: Any = None,
) -> QuantileLossObjective:
    return QuantileLossObjective(target, predictor, quantile, penalty, mean, symbolic_predictor)


# logistic loss factory
def logistic_negative_log_likelihood(
    target: ArrayLike,
    linear_predictor: Callable,
    *,
    sample_weight: ArrayLike | None = None,
    penalty: Callable | None = None,
    mean: bool = True,
    symbolic_predictor: Any = None,
) -> LogisticNegativeLogLikelihood:
    return LogisticNegativeLogLikelihood(
        target,
        linear_predictor,
        sample_weight,
        penalty,
        mean,
        symbolic_predictor,
    )


# zero-one loss factory
def zero_one_loss_objective(
    target: ArrayLike,
    classifier: Callable,
    *,
    mean: bool = False,
) -> ZeroOneLossObjective:
    return ZeroOneLossObjective(target, classifier, mean)
