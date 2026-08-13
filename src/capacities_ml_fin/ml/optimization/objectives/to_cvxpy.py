# imports
from __future__ import annotations
from typing import Any
import numpy as np
import cvxpy as cp

# modules
from capacities_ml_fin.ml.optimization.objectives.objectives import (
    AbsoluteErrorObjective,
    LogisticNegativeLogLikelihood,
    ObjectiveSpec,
    QuantileLossObjective,
    SquaredErrorObjective,
    ZeroOneLossObjective,
)
from capacities_ml_fin.ml.optimization.penalties import L1Penalty, L2Penalty, selection_mask

# CVXPY penalty translation
def _penalty_expression(penalty: Any, variable: Any) -> Any:
    if penalty is None:
        return 0.0
    if isinstance(penalty, (L1Penalty, L2Penalty)):
        mask = selection_mask(penalty.selection, int(variable.shape[0]))
        selected = variable[mask]
        if isinstance(penalty, L1Penalty):
            return penalty.weight * cp.sum(cp.abs(selected))
        return penalty.weight * cp.sum_squares(selected)
    translator = getattr(penalty, "to_cvxpy", None)
    if translator is None:
        raise TypeError("The penalty does not provide a CVXPY translation.")
    return translator(variable)


# symbolic predictor validation
def _symbolic_prediction(objective: Any, variable: Any) -> Any:
    if objective.symbolic_predictor is None:
        raise TypeError(
            "The objective requires a symbolic_predictor for the CVXPY solver."
        )
    return objective.symbolic_predictor(variable)


# squared error translation
def _squared_error(objective: SquaredErrorObjective, variable: Any) -> Any:
    prediction = _symbolic_prediction(objective, variable)
    loss = cp.sum_squares(np.asarray(objective.target) - prediction)
    if objective.mean:
        loss /= objective.target.size
    return loss + _penalty_expression(objective.penalty, variable)


# absolute error translation
def _absolute_error(objective: AbsoluteErrorObjective, variable: Any) -> Any:
    prediction = _symbolic_prediction(objective, variable)
    loss = cp.sum(cp.abs(np.asarray(objective.target) - prediction))
    if objective.mean:
        loss /= objective.target.size
    return loss + _penalty_expression(objective.penalty, variable)


# quantile loss translation
def _quantile_loss(objective: QuantileLossObjective, variable: Any) -> Any:
    prediction = _symbolic_prediction(objective, variable)
    residual = np.asarray(objective.target) - prediction
    losses = cp.maximum(objective.quantile * residual, (objective.quantile - 1.0) * residual)
    loss = cp.sum(losses)
    if objective.mean:
        loss /= objective.target.size
    return loss + _penalty_expression(objective.penalty, variable)


# logistic loss translation
def _logistic_loss(objective: LogisticNegativeLogLikelihood, variable: Any) -> Any:
    logits = _symbolic_prediction(objective, variable)
    losses = cp.multiply(
        np.asarray(objective.sample_weight),
        cp.logistic(logits) - cp.multiply(np.asarray(objective.target), logits),
    )
    loss = cp.sum(losses)
    if objective.mean:
        loss /= objective.target.size
    return loss + _penalty_expression(objective.penalty, variable)


# objective translation dispatcher
def to_cvxpy(objective: ObjectiveSpec):
    """Return a CVXPY expression builder for a supported objective."""
    if isinstance(objective, SquaredErrorObjective):
        return lambda variable: _squared_error(objective, variable)
    if isinstance(objective, AbsoluteErrorObjective):
        return lambda variable: _absolute_error(objective, variable)
    if isinstance(objective, QuantileLossObjective):
        return lambda variable: _quantile_loss(objective, variable)
    if isinstance(objective, LogisticNegativeLogLikelihood):
        return lambda variable: _logistic_loss(objective, variable)
    if isinstance(objective, ZeroOneLossObjective):
        raise NotImplementedError(
            "Zero-one loss needs a mixed-integer CVXPY formulation and is not "
            "available in the continuous CVXPY backend."
        )
    raise TypeError(f"Unsupported objective type for CVXPY: {type(objective).__name__}.")
