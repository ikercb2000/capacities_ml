# imports
from __future__ import annotations
from typing import Any
import numpy as np


# regression prediction callback
def regression_predictor(
    parameters: Any,
    *,
    design: np.ndarray,
    capacity_slice: slice,
    intercept_slice: slice | None,
) -> Any:
    """Evaluate a Choquet regression parameter vector."""
    prediction = design @ parameters[capacity_slice]
    if intercept_slice is not None:
        prediction = prediction + parameters[intercept_slice][0]
    return prediction


# scaled Choquet regression prediction callback
def scaled_regression_predictor(
    parameters: Any,
    *,
    design: np.ndarray,
    capacity_slice: slice,
    q_slice: slice,
    intercept_slice: slice,
) -> Any:
    """Evaluate ``intercept + q * C_mu(X)`` from a parameter vector."""
    aggregate = design @ parameters[capacity_slice]
    return parameters[intercept_slice][0] + parameters[q_slice][0] * aggregate
