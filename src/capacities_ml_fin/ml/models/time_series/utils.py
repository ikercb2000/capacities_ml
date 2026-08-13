# imports
from __future__ import annotations

from typing import Any

import numpy as np

# modules


# lagged supervised sample
def lag_matrix(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create autoregressive predictors ordered from lag one to lag ``p``."""
    rows = np.asarray(
        [
            [series[time - lag] for lag in range(1, lags + 1)]
            for time in range(lags, series.size)
        ],
        dtype=float,
    )
    return rows, series[lags:].copy()


# autoregression optimization callback
def autoregression_predictor(
    parameters: Any,
    *,
    design: np.ndarray,
    exogenous: np.ndarray | None,
    capacity_slice: slice,
    phi_slice: slice,
    intercept_slice: slice | None,
    exogenous_slice: slice | None,
) -> Any:
    """Evaluate the conditional mean for an optimization parameter vector."""
    prediction = parameters[phi_slice][0] * (design @ parameters[capacity_slice])
    if intercept_slice is not None:
        prediction = prediction + parameters[intercept_slice][0]
    if exogenous is not None and exogenous_slice is not None:
        prediction = prediction + exogenous @ parameters[exogenous_slice]
    return prediction
