# imports
from collections.abc import Mapping
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


# series validation
def _series_values(losses: ArrayLike) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("losses must contain only finite values.")
    return values


# multiple rolling estimates
def rolling_risk_estimates(
    losses: ArrayLike,
    risk_measures: Mapping[str, Any],
    **estimator_parameters: Any,
) -> pd.DataFrame:
    """Estimate several aligned rolling or expanding capital series."""
    from capacities_ml_fin.risk.rolling.rolling import RollingRiskEstimator

    if not risk_measures:
        raise ValueError("risk_measures must contain at least one named measure.")
    estimates: dict[str, np.ndarray] = {}
    for name, risk_measure in risk_measures.items():
        if not isinstance(name, str) or not name:
            raise ValueError("risk-measure names must be non-empty strings.")
        estimator = RollingRiskEstimator(
            risk_measure,
            **estimator_parameters,
        ).fit(losses)
        estimates[name] = np.asarray(estimator.predict_in_sample(), dtype=float)
    index = losses.index if isinstance(losses, pd.Series) else None
    return pd.DataFrame(estimates, index=index)
