# imports
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

# modules
from capacities_ml_fin.base.capacities import BaseCapacity
from capacities_ml_fin.base.integrals.choquet import ordered_choquet
from capacities_ml_fin.risk.capacities import (
    DistortedCapacity,
    ProbabilityCapacity,
)
from capacities_ml_fin.risk.distributions import EmpiricalLossDistribution
from capacities_ml_fin.risk.distortions import Distortion, ExpectedShortfallDistortion


# capacity resolution
def _base_capacity(
    losses: ArrayLike,
    sample_weight: ArrayLike | None,
    capacity: BaseCapacity | None,
) -> tuple[np.ndarray, BaseCapacity]:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite one-dimensional array.")
    if capacity is not None and sample_weight is not None:
        raise ValueError("Use either sample_weight or capacity, not both.")
    if capacity is None:
        weights = np.ones(values.size) if sample_weight is None else sample_weight
        capacity = ProbabilityCapacity(weights)
    if not isinstance(capacity, BaseCapacity):
        raise TypeError("capacity must be a BaseCapacity instance.")
    if capacity.n_elements != values.size:
        raise ValueError("capacity and losses must use the same number of scenarios.")
    return values, capacity


# Choquet components
def _choquet_components(
    losses: np.ndarray,
    capacity: BaseCapacity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = np.unique(losses)
    increments = np.empty(support.size, dtype=float)
    event_values = np.empty(support.size, dtype=float)
    contributions = np.empty(support.size, dtype=float)
    increments[0] = support[0]
    event_values[0] = 1.0
    contributions[0] = support[0]
    for index in range(1, support.size):
        increments[index] = support[index] - support[index - 1]
        event_values[index] = capacity.event_value(losses >= support[index])
        contributions[index] = increments[index] * event_values[index]
    return support, increments, event_values, contributions


# risk contributions
def risk_contributions(
    losses: ArrayLike,
    capacity: BaseCapacity,
) -> pd.DataFrame:
    """Decompose a finite Choquet risk measure into ordered increments."""
    values, event_capacity = _base_capacity(losses, None, capacity)
    support, increments, event_values, contributions = _choquet_components(
        values,
        event_capacity,
    )
    return pd.DataFrame(
        {
            "loss": support,
            "increment": increments,
            "capacity": event_values,
            "contribution": contributions,
        }
    )


# Choquet risk measure
def choquet_risk_measure(losses: ArrayLike, capacity: BaseCapacity) -> float:
    """Evaluate the Choquet integral of finite losses under an event capacity."""
    values, event_capacity = _base_capacity(losses, None, capacity)
    return ordered_choquet(event_capacity, values)


# distortion risk measure
def distortion_risk_measure(
    losses: ArrayLike,
    distortion: Distortion,
    *,
    sample_weight: ArrayLike | None = None,
    capacity: BaseCapacity | None = None,
) -> float:
    """Evaluate ``E_(distortion o capacity)(losses)``."""
    if not isinstance(distortion, Distortion):
        raise TypeError("distortion must be a Distortion instance.")
    values, base_capacity = _base_capacity(losses, sample_weight, capacity)
    return choquet_risk_measure(
        values,
        DistortedCapacity(base_capacity, distortion),
    )


# value at risk
def value_at_risk(
    losses: ArrayLike,
    alpha: float,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Return the lower empirical quantile at confidence level ``alpha``."""
    distribution = EmpiricalLossDistribution(losses, sample_weight=sample_weight)
    return float(distribution.lower_quantile(alpha))


# expected shortfall
def expected_shortfall(
    losses: ArrayLike,
    alpha: float,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Return empirical Expected Shortfall using its distortion representation."""
    return distortion_risk_measure(
        losses,
        ExpectedShortfallDistortion(alpha),
        sample_weight=sample_weight,
    )


# generalized value at risk
def generalized_value_at_risk(
    losses: ArrayLike,
    alpha: float,
    capacity: BaseCapacity,
    *,
    quantile: str = "lower",
) -> float:
    """Return a lower or upper quantile with respect to an event capacity."""
    distribution = EmpiricalLossDistribution(losses, capacity=capacity)
    if quantile == "lower":
        return float(distribution.lower_quantile(alpha))
    if quantile == "upper":
        return float(distribution.upper_quantile(alpha))
    raise ValueError("quantile must be 'lower' or 'upper'.")


# generalized tail value at risk
def generalized_tail_value_at_risk(
    losses: ArrayLike,
    alpha: float,
    capacity: BaseCapacity,
) -> float:
    """Return generalized Tail-VaR through the paper's distorted capacity."""
    return distortion_risk_measure(
        losses,
        ExpectedShortfallDistortion(alpha),
        capacity=capacity,
    )
