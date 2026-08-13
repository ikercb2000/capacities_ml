# imports
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml.capacities import BaseCapacity
from capacities_ml.risk.capacities import ProbabilityCapacity
from capacities_ml.risk.distributions.utils import _loss_vector


# empirical loss distribution
@dataclass(slots=True)
class EmpiricalLossDistribution:
    """Finite loss distribution evaluated under an event capacity."""

    losses: ArrayLike
    sample_weight: ArrayLike | None = None
    capacity: BaseCapacity | None = None

    def __post_init__(self) -> None:
        self.losses = _loss_vector(self.losses)
        if self.capacity is not None and self.sample_weight is not None:
            raise ValueError("Use either sample_weight or capacity, not both.")
        if self.capacity is None:
            weights = (
                np.ones(self.losses.size, dtype=float)
                if self.sample_weight is None
                else self.sample_weight
            )
            self.capacity = ProbabilityCapacity(weights)
        elif not isinstance(self.capacity, BaseCapacity):
            raise TypeError("capacity must be a BaseCapacity instance.")
        if self.capacity.n_elements != self.losses.size:
            raise ValueError("capacity and losses must use the same number of scenarios.")
        self.sample_weight = (
            self.capacity.weights.copy()
            if isinstance(self.capacity, ProbabilityCapacity)
            else None
        )

    @property
    def support(self) -> np.ndarray:
        """Return the sorted distinct loss values."""
        return np.unique(self.losses)

    def survival(self, threshold: ArrayLike) -> float | np.ndarray:
        """Return ``capacity(loss > threshold)``."""
        thresholds = np.asarray(threshold, dtype=float)
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("thresholds must be finite.")
        flat = thresholds.reshape(-1)
        values = np.asarray(
            [self.capacity.event_value(self.losses > value) for value in flat],
            dtype=float,
        ).reshape(thresholds.shape)
        return float(values) if values.ndim == 0 else values

    def distribution(self, threshold: ArrayLike) -> float | np.ndarray:
        """Return the capacity distribution ``G(x) = 1 - capacity(loss > x)``."""
        values = 1.0 - np.asarray(self.survival(threshold), dtype=float)
        return float(values) if values.ndim == 0 else values

    def lower_quantile(self, probability: ArrayLike) -> float | np.ndarray:
        """Return ``inf{x: G(x) >= probability}``."""
        return self._quantile(probability, upper=False)

    def upper_quantile(self, probability: ArrayLike) -> float | np.ndarray:
        """Return ``inf{x: G(x) > probability}``."""
        return self._quantile(probability, upper=True)

    def _quantile(self, probability: ArrayLike, *, upper: bool) -> float | np.ndarray:
        probabilities = np.asarray(probability, dtype=float)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite.")
        if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
            raise ValueError("quantile probabilities must lie strictly between zero and one.")
        support = self.support
        distribution_values = np.asarray(self.distribution(support), dtype=float)
        side = "right" if upper else "left"
        indices = np.searchsorted(distribution_values, probabilities, side=side)
        indices = np.minimum(indices, support.size - 1)
        result = support[indices]
        return float(result) if result.ndim == 0 else result

    def event_probability(self, event: ArrayLike) -> float:
        """Evaluate an event under the distribution's capacity."""
        return self.capacity.event_value(event)
