# imports
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities import BaseCapacity
from capacities_ml_fin.risk.distributions import EmpiricalLossDistribution
from capacities_ml_fin.risk.measures import generalized_tail_value_at_risk
from capacities_ml_fin.risk.spectral.utils import _normalized_weights


# discrete spectral risk measure
@dataclass(frozen=True, slots=True)
class SpectralRiskMeasure:
    """Finite normalized mixture of capacity quantiles."""

    levels: Sequence[float]
    weights: Sequence[float]
    quantile: str = "upper"

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels, dtype=float)
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("levels must be a non-empty one-dimensional sequence.")
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValueError("levels must lie strictly between zero and one.")
        if np.any(np.diff(levels) < 0.0):
            raise ValueError("levels must be non-decreasing.")
        weights = _normalized_weights(self.weights, levels.size)
        if self.quantile not in {"lower", "upper"}:
            raise ValueError("quantile must be 'lower' or 'upper'.")
        object.__setattr__(self, "levels", tuple(float(value) for value in levels))
        object.__setattr__(self, "weights", tuple(float(value) for value in weights))

    def __call__(
        self,
        losses: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        capacity: BaseCapacity | None = None,
    ) -> float:
        distribution = EmpiricalLossDistribution(
            losses,
            sample_weight=sample_weight,
            capacity=capacity,
        )
        levels = np.asarray(self.levels)
        quantiles = (
            distribution.lower_quantile(levels)
            if self.quantile == "lower"
            else distribution.upper_quantile(levels)
        )
        return float(np.dot(self.weights, quantiles))


# finite Kusuoka representation
@dataclass(frozen=True, slots=True)
class KusuokaRiskMeasure:
    """Finite mixture of generalized Tail-VaR and worst-case loss."""

    levels: Sequence[float]
    weights: Sequence[float]
    worst_case_weight: float = 0.0

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels, dtype=float)
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("levels must be a non-empty one-dimensional sequence.")
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValueError("levels must lie strictly between zero and one.")
        if not 0.0 <= self.worst_case_weight <= 1.0:
            raise ValueError("worst_case_weight must lie in [0, 1].")
        weights = _normalized_weights(self.weights, levels.size)
        object.__setattr__(self, "levels", tuple(float(value) for value in levels))
        object.__setattr__(self, "weights", tuple(float(value) for value in weights))

    def __call__(
        self,
        losses: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        capacity: BaseCapacity | None = None,
    ) -> float:
        values = np.asarray(losses, dtype=float)
        if capacity is None:
            distribution = EmpiricalLossDistribution(values, sample_weight=sample_weight)
            capacity = distribution.capacity
        elif sample_weight is not None:
            raise ValueError("Use either sample_weight or capacity, not both.")
        tail_values = np.asarray(
            [
                generalized_tail_value_at_risk(values, level, capacity)
                for level in self.levels
            ]
        )
        mixture = float(np.dot(self.weights, tail_values))
        return float(
            self.worst_case_weight * np.max(values)
            + (1.0 - self.worst_case_weight) * mixture
        )
