# imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# modules
from capacities_ml_fin.base.capacities import BaseCapacity
from capacities_ml_fin.ml.optimization import OptimizationResult


# regression aggregation result
@dataclass(frozen=True, slots=True)
class RegressionAggregationResult:
    """Output and fitted state from Choquet regression aggregation."""

    predictions: NDArray[np.float64]
    capacity: BaseCapacity
    model_names: tuple[str, ...]
    optimization_result: OptimizationResult
    fitted_model: Any


# binary probability aggregation result
@dataclass(frozen=True, slots=True)
class BinaryProbabilityAggregationResult:
    """Positive-class probabilities and fitted Choquistic aggregation state."""

    probabilities: NDArray[np.float64]
    capacity: BaseCapacity
    model_names: tuple[str, ...]
    optimization_result: OptimizationResult
    fitted_model: Any
    classes: NDArray[Any]
    positive_class: Any
