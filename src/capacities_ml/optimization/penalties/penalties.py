# imports
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

# modules
from capacities_ml.optimization.penalties.utils import selection_mask

# penalty aliases
FloatArray = NDArray[np.float64]

# L1 penalty
@dataclass(frozen=True, slots=True)
class L1Penalty:
    """L1 penalty applied to a selected set of parameters."""

    weight: float
    selection: ArrayLike

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Penalty weight must be non-negative.")

    def __call__(self, parameters: ArrayLike) -> float:
        vector = np.asarray(parameters, dtype=float).reshape(-1)
        mask = selection_mask(self.selection, vector.size)
        return float(self.weight * np.sum(np.abs(vector[mask])))


# L2 penalty
@dataclass(frozen=True, slots=True)
class L2Penalty:
    """Squared-L2 penalty applied to a selected set of parameters."""

    weight: float
    selection: ArrayLike

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Penalty weight must be non-negative.")

    def __call__(self, parameters: ArrayLike) -> float:
        vector = np.asarray(parameters, dtype=float).reshape(-1)
        mask = selection_mask(self.selection, vector.size)
        return float(self.weight * np.dot(vector[mask], vector[mask]))

    def gradient(self, parameters: ArrayLike) -> FloatArray:
        vector = np.asarray(parameters, dtype=float).reshape(-1)
        mask = selection_mask(self.selection, vector.size)
        gradient = np.zeros_like(vector)
        gradient[mask] = 2.0 * self.weight * vector[mask]
        return gradient
