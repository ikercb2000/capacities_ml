# imports
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.risk.distortions.utils import _probabilities, _result


# distortion base class
class Distortion(ABC):
    """Increasing transformation of event probabilities."""

    @abstractmethod
    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        """Evaluate the distortion on probabilities in ``[0, 1]``."""

    def validate(self, grid_size: int = 1001, tolerance: float = 1e-10) -> None:
        """Check the endpoint and monotonicity conditions of a distortion."""
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3.")
        grid = np.linspace(0.0, 1.0, grid_size)
        values = np.asarray(self(grid), dtype=float)
        if values.shape != grid.shape or not np.all(np.isfinite(values)):
            raise ValueError("A distortion must return one finite value per probability.")
        if not np.isclose(values[0], 0.0, atol=tolerance):
            raise ValueError("A distortion must satisfy g(0) = 0.")
        if not np.isclose(values[-1], 1.0, atol=tolerance):
            raise ValueError("A distortion must satisfy g(1) = 1.")
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError("A distortion must take values in [0, 1].")
        if np.any(np.diff(values) < -tolerance):
            raise ValueError("A distortion must be non-decreasing.")

    def is_concave(self, grid_size: int = 1001, tolerance: float = 1e-10) -> bool:
        """Return whether sampled slopes are non-increasing."""
        grid = np.linspace(0.0, 1.0, grid_size)
        values = np.asarray(self(grid), dtype=float)
        return bool(np.all(np.diff(values, n=2) <= tolerance))

    def is_convex(self, grid_size: int = 1001, tolerance: float = 1e-10) -> bool:
        """Return whether sampled slopes are non-decreasing."""
        grid = np.linspace(0.0, 1.0, grid_size)
        values = np.asarray(self(grid), dtype=float)
        return bool(np.all(np.diff(values, n=2) >= -tolerance))

# identity distortion
@dataclass(frozen=True, slots=True)
class IdentityDistortion(Distortion):
    """Leave event probabilities unchanged."""

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        values, scalar = _probabilities(probability)
        return _result(values.copy(), scalar)


# value-at-risk distortion
@dataclass(frozen=True, slots=True)
class ValueAtRiskDistortion(Distortion):
    """Step distortion associated with lower Value-at-Risk."""

    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        values, scalar = _probabilities(probability)
        distorted = (values > 1.0 - self.alpha).astype(float)
        return _result(distorted, scalar)


# expected-shortfall distortion
@dataclass(frozen=True, slots=True)
class ExpectedShortfallDistortion(Distortion):
    """Concave distortion associated with Expected Shortfall."""

    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        values, scalar = _probabilities(probability)
        distorted = np.minimum(values / (1.0 - self.alpha), 1.0)
        return _result(distorted, scalar)


# proportional-hazards distortion
@dataclass(frozen=True, slots=True)
class ProportionalHazardsDistortion(Distortion):
    """Power distortion ``g(p) = p**gamma``."""

    gamma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("gamma must be a finite positive number.")

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        values, scalar = _probabilities(probability)
        return _result(np.power(values, self.gamma), scalar)


# piecewise-linear distortion
@dataclass(frozen=True, slots=True)
class PiecewiseLinearDistortion(Distortion):
    """Distortion obtained by linear interpolation between supplied knots."""

    probabilities: Sequence[float]
    values: Sequence[float]

    def __post_init__(self) -> None:
        probabilities = tuple(float(value) for value in self.probabilities)
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "values", values)
        if len(probabilities) < 2 or len(probabilities) != len(values):
            raise ValueError("probabilities and values must have the same length of at least two.")
        if not np.all(np.diff(probabilities) > 0.0):
            raise ValueError("probability knots must be strictly increasing.")
        if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
            raise ValueError("probability knots must start at zero and end at one.")
        self.validate(grid_size=max(1001, len(probabilities)))

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        inputs, scalar = _probabilities(probability)
        values = np.interp(inputs, self.probabilities, self.values)
        return _result(values, scalar)


# custom distortion
@dataclass(frozen=True, slots=True)
class CustomDistortion(Distortion):
    """Validated distortion backed by a user-provided callable."""

    function: Callable[[ArrayLike], ArrayLike]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable.")
        self.validate()

    def __call__(self, probability: ArrayLike) -> float | np.ndarray:
        inputs, scalar = _probabilities(probability)
        values = np.asarray(self.function(inputs), dtype=float)
        if values.shape != inputs.shape:
            raise ValueError("The custom distortion must preserve the input shape.")
        return _result(values, scalar)
