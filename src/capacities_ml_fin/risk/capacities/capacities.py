# imports
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities import BaseCapacity
from capacities_ml_fin.base.capacities.utils import _event_mask, _order_permutation
from capacities_ml_fin.risk.distortions import Distortion
from capacities_ml_fin.risk.capacities.utils import (
    _immutable_array,
    _probability_weights,
)


# probability capacity
@dataclass(slots=True)
class ProbabilityCapacity(BaseCapacity):
    """Additive event capacity defined by normalized scenario weights."""

    weights: ArrayLike

    def __post_init__(self) -> None:
        self.weights = _probability_weights(self.weights)
        self._freeze()

    def __reduce__(self) -> tuple[object, tuple[np.ndarray]]:
        """Reconstruct through the validating public constructor."""
        return type(self), (self.weights.copy(),)

    @property
    def n_elements(self) -> int:
        return int(self.weights.size)

    def event_value(self, event: ArrayLike) -> float:
        mask = _event_mask(event, self.n_elements)
        return float(np.sum(self.weights[mask]))

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        permutation = _order_permutation(order, self.n_elements)
        ordered_weights = self.weights[permutation]
        return np.cumsum(ordered_weights[::-1])[::-1]


# distorted event capacity
@dataclass(slots=True)
class DistortedCapacity(BaseCapacity):
    """Composition of an event capacity with a distortion function."""

    base_capacity: BaseCapacity
    distortion: Distortion

    def __post_init__(self) -> None:
        if not isinstance(self.base_capacity, BaseCapacity):
            raise TypeError("base_capacity must be a BaseCapacity instance.")
        if not isinstance(self.distortion, Distortion):
            raise TypeError("distortion must be a Distortion instance.")
        self._freeze()

    def __reduce__(self) -> tuple[object, tuple[BaseCapacity, Distortion]]:
        """Reconstruct through the validating public constructor."""
        return type(self), (self.base_capacity, self.distortion)

    @property
    def n_elements(self) -> int:
        return self.base_capacity.n_elements

    def event_value(self, event: ArrayLike) -> float:
        return float(self.distortion(self.base_capacity.event_value(event)))

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        values = self.base_capacity.nested_event_values(order)
        return np.asarray(self.distortion(values), dtype=float)


# upper probability envelope
@dataclass(slots=True)
class UpperEnvelopeCapacity(BaseCapacity):
    """Upper envelope of a finite collection of prior probabilities."""

    prior_weights: ArrayLike

    def __post_init__(self) -> None:
        priors = np.asarray(self.prior_weights, dtype=float)
        if priors.ndim != 2 or priors.shape[0] == 0 or priors.shape[1] == 0:
            raise ValueError("prior_weights must be a non-empty two-dimensional array.")
        self.prior_weights = _immutable_array(
            np.vstack([_probability_weights(row) for row in priors])
        )
        self._freeze()

    def __reduce__(self) -> tuple[object, tuple[np.ndarray]]:
        """Reconstruct through the validating public constructor."""
        return type(self), (self.prior_weights.copy(),)

    @property
    def n_elements(self) -> int:
        return int(self.prior_weights.shape[1])

    def event_value(self, event: ArrayLike) -> float:
        mask = _event_mask(event, self.n_elements)
        return float(np.max(np.sum(self.prior_weights[:, mask], axis=1)))

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        permutation = _order_permutation(order, self.n_elements)
        ordered_weights = self.prior_weights[:, permutation]
        tail_probabilities = np.cumsum(ordered_weights[:, ::-1], axis=1)[:, ::-1]
        return np.max(tail_probabilities, axis=0)


# lower probability envelope
@dataclass(slots=True)
class LowerEnvelopeCapacity(BaseCapacity):
    """Lower envelope of a finite collection of prior probabilities."""

    prior_weights: ArrayLike

    def __post_init__(self) -> None:
        priors = np.asarray(self.prior_weights, dtype=float)
        if priors.ndim != 2 or priors.shape[0] == 0 or priors.shape[1] == 0:
            raise ValueError("prior_weights must be a non-empty two-dimensional array.")
        self.prior_weights = _immutable_array(
            np.vstack([_probability_weights(row) for row in priors])
        )
        self._freeze()

    def __reduce__(self) -> tuple[object, tuple[np.ndarray]]:
        """Reconstruct through the validating public constructor."""
        return type(self), (self.prior_weights.copy(),)

    @property
    def n_elements(self) -> int:
        return int(self.prior_weights.shape[1])

    def event_value(self, event: ArrayLike) -> float:
        mask = _event_mask(event, self.n_elements)
        return float(np.min(np.sum(self.prior_weights[:, mask], axis=1)))

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        permutation = _order_permutation(order, self.n_elements)
        ordered_weights = self.prior_weights[:, permutation]
        tail_probabilities = np.cumsum(ordered_weights[:, ::-1], axis=1)[:, ::-1]
        return np.min(tail_probabilities, axis=0)
