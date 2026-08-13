# imports
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities.base import BaseCapacity
from capacities_ml_fin.base.capacities.capacities import ExplicitCapacity, VariableUniverse
from capacities_ml_fin.base.capacities.types import CoalitionValue, MobiusMap
from capacities_ml_fin.base.capacities.utils import (
    CoalitionInput,
    _event_mask,
    _mobius_transform_map,
    _order_permutation,
    named_coalition,
    normalize_coalition,
    powerset,
)
from capacities_ml_fin.base.capacities.validation import check_mobius_capacity


# Mobius capacity class
@dataclass
class MobiusCapacity(BaseCapacity):
    """Sparse Mobius coefficients defining an evaluable capacity."""

    universe: VariableUniverse
    coefficients: InitVar[Mapping[CoalitionInput, float]]
    validation_tolerance: float = field(default=1e-9, repr=False)
    _coefficient_map: MobiusMap = field(init=False, repr=False)

    def __post_init__(
        self,
        coefficients: Mapping[CoalitionInput, float],
    ) -> None:
        if not isinstance(self.universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")

        self._coefficient_map = MobiusMap(
            mobius_coefficients=[
                CoalitionValue(
                    normalize_coalition(self.universe, coalition),
                    value,
                )
                for coalition, value in coefficients.items()
            ]
        )
        self.validate()

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.universe.var_names)

    @property
    def n_elements(self) -> int:
        return self.universe.n_elements

    def value(self, coalition: CoalitionInput) -> float:
        """Return a Mobius coefficient, or zero when it is not specified."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self._coefficient_map.get_value(normalized_coalition)
        return 0.0 if value is None else value

    def event_value(self, event: ArrayLike) -> float:
        """Evaluate the capacity of an event from its Mobius coefficients."""
        mask = _event_mask(event, self.n_elements)
        coalition = frozenset(int(index) for index in np.flatnonzero(mask))
        return float(
            sum(
                coefficient.value
                for coefficient in self._coefficient_map
                if coefficient.coalition <= coalition
            )
        )

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        """Evaluate nested events without expanding the capacity table."""
        permutation = _order_permutation(order, self.n_elements)
        ranks = np.empty(self.n_elements, dtype=int)
        ranks[permutation] = np.arange(self.n_elements)
        changes = np.zeros(self.n_elements + 1, dtype=float)

        for coefficient in self._coefficient_map:
            coalition = coefficient.coalition
            boundary = (
                self.n_elements - 1
                if not coalition
                else min(ranks[index] for index in coalition)
            )
            changes[0] += coefficient.value
            changes[boundary + 1] -= coefficient.value

        return np.cumsum(changes[:-1])

    def validate(self) -> None:
        """Check normalization and monotonicity of the Mobius capacity."""
        check_mobius_capacity(
            self._coefficient_map,
            self.n_elements,
            tolerance=self.validation_tolerance,
        )

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the specified coefficients keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self._coefficient_map.lookup_dict.items()
        }


# Mobius transform
def mobius_transform(capacity: ExplicitCapacity) -> MobiusCapacity:
    """Compute the Mobius representation of an explicit capacity."""
    if not isinstance(capacity, ExplicitCapacity):
        raise TypeError("capacity must be an ExplicitCapacity instance.")

    coefficient_map = _mobius_transform_map(capacity.values)
    return MobiusCapacity(
        universe=capacity.universe,
        coefficients=coefficient_map.to_lookup(),
    )


# inverse Mobius transform
def inverse_mobius_transform(
    mobius_capacity: MobiusCapacity,
) -> ExplicitCapacity:
    """Compute an explicit capacity from its Mobius representation."""
    if not isinstance(mobius_capacity, MobiusCapacity):
        raise TypeError("mobius_capacity must be a MobiusCapacity instance.")

    features = frozenset(range(mobius_capacity.n_elements))
    values: dict[frozenset[int], float] = {}
    for coalition in powerset(features):
        if coalition:
            values[coalition] = sum(
                mobius_capacity.value(subset)
                for subset in powerset(coalition)
            )

    return ExplicitCapacity(universe=mobius_capacity.universe, values=values)
