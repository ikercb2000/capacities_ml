# imports
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml.capacities.base import BaseCapacity
from capacities_ml.capacities.capacities import VariableUniverse
from capacities_ml.capacities.types import CoalitionValue
from capacities_ml.capacities.utils import (
    CoalitionInput,
    _event_mask,
    _order_permutation,
    named_coalition,
    normalize_coalition,
)
from capacities_ml.mobius.types import MobiusMap


# mobius representation class
@dataclass
class MobiusRepresentation(BaseCapacity):
    """Sparse Möbius coefficients defining an evaluable capacity."""

    universe: VariableUniverse
    coefficients: InitVar[Mapping[CoalitionInput, float]]
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

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.universe.var_names)

    @property
    def n_elements(self) -> int:
        return self.universe.n_elements

    def value(self, coalition: CoalitionInput) -> float:
        """Return a Möbius coefficient, or zero when it is not specified."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self._coefficient_map.get_value(normalized_coalition)
        return 0.0 if value is None else value

    def event_value(self, event: ArrayLike) -> float:
        """Evaluate the capacity of an event from its Möbius coefficients."""
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

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the specified coefficients keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self._coefficient_map.lookup_dict.items()
        }
