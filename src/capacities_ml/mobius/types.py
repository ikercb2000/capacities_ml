# imports
from collections.abc import Iterable
from dataclasses import InitVar, dataclass

# modules
from capacities_ml.capacities.types import CoalitionMap, CoalitionValue


# mobius mapping
@dataclass
class MobiusMap(CoalitionMap):
    """Mutable, potentially sparse Möbius coefficients."""

    mobius_coefficients: InitVar[Iterable[CoalitionValue]]

    def __post_init__(
        self,
        mobius_coefficients: Iterable[CoalitionValue],
    ) -> None:
        self.lookup_dict = {}

        for coalition_value in mobius_coefficients:
            coalition = coalition_value.coalition
            if coalition in self.lookup_dict:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")
            self.set_value(coalition, coalition_value.value)
