# imports
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field

# modules
from capacities_ml.capacities.capacities import VariableUniverse
from capacities_ml.capacities.types import CoalitionValue
from capacities_ml.capacities.utils import (
    CoalitionInput,
    named_coalition,
    normalize_coalition,
)
from capacities_ml.mobius.types import MobiusMap


# mobius representation class
@dataclass
class MobiusRepresentation:
    """Möbius coefficients over a fixed variable universe."""

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
    def n_vars(self) -> int:
        return self.universe.n_vars

    def value(self, coalition: CoalitionInput) -> float:
        """Return a Möbius coefficient, or zero when it is not specified."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self._coefficient_map.get_value(normalized_coalition)
        return 0.0 if value is None else value

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the specified coefficients keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self._coefficient_map.lookup_dict.items()
        }
