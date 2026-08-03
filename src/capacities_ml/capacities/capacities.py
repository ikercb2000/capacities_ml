# imports
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field

# modules
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
from capacities_ml.capacities.utils import (
    CoalitionInput,
    named_coalition,
    normalize_coalition,
)
from capacities_ml.capacities.validation import check_monotonicity

# variable universe class
@dataclass
class VariableUniverse:
    """Names and index mapping of the variables used by a capacity."""

    var_names: Iterable[str]
    n_vars: int = field(init=False)
    name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        var_names = tuple(self.var_names)
        self.var_names = var_names
        self.n_vars = len(var_names)

        if self.n_vars < 1:
            raise ValueError("var_names must contain at least one variable.")
        if not all(isinstance(name, str) and name for name in var_names):
            raise ValueError("var_names must contain non-empty strings.")
        if len(set(var_names)) != self.n_vars:
            raise ValueError("var_names must be unique.")

        self.name_to_index = {
            name: index for index, name in enumerate(var_names)
        }


# capacity class
@dataclass
class Capacity:
    """A normalized monotone set function over a fixed feature set."""

    universe: VariableUniverse
    values: InitVar[Mapping[CoalitionInput, float]]

    def __post_init__(self, values: Mapping[CoalitionInput, float]) -> None:
        if not isinstance(self.universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")
        self.values = CapacityMap(
            capacities=[
                CoalitionValue(normalize_coalition(self.universe, coalition), value)
                for coalition, value in values.items()
            ]
        )
        self.validate()

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.universe.var_names)

    @property
    def n_vars(self) -> int:
        return self.universe.n_vars

    def value(self, coalition: CoalitionInput) -> float:
        """Return the value of a coalition expressed by names or indices."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self.values.get_value(normalized_coalition)
        if value is None:
            raise KeyError(f"Unknown coalition: {coalition}.")
        return value

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the tabular capacity values keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self.values.lookup_dict.items()
        }

    def validate(self) -> None:
        """Check the mathematical invariants of the capacity."""
        check_monotonicity(
            self.values,
            frozenset(range(self.n_vars)),
        )
