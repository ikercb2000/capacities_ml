# imports
from collections.abc import Iterable, Iterator, Mapping, Set
from dataclasses import InitVar, dataclass, field
from typing import Protocol

# variable universe protocol
class VariableUniverseLike(Protocol):
    var_names: Iterable[str]
    n_elements: int
    name_to_index: Mapping[str, int]

# coalition value
@dataclass
class CoalitionValue:
    coalition: frozenset[int]
    value: float

    def __post_init__(self) -> None:
        self.coalition = frozenset(self.coalition)
        self.value = float(self.value)

    def __str__(self):
        return f"Coalition {set(self.coalition)} with value {self.value}"

# coalition map main class
@dataclass
class CoalitionMap:
    lookup_dict: dict[frozenset[int], float] = field(init=False, repr=False)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[CoalitionValue]:
        return (
            CoalitionValue(coalition, value)
            for coalition, value in self.lookup_dict.items()
        )

    @property
    def coalitions(self) -> tuple[CoalitionValue, ...]:
        return tuple(self)

    def str_map(self) -> dict[str, float]:
        return {
            str(set(coalition)): value
            for coalition, value in self.lookup_dict.items()
        }

    def to_lookup(self) -> dict[frozenset[int], float]:
        return dict(self.lookup_dict)

    def get_value(self, subset: Set[int]) -> float | None:
        return self.lookup_dict.get(frozenset(subset))

    def set_value(self, coalition: Set[int], value: float) -> None:
        self.lookup_dict[frozenset(coalition)] = float(value)

    def remove_value(self, coalition: Set[int]) -> None:
        del self.lookup_dict[frozenset(coalition)]

    def get_coalition(self, value: float) -> list[frozenset[int]]:
        return [
            coalition
            for coalition, coalition_value in self.lookup_dict.items()
            if coalition_value == value
        ]


# capacity mapping
@dataclass
class CapacityMap(CoalitionMap):
    """Mutable, unvalidated tabular values of a set function."""

    capacities: InitVar[Iterable[CoalitionValue]]

    def __post_init__(self, capacities: Iterable[CoalitionValue]) -> None:
        self.lookup_dict = {frozenset(): 0.0}

        for coalition_value in capacities:
            coalition = coalition_value.coalition

            if not coalition:
                raise ValueError(
                    "The empty coalition must not be included explicitly; its value is assumed to be zero."
                )

            if coalition in self.lookup_dict:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")

            self.set_value(coalition, coalition_value.value)

    def __iter__(self) -> Iterator[CoalitionValue]:
        return (
            CoalitionValue(coalition, value)
            for coalition, value in self.lookup_dict.items()
            if coalition
        )

    def set_value(self, coalition: Set[int], value: float) -> None:
        frozen_coalition = frozenset(coalition)
        if not frozen_coalition:
            raise ValueError("The empty coalition always has value zero.")
        super().set_value(frozen_coalition, value)

    def remove_value(self, coalition: Set[int]) -> None:
        frozen_coalition = frozenset(coalition)
        if not frozen_coalition:
            raise ValueError("The empty coalition cannot be removed.")
        super().remove_value(frozen_coalition)


# Mobius mapping
@dataclass
class MobiusMap(CoalitionMap):
    """Mutable, potentially sparse Mobius coefficients."""

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
