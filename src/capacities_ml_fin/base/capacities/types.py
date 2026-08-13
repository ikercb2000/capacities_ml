# imports
from collections.abc import Iterable, Iterator, Mapping, Set
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Protocol

# variable universe protocol
class VariableUniverseLike(Protocol):
    var_names: Iterable[str]
    n_elements: int
    name_to_index: Mapping[str, int]

# coalition value
@dataclass(frozen=True, slots=True)
class CoalitionValue:
    coalition: frozenset[int]
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "coalition", frozenset(self.coalition))
        object.__setattr__(self, "value", float(self.value))

    def __str__(self):
        return f"Coalition {set(self.coalition)} with value {self.value}"

# coalition map main class
@dataclass(frozen=True, slots=True)
class CoalitionMap:
    lookup_dict: Mapping[frozenset[int], float] = field(init=False, repr=False)

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
        raise AttributeError(
            f"{type(self).__name__} is immutable; create a new map instead."
        )

    def remove_value(self, coalition: Set[int]) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; create a new map instead."
        )

    def get_coalition(self, value: float) -> list[frozenset[int]]:
        return [
            coalition
            for coalition, coalition_value in self.lookup_dict.items()
            if coalition_value == value
        ]


# capacity mapping
@dataclass(frozen=True, slots=True)
class CapacityMap(CoalitionMap):
    """Immutable, unvalidated tabular values of a set function."""

    capacities: InitVar[Iterable[CoalitionValue]]

    def __post_init__(self, capacities: Iterable[CoalitionValue]) -> None:
        lookup = {frozenset(): 0.0}

        for coalition_value in capacities:
            coalition = coalition_value.coalition

            if not coalition:
                raise ValueError(
                    "The empty coalition must not be included explicitly; its value is assumed to be zero."
                )

            if coalition in lookup:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")

            lookup[coalition] = float(coalition_value.value)

        object.__setattr__(self, "lookup_dict", MappingProxyType(lookup))

    def __iter__(self) -> Iterator[CoalitionValue]:
        return (
            CoalitionValue(coalition, value)
            for coalition, value in self.lookup_dict.items()
            if coalition
        )

    def __reduce__(self) -> tuple[object, tuple[tuple[CoalitionValue, ...]]]:
        return type(self), (self.coalitions,)


# Mobius mapping
@dataclass(frozen=True, slots=True)
class MobiusMap(CoalitionMap):
    """Immutable, potentially sparse Mobius coefficients."""

    mobius_coefficients: InitVar[Iterable[CoalitionValue]]

    def __post_init__(
        self,
        mobius_coefficients: Iterable[CoalitionValue],
    ) -> None:
        lookup: dict[frozenset[int], float] = {}

        for coalition_value in mobius_coefficients:
            coalition = coalition_value.coalition
            if coalition in lookup:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")
            lookup[coalition] = float(coalition_value.value)

        object.__setattr__(self, "lookup_dict", MappingProxyType(lookup))

    def __reduce__(self) -> tuple[object, tuple[tuple[CoalitionValue, ...]]]:
        return type(self), (self.coalitions,)
