# imports
from collections.abc import Iterable, Iterator, Set as AbstractSet
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List


# capacity value
@dataclass(frozen=True)
class CoalitionValue:
    coalition: FrozenSet[int]
    value: float

    def __post_init__(self):
        object.__setattr__(self, "coalition", frozenset(self.coalition))

    def __str__(self):
        return f"Coalition {set(self.coalition)} with value {self.value}"

# coalition map main class
@dataclass
class CoalitionMap:
    lookup_dict: Dict[FrozenSet[int], float] = field(init=False, repr=False)

    def __len__(self) -> int:
        return len(self.coalitions)

    def __iter__(self) -> Iterator[CoalitionValue]:
        return iter(self.coalitions)

    def str_map(self) -> Dict[str, float]:
        return {
            str(set(coalition)): value
            for coalition, value in self.lookup_dict.items()
        }

    def to_lookup(self) -> Dict[FrozenSet[int], float]:
        return dict(self.lookup_dict)

    def get_value(self, subset: AbstractSet[int]) -> float | None:
        return self.lookup_dict.get(frozenset(subset))

    def get_coalition(self, value: float) -> List[FrozenSet[int]]:
        return [
            coalition
            for coalition, coalition_value in self.lookup_dict.items()
            if coalition_value == value
        ]


# capacity mapping
@dataclass
class CapacityMap(CoalitionMap):
    capacities: Iterable[CoalitionValue]
    coalitions: List[CoalitionValue] = field(init=False, repr=False)

    def __post_init__(self):
        self.coalitions = list(self.capacities)
        self.lookup_dict = {frozenset(): 0.0}

        for coalition_value in self.coalitions:
            coalition = coalition_value.coalition

            if not coalition:
                raise ValueError(
                    "The empty coalition must not be included explicitly; its value is assumed to be zero."
                )

            if coalition in self.lookup_dict:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")

            self.lookup_dict[coalition] = float(coalition_value.value)

        self.capacities = self.coalitions


# möbius mapping
@dataclass
class MobiusMap(CoalitionMap):
    mobius_coefficients: Iterable[CoalitionValue]
    coalitions: List[CoalitionValue] = field(init=False, repr=False)

    def __post_init__(self):
        self.coalitions = list(self.mobius_coefficients)
        self.lookup_dict = {}

        for coalition_value in self.coalitions:
            coalition = coalition_value.coalition

            if coalition in self.lookup_dict:
                raise ValueError(f"Duplicate coalition found: {set(coalition)}.")

            self.lookup_dict[coalition] = float(coalition_value.value)

        self.mobius_coefficients = self.coalitions
