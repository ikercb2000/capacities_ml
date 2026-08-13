# imports
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

# modules
from capacities_ml.capacities.capacities import ExplicitCapacity, VariableUniverse
from capacities_ml.capacities.utils import CoalitionInput, normalize_coalition
from capacities_ml.capacities.validation import check_k_additivity

# k-Additive capacity dataclass
@dataclass
class KAdditiveCapacity(ExplicitCapacity):
    k: int

    def __post_init__(self, values: Mapping[CoalitionInput, float]) -> None:
        if not isinstance(self.universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")
        if not isinstance(self.k, int):
            raise TypeError("k must be an integer.")
        if not 1 <= self.k <= self.universe.n_elements:
            raise ValueError(
                f"k must satisfy 1 <= k <= n_elements, received k={self.k} "
                f"and n_elements={self.universe.n_elements}."
            )

        normalized_values: dict[frozenset[int], float] = {}
        for coalition, value in values.items():
            normalized = normalize_coalition(self.universe, coalition)
            if normalized in normalized_values:
                raise ValueError(f"Duplicate coalition found: {set(normalized)}.")
            normalized_values[normalized] = float(value)

        completed_values = self._complete_k_additive_values(normalized_values)
        super().__post_init__(completed_values)

    def _complete_k_additive_values(self, values: dict[frozenset[int], float]) -> dict[frozenset[int], float]:
        variables = tuple(range(self.universe.n_elements))
        required_coalitions = [
            frozenset(coalition)
            for size in range(1, self.k + 1)
            for coalition in combinations(variables, size)
        ]
        missing = [
            coalition
            for coalition in required_coalitions
            if coalition not in values
        ]
        if missing:
            raise ValueError(
                f"Missing coalitions of order at most k={self.k}: "
                f"{[set(coalition) for coalition in missing]}."
            )

        mobius_coefficients: dict[frozenset[int], float] = {}
        for coalition in required_coalitions:
            lower_order_sum = sum(
                coefficient
                for subset, coefficient in mobius_coefficients.items()
                if subset < coalition
            )
            mobius_coefficients[coalition] = values[coalition] - lower_order_sum

        completed_values = dict(values)
        for size in range(self.k + 1, self.universe.n_elements + 1):
            for coalition_items in combinations(variables, size):
                coalition = frozenset(coalition_items)
                completed_values.setdefault(
                    coalition,
                    sum(
                        coefficient
                        for subset, coefficient in mobius_coefficients.items()
                        if subset <= coalition
                    ),
                )

        return completed_values

    def validate(self) -> None:
        super().validate()
        check_k_additivity(self.values, self.k, self.n_elements)
