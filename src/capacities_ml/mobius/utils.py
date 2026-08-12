# imports
from collections.abc import Set
from itertools import combinations

# modules
from capacities_ml.capacities.capacities import Capacity
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
from capacities_ml.mobius.mobius import MobiusRepresentation
from capacities_ml.mobius.types import MobiusMap


# powerset
def powerset(elements: Set[int]) -> list[frozenset[int]]:
    """Compute the power set of a coalition."""
    items = tuple(elements)
    return [
        frozenset(subset)
        for size in range(len(items) + 1)
        for subset in combinations(items, size)
    ]


# internal mobius transform
def _mobius_transform_map(capacities: CapacityMap) -> MobiusMap:
    """Transform an internal capacity map into an internal Möbius map."""
    mobius_coefficients: list[CoalitionValue] = []
    for capacity_value in capacities:
        mobius_coefficient = 0.0
        for subset in powerset(capacity_value.coalition):
            subset_value = capacities.get_value(subset)
            if subset_value is None:
                raise ValueError(
                    f"Missing coalition in capacity map: {set(subset)}."
                )
            mobius_coefficient += (
                (-1) ** (len(capacity_value.coalition) - len(subset))
            ) * subset_value
        mobius_coefficients.append(
            CoalitionValue(capacity_value.coalition, mobius_coefficient)
        )

    return MobiusMap(mobius_coefficients=mobius_coefficients)


# mobius transform
def mobius_transform(capacity: Capacity) -> MobiusRepresentation:
    """Compute the Möbius representation of a capacity."""
    if not isinstance(capacity, Capacity):
        raise TypeError("capacity must be a Capacity instance.")

    coefficient_map = _mobius_transform_map(capacity.values)
    return MobiusRepresentation(
        universe=capacity.universe,
        coefficients=coefficient_map.to_lookup(),
    )


# inverse mobius transform
def inverse_mobius_transform(mobius_rep: MobiusRepresentation) -> Capacity:
    """Compute a capacity from its Möbius representation."""
    if not isinstance(mobius_rep, MobiusRepresentation):
        raise TypeError("mobius_rep must be a MobiusRepresentation instance.")

    features = frozenset(range(mobius_rep.n_elements))
    values: dict[frozenset[int], float] = {}
    for coalition in powerset(features):
        if coalition:
            values[coalition] = sum(
                mobius_rep.value(subset)
                for subset in powerset(coalition)
            )

    return Capacity(universe=mobius_rep.universe, values=values)
