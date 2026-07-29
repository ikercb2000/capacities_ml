# imports
from collections.abc import Iterable, Set as AbstractSet
from itertools import combinations

# modules
from capacities_ml.capacities.types import CapacityMap, CoalitionValue, MobiusMap


# powerset
def powerset(elements: AbstractSet[int]) -> list[frozenset[int]]:
    """
    Computes the power set for any set.
    """
    items = tuple(elements)
    return [
        frozenset(subset)
        for size in range(len(items) + 1)
        for subset in combinations(items, size)
    ]


# möbius transform from capacities
def mobius_transform(capacities: CapacityMap, n_features: int):
    """
    Computes the Möbius transform representation for all coalitions.
    """
    del n_features

    mobius_rep = []
    for cap in capacities:
        subsets = powerset(cap.coalition)
        mobius_coef = 0
        for sub in subsets:
            sub_value = capacities.get_value(sub)
            if sub_value is None:
                raise ValueError(f"Missing coalition in capacity map: {set(sub)}.")
            mobius_coef += ((-1) ** (len(cap.coalition) - len(sub))) * sub_value
        mobius_rep.append(CoalitionValue(cap.coalition, mobius_coef))
    return MobiusMap(mobius_coefficients=mobius_rep)


# inverse möbius transform
def inverse_mobius_transform(mobius_rep: MobiusMap):
    """
    Computes the inverse Möbius transform representation for all coalitions.
    """
    capacities = []
    for mob in mobius_rep:
        subsets = powerset(mob.coalition)
        cap_value = 0
        for sub in subsets:
            sub_value = mobius_rep.get_value(sub)
            if sub_value is None:
                continue
            cap_value += sub_value
        capacities.append(CoalitionValue(mob.coalition, cap_value))
    return CapacityMap(capacities=capacities)
