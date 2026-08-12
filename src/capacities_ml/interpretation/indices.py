# imports
from __future__ import annotations
from itertools import combinations
from math import factorial
from typing import TypeAlias
import numpy as np

# modules
from capacities_ml.capacities import Capacity
from capacities_ml.capacities.utils import normalize_coalition
from capacities_ml.mobius import MobiusRepresentation, mobius_transform

# interpretation aliases
CapacityLike: TypeAlias = Capacity | MobiusRepresentation


# representation conversion
def _as_mobius(capacity: CapacityLike) -> MobiusRepresentation:
    if isinstance(capacity, MobiusRepresentation):
        return capacity
    if isinstance(capacity, Capacity):
        return mobius_transform(capacity)
    raise TypeError("Expected a Capacity or MobiusRepresentation.")


# mobius terms
def _mobius_terms(mobius: MobiusRepresentation):
    for mask in range(1, 1 << mobius.n_elements):
        coalition = frozenset(
            feature
            for feature in range(mobius.n_elements)
            if mask & (1 << feature)
        )
        yield coalition, mobius.value(coalition)


# feature index
def _feature_index(capacity: CapacityLike, feature: int | str) -> int:
    return next(
        iter(normalize_coalition(capacity.universe, feature))
    )


# shapley index
def shapley_index(capacity: CapacityLike, feature: int | str) -> float:
    """Compute the Shapley importance index of one feature."""
    index = _feature_index(capacity, feature)
    mobius = _as_mobius(capacity)
    value = 0.0
    for coalition, coefficient in _mobius_terms(mobius):
        if index in coalition:
            value += coefficient / len(coalition)
    return float(value)


# shapley indices
def shapley_indices(capacity: CapacityLike) -> dict[str, float]:
    """Return Shapley importance indexed by variable name."""
    return {
        name: shapley_index(capacity, index)
        for index, name in enumerate(capacity.var_names)
    }


# pairwise interaction index
def pairwise_interaction_index(
    capacity: CapacityLike,
    first: int | str,
    second: int | str,
) -> float:
    """Compute the Shapley pairwise interaction index ``I_ij``."""
    first_index = _feature_index(capacity, first)
    second_index = _feature_index(capacity, second)
    if first_index == second_index:
        raise ValueError("Pairwise interaction requires two distinct features.")

    mobius = _as_mobius(capacity)
    value = 0.0
    for coalition, coefficient in _mobius_terms(mobius):
        if first_index in coalition and second_index in coalition:
            value += coefficient / (len(coalition) - 1)
    return float(value)


# pairwise interactions
def pairwise_interactions(capacity: CapacityLike) -> dict[tuple[str, str], float]:
    """Return all pairwise interaction indices keyed by variable names."""
    return {
        (capacity.var_names[first], capacity.var_names[second]): pairwise_interaction_index(
            capacity,
            first,
            second,
        )
        for first, second in combinations(range(capacity.n_elements), 2)
    }


# pairwise interaction matrix
def pairwise_interaction_matrix(capacity: CapacityLike) -> np.ndarray:
    """Return a symmetric matrix of pairwise interaction indices."""
    matrix = np.zeros((capacity.n_elements, capacity.n_elements), dtype=float)
    for (first, second), value in pairwise_interactions(capacity).items():
        first_index = capacity.universe.name_to_index[first]
        second_index = capacity.universe.name_to_index[second]
        matrix[first_index, second_index] = value
        matrix[second_index, first_index] = value
    return matrix


# interaction signs
def interaction_signs(capacity: CapacityLike) -> dict[tuple[str, str], int]:
    """Classify pairwise interactions as redundant, neutral or complementary."""
    return {
        pair: int(np.sign(value))
        for pair, value in pairwise_interactions(capacity).items()
    }
