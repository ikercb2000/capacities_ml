# imports
import numpy as np

# modules
from capacities_ml.capacities import BaseCapacity, MobiusCapacity
from capacities_ml.integrals.utils import as_vector, coalition_indices

# ordered choquet integral
def ordered_choquet(capacity: BaseCapacity, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a capacity.
    """
    if not isinstance(capacity, BaseCapacity):
        raise TypeError("capacity must be a BaseCapacity instance.")
    vector = as_vector(x)
    if vector.size != capacity.n_elements:
        raise ValueError(
            f"The capacity expects {capacity.n_elements} elements, "
            f"but x has {vector.size}."
        )
    permutation = np.argsort(vector, kind="stable")
    sorted_values = vector[permutation]
    increments = np.diff(np.concatenate(([0.0], sorted_values)))
    capacity_values = np.asarray(
        capacity.nested_event_values(permutation),
        dtype=float,
    )
    if capacity_values.shape != vector.shape:
        raise ValueError("nested_event_values returned an incompatible shape.")
    return float(np.dot(increments, capacity_values))

# integral with mobius coefficients
def mobius_choquet(mobius_capacity: MobiusCapacity, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a Möbius representation.
    """
    vector = as_vector(x)
    if vector.size != mobius_capacity.n_elements:
        raise ValueError(
            f"The Möbius capacity expects {mobius_capacity.n_elements} variables, "
            f"but x has {vector.size}."
        )

    choquet_value = 0.0

    for coefficient in mobius_capacity._coefficient_map:
        coalition = coefficient.coalition

        if not coalition:
            continue
        feature_indices = coalition_indices(coalition, vector.size)
        minimum_value = float(np.min(vector[list(feature_indices)]))
        choquet_value += coefficient.value * minimum_value

    return float(choquet_value)
