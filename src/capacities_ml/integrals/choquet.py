# imports
import numpy as np

# modules
from capacities_ml.capacities.capacities import Capacity
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.integrals.utils import as_vector, coalition_indices

# ordered choquet integral
def ordered_choquet(capacity: Capacity, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a capacity.
    """
    vector = as_vector(x)
    permutation = np.argsort(vector)
    sorted_x = vector[permutation] # ordered vector

    choquet_value = 0.0
    previous = 0.0

    for position, value in enumerate(sorted_x):
        coalition = frozenset(int(index) for index in permutation[position:])
        capacity_value = capacity.value(coalition)
        choquet_value += (value - previous) * capacity_value
        previous = value

    return float(choquet_value)

# integral with mobius coefficients
def mobius_choquet(mobius_rep: MobiusRepresentation, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a Möbius representation.
    """
    vector = as_vector(x)
    if vector.size != mobius_rep.n_vars:
        raise ValueError(
            f"The Möbius representation expects {mobius_rep.n_vars} variables, "
            f"but x has {vector.size}."
        )

    choquet_value = 0.0

    for coefficient in mobius_rep._coefficient_map:
        coalition = coefficient.coalition

        if not coalition:
            continue
        feature_indices = coalition_indices(coalition, vector.size)
        minimum_value = float(np.min(vector[list(feature_indices)]))
        choquet_value += coefficient.value * minimum_value

    return float(choquet_value)
