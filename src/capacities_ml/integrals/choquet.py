# imports
import numpy as np

# modules
from capacities_ml.capacities.base import Capacity
from capacities_ml.integrals.utils import as_vector

# ordered choquet integral
def ordered_choquet(capacity: Capacity, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a capacity.
    """
    vector = as_vector(x)
    permutation = np.argsort(vector)
    sorted_x = vector[permutation]

    choquet_value = 0.0
    previous = 0.0

    for position, value in enumerate(sorted_x):
        coalition = frozenset(int(index) for index in permutation[position:])
        capacity_value = capacity.subset_values.get_value(coalition)

        if capacity_value is None:
            raise ValueError(
                f"Missing coalition in capacity map: {set(coalition)}."
            )

        choquet_value += (value - previous) * capacity_value
        previous = value

    return float(choquet_value)

# integral with mobius coefficients
def mobius_choquet(capacity: Capacity, x: np.ndarray) -> float:
    """
    Compute the discrete Choquet integral from a Möbius representation.
    """
    vector = as_vector(x)
    mobius_rep = capacity.mobius_rep()
    choquet_value = 0.0

    for coefficient in mobius_rep:
        coalition = coefficient.coalition

        if not coalition:
            continue

        minimum_value = float(np.min(vector[list(coalition)]))
        choquet_value += coefficient.value * minimum_value

    return float(choquet_value)
