# imports
from collections.abc import Set

import numpy as np

# modules
from capacities_ml.capacities.capacities import Capacity
from capacities_ml.capacities.utils import subset_encoding
from capacities_ml.mobius import MobiusRepresentation

# transform array as vector
def as_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert the input into a one-dimensional numeric vector.
    """
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    return vector


# transform array as matrix
def as_matrix(X: np.ndarray) -> np.ndarray:
    """
    Convert the input into a two-dimensional numeric matrix.
    """
    matrix = np.asarray(X, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")

    return matrix


# capacity values by bitmask
def capacity_values_by_mask(capacity: Capacity) -> np.ndarray:
    """
    Return the capacity values indexed by subset bitmasks.
    """
    n_features = capacity.n_vars
    values = np.empty(1 << n_features, dtype=float)
    values[0] = 0.0

    for coalition_value in capacity.values:
        mask = subset_encoding(
            coalition_value.coalition,
            n_features,
        )
        values[mask] = float(coalition_value.value)

    return values


# mobius terms and coefficients
def mobius_coalitions_and_coefficients(mobius_rep: MobiusRepresentation) -> tuple[tuple[frozenset[int], ...], np.ndarray]:
    """
    Return aligned non-empty coalitions and Möbius coefficients.
    """
    terms = tuple(
        coefficient
        for coefficient in mobius_rep._coefficient_map
        if coefficient.coalition
    )
    coalitions = tuple(term.coalition for term in terms)
    coefficients = np.asarray([term.value for term in terms], dtype=float)
    return coalitions, coefficients


# coalition indices
def coalition_indices(coalition: Set[int], n_features: int) -> tuple[int, ...]:
    """Return validated feature indices for a non-empty coalition."""
    frozen_coalition = frozenset(coalition)
    if not frozen_coalition:
        raise ValueError(
            "The empty coalition cannot be used in the Möbius design matrix."
        )
    if any(
        not isinstance(index, int) or not 0 <= index < n_features
        for index in frozen_coalition
    ):
        raise ValueError(
            f"Coalition {set(frozen_coalition)} is invalid for "
            f"{n_features} features."
        )
    return tuple(sorted(frozen_coalition))
