# imports
from collections.abc import Sequence, Set
import numpy as np

# modules
from capacities_ml.capacities import BaseCapacity, ExplicitCapacity
from capacities_ml.integrals.choquet import ordered_choquet
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.integrals.utils import (
    as_matrix,
    capacity_values_by_mask,
    coalition_indices,
    mobius_coalitions_and_coefficients,
)

# matrix-like choquet integral
def capacity_design_matrix(X: np.ndarray) -> np.ndarray:
    """Build the linear design matrix for direct capacity values."""
    matrix = as_matrix(X)
    n_samples, n_features = matrix.shape
    order = np.argsort(matrix, axis=1, kind="stable")
    sorted_values = np.take_along_axis(matrix, order, axis=1)
    previous_values = np.concatenate(
        [
            np.zeros((n_samples, 1), dtype=float),
            sorted_values[:, :-1],
        ],
        axis=1,
    )
    sorted_feature_bits = 1 << order
    masks = np.flip(
        np.cumsum(np.flip(sorted_feature_bits, axis=1), axis=1),
        axis=1,
    )
    design = np.zeros((n_samples, 1 << n_features), dtype=float)
    rows = np.arange(n_samples)[:, None]
    design[rows, masks] = sorted_values - previous_values
    return design


# matrix-like choquet integral
def batch_choquet_integral(X: np.ndarray, capacity: BaseCapacity) -> np.ndarray:
    """
    Evaluate the discrete Choquet integral row by row.
    """
    matrix = as_matrix(X)
    _, n_features = matrix.shape
    if not isinstance(capacity, BaseCapacity):
        raise TypeError("capacity must be a BaseCapacity instance.")
    if n_features != capacity.n_elements:
        raise ValueError(
            f"The capacity expects {capacity.n_elements} elements, but X has {n_features}."
        )

    if isinstance(capacity, ExplicitCapacity):
        values_by_mask = capacity_values_by_mask(capacity)
        return capacity_design_matrix(matrix) @ values_by_mask
    return np.asarray([ordered_choquet(capacity, row) for row in matrix])

# mobius design matrix
def mobius_design_matrix(X: np.ndarray,coalitions: Sequence[Set[int]]) -> np.ndarray:
    """
    Construct Z where Z[t, A] = min(X[t, i] : i in A).
    """
    matrix = as_matrix(X)
    n_samples, n_features = matrix.shape
    design = np.empty(shape=(n_samples, len(coalitions)), dtype=float)

    for column, coalition in enumerate(coalitions):
        feature_indices = coalition_indices(coalition, n_features)
        design[:, column] = np.min(matrix[:, feature_indices], axis=1)
    return design


# batch integral with mobius coefficients
def batch_choquet_integral_mobius(X: np.ndarray, mobius_rep: MobiusRepresentation) -> np.ndarray:
    """
    Evaluate several Choquet integrals using the Möbius representation.
    """
    matrix = as_matrix(X)
    if matrix.shape[1] != mobius_rep.n_elements:
        raise ValueError(
            f"The Möbius representation expects {mobius_rep.n_elements} variables, "
            f"but X has {matrix.shape[1]}."
        )

    coalitions, coefficients = mobius_coalitions_and_coefficients(mobius_rep)
    design = mobius_design_matrix(matrix, coalitions=coalitions)

    if coefficients.shape != (design.shape[1],):
        raise ValueError(
            "The number of Möbius coefficients does not match the number of coalitions."
        )

    return design @ coefficients
