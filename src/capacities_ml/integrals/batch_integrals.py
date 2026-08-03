# imports
from collections.abc import Sequence, Set
import numpy as np

# modules
from capacities_ml.capacities.capacities import Capacity
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.integrals.utils import (
    as_matrix,
    capacity_values_by_mask,
    coalition_indices,
    mobius_coalitions_and_coefficients,
)

# matrix-like choquet integral
def batch_choquet_integral(X: np.ndarray, capacity: Capacity) -> np.ndarray:
    """
    Evaluate the discrete Choquet integral row by row.
    """
    matrix = as_matrix(X)
    n_samples, n_features = matrix.shape
    if n_features != capacity.n_vars:
        raise ValueError(
            f"The capacity expects {capacity.n_vars} features, but X has {n_features}."
        )

    order = np.argsort(matrix,axis=1,kind="stable") # indices in order
    sorted_X = np.take_along_axis(matrix, order, axis=1) # order inside matrix
    previous_values = np.concatenate(
        [
            np.zeros((n_samples, 1), dtype=float),
            sorted_X[:, :-1],
        ],
        axis=1,
    ) # previous values to compute difference for integral
    sorted_feature_bits = 1 << order
    coalition_masks = np.flip(np.cumsum(np.flip(sorted_feature_bits, axis=1),axis=1),axis=1)
    values_by_mask = capacity_values_by_mask(capacity)
    coalition_values = values_by_mask[coalition_masks]
    return np.einsum("ti,ti->t", sorted_X - previous_values, coalition_values)

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
    if matrix.shape[1] != mobius_rep.n_vars:
        raise ValueError(
            f"The Möbius representation expects {mobius_rep.n_vars} variables, "
            f"but X has {matrix.shape[1]}."
        )

    coalitions, coefficients = mobius_coalitions_and_coefficients(mobius_rep)
    design = mobius_design_matrix(matrix, coalitions=coalitions)

    if coefficients.shape != (design.shape[1],):
        raise ValueError(
            "The number of Möbius coefficients does not match the number of coalitions."
        )

    return design @ coefficients
