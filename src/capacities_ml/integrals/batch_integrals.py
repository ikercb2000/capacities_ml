# imports
from collections.abc import Sequence
import numpy as np

# modules
from capacities_ml.capacities.base import Capacity
from capacities_ml.integrals.utils import (
    as_matrix,
    capacity_values_by_mask,
    indices_from_mask,
    mobius_masks_and_coefficients,
)


def batch_choquet_integral(X: np.ndarray, capacity: Capacity) -> np.ndarray:
    """
    Evaluate the discrete Choquet integral row by row.
    """
    matrix = as_matrix(X)
    n_samples, n_features = matrix.shape

    if n_features != capacity.n_features:
        raise ValueError(
            f"The capacity expects {capacity.n_features} features, but X has {n_features}."
        )

    order = np.argsort(matrix,axis=1,kind="stable")
    sorted_X = np.take_along_axis(matrix, order, axis=1)
    previous_values = np.concatenate(
        [
            np.zeros((n_samples, 1), dtype=float),
            sorted_X[:, :-1],
        ],
        axis=1,
    )
    sorted_feature_bits = 1 << order
    coalition_masks = np.flip(
        np.cumsum(
            np.flip(sorted_feature_bits, axis=1),
            axis=1,
        ),
        axis=1,
    )
    values_by_mask = capacity_values_by_mask(capacity)
    coalition_values = values_by_mask[coalition_masks]
    return np.einsum("ti,ti->t", sorted_X - previous_values, coalition_values)

def mobius_design_matrix(X: np.ndarray, coalition_masks: Sequence[int]) -> np.ndarray:
    """
    Construct Z where Z[t, A] = min(X[t, i] : i in A).
    """
    matrix = as_matrix(X)
    n_samples, n_features = matrix.shape
    design = np.empty(shape=(n_samples, len(coalition_masks)),dtype=float)

    for column, mask in enumerate(coalition_masks):
        feature_indices = indices_from_mask(mask,n_features)
        design[:, column] = np.min(matrix[:, feature_indices],axis=1)
    return design


def batch_choquet_integral_mobius(X: np.ndarray, capacity: Capacity, coalition_masks: Sequence[int] | None = None) -> np.ndarray:
    """
    Evaluate several Choquet integrals using the Möbius representation.
    """
    matrix = as_matrix(X)

    if matrix.shape[1] != capacity.n_features:
        raise ValueError(
            f"The capacity expects {capacity.n_features} features, but X has {matrix.shape[1]}."
        )

    masks, coefficients = mobius_masks_and_coefficients(capacity,coalition_masks=coalition_masks)
    design = mobius_design_matrix(matrix,coalition_masks=masks)

    if coefficients.shape != (design.shape[1],):
        raise ValueError(
            "The number of Möbius coefficients does not match the number of coalitions."
        )

    return design @ coefficients
