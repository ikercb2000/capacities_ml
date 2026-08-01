# imports
from collections.abc import Sequence

import numpy as np

# modules
from capacities_ml.capacities.base import Capacity
from capacities_ml.capacities.utils import subset_decoding, subset_encoding

# transform array as vector
def as_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert the input into a one-dimensional numeric vector.
    """
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    return vector


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


def capacity_values_by_mask(capacity: Capacity) -> np.ndarray:
    """
    Return the capacity values indexed by subset bitmasks.
    """
    n_features = capacity.n_features
    values = np.empty(1 << n_features, dtype=float)
    values[0] = 0.0

    for coalition_value in capacity.subset_values:
        mask = subset_encoding(
            coalition_value.coalition,
            n_features,
        )
        values[mask] = float(coalition_value.value)

    return values


def mobius_masks_and_coefficients(
    capacity: Capacity,
    coalition_masks: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], np.ndarray]:
    """
    Return aligned coalition masks and Möbius coefficients.
    """
    mobius_rep = capacity.mobius_rep()

    if coalition_masks is None:
        masks = tuple(
            subset_encoding(coefficient.coalition, capacity.n_features)
            for coefficient in mobius_rep
        )
        coefficients = np.asarray(
            [coefficient.value for coefficient in mobius_rep],
            dtype=float,
        )
        return masks, coefficients

    masks = tuple(coalition_masks)
    coefficients = np.empty(len(masks), dtype=float)

    for index, mask in enumerate(masks):
        coalition = subset_decoding(mask, capacity.n_features)
        coefficient = mobius_rep.get_value(coalition)

        if coefficient is None:
            raise ValueError(
                f"Missing Möbius coefficient for coalition {set(coalition)}."
            )

        coefficients[index] = coefficient

    return masks, coefficients


def indices_from_mask(mask: int, n_features: int) -> tuple[int, ...]:
    """
    Return the feature indices contained in a bitmask.
    """
    if mask <= 0:
        raise ValueError(
            "The empty coalition cannot be used in the Möbius design matrix."
        )

    if mask >= 1 << n_features:
        raise ValueError(
            f"Mask {mask} is invalid for {n_features} features."
        )

    return tuple(
        index
        for index in range(n_features)
        if mask & (1 << index)
    )
