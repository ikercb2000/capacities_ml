# imports
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_array

# modules
from capacities_ml.capacities import VariableUniverse
from capacities_ml.integrals.batch_integrals import (
    capacity_design_matrix,
    mobius_design_matrix,
)
from capacities_ml.optimization.enums import CapacityRepresentation


# feature validation
def validate_features(
    X: ArrayLike,
    universe: VariableUniverse,
    *,
    fitting: bool = False,
) -> np.ndarray:
    """Validate feature matrices using scikit-learn conventions."""
    matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=1)
    if matrix.shape[1] != universe.n_elements:
        raise ValueError(
            f"X has {matrix.shape[1]} features; expected {universe.n_elements}."
        )
    if fitting and not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")
    return matrix


# capacity design matrix
def capacity_design(
    X: np.ndarray,
    parameter_masks: tuple[int, ...],
    representation: CapacityRepresentation,
) -> np.ndarray:
    """Build a design matrix for the selected capacity representation."""
    if representation is CapacityRepresentation.VALUES:
        return capacity_design_matrix(X)[:, parameter_masks]
    coalitions = [
        frozenset(
            feature for feature in range(X.shape[1]) if mask & (1 << feature)
        )
        for mask in parameter_masks
    ]
    return mobius_design_matrix(X, coalitions)
