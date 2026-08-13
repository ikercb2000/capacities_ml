# imports
from __future__ import annotations
from typing import Any
import numpy as np

# modules
from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.base.integrals.batch_integrals import (
    capacity_design_matrix,
    mobius_design_matrix,
)
from capacities_ml_fin.ml.optimization.enums import CapacityRepresentation


# fitted universe
def fitted_universe(estimator: Any) -> VariableUniverse:
    """Build a variable universe from validated estimator input metadata."""
    if hasattr(estimator, "feature_names_in_"):
        names = tuple(str(name) for name in estimator.feature_names_in_)
    else:
        return VariableUniverse.from_size(estimator.n_features_in_)
    return VariableUniverse(names)


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
