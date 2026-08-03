# imports
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np

# modules
from capacities_ml.optimization.capacity_constraints.capacity_constraints import (
    CapacityParameterization,
)
from capacities_ml.optimization.constraints import (
    ConstraintBundle,
    LinearConstraintSystem,
    VariableBounds,
)
from capacities_ml.optimization.enums import CapacityRepresentation, CapacityShape

# coalition mask
def coalition_masks(n_features: int, max_order: int | None = None) -> tuple[int, ...]:
    """Return non-empty coalition bitmasks in cardinality/lexicographic order."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if max_order is None:
        max_order = n_features
    if not 1 <= max_order <= n_features:
        raise ValueError("max_order must satisfy 1 <= max_order <= n_features.")
    masks: list[int] = []
    for order in range(1, max_order + 1):
        for indices in combinations(range(n_features), order):
            mask = 0
            for index in indices:
                mask |= 1 << index
            masks.append(mask)
    return tuple(masks)


# immediate coalition supersets
def immediate_supersets(mask: int, n_features: int) -> tuple[int, ...]:
    return tuple(
        mask | (1 << feature)
        for feature in range(n_features)
        if not mask & (1 << feature)
    )


# capacity value constraints
def capacity_value_constraints(n_features: int) -> CapacityParameterization:
    """Build normalization, range and monotonicity constraints for ``nu(A)``."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    n_subsets = 1 << n_features
    full_mask = n_subsets - 1

    lower = np.zeros(n_subsets, dtype=float)
    upper = np.ones(n_subsets, dtype=float)
    lower[0] = upper[0] = 0.0
    lower[full_mask] = upper[full_mask] = 1.0
    bounds = VariableBounds(lower, upper)

    rows: list[np.ndarray] = []
    for mask in range(n_subsets):
        for larger_mask in immediate_supersets(mask, n_features):
            row = np.zeros(n_subsets, dtype=float)
            row[larger_mask] = 1.0
            row[mask] = -1.0
            rows.append(row)

    monotonicity = LinearConstraintSystem.lower_bounded(
        np.vstack(rows),
        np.zeros(len(rows), dtype=float),
        name="capacity_value_monotonicity",
    )
    return CapacityParameterization(
        constraints=ConstraintBundle(
            bounds=bounds,
            linear_constraints=(monotonicity,),
        ),
        parameter_masks=tuple(range(n_subsets)),
        representation=CapacityRepresentation.VALUES,
        n_features=n_features,
        max_order=n_features,
    )


# Mobius capacity constraints
def mobius_capacity_constraints(
    n_features: int,
    max_order: int,
    *,
    shape: CapacityShape = CapacityShape.GENERAL,
) -> CapacityParameterization:
    """Build constraints for an at-most-``k`` additive capacity."""
    if not isinstance(shape, CapacityShape):
        raise TypeError("shape must be a CapacityShape enum member.")

    masks = coalition_masks(n_features, max_order)
    position = {mask: index for index, mask in enumerate(masks)}
    n_parameters = len(masks)

    normalization = LinearConstraintSystem.equality(
        np.ones((1, n_parameters), dtype=float),
        np.ones(1, dtype=float),
        name="mobius_normalization",
    )

    rows: list[np.ndarray] = []
    full_mask = (1 << n_features) - 1
    for feature in range(n_features):
        feature_bit = 1 << feature
        remaining_mask = full_mask ^ feature_bit
        submask = remaining_mask
        while True:
            # Delta_i nu(A) = sum_{B subseteq A} m(B union {i}) >= 0.
            row = np.zeros(n_parameters, dtype=float)
            candidate = submask
            while True:
                term = candidate | feature_bit
                term_position = position.get(term)
                if term_position is not None:
                    row[term_position] = 1.0
                if candidate == 0:
                    break
                candidate = (candidate - 1) & submask
            rows.append(row)
            if submask == 0:
                break
            submask = (submask - 1) & remaining_mask

    monotonicity = LinearConstraintSystem.lower_bounded(
        np.vstack(rows),
        np.zeros(len(rows), dtype=float),
        name="mobius_capacity_monotonicity",
    )

    systems: list[LinearConstraintSystem] = [normalization, monotonicity]
    if shape is not CapacityShape.GENERAL:
        if max_order != 2:
            raise ValueError(
                "Convex/concave sign constraints are implemented only for "
                "2-additive capacities."
            )
        pair_positions = [
            index for index, mask in enumerate(masks) if mask.bit_count() == 2
        ]
        matrix = np.zeros((len(pair_positions), n_parameters), dtype=float)
        matrix[np.arange(len(pair_positions)), pair_positions] = 1.0
        if shape is CapacityShape.CONVEX:
            systems.append(
                LinearConstraintSystem.lower_bounded(
                    matrix,
                    np.zeros(len(pair_positions)),
                    name="convex_pairwise_mobius",
                )
            )
        elif shape is CapacityShape.CONCAVE:
            systems.append(
                LinearConstraintSystem.upper_bounded(
                    matrix,
                    np.zeros(len(pair_positions)),
                    name="concave_pairwise_mobius",
                )
            )

    # A normalized capacity has coefficients bounded by the number of
    # alternating terms in the corresponding Mobius transform.
    coefficient_bound = np.asarray(
        [2.0 ** (mask.bit_count() - 1) for mask in masks],
        dtype=float,
    )
    return CapacityParameterization(
        constraints=ConstraintBundle(
            bounds=VariableBounds(-coefficient_bound, coefficient_bound),
            linear_constraints=tuple(systems),
        ),
        parameter_masks=masks,
        representation=CapacityRepresentation.MOBIUS,
        n_features=n_features,
        max_order=max_order,
    )


# interaction parameter mask
def interaction_parameter_mask(parameter_masks: tuple[int, ...]) -> np.ndarray:
    return np.asarray([mask.bit_count() >= 2 for mask in parameter_masks], dtype=bool)


# stable autoregressive bounds
def stable_ar_bounds(
    n_capacity_parameters: int,
    *,
    phi_position: int,
    total_parameters: int,
    epsilon: float = 1e-6,
) -> VariableBounds:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1).")
    if not 0 <= phi_position < total_parameters:
        raise ValueError("phi_position is outside the parameter vector.")
    if n_capacity_parameters > total_parameters:
        raise ValueError("n_capacity_parameters cannot exceed total_parameters.")
    lower = np.full(total_parameters, -np.inf)
    upper = np.full(total_parameters, np.inf)
    lower[phi_position] = -1.0 + epsilon
    upper[phi_position] = 1.0 - epsilon
    return VariableBounds(lower, upper)


# direct Choquet threshold bounds
def direct_choquet_threshold_bounds(
    *,
    total_parameters: int,
    threshold_positions: tuple[int, ...],
) -> VariableBounds:
    lower = np.full(total_parameters, -np.inf)
    upper = np.full(total_parameters, np.inf)
    for position in threshold_positions:
        if not 0 <= position < total_parameters:
            raise ValueError("A threshold position is outside the parameter vector.")
        lower[position] = 0.0
        upper[position] = 1.0
    return VariableBounds(lower, upper)
