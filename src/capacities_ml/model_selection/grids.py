# imports
from __future__ import annotations
from collections.abc import Iterable

# modules
from capacities_ml.optimization.enums import CapacityShape
from capacities_ml.optimization.sparsity import (
    CapacitySparsity,
    KAdditivity,
    PairwiseInteractionSparsity,
)


# capacity sparsity grid
def capacity_sparsity_grid(
    orders: Iterable[int],
    *,
    shapes: Iterable[CapacityShape] = (CapacityShape.GENERAL,),
) -> list[CapacitySparsity]:
    """Build sparsity candidates for capacity model selection."""
    order_values = tuple(orders)
    shape_values = tuple(shapes)
    if not order_values:
        raise ValueError("orders must contain at least one value.")
    if not shape_values:
        raise ValueError("shapes must contain at least one value.")
    if any(not isinstance(order, int) or order < 1 for order in order_values):
        raise ValueError("orders must contain positive integers.")
    if any(not isinstance(shape, CapacityShape) for shape in shape_values):
        raise TypeError("shapes must contain CapacityShape enum members.")
    if any(
        shape is not CapacityShape.GENERAL and order != 2
        for shape in shape_values
        for order in order_values
    ):
        raise ValueError(
            "Convex and concave capacity shapes are currently available only "
            "for order=2."
        )
    return [
        KAdditivity(order=order, shape=shape)
        for order in order_values
        for shape in shape_values
    ]


# capacity shape grid
def capacity_shape_grid(
    *,
    order: int = 2,
    shapes: Iterable[CapacityShape] = (
        CapacityShape.GENERAL,
        CapacityShape.CONVEX,
        CapacityShape.CONCAVE,
    ),
) -> list[CapacitySparsity]:
    """Build candidates that compare capacity shapes at one order."""
    return capacity_sparsity_grid((order,), shapes=shapes)


# interaction order grid
def interaction_order_grid(
    orders: Iterable[int],
    *,
    shape: CapacityShape = CapacityShape.GENERAL,
) -> list[CapacitySparsity]:
    """Build candidates that compare the maximum interaction order."""
    return capacity_sparsity_grid(orders, shapes=(shape,))


# pairwise interaction grid
def pairwise_interaction_grid(
    orders: Iterable[int],
    *,
    pairs: tuple[tuple[int, int], ...] | None = None,
    target: float = 0.0,
    shape: CapacityShape = CapacityShape.GENERAL,
) -> list[CapacitySparsity]:
    """Build candidates with selected pairwise interactions fixed."""
    order_values = tuple(orders)
    if any(order < 2 for order in order_values):
        raise ValueError("Pairwise interaction grids require order >= 2.")
    if shape is not CapacityShape.GENERAL and any(order != 2 for order in order_values):
        raise ValueError(
            "Convex and concave capacity shapes are currently available only "
            "for order=2."
        )
    return [
        PairwiseInteractionSparsity(
            order=order,
            pairs=pairs,
            target=target,
            shape=shape,
        )
        for order in order_values
    ]


# sklearn parameter grid
def capacity_parameter_grid(
    *,
    parameter_name: str = "sparsity",
    orders: Iterable[int] = (1, 2),
    shapes: Iterable[CapacityShape] = (CapacityShape.GENERAL,),
) -> dict[str, list[CapacitySparsity]]:
    """Return a parameter grid ready for ``GridSearchCV``."""
    if not parameter_name:
        raise ValueError("parameter_name cannot be empty.")
    return {
        parameter_name: capacity_sparsity_grid(orders, shapes=shapes),
    }
