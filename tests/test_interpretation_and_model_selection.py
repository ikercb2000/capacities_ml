import numpy as np

from capacities_ml.capacities import VariableUniverse
from capacities_ml.interpretation import (
    pairwise_interaction_index,
    shapley_indices,
)
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.model_selection import (
    capacity_parameter_grid,
    capacity_shape_grid,
    interaction_order_grid,
)
from capacities_ml.optimization import (
    CapacityShape,
    PairwiseInteractionSparsity,
)


def test_shapley_and_pairwise_interaction_indices_use_mobius_coefficients():
    universe = VariableUniverse(("x0", "x1", "x2"))
    mobius = MobiusRepresentation(
        universe,
        {
            frozenset({0}): 0.3,
            frozenset({1}): 0.3,
            frozenset({2}): 0.2,
            frozenset({0, 1}): 0.2,
        },
    )

    shapley = shapley_indices(mobius)

    np.testing.assert_allclose(shapley["x0"], 0.4)
    np.testing.assert_allclose(shapley["x1"], 0.4)
    np.testing.assert_allclose(shapley["x2"], 0.2)
    np.testing.assert_allclose(pairwise_interaction_index(mobius, "x0", "x1"), 0.2)


def test_model_selection_builds_sklearn_ready_capacity_grids():
    shapes = capacity_shape_grid(order=2)
    orders = interaction_order_grid((1, 2, 3))
    grid = capacity_parameter_grid(
        parameter_name="model__sparsity",
        orders=(1, 2),
        shapes=(CapacityShape.GENERAL,),
    )

    assert len(shapes) == 3
    assert len(orders) == 3
    assert len(grid["model__sparsity"]) == 2


def test_pairwise_interaction_grid_returns_capacity_sparsity_objects():
    candidates = [
        PairwiseInteractionSparsity(order=2, pairs=((0, 1),)),
    ]

    compilation = candidates[0].compile(3)

    assert compilation.bundle.constraints.linear_constraints[-1].name == (
        "pairwise_interaction_constraints"
    )
