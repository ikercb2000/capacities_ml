import numpy as np

from capacities_ml.capacities import MobiusRepresentation, VariableUniverse
from capacities_ml.interpretation import (
    interaction_signs,
    pairwise_interaction_index,
    pairwise_interaction_matrix,
    pairwise_interactions,
    shapley_index,
    shapley_indices,
)
from capacities_ml.model_selection import (
    capacity_parameter_grid,
    capacity_shape_grid,
    interaction_order_grid,
)
from capacities_ml.optimization import (
    CapacityShape,
    PairwiseInteractionSparsity,
)
from capacities_ml.risk import ProbabilityCapacity


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


def test_interpretation_accepts_dynamic_base_capacity_via_event_values():
    capacity = ProbabilityCapacity([0.2, 0.3, 0.5])

    assert np.isclose(shapley_index(capacity, 2), 0.5)
    shapley = shapley_indices(capacity)
    interactions = pairwise_interactions(capacity)

    assert set(shapley) == {0, 1, 2}
    np.testing.assert_allclose(list(shapley.values()), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(list(interactions.values()), 0.0, atol=1e-12)
    np.testing.assert_allclose(pairwise_interaction_matrix(capacity), 0.0, atol=1e-12)
    assert interaction_signs(capacity) == {
        (0, 1): 0,
        (0, 2): 0,
        (1, 2): 0,
    }


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
