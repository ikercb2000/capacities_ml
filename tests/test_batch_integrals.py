import numpy as np
import pytest

from capacities_ml.capacities import (
    ExplicitCapacity,
    VariableUniverse,
    mobius_transform,
)
from capacities_ml.integrals.batch_integrals import (
    batch_choquet_integral,
    batch_choquet_integral_mobius,
    mobius_design_matrix,
)
from capacities_ml.integrals.choquet import mobius_choquet, ordered_choquet
from capacities_ml.integrals.utils import coalition_indices
from capacities_ml.risk import ProbabilityCapacity


def build_capacity() -> ExplicitCapacity:
    return ExplicitCapacity(
        universe=VariableUniverse(("x0", "x1", "x2")),
        values={
            ("x0",): 0.2,
            ("x1",): 0.3,
            ("x2",): 0.4,
            ("x0", "x1"): 0.6,
            ("x0", "x2"): 0.7,
            ("x1", "x2"): 0.8,
            ("x0", "x1", "x2"): 1.0,
        },
    )


def test_batch_choquet_integral_matches_rowwise_ordered_choquet():
    capacity = build_capacity()
    X = np.array(
        [
            [0.2, 0.8, 0.6],
            [0.7, 0.4, 0.9],
        ]
    )

    batch_values = batch_choquet_integral(X, capacity)
    rowwise_values = np.array(
        [ordered_choquet(capacity, row) for row in X],
        dtype=float,
    )

    assert np.allclose(batch_values, rowwise_values)


def test_batch_choquet_integral_supports_dynamic_capacities():
    X = np.array([[1.0, 3.0], [2.0, 4.0]])
    capacity = ProbabilityCapacity([0.25, 0.75])

    values = batch_choquet_integral(X, capacity)

    np.testing.assert_allclose(values, [2.5, 3.5])


def test_coalition_indices_returns_expected_indices():
    assert coalition_indices(frozenset({1, 2}), 3) == (1, 2)


def test_coalition_indices_rejects_empty_coalition():
    with pytest.raises(ValueError, match="empty coalition"):
        coalition_indices(frozenset(), 3)


def test_mobius_design_matrix_matches_expected_example():
    X = np.array(
        [
            [0.2, 0.8, 0.6],
            [0.7, 0.4, 0.9],
        ]
    )
    coalitions = (
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 2}),
    )

    design = mobius_design_matrix(X, coalitions)

    expected = np.array(
        [
            [0.2, 0.8, 0.6, 0.2, 0.2, 0.6],
            [0.7, 0.4, 0.9, 0.4, 0.7, 0.4],
        ]
    )

    assert np.allclose(design, expected)


def test_batch_choquet_integral_mobius_matches_rowwise_mobius_choquet():
    capacity = build_capacity()
    X = np.array(
        [
            [0.2, 0.8, 0.6],
            [0.7, 0.4, 0.9],
        ]
    )

    mobius_capacity = mobius_transform(capacity)
    batch_values = batch_choquet_integral_mobius(X, mobius_capacity)
    rowwise_values = np.array(
        [mobius_choquet(mobius_capacity, row) for row in X],
        dtype=float,
    )

    assert np.allclose(batch_values, rowwise_values)
    assert np.allclose(batch_choquet_integral(X, mobius_capacity), rowwise_values)
