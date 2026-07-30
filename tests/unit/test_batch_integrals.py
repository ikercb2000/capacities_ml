from dataclasses import dataclass

import numpy as np
import pytest

from capacities_ml.capacities.base import Capacity
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
from capacities_ml.integrals.batch_integrals import (
    batch_choquet_integral,
    batch_choquet_integral_mobius,
    indices_from_mask,
    mobius_design_matrix,
)
from capacities_ml.integrals.choquet import mobius_choquet, ordered_choquet


@dataclass
class DummyCapacity(Capacity):
    subset_values: CapacityMap
    n_features: int
    feature_names: tuple[str, ...]

    def value(self, subset):
        return super().value(subset)

    def values(self):
        return super().values()

    def mobius_rep(self):
        return super().mobius_rep()

    def validate(self):
        return None


def build_capacity() -> DummyCapacity:
    return DummyCapacity(
        subset_values=CapacityMap(
            capacities=[
                CoalitionValue(frozenset({0}), 0.2),
                CoalitionValue(frozenset({1}), 0.3),
                CoalitionValue(frozenset({2}), 0.4),
                CoalitionValue(frozenset({0, 1}), 0.6),
                CoalitionValue(frozenset({0, 2}), 0.7),
                CoalitionValue(frozenset({1, 2}), 0.8),
                CoalitionValue(frozenset({0, 1, 2}), 1.0),
            ]
        ),
        n_features=3,
        feature_names=("x0", "x1", "x2"),
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


def test_indices_from_mask_returns_expected_indices():
    assert indices_from_mask(6, 3) == (1, 2)


def test_indices_from_mask_rejects_empty_coalition():
    with pytest.raises(ValueError, match="empty coalition"):
        indices_from_mask(0, 3)


def test_mobius_design_matrix_matches_expected_example():
    X = np.array(
        [
            [0.2, 0.8, 0.6],
            [0.7, 0.4, 0.9],
        ]
    )
    coalition_masks = (1, 2, 4, 3, 5, 6)

    design = mobius_design_matrix(X, coalition_masks)

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

    batch_values = batch_choquet_integral_mobius(X, capacity)
    rowwise_values = np.array(
        [mobius_choquet(capacity, row) for row in X],
        dtype=float,
    )

    assert np.allclose(batch_values, rowwise_values)
