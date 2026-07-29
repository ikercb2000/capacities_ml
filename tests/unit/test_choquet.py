from dataclasses import dataclass

import numpy as np

from capacities_ml.capacities.base import Capacity
from capacities_ml.capacities.mobius import mobius_transform
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
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
                CoalitionValue(frozenset({1}), 0.5),
                CoalitionValue(frozenset({0, 1}), 1.0),
            ]
        ),
        n_features=2,
        feature_names=("x0", "x1"),
    )


def test_ordered_choquet_matches_manual_value():
    capacity = build_capacity()
    x = np.array([3.0, 1.0])

    choquet_value = ordered_choquet(capacity, x)

    assert np.isclose(choquet_value, 1.4)


def test_mobius_choquet_matches_ordered_choquet():
    capacity = build_capacity()
    x = np.array([3.0, 1.0])

    ordered_value = ordered_choquet(capacity, x)
    mobius_value = mobius_choquet(capacity, x)

    assert np.isclose(mobius_value, ordered_value)


def test_mobius_transform_recovers_expected_singletons():
    capacity = build_capacity()
    mobius_rep = mobius_transform(capacity.subset_values, n_features=2)

    assert np.isclose(mobius_rep.get_value({0}), 0.2)
    assert np.isclose(mobius_rep.get_value({1}), 0.5)
    assert np.isclose(mobius_rep.get_value({0, 1}), 0.3)
