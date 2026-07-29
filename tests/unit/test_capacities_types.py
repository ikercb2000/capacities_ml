# modules
from capacities_ml.capacities.types import CapacityMap, CoalitionValue


def test_coalition_value_normalizes_to_frozenset():
    coalition_value = CoalitionValue({0, 1}, 0.7)

    assert coalition_value.coalition == frozenset({0, 1})


def test_get_value_uses_internal_lookup():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.2),
            CoalitionValue(frozenset({1}), 0.4),
            CoalitionValue(frozenset({0, 1}), 1.0),
        ]
    )

    assert capacities.get_value(set()) == 0.0
    assert capacities.get_value({0}) == 0.2
    assert capacities.get_value({0, 1}) == 1.0


def test_get_coalition_returns_all_matching_coalitions():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.5),
            CoalitionValue(frozenset({1}), 0.5),
            CoalitionValue(frozenset({0, 1}), 1.0),
        ]
    )

    coalitions = capacities.get_coalition(0.5)

    assert set(coalitions) == {frozenset({0}), frozenset({1})}
