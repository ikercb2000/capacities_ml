# import
import pytest

# modules
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
from capacities_ml.capacities.validation import check_k_additivity, check_monotonicity


def test_accepts_a_monotone_capacity():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.2),
            CoalitionValue(frozenset({1}), 0.4),
            CoalitionValue(frozenset({0, 1}), 1.0),
        ]
    )

    check_monotonicity(capacities, {0, 1})


def test_rejects_when_a_superset_has_lower_value():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.8),
            CoalitionValue(frozenset({1}), 0.3),
            CoalitionValue(frozenset({0, 1}), 0.7),
        ]
    )

    with pytest.raises(ValueError, match="Breach in monotonicity"):
        check_monotonicity(capacities, {0, 1})


def test_accepts_a_1_additive_capacity():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.3),
            CoalitionValue(frozenset({1}), 0.7),
            CoalitionValue(frozenset({0, 1}), 1.0),
        ]
    )

    check_k_additivity(capacities, k=1, n_features=2)


def test_rejects_when_a_higher_order_mobius_term_is_non_zero():
    capacities = CapacityMap(
        capacities=[
            CoalitionValue(frozenset({0}), 0.2),
            CoalitionValue(frozenset({1}), 0.3),
            CoalitionValue(frozenset({0, 1}), 1.0),
        ]
    )

    with pytest.raises(ValueError, match="Breach in k-additivity"):
        check_k_additivity(capacities, k=1, n_features=2)
