import numpy as np

from capacities_ml.capacities import (
    BaseCapacity,
    Capacity,
    VariableUniverse,
)
from capacities_ml.mobius import (
    MobiusRepresentation,
    inverse_mobius_transform,
    mobius_transform,
)
from capacities_ml.integrals.choquet import mobius_choquet, ordered_choquet


def build_capacity() -> Capacity:
    return Capacity(
        universe=VariableUniverse(("x0", "x1")),
        values={
            ("x0",): 0.2,
            ("x1",): 0.5,
            ("x0", "x1"): 1.0,
        },
    )


def test_ordered_choquet_matches_manual_value():
    capacity = build_capacity()
    x = np.array([3.0, 1.0])

    choquet_value = ordered_choquet(capacity, x)

    assert np.isclose(choquet_value, 1.4)


def test_tabular_capacity_implements_common_event_interface():
    capacity = build_capacity()

    assert isinstance(capacity, BaseCapacity)
    assert capacity.n_elements == 2
    assert capacity.event_value([True, False]) == 0.2
    np.testing.assert_allclose(capacity.nested_event_values([1, 0]), [1.0, 0.2])


def test_mobius_choquet_matches_ordered_choquet():
    capacity = build_capacity()
    x = np.array([3.0, 1.0])

    ordered_value = ordered_choquet(capacity, x)
    mobius_rep = mobius_transform(capacity)
    mobius_value = mobius_choquet(mobius_rep, x)

    assert np.isclose(mobius_value, ordered_value)


def test_mobius_transform_recovers_expected_singletons():
    capacity = build_capacity()
    mobius_rep = mobius_transform(capacity)

    assert mobius_rep.universe is capacity.universe
    assert np.isclose(mobius_rep.value({0}), 0.2)
    assert np.isclose(mobius_rep.value({1}), 0.5)
    assert np.isclose(mobius_rep.value({0, 1}), 0.3)


def test_inverse_mobius_transform_expands_a_sparse_representation():
    universe = VariableUniverse(("x0", "x1", "x2"))
    mobius_rep = MobiusRepresentation(
        universe=universe,
        coefficients={
            ("x0",): 0.2,
            ("x1",): 0.3,
            ("x2",): 0.1,
            ("x0", "x1"): 0.1,
            ("x0", "x2"): 0.1,
            ("x1", "x2"): 0.2,
        },
    )

    capacity = inverse_mobius_transform(mobius_rep)

    assert capacity.universe is universe
    assert np.isclose(capacity.value({0, 1, 2}), 1.0)


def test_mobius_representation_supports_names_and_sparse_coefficients():
    mobius_rep = MobiusRepresentation(
        universe=VariableUniverse(("price", "quality")),
        coefficients={
            "price": 0.2,
            ("price", "quality"): 0.3,
        },
    )

    assert mobius_rep.value("price") == 0.2
    assert mobius_rep.value({0, "quality"}) == 0.3
    assert mobius_rep.value("quality") == 0.0
    assert mobius_rep.to_named_dict() == {
        ("price",): 0.2,
        ("price", "quality"): 0.3,
    }
