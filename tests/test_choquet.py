import numpy as np
import pytest

from capacities_ml_fin.base.capacities import (
    BaseCapacity,
    ExplicitCapacity,
    MobiusCapacity,
    VariableUniverse,
    inverse_mobius_transform,
    mobius_transform,
)
from capacities_ml_fin.base.integrals.choquet import mobius_choquet, ordered_choquet


def build_capacity() -> ExplicitCapacity:
    return ExplicitCapacity(
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
    mobius_capacity = mobius_transform(capacity)
    mobius_value = mobius_choquet(mobius_capacity, x)

    assert np.isclose(mobius_value, ordered_value)
    assert np.isclose(ordered_choquet(mobius_capacity, x), ordered_value)


def test_mobius_capacity_implements_common_event_interface():
    mobius_capacity = MobiusCapacity(
        universe=VariableUniverse(("x0", "x1", "x2")),
        coefficients={
            ("x0",): 0.2,
            ("x1",): 0.3,
            ("x2",): 0.1,
            ("x0", "x1"): 0.1,
            ("x0", "x2"): 0.1,
            ("x1", "x2"): 0.2,
        },
    )

    assert isinstance(mobius_capacity, BaseCapacity)
    assert mobius_capacity.event_value([True, False, True]) == 0.4
    np.testing.assert_allclose(
        mobius_capacity.nested_event_values([2, 0, 1]),
        [1.0, 0.6, 0.3],
    )


def test_mobius_transform_recovers_expected_singletons():
    capacity = build_capacity()
    mobius_capacity = mobius_transform(capacity)

    assert mobius_capacity.universe is capacity.universe
    assert np.isclose(mobius_capacity.value({0}), 0.2)
    assert np.isclose(mobius_capacity.value({1}), 0.5)
    assert np.isclose(mobius_capacity.value({0, 1}), 0.3)


def test_inverse_mobius_transform_expands_a_sparse_representation():
    universe = VariableUniverse(("x0", "x1", "x2"))
    mobius_capacity = MobiusCapacity(
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

    capacity = inverse_mobius_transform(mobius_capacity)

    assert capacity.universe is universe
    assert np.isclose(capacity.value({0, 1, 2}), 1.0)


def test_mobius_capacity_supports_names_and_sparse_coefficients():
    mobius_capacity = MobiusCapacity(
        coefficients={
            "price": 0.2,
            ("price", "quality"): 0.8,
        },
    )

    assert mobius_capacity.value("price") == 0.2
    assert mobius_capacity.value({0, "quality"}) == 0.8
    assert mobius_capacity.value("quality") == 0.0
    assert mobius_capacity.to_named_dict() == {
        ("price",): 0.2,
        ("price", "quality"): 0.8,
    }


def test_mobius_capacity_infers_minimum_integer_universe():
    capacity = MobiusCapacity(
        coefficients={(0,): 0.2, (2,): 0.8},
    )

    assert capacity.n_elements == 3
    assert capacity.var_names == ("x0", "x1", "x2")


def test_mobius_capacity_accepts_valid_negative_interaction():
    mobius_capacity = MobiusCapacity(
        universe=VariableUniverse(("x0", "x1")),
        coefficients={
            "x0": 0.6,
            "x1": 0.6,
            ("x0", "x1"): -0.2,
        },
    )

    assert mobius_capacity.event_value([True, True]) == pytest.approx(1.0)
    assert mobius_capacity.event_value([True, False]) == pytest.approx(0.6)


def test_mobius_capacity_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="must sum to 1"):
        MobiusCapacity(
            universe=VariableUniverse(("x0", "x1")),
            coefficients={"x0": 0.2, "x1": 0.3},
        )


def test_mobius_capacity_rejects_non_monotone_coefficients():
    with pytest.raises(ValueError, match="Breach in monotonicity"):
        MobiusCapacity(
            universe=VariableUniverse(("x0", "x1")),
            coefficients={
                "x0": 0.1,
                "x1": 1.1,
                ("x0", "x1"): -0.2,
            },
        )


def test_mobius_capacity_checks_higher_order_marginals():
    with pytest.raises(ValueError, match="Breach in monotonicity"):
        MobiusCapacity(
            universe=VariableUniverse(("x0", "x1", "x2")),
            coefficients={
                "x0": 0.1,
                "x1": 0.6,
                "x2": 0.6,
                ("x0", "x1", "x2"): -0.3,
            },
        )


def test_mobius_capacity_rejects_nonzero_empty_coefficient():
    with pytest.raises(ValueError, match="empty-coalition"):
        MobiusCapacity(
            universe=VariableUniverse(("x0",)),
            coefficients={(): 0.1, "x0": 0.9},
        )


def test_mobius_capacity_rejects_non_finite_coefficient():
    with pytest.raises(ValueError, match="must be finite"):
        MobiusCapacity(
            universe=VariableUniverse(("x0",)),
            coefficients={"x0": np.nan},
        )
