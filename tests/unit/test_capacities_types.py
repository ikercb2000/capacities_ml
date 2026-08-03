import pytest

# modules
from capacities_ml.capacities import Capacity, KAdditiveCapacity, VariableUniverse
from capacities_ml.capacities.types import CapacityMap, CoalitionValue
from capacities_ml.mobius.types import MobiusMap
from capacities_ml.mobius import mobius_transform


def test_coalition_value_normalizes_to_frozenset():
    coalition_value = CoalitionValue({0, 1}, 0.7)

    assert coalition_value.coalition == frozenset({0, 1})

    coalition_value.value = 0.8

    assert coalition_value.value == 0.8


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


def test_capacity_validates_automatically_on_construction():
    with pytest.raises(ValueError, match="Breach in monotonicity"):
        Capacity(
            universe=VariableUniverse(("x0", "x1")),
            values={
                ("x0",): 0.8,
                ("x1",): 0.3,
                ("x0", "x1"): 0.7,
            },
        )


def test_capacity_and_its_internal_map_are_mutable():
    capacity = Capacity(
        universe=VariableUniverse(("x0", "x1")),
        values={
            ("x0",): 0.2,
            ("x1",): 0.4,
            ("x0", "x1"): 1.0,
        },
    )

    capacity.universe = VariableUniverse(("x0", "x1", "x2"))
    capacity._subset_values.set_value({0}, 0.9)

    assert capacity.universe.n_vars == 3
    assert capacity.value({0}) == 0.9


def test_mobius_map_is_mutable():
    coefficients = MobiusMap(
        mobius_coefficients=[CoalitionValue({0}, 0.2)]
    )

    coefficients.set_value({0}, 0.4)
    coefficients.set_value({0, 1}, 0.3)

    assert coefficients.get_value({0}) == 0.4
    assert coefficients.get_value({0, 1}) == 0.3


def test_variable_universe_derives_cardinality_from_names():
    universe = VariableUniverse(("x0", "x1", "x2"))

    assert universe.var_names == ("x0", "x1", "x2")
    assert universe.n_vars == 3
    assert universe.name_to_index == {"x0": 0, "x1": 1, "x2": 2}


def test_variable_universe_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        VariableUniverse(("x0", "x0"))


def test_variable_universe_allows_mutation():
    universe = VariableUniverse(("price", "quality", "risk"))

    universe.var_names = ("price", "quality")
    universe.n_vars = 2
    universe.name_to_index["extra"] = 3

    assert universe.var_names == ("price", "quality")
    assert universe.n_vars == 2
    assert universe.name_to_index["extra"] == 3


def test_capacity_supports_user_friendly_access():
    universe = VariableUniverse(("price", "quality"))
    capacity = Capacity(
        universe=universe,
        values={
            ("price",): 0.2,
            ("quality",): 0.5,
            ("price", "quality"): 1.0,
        },
    )

    assert capacity.value("price") == 0.2
    assert capacity.value(("price", "quality")) == 1.0
    assert capacity.value({0, "quality"}) == 1.0
    assert capacity.to_named_dict() == {
        (): 0.0,
        ("price",): 0.2,
        ("quality",): 0.5,
        ("price", "quality"): 1.0,
    }


def test_k_additive_capacity_derives_coalitions_above_k():
    capacity = KAdditiveCapacity(
        universe=VariableUniverse(("x0", "x1", "x2")),
        values={
            "x0": 0.2,
            "x1": 0.3,
            "x2": 0.1,
            ("x0", "x1"): 0.6,
            ("x0", "x2"): 0.45,
            ("x1", "x2"): 0.55,
        },
        k=2,
    )

    mobius_rep = mobius_transform(capacity)

    assert capacity.value(("x0", "x1", "x2")) == pytest.approx(1.0)
    assert mobius_rep.value(("x0", "x1", "x2")) == pytest.approx(0.0)
