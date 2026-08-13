import pytest

# modules
from capacities_ml_fin.base.capacities import (
    ExplicitCapacity,
    KAdditiveCapacity,
    MobiusCapacity,
    VariableUniverse,
)
from capacities_ml_fin.base.capacities import mobius_transform
from capacities_ml_fin.base.capacities.types import CapacityMap, CoalitionValue, MobiusMap


def test_coalition_value_normalizes_to_frozenset_and_is_immutable():
    coalition_value = CoalitionValue({0, 1}, 0.7)

    assert coalition_value.coalition == frozenset({0, 1})

    with pytest.raises(AttributeError):
        coalition_value.value = 0.8


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
        ExplicitCapacity(
            universe=VariableUniverse(("x0", "x1")),
            values={
                ("x0",): 0.8,
                ("x1",): 0.3,
                ("x0", "x1"): 0.7,
            },
        )


def test_capacity_and_its_internal_map_are_immutable():
    capacity = ExplicitCapacity(
        universe=VariableUniverse(("x0", "x1")),
        values={
            ("x0",): 0.2,
            ("x1",): 0.4,
            ("x0", "x1"): 1.0,
        },
    )

    with pytest.raises(AttributeError, match="immutable"):
        capacity.universe = VariableUniverse(("x0", "x1", "x2"))
    with pytest.raises(AttributeError, match="immutable"):
        capacity.values.set_value({0}, 0.9)
    with pytest.raises(TypeError):
        capacity.values.lookup_dict[frozenset({0})] = 0.9

    assert capacity.universe.n_elements == 2
    assert capacity.value({0}) == 0.2


def test_mobius_map_is_immutable():
    coefficients = MobiusMap(
        mobius_coefficients=[CoalitionValue({0}, 0.2)]
    )

    with pytest.raises(AttributeError, match="immutable"):
        coefficients.set_value({0}, 0.4)
    with pytest.raises(TypeError):
        coefficients.lookup_dict[frozenset({0})] = 0.4

    assert coefficients.get_value({0}) == 0.2
    assert coefficients.get_value({0, 1}) is None


def test_variable_universe_derives_cardinality_from_names():
    universe = VariableUniverse(("x0", "x1", "x2"))

    assert universe.var_names == ("x0", "x1", "x2")
    assert universe.n_elements == 3
    assert universe.name_to_index == {"x0": 0, "x1": 1, "x2": 2}


def test_variable_universe_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        VariableUniverse(("x0", "x0"))


def test_variable_universe_can_be_inferred_from_size_and_coalitions():
    by_size = VariableUniverse.from_size(3)
    by_names = VariableUniverse.from_coalitions(
        [("quality",), ("quality", "price")]
    )

    assert by_size.var_names == ("x0", "x1", "x2")
    assert by_names.var_names == ("quality", "price")


def test_variable_universe_is_immutable():
    universe = VariableUniverse(("price", "quality", "risk"))

    with pytest.raises(AttributeError):
        universe.var_names = ("price", "quality")
    with pytest.raises(AttributeError):
        universe.n_elements = 2
    with pytest.raises(TypeError):
        universe.name_to_index["extra"] = 3

    assert universe.var_names == ("price", "quality", "risk")
    assert universe.n_elements == 3


def test_capacity_supports_user_friendly_access():
    capacity = ExplicitCapacity(
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


def test_explicit_capacity_infers_named_universe_from_values():
    capacity = ExplicitCapacity(
        values={
            ("quality",): 0.3,
            ("price",): 0.4,
            ("quality", "price"): 1.0,
        }
    )

    assert capacity.var_names == ("quality", "price")
    assert capacity.n_elements == 2


def test_mobius_capacity_can_include_omitted_null_elements_by_size():
    capacity = MobiusCapacity(
        coefficients={(0,): 1.0},
        n_elements=3,
    )

    assert capacity.var_names == ("x0", "x1", "x2")
    assert capacity.n_elements == 3
    assert capacity.event_value([True, False, False]) == 1.0
    assert capacity.event_value([False, True, True]) == 0.0


def test_capacity_accepts_names_without_a_variable_universe_object():
    capacity = MobiusCapacity(
        coefficients={"quality": 0.4, "price": 0.6},
        var_names=("quality", "price", "unused"),
    )

    assert capacity.var_names == ("quality", "price", "unused")
    assert capacity.event_value([False, False, True]) == 0.0


def test_capacity_rejects_ambiguous_mixed_universe_inference():
    with pytest.raises(ValueError, match="mixed names and indices"):
        MobiusCapacity(
            coefficients={"quality": 0.5, 1: 0.5},
        )


def test_k_additive_capacity_derives_coalitions_above_k():
    capacity = KAdditiveCapacity(
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

    mobius_capacity = mobius_transform(capacity)

    assert capacity.value(("x0", "x1", "x2")) == pytest.approx(1.0)
    assert mobius_capacity.value(("x0", "x1", "x2")) == pytest.approx(0.0)
