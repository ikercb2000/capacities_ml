# imports
from __future__ import annotations
from itertools import combinations
from math import factorial
from numbers import Integral
from typing import TypeAlias
import numpy as np

# modules
from capacities_ml_fin.base.capacities import BaseCapacity


# interpretation aliases
ElementLabel: TypeAlias = int | str
EventCache: TypeAlias = dict[int, float]


# capacity validation
def _check_capacity(capacity: BaseCapacity) -> None:
    if not isinstance(capacity, BaseCapacity):
        raise TypeError("capacity must be a BaseCapacity instance.")


# element index
def _element_index(capacity: BaseCapacity, element: ElementLabel) -> int:
    _check_capacity(capacity)
    if isinstance(element, str):
        universe = getattr(capacity, "universe", None)
        if universe is None:
            raise TypeError(
                "String elements require a capacity with a named universe."
            )
        try:
            return int(universe.name_to_index[element])
        except KeyError as error:
            raise KeyError(f"Unknown element name: {element}.") from error
    if isinstance(element, bool) or not isinstance(element, Integral):
        raise TypeError("element must be an integer index or a variable name.")
    index = int(element)
    if not 0 <= index < capacity.n_elements:
        raise ValueError(
            f"Element index {index} is invalid for "
            f"{capacity.n_elements} elements."
        )
    return index


# result labels
def _element_labels(capacity: BaseCapacity) -> tuple[ElementLabel, ...]:
    var_names = getattr(capacity, "var_names", None)
    if var_names is None:
        return tuple(range(capacity.n_elements))
    labels = tuple(var_names)
    if len(labels) != capacity.n_elements:
        raise ValueError("var_names has an incompatible number of elements.")
    return labels


# event evaluation
def _event_value(
    capacity: BaseCapacity,
    mask: int,
    cache: EventCache,
) -> float:
    if mask not in cache:
        event = np.asarray(
            [bool(mask & (1 << index)) for index in range(capacity.n_elements)],
            dtype=bool,
        )
        cache[mask] = float(capacity.event_value(event))
    return cache[mask]


# subset masks
def _subset_masks(
    n_elements: int,
    excluded: frozenset[int],
):
    remaining = tuple(
        index for index in range(n_elements) if index not in excluded
    )
    for local_mask in range(1 << len(remaining)):
        mask = 0
        for position, index in enumerate(remaining):
            if local_mask & (1 << position):
                mask |= 1 << index
        yield mask, local_mask.bit_count()


# internal Shapley index
def _shapley_index(
    capacity: BaseCapacity,
    index: int,
    cache: EventCache,
) -> float:
    n_elements = capacity.n_elements
    element_mask = 1 << index
    denominator = factorial(n_elements)
    value = 0.0
    for mask, size in _subset_masks(n_elements, frozenset({index})):
        weight = (
            factorial(size)
            * factorial(n_elements - size - 1)
            / denominator
        )
        marginal = (
            _event_value(capacity, mask | element_mask, cache)
            - _event_value(capacity, mask, cache)
        )
        value += weight * marginal
    return float(value)


# Shapley index
def shapley_index(
    capacity: BaseCapacity,
    element: ElementLabel,
) -> float:
    """Compute one exact Shapley index through event evaluations."""
    index = _element_index(capacity, element)
    return _shapley_index(capacity, index, {})


# Shapley indices
def shapley_indices(
    capacity: BaseCapacity,
) -> dict[ElementLabel, float]:
    """Return exact Shapley indices, using names when they are available."""
    _check_capacity(capacity)
    labels = _element_labels(capacity)
    cache: EventCache = {}
    return {
        label: _shapley_index(capacity, index, cache)
        for index, label in enumerate(labels)
    }


# internal pairwise interaction index
def _pairwise_interaction_index(
    capacity: BaseCapacity,
    first: int,
    second: int,
    cache: EventCache,
) -> float:
    n_elements = capacity.n_elements
    first_mask = 1 << first
    second_mask = 1 << second
    denominator = factorial(n_elements - 1)
    value = 0.0
    for mask, size in _subset_masks(
        n_elements,
        frozenset({first, second}),
    ):
        weight = (
            factorial(size)
            * factorial(n_elements - size - 2)
            / denominator
        )
        second_difference = (
            _event_value(
                capacity,
                mask | first_mask | second_mask,
                cache,
            )
            - _event_value(capacity, mask | first_mask, cache)
            - _event_value(capacity, mask | second_mask, cache)
            + _event_value(capacity, mask, cache)
        )
        value += weight * second_difference
    return float(value)


# pairwise interaction index
def pairwise_interaction_index(
    capacity: BaseCapacity,
    first: ElementLabel,
    second: ElementLabel,
) -> float:
    """Compute one exact Shapley pairwise interaction from event values."""
    first_index = _element_index(capacity, first)
    second_index = _element_index(capacity, second)
    if first_index == second_index:
        raise ValueError("Pairwise interaction requires two distinct elements.")
    return _pairwise_interaction_index(
        capacity,
        first_index,
        second_index,
        {},
    )


# pairwise interactions
def pairwise_interactions(
    capacity: BaseCapacity,
) -> dict[tuple[ElementLabel, ElementLabel], float]:
    """Return all exact pairwise interactions from event evaluations."""
    _check_capacity(capacity)
    labels = _element_labels(capacity)
    cache: EventCache = {}
    return {
        (labels[first], labels[second]): _pairwise_interaction_index(
            capacity,
            first,
            second,
            cache,
        )
        for first, second in combinations(range(capacity.n_elements), 2)
    }


# pairwise interaction matrix
def pairwise_interaction_matrix(capacity: BaseCapacity) -> np.ndarray:
    """Return a symmetric matrix of exact pairwise interactions."""
    _check_capacity(capacity)
    matrix = np.zeros((capacity.n_elements, capacity.n_elements), dtype=float)
    cache: EventCache = {}
    for first, second in combinations(range(capacity.n_elements), 2):
        value = _pairwise_interaction_index(
            capacity,
            first,
            second,
            cache,
        )
        matrix[first, second] = value
        matrix[second, first] = value
    return matrix


# interaction signs
def interaction_signs(
    capacity: BaseCapacity,
) -> dict[tuple[ElementLabel, ElementLabel], int]:
    """Classify interactions as redundant, neutral or complementary."""
    return {
        pair: 0 if np.isclose(value, 0.0, atol=1e-12) else int(np.sign(value))
        for pair, value in pairwise_interactions(capacity).items()
    }
