# imports
from collections.abc import Set
from math import isclose, isfinite
from numbers import Real
import numpy as np

# modules
from capacities_ml_fin.base.capacities.base import BaseCapacity
from capacities_ml_fin.base.capacities.types import CapacityMap, CoalitionMap


# exhaustive capacity values
def _all_event_values(capacity: BaseCapacity, max_elements: int) -> np.ndarray:
    if not isinstance(capacity, BaseCapacity):
        raise TypeError("capacity must be a BaseCapacity instance.")
    if not isinstance(max_elements, int):
        raise TypeError("max_elements must be an integer.")
    if max_elements < 1:
        raise ValueError("max_elements must be positive.")
    if capacity.n_elements > max_elements:
        raise ValueError(
            f"Exact validation is limited to {max_elements} elements; "
            f"received {capacity.n_elements}."
        )

    values = np.empty(1 << capacity.n_elements, dtype=float)
    positions = np.arange(capacity.n_elements)
    for encoding in range(values.size):
        event = (encoding & (1 << positions)) != 0
        values[encoding] = capacity.event_value(event)
    return values


# capacity validation
def validate_capacity(
    capacity: BaseCapacity,
    *,
    max_elements: int = 16,
    tolerance: float = 1e-10,
) -> None:
    """Check normalization and monotonicity on a finite capacity domain."""
    values = _all_event_values(capacity, max_elements)
    if not np.isclose(values[0], 0.0, atol=tolerance):
        raise ValueError("A capacity must assign zero to the empty event.")
    if not np.isclose(values[-1], 1.0, atol=tolerance):
        raise ValueError("A capacity must assign one to the full event.")

    for event in range(values.size):
        for element in range(capacity.n_elements):
            if event & (1 << element):
                continue
            superset = event | (1 << element)
            if values[event] > values[superset] + tolerance:
                raise ValueError("The capacity is not monotone.")


# concave capacity
def is_concave_capacity(
    capacity: BaseCapacity,
    *,
    max_elements: int = 8,
    tolerance: float = 1e-10,
) -> bool:
    """Check submodularity of a capacity by exhaustive enumeration."""
    values = _all_event_values(capacity, max_elements)
    for first in range(values.size):
        for second in range(values.size):
            if values[first | second] + values[first & second] > (
                values[first] + values[second] + tolerance
            ):
                return False
    return True


# convex capacity
def is_convex_capacity(
    capacity: BaseCapacity,
    *,
    max_elements: int = 8,
    tolerance: float = 1e-10,
) -> bool:
    """Check supermodularity of a capacity by exhaustive enumeration."""
    values = _all_event_values(capacity, max_elements)
    for first in range(values.size):
        for second in range(values.size):
            if values[first | second] + values[first & second] < (
                values[first] + values[second] - tolerance
            ):
                return False
    return True


# monotonicity
def check_monotonicity(capacities: CapacityMap, features: Set[int], tolerance: float = 1e-12) -> None:
    """
    Check that a capacity is normalized and monotone.
    """
    from capacities_ml_fin.base.capacities.utils import powerset

    if not isinstance(tolerance, Real):
        raise TypeError("tolerance must be numeric.")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")

    feature_set = frozenset(features)
    lookup = capacities.to_lookup()
    all_subsets = [frozenset(subset) for subset in powerset(feature_set)]
    expected_subsets = set(all_subsets)

    for subset in all_subsets:
        if subset not in lookup:
            raise ValueError(f"Missing coalition: {set(subset)}.")
        if not isfinite(lookup[subset]):
            raise ValueError(
                f"Capacity value for coalition {set(subset)} must be finite."
            )

    unexpected_subsets = set(lookup) - expected_subsets
    if unexpected_subsets:
        raise ValueError(
            "Coalitions outside the feature set: "
            f"{[set(subset) for subset in unexpected_subsets]}."
        )

    grand_coalition = frozenset(feature_set)
    if not isclose(lookup[grand_coalition], 1.0, abs_tol=tolerance):
        raise ValueError("Breach in monotonicity: the full coalition must have value 1.")

    for subset in all_subsets:
        subset_value = lookup[subset]

        for feature in grand_coalition - subset:
            extended_subset = subset | {feature}
            extended_value = lookup[extended_subset]

            if subset_value > extended_value + tolerance:
                raise ValueError(
                    "Breach in monotonicity: "
                    f"{set(subset)} has value {subset_value} but "
                    f"{set(extended_subset)} has value {extended_value}."
                )


# Mobius capacity validation
def check_mobius_capacity(
    coefficients: CoalitionMap,
    n_elements: int,
    tolerance: float = 1e-9,
) -> None:
    """Check normalization and monotonicity directly from Mobius terms."""
    if not isinstance(n_elements, int):
        raise TypeError("n_elements must be an integer.")
    if n_elements < 1:
        raise ValueError("n_elements must be positive.")
    if not isinstance(tolerance, Real):
        raise TypeError("tolerance must be numeric.")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")

    terms = tuple(coefficients)
    for coefficient in terms:
        if not isfinite(coefficient.value):
            raise ValueError(
                "Mobius coefficient for coalition "
                f"{set(coefficient.coalition)} must be finite."
            )

    empty_value = coefficients.get_value(frozenset())
    if empty_value is not None and not isclose(
        empty_value,
        0.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("The empty-coalition Mobius coefficient must be zero.")

    total = sum(coefficient.value for coefficient in terms)
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            "Breach in normalization: Mobius coefficients must sum to 1, "
            f"received {total}."
        )

    # Delta_i mu(A) only depends on elements that share a stored term with i.
    for element in range(n_elements):
        marginal_terms: list[tuple[frozenset[int], float]] = []
        related_elements: set[int] = set()
        for coefficient in terms:
            if coefficient.value == 0.0 or element not in coefficient.coalition:
                continue
            support = coefficient.coalition - {element}
            marginal_terms.append((support, coefficient.value))
            related_elements.update(support)

        if not marginal_terms or all(value >= 0.0 for _, value in marginal_terms):
            continue

        # For a 2-additive capacity, every pairwise term can be selected
        # independently, so the minimum marginal has a closed form.
        if all(len(support) <= 1 for support, _ in marginal_terms):
            event = frozenset(
                next(iter(support))
                for support, value in marginal_terms
                if support and value < 0.0
            )
            marginal = sum(
                value
                for support, value in marginal_terms
                if not support or value < 0.0
            )
            if marginal < -tolerance:
                raise ValueError(
                    "Breach in monotonicity: adding element "
                    f"{element} to event {set(event)} has marginal value "
                    f"{marginal}."
                )
            continue

        related = tuple(sorted(related_elements))
        for mask in range(1 << len(related)):
            event = frozenset(
                related[position]
                for position in range(len(related))
                if mask & (1 << position)
            )
            marginal = sum(
                value
                for support, value in marginal_terms
                if support <= event
            )
            if marginal < -tolerance:
                raise ValueError(
                    "Breach in monotonicity: adding element "
                    f"{element} to event {set(event)} has marginal value "
                    f"{marginal}."
                )


def check_k_additivity(capacities: CapacityMap, k: int, n_features: int, tolerance: float = 1e-12) -> None:
    """
    Check that a capacity is exactly k-additive.
    """
    from capacities_ml_fin.base.capacities.utils import _mobius_transform_map

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if not isinstance(n_features, int):
        raise TypeError("n_features must be an integer.")

    if not 1 <= k <= n_features:
        raise ValueError(
            f"k must satisfy 1 <= k <= n_features, received k={k} and n_features={n_features}."
        )

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    mobius_map = _mobius_transform_map(capacities)
    has_non_zero_order_k_term = False

    for coefficient in mobius_map:
        coalition_size = len(coefficient.coalition)

        if coalition_size == k and not isclose(coefficient.value,0.0,abs_tol=tolerance):
            has_non_zero_order_k_term = True

        if coalition_size > k and not isclose(coefficient.value,0.0,abs_tol=tolerance):
            raise ValueError(
                "Breach in k-additivity: "
                f"Möbius coefficient of coalition {set(coefficient.coalition)} "
                f"must be zero for k={k}, received {coefficient.value}."
            )

    if not has_non_zero_order_k_term:
        raise ValueError(
            "Breach in k-additivity: "
            f"at least one Möbius coefficient of order {k} must be non-zero."
        )
