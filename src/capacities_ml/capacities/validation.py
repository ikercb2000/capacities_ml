# imports
from collections.abc import Set
from math import isclose, isfinite
from numbers import Real

# modules
from capacities_ml.capacities.types import CapacityMap, CoalitionMap


# monotonicity
def check_monotonicity(capacities: CapacityMap, features: Set[int], tolerance: float = 1e-12) -> None:
    """
    Check that a capacity is normalized and monotone.
    """
    from capacities_ml.capacities.utils import powerset

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
    from capacities_ml.capacities.utils import _mobius_transform_map

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

    mobius_rep = _mobius_transform_map(capacities)
    has_non_zero_order_k_term = False

    for coefficient in mobius_rep:
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
