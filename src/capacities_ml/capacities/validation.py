# imports
from collections.abc import Set
from math import isclose, isfinite

# modules
from capacities_ml.capacities.types import CapacityMap


# monotonicity
def check_monotonicity(capacities: CapacityMap, features: Set[int], tolerance: float = 1e-12) -> None:
    """
    Check that a capacity is normalized and monotone.
    """
    from capacities_ml.mobius.utils import powerset

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

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


def check_k_additivity(capacities: CapacityMap, k: int, n_features: int, tolerance: float = 1e-12) -> None:
    """
    Check that a capacity is exactly k-additive.
    """
    from capacities_ml.mobius.utils import _mobius_transform_map

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
