# imports
from collections.abc import Set as AbstractSet
from math import isclose

# modules
from capacities_ml.capacities.mobius import mobius_transform, powerset
from capacities_ml.capacities.types import CapacityMap


# monotonicity
def check_monotonicity(capacities: CapacityMap, features: AbstractSet[int]):
    """
    Check that a capacity is normalized and monotone.
    """
    feature_set = set(features)
    lookup = capacities.to_lookup()
    all_subsets = [frozenset(subset) for subset in powerset(feature_set)]

    for subset in all_subsets:
        if subset not in lookup:
            raise ValueError(f"Missing coalition: {set(subset)}.")

    grand_coalition = frozenset(feature_set)
    if lookup[grand_coalition] != 1:
        raise ValueError("Breach in monotonicity: the full coalition must have value 1.")

    for subset in all_subsets:
        subset_value = lookup[subset]

        for feature in grand_coalition - subset:
            extended_subset = subset | {feature}
            extended_value = lookup[extended_subset]

            if subset_value > extended_value:
                raise ValueError(
                    "Breach in monotonicity: "
                    f"{set(subset)} has value {subset_value} but "
                    f"{set(extended_subset)} has value {extended_value}."
                )


def check_k_additivity(capacities: CapacityMap, k: int, n_features: int, tolerance: float = 1e-12):
    """
    Check that a capacity is exactly k-additive.
    """
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

    mobius_rep = mobius_transform(capacities, n_features)
    has_non_zero_order_k_term = False

    for coefficient in mobius_rep:
        coalition_size = len(coefficient.coalition)

        if coalition_size == k and not isclose(
            coefficient.value,
            0.0,
            abs_tol=tolerance,
        ):
            has_non_zero_order_k_term = True

        if coalition_size > k and not isclose(
            coefficient.value,
            0.0,
            abs_tol=tolerance,
        ):
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
