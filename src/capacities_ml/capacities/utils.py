# imports
from collections.abc import Set as AbstractSet
from typing import Dict

# modules
from capacities_ml.capacities.types import CapacityMap

# subset encoding
def subset_encoding(elements: AbstractSet[int], n_features: int) -> int:
    """
    Encode a subset using an integer bitmask.
    """
    if n_features < 1:
        raise ValueError("n_features must be positive.")

    mask = 0

    for index in elements:
        if not isinstance(index, int):
            raise TypeError("Feature indices must be integers.")

        if not 0 <= index < n_features:
            raise ValueError(
                f"Feature index {index} is invalid for "
                f"n_features={n_features}."
            )

        mask |= 1 << index

    return mask

# subset decoding
def subset_decoding(index: int, n_features: int) -> frozenset[int]:
    """
    Decode an integer bitmask into a subset.
    """
    if not isinstance(index, int):
        raise TypeError("index must be an integer.")

    maximum_index = (1 << n_features) - 1

    if not 0 <= index <= maximum_index:
        raise ValueError(
            f"index must be between 0 and {maximum_index}."
        )

    return frozenset(
        feature
        for feature in range(n_features)
        if index & (1 << feature)
    )

# capacity mapping
def map_capacities(capacities: CapacityMap, n_features: int) -> Dict[int, float]:
    """
    Map every coalition bitmask to its capacity value. The empty coalition is inserted automatically with value zero.
    """
    expected_non_empty = (1 << n_features) - 1

    if len(capacities) != expected_non_empty:
        raise ValueError(
            "The number of capacity values must equal the number "
            f"of non-empty subsets: expected {expected_non_empty}, "
            f"received {len(capacities)}."
        )

    mapping: Dict[int, float] = {0: 0.0}

    for capacity in capacities:
        encoding = subset_encoding(
            capacity.coalition,
            n_features,
        )

        if encoding == 0:
            raise ValueError(
                "The empty coalition must not be included in capacities; "
                "its value is inserted automatically as zero."
            )

        if encoding in mapping:
            raise ValueError(
                f"Duplicate coalition: {capacity.coalition}."
            )

        mapping[encoding] = float(capacity.value)

    expected_indices = set(range(1 << n_features))
    actual_indices = set(mapping)

    if actual_indices != expected_indices:
        missing_indices = expected_indices - actual_indices
        missing_coalitions = [
            subset_decoding(index, n_features)
            for index in sorted(missing_indices)
        ]

        raise ValueError(
            f"Missing coalitions: {missing_coalitions}."
        )

    return mapping
