# imports
from collections.abc import Iterable, Set
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml.capacities.types import CapacityMap, VariableUniverseLike


# aliases
CoalitionMember = int | str
CoalitionInput = CoalitionMember | Iterable[CoalitionMember]

# event mask validation
def _event_mask(event: ArrayLike, n_elements: int) -> np.ndarray:
    mask = np.asarray(event)
    if mask.ndim != 1 or mask.size != n_elements:
        raise ValueError(
            f"event must be a one-dimensional mask of length {n_elements}."
        )
    if mask.dtype != bool:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("event must contain only boolean or binary values.")
        mask = mask.astype(bool)
    return mask


# element order validation
def _order_permutation(order: ArrayLike, n_elements: int) -> np.ndarray:
    permutation = np.asarray(order)
    if permutation.ndim != 1 or permutation.size != n_elements:
        raise ValueError(
            f"order must be a one-dimensional permutation of length {n_elements}."
        )
    if not np.issubdtype(permutation.dtype, np.integer):
        raise TypeError("order must contain integer indices.")
    permutation = permutation.astype(int, copy=False)
    if not np.array_equal(np.sort(permutation), np.arange(n_elements)):
        raise ValueError("order must contain every element index exactly once.")
    return permutation


# coalition normalization
def normalize_coalition(
    universe: VariableUniverseLike,
    coalition: CoalitionInput,
) -> frozenset[int]:
    """Convert a coalition expressed with names or indices into indices."""
    members = (coalition,) if isinstance(coalition, (str, int)) else tuple(coalition)
    normalized: set[int] = set()

    for member in members:
        if isinstance(member, str):
            try:
                normalized.add(universe.name_to_index[member])
            except KeyError as exc:
                raise KeyError(f"Unknown variable name: {member}.") from exc
            continue

        if isinstance(member, int):
            if not 0 <= member < universe.n_elements:
                raise ValueError(
                    f"Variable index {member} is invalid for "
                    f"{universe.n_elements} variables."
                )
            normalized.add(member)
            continue

        raise TypeError(
            "Coalitions must contain only variable names or integer indices."
        )

    return frozenset(normalized)


# named coalition
def named_coalition(
    universe: VariableUniverseLike,
    coalition: Set[int],
) -> tuple[str, ...]:
    """Convert an index coalition into ordered variable names."""
    return tuple(universe.var_names[index] for index in sorted(coalition))


# subset encoding
def subset_encoding(elements: Set[int], n_elements: int) -> int:
    """Encode a coalition using an integer bitmask."""
    if n_elements < 1:
        raise ValueError("n_elements must be positive.")

    mask = 0
    for index in elements:
        if not isinstance(index, int):
            raise TypeError("Variable indices must be integers.")
        if not 0 <= index < n_elements:
            raise ValueError(
                f"Variable index {index} is invalid for n_elements={n_elements}."
            )
        mask |= 1 << index

    return mask


# subset decoding
def subset_decoding(index: int, n_elements: int) -> frozenset[int]:
    """Decode an integer bitmask into a coalition."""
    if not isinstance(index, int):
        raise TypeError("index must be an integer.")

    maximum_index = (1 << n_elements) - 1
    if not 0 <= index <= maximum_index:
        raise ValueError(f"index must be between 0 and {maximum_index}.")

    return frozenset(
        variable
        for variable in range(n_elements)
        if index & (1 << variable)
    )


# capacity mapping
def map_capacities(capacities: CapacityMap, n_elements: int) -> dict[int, float]:
    """Map every coalition bitmask to its capacity value."""
    expected_non_empty = (1 << n_elements) - 1
    if len(capacities) != expected_non_empty:
        raise ValueError(
            "The number of capacity values must equal the number "
            f"of non-empty subsets: expected {expected_non_empty}, "
            f"received {len(capacities)}."
        )

    mapping: dict[int, float] = {0: 0.0}
    for capacity_value in capacities:
        encoding = subset_encoding(capacity_value.coalition, n_elements)
        if encoding in mapping:
            raise ValueError(
                f"Duplicate coalition: {capacity_value.coalition}."
            )
        mapping[encoding] = float(capacity_value.value)

    expected_indices = set(range(1 << n_elements))
    missing_indices = expected_indices - set(mapping)
    if missing_indices:
        missing_coalitions = [
            subset_decoding(index, n_elements)
            for index in sorted(missing_indices)
        ]
        raise ValueError(f"Missing coalitions: {missing_coalitions}.")

    return mapping
