# imports
from collections.abc import Iterable, Mapping, Set as AbstractSet
from typing import Protocol

# modules
from capacities_ml.capacities.types import CapacityMap


# aliases
CoalitionMember = int | str
CoalitionInput = CoalitionMember | Iterable[CoalitionMember]


# variable universe protocol
class VariableUniverseLike(Protocol):
    var_names: Iterable[str]
    n_vars: int
    name_to_index: Mapping[str, int]


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
            if not 0 <= member < universe.n_vars:
                raise ValueError(
                    f"Variable index {member} is invalid for "
                    f"{universe.n_vars} variables."
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
    coalition: AbstractSet[int],
) -> tuple[str, ...]:
    """Convert an index coalition into ordered variable names."""
    return tuple(universe.var_names[index] for index in sorted(coalition))


# subset encoding
def subset_encoding(elements: AbstractSet[int], n_vars: int) -> int:
    """Encode a coalition using an integer bitmask."""
    if n_vars < 1:
        raise ValueError("n_vars must be positive.")

    mask = 0
    for index in elements:
        if not isinstance(index, int):
            raise TypeError("Variable indices must be integers.")
        if not 0 <= index < n_vars:
            raise ValueError(
                f"Variable index {index} is invalid for n_vars={n_vars}."
            )
        mask |= 1 << index

    return mask


# subset decoding
def subset_decoding(index: int, n_vars: int) -> frozenset[int]:
    """Decode an integer bitmask into a coalition."""
    if not isinstance(index, int):
        raise TypeError("index must be an integer.")

    maximum_index = (1 << n_vars) - 1
    if not 0 <= index <= maximum_index:
        raise ValueError(f"index must be between 0 and {maximum_index}.")

    return frozenset(
        variable
        for variable in range(n_vars)
        if index & (1 << variable)
    )


# capacity mapping
def map_capacities(capacities: CapacityMap, n_vars: int) -> dict[int, float]:
    """Map every coalition bitmask to its capacity value."""
    expected_non_empty = (1 << n_vars) - 1
    if len(capacities) != expected_non_empty:
        raise ValueError(
            "The number of capacity values must equal the number "
            f"of non-empty subsets: expected {expected_non_empty}, "
            f"received {len(capacities)}."
        )

    mapping: dict[int, float] = {0: 0.0}
    for capacity_value in capacities:
        encoding = subset_encoding(capacity_value.coalition, n_vars)
        if encoding in mapping:
            raise ValueError(
                f"Duplicate coalition: {capacity_value.coalition}."
            )
        mapping[encoding] = float(capacity_value.value)

    expected_indices = set(range(1 << n_vars))
    missing_indices = expected_indices - set(mapping)
    if missing_indices:
        missing_coalitions = [
            subset_decoding(index, n_vars)
            for index in sorted(missing_indices)
        ]
        raise ValueError(f"Missing coalitions: {missing_coalitions}.")

    return mapping
