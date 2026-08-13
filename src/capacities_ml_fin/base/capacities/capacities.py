# imports
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities.base import BaseCapacity
from capacities_ml_fin.base.capacities.types import CapacityMap, CoalitionValue
from capacities_ml_fin.base.capacities.utils import (
    CoalitionInput,
    _event_mask,
    named_coalition,
    normalize_coalition,
)
from capacities_ml_fin.base.capacities.validation import check_monotonicity

# variable universe class
@dataclass(frozen=True, slots=True)
class VariableUniverse:
    """Immutable names and index mapping for a finite universe.

    Parameters
    ----------
    var_names : iterable of str
        Unique, non-empty variable names in their canonical order.

    Attributes
    ----------
    var_names : tuple of str
        Canonical ordered variable names.
    n_elements : int
        Size of the universe.
    name_to_index : mapping of str to int
        Read-only lookup from each name to its zero-based index.
    """

    var_names: Iterable[str]
    n_elements: int = field(init=False)
    name_to_index: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        var_names = tuple(self.var_names)
        object.__setattr__(self, "var_names", var_names)
        object.__setattr__(self, "n_elements", len(var_names))

        if self.n_elements < 1:
            raise ValueError("var_names must contain at least one variable.")
        if not all(isinstance(name, str) and name for name in var_names):
            raise ValueError("var_names must contain non-empty strings.")
        if len(set(var_names)) != self.n_elements:
            raise ValueError("var_names must be unique.")

        object.__setattr__(
            self,
            "name_to_index",
            MappingProxyType(
                {name: index for index, name in enumerate(var_names)}
            ),
        )

    def __reduce__(self) -> tuple[object, tuple[tuple[str, ...]]]:
        """Reconstruct without serializing the read-only mapping proxy."""
        return type(self), (tuple(self.var_names),)

    @classmethod
    def from_size(cls, n_elements: int) -> "VariableUniverse":
        """Create a universe with generated names ``x0``, ``x1``, ... ."""
        if isinstance(n_elements, bool) or not isinstance(n_elements, Integral):
            raise TypeError("n_elements must be an integer.")
        if int(n_elements) < 1:
            raise ValueError("n_elements must be positive.")
        return cls(tuple(f"x{index}" for index in range(int(n_elements))))

    @classmethod
    def from_coalitions(
        cls,
        coalitions: Iterable[CoalitionInput],
    ) -> "VariableUniverse":
        """Infer the smallest universe containing all observed members."""
        observed: list[int | str] = []
        seen: set[int | str] = set()
        for coalition in coalitions:
            members = (
                (coalition,)
                if isinstance(coalition, (str, Integral))
                else tuple(coalition)
            )
            for member in members:
                if (
                    isinstance(member, bool)
                    or not isinstance(member, (str, Integral))
                ):
                    raise TypeError(
                        "Coalitions must contain only variable names or "
                        "integer indices."
                    )
                normalized_member = (
                    int(member) if isinstance(member, Integral) else member
                )
                if normalized_member not in seen:
                    seen.add(normalized_member)
                    observed.append(normalized_member)

        if not observed:
            raise ValueError(
                "The universe cannot be inferred from empty coalitions. "
                "Provide n_elements or var_names."
            )
        if all(isinstance(member, str) for member in observed):
            return cls(tuple(observed))
        if all(isinstance(member, int) for member in observed):
            if any(member < 0 for member in observed):
                raise ValueError("Variable indices must be non-negative.")
            return cls.from_size(max(observed) + 1)
        raise ValueError(
            "A universe cannot be inferred from mixed names and indices. "
            "Provide universe or var_names explicitly."
        )


# universe resolution
def resolve_universe(
    coalitions: Iterable[CoalitionInput],
    *,
    universe: VariableUniverse | None = None,
    n_elements: int | None = None,
    var_names: Iterable[str] | None = None,
) -> VariableUniverse:
    """Resolve an explicit universe or infer the smallest observed universe."""
    specified = sum(
        option is not None
        for option in (universe, n_elements, var_names)
    )
    if specified > 1:
        raise ValueError(
            "Use only one of universe, n_elements or var_names."
        )
    if universe is not None:
        if not isinstance(universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")
        return universe
    if var_names is not None:
        return VariableUniverse(var_names)
    if n_elements is not None:
        return VariableUniverse.from_size(n_elements)
    return VariableUniverse.from_coalitions(coalitions)


# explicit capacity class
class ExplicitCapacity(BaseCapacity):
    """Explicit normalized monotone capacity over a finite universe.

    Parameters
    ----------
    values : mapping
        Capacity values keyed by coalitions expressed as names, indices, or
        iterables thereof. Every non-empty coalition must be present; the empty
        coalition is implicit and has value zero.
    universe : VariableUniverse, optional
        Existing universe. Mutually exclusive with ``n_elements`` and
        ``var_names``.
    n_elements : int, optional
        Universe size used to generate names ``x0``, ``x1``, and so on.
    var_names : iterable of str, optional
        Ordered variable names. If no universe argument is supplied, the
        smallest universe is inferred from ``values``.

    Attributes
    ----------
    universe : VariableUniverse
        Resolved event universe.
    values : CapacityMap
        Immutable table of coalition values.

    Raises
    ------
    ValueError
        If the table is incomplete, non-normalized, or non-monotone.
    """

    def __init__(
        self,
        values: Mapping[CoalitionInput, float],
        *,
        universe: VariableUniverse | None = None,
        n_elements: int | None = None,
        var_names: Iterable[str] | None = None,
    ) -> None:
        self.universe = resolve_universe(
            values,
            universe=universe,
            n_elements=n_elements,
            var_names=var_names,
        )
        self.values = CapacityMap(
            capacities=[
                CoalitionValue(normalize_coalition(self.universe, coalition), value)
                for coalition, value in values.items()
            ]
        )
        self.validate()
        self._freeze()

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.universe.var_names)

    @property
    def n_elements(self) -> int:
        return self.universe.n_elements

    def value(self, coalition: CoalitionInput) -> float:
        """Return the value of a coalition expressed by names or indices."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self.values.get_value(normalized_coalition)
        if value is None:
            raise KeyError(f"Unknown coalition: {coalition}.")
        return value

    def event_value(self, event: ArrayLike) -> float:
        """Return the stored value of a boolean event mask."""
        mask = _event_mask(event, self.n_elements)
        coalition = frozenset(int(index) for index in np.flatnonzero(mask))
        return self.value(coalition)

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the tabular capacity values keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self.values.lookup_dict.items()
        }

    def validate(self) -> None:
        """Check the mathematical invariants of the capacity."""
        check_monotonicity(
            self.values,
            frozenset(range(self.n_elements)),
        )
