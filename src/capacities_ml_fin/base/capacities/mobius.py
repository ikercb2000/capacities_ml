# imports
from collections.abc import Iterable, Mapping
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities.base import BaseCapacity
from capacities_ml_fin.base.capacities.capacities import (
    ExplicitCapacity,
    VariableUniverse,
    resolve_universe,
)
from capacities_ml_fin.base.capacities.types import CoalitionValue, MobiusMap
from capacities_ml_fin.base.capacities.utils import (
    CoalitionInput,
    _event_mask,
    _mobius_transform_map,
    _order_permutation,
    named_coalition,
    normalize_coalition,
    powerset,
)
from capacities_ml_fin.base.capacities.validation import check_mobius_capacity


# Mobius capacity class
class MobiusCapacity(BaseCapacity):
    """Sparse Möbius representation of an evaluable finite capacity.

    Parameters
    ----------
    coefficients : mapping
        Möbius coefficients keyed by coalitions. Unspecified coefficients are
        interpreted as zero; the empty coalition must not be supplied.
    universe : VariableUniverse, optional
        Existing universe. Mutually exclusive with ``n_elements`` and
        ``var_names``.
    n_elements : int, optional
        Universe size when names are not supplied.
    var_names : iterable of str, optional
        Ordered variable names.
    validation_tolerance : float, default=1e-9
        Numerical tolerance for normalization and monotonicity checks.

    Attributes
    ----------
    universe : VariableUniverse
        Resolved event universe.
    validation_tolerance : float
        Tolerance used during validation.

    Notes
    -----
    :meth:`value` returns a Möbius coefficient ``m(A)`` whereas
    :meth:`event_value` reconstructs ``mu(A) = sum_{B subseteq A} m(B)``.
    The full capacity table is not stored.
    """

    def __init__(
        self,
        coefficients: Mapping[CoalitionInput, float],
        *,
        universe: VariableUniverse | None = None,
        n_elements: int | None = None,
        var_names: Iterable[str] | None = None,
        validation_tolerance: float = 1e-9,
    ) -> None:
        self.universe = resolve_universe(
            coefficients,
            universe=universe,
            n_elements=n_elements,
            var_names=var_names,
        )
        self.validation_tolerance = validation_tolerance
        self._coefficient_map = MobiusMap(
            mobius_coefficients=[
                CoalitionValue(
                    normalize_coalition(self.universe, coalition),
                    value,
                )
                for coalition, value in coefficients.items()
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
        """Return a Mobius coefficient, or zero when it is not specified."""
        normalized_coalition = normalize_coalition(self.universe, coalition)
        value = self._coefficient_map.get_value(normalized_coalition)
        return 0.0 if value is None else value

    def event_value(self, event: ArrayLike) -> float:
        """Evaluate the capacity of an event from its Mobius coefficients."""
        mask = _event_mask(event, self.n_elements)
        coalition = frozenset(int(index) for index in np.flatnonzero(mask))
        return float(
            sum(
                coefficient.value
                for coefficient in self._coefficient_map
                if coefficient.coalition <= coalition
            )
        )

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        """Evaluate nested events without expanding the capacity table."""
        permutation = _order_permutation(order, self.n_elements)
        ranks = np.empty(self.n_elements, dtype=int)
        ranks[permutation] = np.arange(self.n_elements)
        changes = np.zeros(self.n_elements + 1, dtype=float)

        for coefficient in self._coefficient_map:
            coalition = coefficient.coalition
            boundary = (
                self.n_elements - 1
                if not coalition
                else min(ranks[index] for index in coalition)
            )
            changes[0] += coefficient.value
            changes[boundary + 1] -= coefficient.value

        return np.cumsum(changes[:-1])

    def validate(self) -> None:
        """Check normalization and monotonicity of the Mobius capacity."""
        check_mobius_capacity(
            self._coefficient_map,
            self.n_elements,
            tolerance=self.validation_tolerance,
        )

    def to_named_dict(self) -> dict[tuple[str, ...], float]:
        """Return the specified coefficients keyed by variable names."""
        return {
            named_coalition(self.universe, coalition): value
            for coalition, value in self._coefficient_map.lookup_dict.items()
        }


# Mobius transform
def mobius_transform(capacity: ExplicitCapacity) -> MobiusCapacity:
    """Compute the Mobius representation of an explicit capacity."""
    if not isinstance(capacity, ExplicitCapacity):
        raise TypeError("capacity must be an ExplicitCapacity instance.")

    coefficient_map = _mobius_transform_map(capacity.values)
    return MobiusCapacity(
        coefficients=coefficient_map.to_lookup(),
        universe=capacity.universe,
    )


# inverse Mobius transform
def inverse_mobius_transform(
    mobius_capacity: MobiusCapacity,
) -> ExplicitCapacity:
    """Compute an explicit capacity from its Mobius representation."""
    if not isinstance(mobius_capacity, MobiusCapacity):
        raise TypeError("mobius_capacity must be a MobiusCapacity instance.")

    features = frozenset(range(mobius_capacity.n_elements))
    values: dict[frozenset[int], float] = {}
    for coalition in powerset(features):
        if coalition:
            values[coalition] = sum(
                mobius_capacity.value(subset)
                for subset in powerset(coalition)
            )

    return ExplicitCapacity(values=values, universe=mobius_capacity.universe)
