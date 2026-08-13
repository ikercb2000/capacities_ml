# imports
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities.utils import _order_permutation


# common capacity base class
class BaseCapacity(ABC):
    """Abstract interface for immutable finite capacities.

    A capacity assigns a normalized, monotone value to every event in a finite
    universe. Concrete implementations may store a full table, sparse Möbius
    coefficients, or compute event values dynamically. Algorithms consume this
    interface through :meth:`event_value` and never require materialization.

    Attributes
    ----------
    n_elements : int
        Number of elements in the event universe.

    Notes
    -----
    Capacity instances become immutable after successful validation. Subclasses
    should call ``self._freeze()`` at the end of construction.
    """

    _capacity_frozen = False

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_capacity_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; create a new capacity instead."
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_capacity_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; create a new capacity instead."
            )
        object.__delattr__(self, name)

    def _freeze(self) -> None:
        """Prevent changes after successful construction and validation."""
        object.__setattr__(self, "_capacity_frozen", True)

    @property
    @abstractmethod
    def n_elements(self) -> int:
        """Number of elements in the finite capacity universe."""

    @abstractmethod
    def event_value(self, event: ArrayLike) -> float:
        """Return the capacity value of a boolean event mask."""

    def nested_event_values(self, order: ArrayLike) -> np.ndarray:
        """Evaluate the nested upper sets induced by an element ordering."""
        permutation = _order_permutation(order, self.n_elements)
        values = np.empty(self.n_elements, dtype=float)
        event = np.zeros(self.n_elements, dtype=bool)

        for position in range(self.n_elements - 1, -1, -1):
            event[permutation[position]] = True
            values[position] = self.event_value(event)

        return values
