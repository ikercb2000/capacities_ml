# imports
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.base.capacities.utils import _order_permutation


# common capacity base class
class BaseCapacity(ABC):
    """Common interface for explicit and dynamically evaluated capacities."""

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
