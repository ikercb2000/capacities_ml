# imports
from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from typing import Dict, Tuple

# modules
from capacities_ml.capacities.types import CapacityMap, MobiusMap
from capacities_ml.capacities.mobius import mobius_transform

# capacity
class Capacity(ABC):
    n_features: int
    subset_values: CapacityMap
    feature_names: Tuple[str]

    @abstractmethod
    def value(self, subset: AbstractSet[int]) -> float:
        """Get the capacity value for any coalition."""
        return self.subset_values.get_value(subset)

    @abstractmethod
    def values(self) -> Dict[str, float]:
        """Get the capacity values for all possible coalitions."""
        return self.subset_values.str_map()

    @abstractmethod
    def mobius_rep(self) -> MobiusMap:
        """Get the mobius representation for all possible coalitions."""
        return mobius_transform(self.subset_values, self.n_features)

    @abstractmethod
    def validate(self) -> None:
        """Check whether this mapping is a valid capacity."""

    
