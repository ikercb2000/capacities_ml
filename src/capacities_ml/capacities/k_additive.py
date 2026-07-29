# imports
from dataclasses import dataclass

# modules
from capacities_ml.capacities.base import Capacity
from capacities_ml.capacities.types import CapacityMap
from capacities_ml.capacities.validation import check_k_additivity, check_monotonicity

# k-additive capacity
@dataclass
class KAdditiveCapacity(Capacity):
    k: int
    subset_values: CapacityMap
    n_features: int
    feature_names: tuple[str, ...]

    def value(self, subset):
        return super().value(subset)

    def values(self):
        return super().values()

    def mobius_rep(self):
        return super().mobius_rep()

    def validate(self):
        check_monotonicity(self.subset_values, frozenset(range(self.n_features)))
        check_k_additivity(self.subset_values, self.k, self.n_features)
