# imports
from dataclasses import dataclass

# modules
from capacities_ml.capacities.capacities import Capacity
from capacities_ml.capacities.validation import check_k_additivity

# k-Additive capacity dataclass
@dataclass
class KAdditiveCapacity(Capacity):
    k: int
    def validate(self) -> None:
        super().validate()
        check_k_additivity(self._subset_values, self.k, self.n_vars)
