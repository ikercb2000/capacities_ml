# imports
from __future__ import annotations
from dataclasses import dataclass

# modules
from capacities_ml.optimization.constraints import ConstraintBundle
from capacities_ml.optimization.enums import CapacityRepresentation

# capacity parameterization
@dataclass(frozen=True, slots=True)
class CapacityParameterization:
    """Capacity metadata combined with a generic constraint bundle."""

    constraints: ConstraintBundle
    parameter_masks: tuple[int, ...]
    representation: CapacityRepresentation
    n_features: int
    max_order: int

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_masks)
