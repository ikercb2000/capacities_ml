# imports
from dataclasses import dataclass
from numpy.typing import ArrayLike

# modules
from capacities_ml.risk.distortions import Distortion
from capacities_ml.risk.measures.utils import distortion_risk_measure


# reusable distortion measure
@dataclass(frozen=True, slots=True)
class DistortionRiskMeasure:
    """Callable distortion risk measure suitable for temporal estimators."""

    distortion: Distortion

    def __post_init__(self) -> None:
        if not isinstance(self.distortion, Distortion):
            raise TypeError("distortion must be a Distortion instance.")

    def __call__(
        self,
        losses: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        return distortion_risk_measure(
            losses,
            self.distortion,
            sample_weight=sample_weight,
        )
