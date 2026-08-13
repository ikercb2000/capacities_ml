# imports
import numpy as np
from numpy.typing import ArrayLike


# probability validation
def _probabilities(probability: ArrayLike) -> tuple[np.ndarray, bool]:
    values = np.asarray(probability, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be finite.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")
    return values, values.ndim == 0


# scalar result restoration
def _result(values: np.ndarray, scalar: bool) -> float | np.ndarray:
    return float(values) if scalar else values


# distortion validation
def validate_distortion(distortion: object) -> None:
    """Validate a distortion through its public numerical contract."""
    from capacities_ml.risk.distortions.distortions import Distortion

    if not isinstance(distortion, Distortion):
        raise TypeError("distortion must be a Distortion instance.")
    distortion.validate()
