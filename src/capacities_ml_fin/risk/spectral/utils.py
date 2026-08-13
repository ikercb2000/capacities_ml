# imports
import numpy as np
from numpy.typing import ArrayLike


# spectral weight normalization
def _normalized_weights(weights: ArrayLike, size: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"weights must contain {size} finite values.")
    if np.any(values < 0.0) or np.sum(values) <= 0.0:
        raise ValueError("weights must be non-negative with a positive sum.")
    return values / np.sum(values)
