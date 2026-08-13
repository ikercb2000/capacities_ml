# imports
import numpy as np
from numpy.typing import ArrayLike

# probability weight normalization
def _probability_weights(weights: ArrayLike) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have a positive sum.")
    return values / total
