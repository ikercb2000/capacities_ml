# imports
import numpy as np
from numpy.typing import ArrayLike


# loss vector validation
def _loss_vector(losses: ArrayLike) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("losses must contain only finite values.")
    return values
