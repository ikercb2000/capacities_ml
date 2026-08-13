# imports
from __future__ import annotations
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray

# constraint aliases
FloatArray = NDArray[np.float64]
ConstraintFunction = Callable[[FloatArray], ArrayLike]
JacobianFunction = Callable[[FloatArray], ArrayLike]

# constraint vector conversion
def _as_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} cannot contain NaN values.")
    return array
