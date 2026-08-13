# imports
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

# penalty selection mask
def selection_mask(selection: ArrayLike, size: int) -> NDArray[np.bool_]:
    """Convert indices or a Boolean selector to a validated parameter mask."""
    array = np.asarray(selection)
    if array.dtype == bool:
        if array.shape != (size,):
            raise ValueError("Boolean penalty mask has an incompatible shape.")
        return array.copy()
    indices = np.asarray(selection, dtype=int).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("Penalty indices are outside the parameter vector.")
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask
