# imports
import numpy as np

# transform array as vector
def as_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert the input into a one-dimensional numeric vector.
    """
    vector = np.asarray(x, dtype=float)

    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    return vector
