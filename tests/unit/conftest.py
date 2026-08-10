import numpy as np
import pytest


@pytest.fixture
def binary_classification_sample() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.8, 0.0],
            [1.0, 1.0],
            [0.9, 0.2],
            [0.1, 0.8],
        ]
    )
    y = np.array(
        ["negative", "negative", "positive", "positive", "positive", "negative"]
    )
    return X, y
