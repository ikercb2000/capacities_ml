import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))


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
