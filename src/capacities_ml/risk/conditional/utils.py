# imports
import numpy as np


# residual resampling
def _bootstrap_residuals(
    residuals: np.ndarray,
    max_scenarios: int | None,
    random_state: int | None,
) -> np.ndarray:
    if max_scenarios is None or max_scenarios >= residuals.size:
        return residuals
    generator = np.random.default_rng(random_state)
    indices = generator.choice(
        residuals.size,
        size=max_scenarios,
        replace=True,
    )
    return residuals[indices]
