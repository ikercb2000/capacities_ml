# imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np

# modules

# optimization result data object
@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Solver result returned by every optimization backend."""
    parameters: np.ndarray
    objective_value: float
    success: bool
    status: str
    message: str = ""
    n_iterations: int | None = None
    n_function_evaluations: int | None = None
    runtime_seconds: float | None = None
    solver_name: str | None = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=float)
        if parameters.ndim != 1:
            raise ValueError("parameters must be one-dimensional.")
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "objective_value", float(self.objective_value))

    @property
    def x(self) -> np.ndarray:
        """Alias matching SciPy and pymoo terminology."""
        return self.parameters.copy()

    @property
    def fun(self) -> float:
        """Alias matching SciPy terminology."""
        return self.objective_value
