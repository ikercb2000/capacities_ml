# imports
from __future__ import annotations

from functools import partial

import numpy as np

# modules
from capacities_ml_fin.ml.optimization.backends.to_scipy import ScipyOptimizer
from capacities_ml_fin.ml.optimization.problem import OptimizationProblem
from capacities_ml_fin.ml.optimization.result import OptimizationResult


# squared distance from a numerical candidate
def _squared_distance(parameters: np.ndarray, *, target: np.ndarray) -> float:
    difference = parameters - target
    return float(0.5 * difference @ difference)


# squared-distance gradient
def _squared_distance_gradient(
    parameters: np.ndarray,
    *,
    target: np.ndarray,
) -> np.ndarray:
    return parameters - target


# feasibility projection
def project_to_feasible(
    problem: OptimizationProblem,
    parameters: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 2_000,
) -> OptimizationResult:
    """Project a numerical candidate onto the problem's feasible set."""
    candidate = np.asarray(parameters, dtype=float).reshape(-1)
    if candidate.shape != (problem.n_parameters,):
        raise ValueError("Projection candidate has an incompatible shape.")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Projection candidate must contain only finite values.")

    projection_problem = OptimizationProblem(
        objective=partial(_squared_distance, target=candidate),
        gradient=partial(_squared_distance_gradient, target=candidate),
        initial_parameters=candidate,
        bounds=problem.bounds,
        linear_constraints=problem.linear_constraints,
        nonlinear_constraints=problem.nonlinear_constraints,
        layout=problem.layout,
        name=f"{problem.name}_feasibility_projection",
        metadata=dict(problem.metadata),
    )
    return ScipyOptimizer(
        method="SLSQP",
        tolerance=tolerance,
        options={"maxiter": max_iterations, "ftol": tolerance},
    ).solve(projection_problem)
